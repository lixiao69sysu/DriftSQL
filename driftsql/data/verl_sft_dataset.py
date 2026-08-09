"""VERL SFT dataset compatibility for nested Arrow tool schemas."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation


class NestedToolsSFTDataset(MultiTurnSFTDataset):
    """Read nested messages/tools with pandas objects instead of ArrowDtype.

    PyArrow-backed pandas frames can abort inside ``iloc`` when a list-of-struct
    column has non-zero child offsets.  VERL accesses every row through iloc,
    so the default object backend is the safe representation for tool schemas.
    """

    def _read_files_and_process(self) -> None:
        dataframes = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {total}")
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"selected {self.max_samples} random samples out of {total}")

        self.messages = self.dataframe[self.messages_key].apply(
            convert_nested_value_to_list_recursive
        ).tolist()
        if self.tools_key in self.dataframe.columns:
            def normalize_tools(value):
                value = convert_nested_value_to_list_recursive(value)
                return json.loads(value) if isinstance(value, str) else value

            self.tools = self.dataframe[self.tools_key].apply(normalize_tools).tolist()
        else:
            self.tools = None
        self.enable_thinking = (
            self.dataframe[self.enable_thinking_key].tolist()
            if self.enable_thinking_key in self.dataframe.columns
            else None
        )
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )


class LastAssistantSFTDataset(NestedToolsSFTDataset):
    """Supervise only the final assistant action in each conversation prefix."""

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant":
            last_assistant = max(
                position
                for position, candidate in enumerate(full_message)
                if candidate["role"] == "assistant"
            )
            if index != last_assistant:
                loss_mask.zero_()
        return input_ids, loss_mask, attention_mask, inputs


class ToolBoundarySFTDataset(LastAssistantSFTDataset):
    """Train only native tool-call boundary markers and the assistant EOS.

    This curriculum stage preserves the SQL/tool arguments learned by the
    previous adapter while giving the three protocol boundary tokens enough
    weight to prevent whole-trajectory emission or premature termination.
    """

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant" and loss_mask.any():
            marker_ids = {
                self.tokenizer.convert_tokens_to_ids("<tool_call>"),
                self.tokenizer.convert_tokens_to_ids("</tool_call>"),
                self.tokenizer.eos_token_id,
            }
            boundary_mask = input_ids == -1
            for token_id in marker_ids:
                boundary_mask |= input_ids == token_id
            loss_mask.mul_(boundary_mask.to(loss_mask.dtype))
            if int(loss_mask.sum()) != 3:
                raise ValueError(
                    f"Expected 3 supervised tool boundary tokens, got {int(loss_mask.sum())}"
                )
        return input_ids, loss_mask, attention_mask, inputs


class JsonActionSFTDataset(LastAssistantSFTDataset):
    """Supervise the final plain-JSON action and EOS, excluding its think text."""

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant" and loss_mask.any():
            # Exclude the closing quote: Qwen may merge it with the following
            # colon, while the ``{"name`` prefix has stable token boundaries.
            marker = self.tokenizer.encode('{"name', add_special_tokens=False)
            values = input_ids.tolist()
            starts = [
                position
                for position in range(len(values) - len(marker) + 1)
                if values[position : position + len(marker)] == marker
            ]
            if len(starts) != 1:
                raise ValueError(f"Expected one plain JSON action marker, got {len(starts)}")
            loss_mask[: starts[0]] = 0
        return input_ids, loss_mask, attention_mask, inputs


class ActionNameSFTDataset(LastAssistantSFTDataset):
    """Supervise only the selected tool-name tokens in a plain-JSON action.

    Recovery examples primarily correct action selection. Long SQL arguments
    otherwise dilute the few decisive tokens, so this objective keeps the
    on-policy state and verified target while treating the action as a
    classification-like intervention.
    """

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant" and loss_mask.any():
            try:
                payload = json.loads(str(message.get("content", "")).rsplit("\n", 1)[-1])
                target_action = str(payload["name"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("Final assistant message lacks a plain-JSON action") from error
            marker = self.tokenizer.encode(target_action, add_special_tokens=False)
            values = input_ids.tolist()
            starts = [
                position
                for position in range(len(values) - len(marker) + 1)
                if values[position : position + len(marker)] == marker
                and bool(loss_mask[position : position + len(marker)].all())
            ]
            if len(starts) != 1:
                raise ValueError(
                    f"Expected one supervised {target_action} marker, got {len(starts)}"
                )
            focused = loss_mask == -1
            start = starts[0]
            focused[start : start + len(marker)] = True
            loss_mask.mul_(focused.to(loss_mask.dtype))
        return input_ids, loss_mask, attention_mask, inputs


class DecisionPrefixSFTDataset(LastAssistantSFTDataset):
    """Supervise the recovery rationale and action name, but not arguments.

    Action-name-only teacher forcing leaks the answer through the preceding
    golden ``<think>`` text: predicting ``execute_sql`` is trivial after the
    model has already been handed a rationale saying it will execute SQL.  At
    rollout time the model must generate that decision itself.  This objective
    therefore keeps every supervised token of the final assistant response up
    to and including the selected tool name, while masking the potentially
    long SQL/tool arguments that would otherwise dominate the correction.
    """

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant" and loss_mask.any():
            try:
                payload = json.loads(str(message.get("content", "")).rsplit("\n", 1)[-1])
                target_action = str(payload["name"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("Final assistant message lacks a plain-JSON action") from error
            marker = self.tokenizer.encode(target_action, add_special_tokens=False)
            values = input_ids.tolist()
            starts = [
                position
                for position in range(len(values) - len(marker) + 1)
                if values[position : position + len(marker)] == marker
                and bool(loss_mask[position : position + len(marker)].all())
            ]
            if len(starts) != 1:
                raise ValueError(
                    f"Expected one supervised {target_action} marker, got {len(starts)}"
                )
            supervised = loss_mask.nonzero().flatten()
            if supervised.numel() == 0:
                raise ValueError("Final assistant message has no supervised prefix")
            focused = loss_mask == -1
            start = int(supervised[0])
            end = starts[0] + len(marker)
            focused[start:end] = True
            loss_mask.mul_(focused.to(loss_mask.dtype))
        return input_ids, loss_mask, attention_mask, inputs


class TerminalActionNameSFTDataset(LastAssistantSFTDataset):
    """Supervise only the terminal tool-name tokens, not its long SQL argument.

    Terminal-action correction is a classification-like intervention.  If the
    copied SQL argument is included in the objective, hundreds of already-easy
    SQL tokens dilute the few tokens that decide between another retrieval and
    ``submit_solution``.
    """

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        if message["role"] == "assistant" and loss_mask.any():
            marker = self.tokenizer.encode("submit_solution", add_special_tokens=False)
            values = input_ids.tolist()
            starts = [
                position
                for position in range(len(values) - len(marker) + 1)
                if values[position : position + len(marker)] == marker
                and bool(loss_mask[position : position + len(marker)].all())
            ]
            if len(starts) != 1:
                raise ValueError(
                    f"Expected one supervised submit_solution marker, got {len(starts)}"
                )
            focused = loss_mask == -1
            start = starts[0]
            focused[start : start + len(marker)] = True
            loss_mask.mul_(focused.to(loss_mask.dtype))
        return input_ids, loss_mask, attention_mask, inputs
