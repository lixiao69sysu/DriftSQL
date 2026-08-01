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
