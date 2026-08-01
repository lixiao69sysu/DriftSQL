#!/usr/bin/env python3
"""Generate one held-out first turn and validate BIRD-RL's tool-call syntax."""

from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from driftsql.tool_calls import find_tool_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--assistant-turn", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    row = pq.read_table(args.data).slice(args.index, 1).to_pylist()[0]
    prompt_messages = row["messages"][: 2 if args.assistant_turn == 1 else 4]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )[0, inputs.shape[-1] :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    calls = find_tool_calls(output)
    parsed = calls[0].as_dict() if calls else None
    valid = (
        isinstance(parsed, dict)
        and parsed.get("name") in {"execute_sql", "submit_solution"}
        and isinstance(parsed.get("arguments"), dict)
    )
    print(json.dumps({"valid_tool_call": valid, "parsed": parsed, "output": output}, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
