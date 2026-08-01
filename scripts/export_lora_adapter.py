#!/usr/bin/env python3
"""Extract a portable PEFT adapter from a VERL LoRA HF checkpoint."""

from __future__ import annotations

import argparse
import json

from driftsql.training import export_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-hf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-meta")
    args = parser.parse_args()
    result = export_lora_adapter(
        checkpoint_hf_dir=args.checkpoint_hf,
        output_dir=args.output,
        base_model_path=args.base_model,
        lora_meta_path=args.lora_meta,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
