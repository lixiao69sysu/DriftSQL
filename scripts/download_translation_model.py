#!/usr/bin/env python3
"""Download the pinned CPU Chinese-to-English translation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=args.output,
        allow_patterns=[
            "README.md",
            "*.json",
            "*.safetensors",
            "*.txt",
        ],
    )
    print(f"translation_model={path}")
    print(f"revision={MODEL_REVISION}")


if __name__ == "__main__":
    main()
