#!/usr/bin/env python3
"""Download the pinned base model and verify its local snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        default="base_model",
        help="Entry in models.lock.json to download (default: base_model).",
    )
    args = parser.parse_args()

    lock = json.loads((PROJECT_ROOT / "models.lock.json").read_text(encoding="utf-8"))
    if args.model_key not in lock:
        parser.error(
            f"Unknown model key {args.model_key!r}; choose one of: "
            + ", ".join(sorted(lock))
        )
    config = lock[args.model_key]
    destination = PROJECT_ROOT / config["local_dir"]

    downloaded = Path(
        snapshot_download(
            repo_id=config["repo_id"],
            revision=config["revision"],
            local_dir=destination,
        )
    )
    weight_files = sorted(downloaded.glob("*.safetensors"))
    actual_size = sum(path.stat().st_size for path in downloaded.rglob("*") if path.is_file())
    report = {
        "model_key": args.model_key,
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "path": str(downloaded),
        "weight_shards": len(weight_files),
        "downloaded_size_bytes": actual_size,
        "config_present": (downloaded / "config.json").is_file(),
        "tokenizer_present": (downloaded / "tokenizer.json").is_file(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not weight_files or not report["config_present"] or not report["tokenizer_present"]:
        raise SystemExit("Base model snapshot is incomplete")


if __name__ == "__main__":
    main()
