#!/usr/bin/env python3
"""Download pinned public datasets and run their integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from driftsql.data import audit_mini_interact

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset key from datasets.lock.json. Repeat for several; default: all.",
    )
    args = parser.parse_args()

    lock = json.loads((PROJECT_ROOT / "datasets.lock.json").read_text(encoding="utf-8"))
    selected = args.datasets or list(lock)
    unknown = sorted(set(selected) - set(lock))
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {', '.join(unknown)}")

    for name in selected:
        config = lock[name]
        destination = PROJECT_ROOT / config["local_dir"]
        downloaded = snapshot_download(
            repo_id=config["repo_id"],
            repo_type=config["repo_type"],
            revision=config["revision"],
            local_dir=destination,
        )
        report = {
            "dataset": name,
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "path": downloaded,
            "task_file_present": (destination / config["task_file"]).is_file(),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["task_file_present"]:
            raise SystemExit(f"{name}: task file is missing")

        if name == "mini_interact":
            audit = audit_mini_interact(destination)
            print(json.dumps(audit, indent=2, ensure_ascii=False))
            if audit["malformed_rows"] or audit["missing_assets"] or audit["invalid_databases"]:
                raise SystemExit("Mini-Interact integrity check failed")


if __name__ == "__main__":
    main()
