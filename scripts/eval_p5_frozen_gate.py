#!/usr/bin/env python3
"""Run the single frozen P5 candidate on the prepared one-shot Gate input."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "reports/p5/final_candidate/frozen_candidate.json")
    parser.add_argument("--gate-dir", type=Path, default=ROOT / "data/processed/p5_gate_eval")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/p5/gate_one_shot")
    parser.add_argument("--gpus", default="0,2,3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("protocol") != "driftsql_p5_tune_frozen_candidate_v1":
        raise RuntimeError("Invalid P5 candidate freeze")
    for relative, expected in freeze["candidate"]["adapter_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen adapter changed: {relative}")
    gate_summary = json.loads((args.gate_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256(args.freeze) != gate_summary["candidate_freeze_sha256"]:
        raise RuntimeError("Gate input was prepared for a different candidate freeze")
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_stage7_process_isolated_eval.py"),
        "--data", str(args.gate_dir / "gate_agent_eval.jsonl"),
        "--output-dir", str(args.output_dir),
        "--adapter-alias", "p5-frozen-gate",
        "--adapter-path", str(ROOT / freeze["candidate"]["adapter_path"]),
        "--drift-type", "add_column",
        "--gpus", args.gpus,
        "--max-turns", "7",
        "--max-new-tokens", "512",
        "--max-model-len", "8192",
    ]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

