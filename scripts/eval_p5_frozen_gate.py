#!/usr/bin/env python3
"""Run the single frozen P5 candidate on the prepared one-shot Gate input."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_lifecycle(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_state_exclusive(path: Path, state: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replace_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def evaluation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another P5 Gate evaluation process is active") from error
        yield


def start_or_resume_evaluation(
    *,
    state_path: Path,
    lifecycle_path: Path,
    identity: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    if not state_path.exists():
        if resume:
            raise RuntimeError("Cannot resume: the P5 Gate evaluation has never started")
        state = {
            "protocol": "driftsql_p5_one_shot_eval_state_v1",
            "status": "started",
            "attempts": 1,
            "started_at_utc": now,
            **identity,
        }
        write_state_exclusive(state_path, state)
        event_name = "gate_eval_started"
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError(
                "P5 Gate evaluation has already started; only --resume may continue an interrupted run"
            )
        if state.get("status") != "started":
            raise RuntimeError("P5 Gate evaluation is already completed and cannot be rerun")
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"P5 Gate resume identity changed: {key}")
        state["attempts"] = int(state.get("attempts", 1)) + 1
        state["resumed_at_utc"] = now
        replace_state(state_path, state)
        event_name = "gate_eval_resumed"
    append_lifecycle(
        lifecycle_path,
        {
            "event": event_name,
            "at": now,
            "eval_state_sha256": sha256(state_path),
            "attempts": state["attempts"],
        },
    )
    return state


def complete_evaluation(
    *,
    state_path: Path,
    lifecycle_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "started":
        raise RuntimeError("P5 Gate evaluation state is not active")
    result_files = [output_dir / "summary.json", output_dir / "p5-frozen-gate.jsonl"]
    for path in result_files:
        if not path.is_file():
            raise FileNotFoundError(f"P5 Gate evaluator did not produce {path}")
    state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(UTC).isoformat()
    state["result_files_sha256"] = {path.name: sha256(path) for path in result_files}
    replace_state(state_path, state)
    append_lifecycle(
        lifecycle_path,
        {
            "event": "gate_eval_completed",
            "at": state["completed_at_utc"],
            "eval_state_sha256": sha256(state_path),
            "result_files_sha256": state["result_files_sha256"],
        },
    )
    return state


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
    for relative, expected in freeze["locked_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen P5 input changed before Gate evaluation: {relative}")
    gate_summary = json.loads((args.gate_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256(args.freeze) != gate_summary["candidate_freeze_sha256"]:
        raise RuntimeError("Gate input was prepared for a different candidate freeze")
    inference = freeze["candidate"]["inference"]
    if freeze["one_shot_gate"].get("allowed_candidate_runs") != 1:
        raise RuntimeError("Frozen P5 Gate policy does not allow exactly one candidate run")
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_stage7_process_isolated_eval.py"),
        "--data", str(args.gate_dir / "gate_agent_eval.jsonl"),
        "--output-dir", str(args.output_dir),
        "--adapter-alias", "p5-frozen-gate",
        "--adapter-path", str(ROOT / freeze["candidate"]["adapter_path"]),
        "--drift-type", "add_column",
        "--gpus", args.gpus,
        "--max-turns", str(inference["max_turns"]),
        "--max-new-tokens", str(inference["max_new_tokens"]),
        "--max-model-len", str(inference["max_model_len"]),
    ]
    if args.resume:
        command.append("--resume")
    state_path = args.freeze.parent / "gate_eval_state.json"
    lifecycle_path = args.freeze.parent / "gate_lifecycle.jsonl"
    identity = {
        "candidate_freeze_sha256": sha256(args.freeze),
        "gate_input_summary_sha256": sha256(args.gate_dir / "summary.json"),
        "gate_eval_input_sha256": sha256(args.gate_dir / "gate_agent_eval.jsonl"),
        "candidate": freeze["candidate"]["name"],
        "inference": inference,
        "gpus": args.gpus,
    }
    with evaluation_lock(args.freeze.parent / "gate_eval.lock"):
        start_or_resume_evaluation(
            state_path=state_path,
            lifecycle_path=lifecycle_path,
            identity=identity,
            resume=args.resume,
        )
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            append_lifecycle(
                lifecycle_path,
                {
                    "event": "gate_eval_interrupted",
                    "at": datetime.now(UTC).isoformat(),
                    "returncode": error.returncode,
                    "resume_required": True,
                },
            )
            raise
        complete_evaluation(
            state_path=state_path,
            lifecycle_path=lifecycle_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
