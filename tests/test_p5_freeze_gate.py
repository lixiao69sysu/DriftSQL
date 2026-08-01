from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def evaluation_rows(alias: str, successes: int, hard_successes: int) -> list[dict]:
    rows = []
    for index in range(18):
        success = index < hard_successes if index < 12 else index < 12 + (successes - hard_successes)
        rows.append(
            {
                "variant": alias,
                "instance_id": f"p5-tune-{index:02d}",
                "task_success": success,
                "termination_reason": "submitted" if success else "turn_limit",
                "safety": {"unsafe": False, "timed_out": False},
                "usage": {"model_calls": 5 if success else 7, "tool_calls": 5 if success else 7},
            }
        )
    return rows


def prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    data = tmp_path / "data"
    protocol = tmp_path / "protocol"
    reports = tmp_path / "reports"
    adapters = tmp_path / "adapters"
    output = tmp_path / "final" / "frozen_candidate.json"
    metadata = [
        {
            "extra_info": {
                "instance_id": f"p5-tune-{index:02d}",
                "p5_turn_limit_focus": index < 12,
            }
        }
        for index in range(18)
    ]
    write_jsonl(data / "tune_agent_eval.jsonl", metadata)
    write_json(data / "summary.json", {"protocol": "fixture"})
    write_json(
        protocol / "summary.json",
        {"gate": {"status": "sealed_unopened", "sha256": "f" * 64}},
    )
    variants = {
        "p5-sft20": (9, 4),
        "p5-grpo-step5": (10, 5),
        "p5-grpo-step10": (12, 7),
    }
    for alias, (successes, hard_successes) in variants.items():
        write_jsonl(reports / alias / f"{alias}.jsonl", evaluation_rows(alias, successes, hard_successes))
    adapter_paths = {
        "p5-sft20": adapters / "sft20",
        "p5-grpo-step5": adapters / "grpo/global_step_5/merged/lora_adapter",
        "p5-grpo-step10": adapters / "grpo/global_step_10/merged/lora_adapter",
    }
    for alias, path in adapter_paths.items():
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(f"fixture:{alias}".encode())
    return data, protocol, reports, adapters, output


def freeze_command(tmp_path: Path) -> tuple[list[str], Path, Path]:
    data, protocol, reports, adapters, output = prepare_fixture(tmp_path)
    command = [
        sys.executable,
        str(ROOT / "scripts/freeze_p5_candidate.py"),
        "--data-dir", str(data),
        "--protocol-dir", str(protocol),
        "--report-root", str(reports),
        "--sft20-adapter", str(adapters / "sft20"),
        "--grpo-root", str(adapters / "grpo"),
        "--output", str(output),
    ]
    return command, protocol, output


def test_p5_freeze_selects_tune_winner_without_opening_gate(tmp_path: Path) -> None:
    command, protocol, output = freeze_command(tmp_path)

    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert frozen["candidate"]["name"] == "p5-grpo-step10"
    assert frozen["tune_selection"]["acceptance"] == {
        "all_candidates_have_18_tune_tasks": True,
        "turn_limit_slice_has_12_tasks": True,
        "selected_overall_not_worse_than_sft20": True,
        "selected_hard_slice_not_worse_than_sft20": True,
        "selected_unsafe_eq_0": True,
        "selected_timeout_eq_0": True,
    }
    assert not (protocol / "sealed_gate.jsonl").exists()


def test_p5_gate_open_attempt_is_fail_closed_and_cannot_repeat(tmp_path: Path) -> None:
    command, protocol, output = freeze_command(tmp_path)
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    gate_command = [
        sys.executable,
        str(ROOT / "scripts/prepare_p5_gate_eval.py"),
        "--freeze", str(output),
        "--protocol-dir", str(protocol),
        "--output-dir", str(tmp_path / "gate-eval"),
    ]

    first = subprocess.run(gate_command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode != 0
    lifecycle = output.parent / "gate_lifecycle.jsonl"
    assert lifecycle.is_file()
    assert "gate_open_started" in lifecycle.read_text(encoding="utf-8")

    second = subprocess.run(gate_command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already been opened or attempted" in second.stderr
