"""Report the runtime components needed before any expensive training run."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def module_version(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def git_revision(path: Path) -> str:
    if not (path / ".git").exists():
        return "missing"
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "modules": {
            name: module_version(name)
            for name in [
                "torch",
                "vllm",
                "ray",
                "verl",
                "tensordict",
                "transformers",
                "peft",
                "datasets",
                "sqlglot",
                "pydantic",
            ]
        },
        "frameworks": {
            name: git_revision(PROJECT_ROOT / "third_party" / name)
            for name in ["BIRD-RL", "BIRD-Interact", "verl"]
        },
    }

    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 1),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:
        report["cuda"] = {"available": False, "error": f"{type(error).__name__}: {error}"}

    print(json.dumps(report, indent=2, ensure_ascii=False))

    missing = [name for name, state in report["modules"].items() if not state["available"]]
    if missing:
        raise SystemExit("Missing runtime modules: {}".format(", ".join(missing)))


if __name__ == "__main__":
    main()
