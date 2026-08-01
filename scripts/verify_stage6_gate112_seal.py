#!/usr/bin/env python3
"""Verify that sealed Stage 6 Gate112 artifacts remain byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEAL = PROJECT_ROOT / "reports/stage7/stage6_gate112_seal.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    mismatches = {}
    for relative, expected in seal["files_sha256"].items():
        actual = sha256(PROJECT_ROOT / relative)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Stage 6 Gate112 seal violation: {mismatches}")
    print(json.dumps({"sealed": True, "files": len(seal["files_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
