#!/usr/bin/env python3
"""Verify that permanently sealed Stage 7 Gate106 artifacts are byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEAL = PROJECT_ROOT / "reports/stage8/stage7_gate106_seal.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("stage7_gate_rows_read") is not False:
        raise RuntimeError("Stage 7 Gate106 seal does not assert hash-only access")
    mismatches = {}
    for relative, expected in seal["files_sha256"].items():
        path = PROJECT_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"Stage 7 Gate106 seal violation: {mismatches}")
    return {"sealed": True, "files": len(seal["files_sha256"])}


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
