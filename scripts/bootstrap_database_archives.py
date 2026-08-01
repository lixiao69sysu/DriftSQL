#!/usr/bin/env python3
"""Download and extract the separately hosted BIRD SQLite archives."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _download(config: dict[str, Any], archive: Path) -> None:
    expected = int(config["database_archive_size_bytes"])
    if archive.is_file() and archive.stat().st_size == expected:
        return
    if archive.exists() and archive.stat().st_size > expected:
        raise RuntimeError(f"Archive is larger than expected: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable database downloads")
    _run(
        curl,
        "-L",
        "--fail",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(archive),
        str(config["database_archive_url"]),
    )
    if archive.stat().st_size != expected:
        raise RuntimeError(
            f"Archive size mismatch for {archive}: expected {expected}, "
            f"got {archive.stat().st_size}"
        )


def _extract_bird23(config: dict[str, Any], root: Path) -> None:
    archive = root / "train.zip"
    _download(config, archive)
    target = root / "full" / "train"
    nested = target / "train_databases.zip"
    if not nested.is_file() or nested.stat().st_size != int(config["nested_archive_size_bytes"]):
        target.parent.mkdir(parents=True, exist_ok=True)
        _run("unzip", "-q", "-o", str(archive), "train/*", "-d", str(target.parent))
    database_root = target / "train_databases"
    database_count = len(list(database_root.glob("*/*.sqlite")))
    if database_count != int(config["expected_database_count"]):
        _run(
            "unzip",
            "-q",
            "-o",
            str(nested),
            "train_databases/*",
            "-x",
            "__MACOSX/*",
            "-d",
            str(target),
        )
    database_count = len(list(database_root.glob("*/*.sqlite")))
    expected = int(config["expected_database_count"])
    if database_count != expected:
        raise RuntimeError(
            f"BIRD23 database count mismatch: expected {expected}, got {database_count}"
        )


def _extract_mini_dev(config: dict[str, Any], root: Path) -> None:
    archive = root / "dev.zip"
    _download(config, archive)
    target = root / "full" / "dev_20240627"
    nested = target / "dev_databases.zip"
    if not nested.is_file() or nested.stat().st_size != int(config["nested_archive_size_bytes"]):
        target.parent.mkdir(parents=True, exist_ok=True)
        _run("unzip", "-q", "-o", str(archive), "-d", str(target.parent))
    database_root = target / "dev_databases"
    database_count = len(list(database_root.glob("*/*.sqlite")))
    if database_count != int(config["expected_database_count"]):
        _run("unzip", "-q", "-o", str(nested), "-d", str(target))
    database_count = len(list(database_root.glob("*/*.sqlite")))
    expected = int(config["expected_database_count"])
    if database_count != expected:
        raise RuntimeError(
            f"Mini-Dev database count mismatch: expected {expected}, got {database_count}"
        )


def main() -> None:
    lock = json.loads((PROJECT_ROOT / "datasets.lock.json").read_text(encoding="utf-8"))
    bird23 = lock["bird23_train_filtered"]
    mini_dev = lock["bird_mini_dev"]
    _extract_bird23(bird23, PROJECT_ROOT / bird23["local_dir"])
    _extract_mini_dev(mini_dev, PROJECT_ROOT / mini_dev["local_dir"])
    print("BIRD database archives are downloaded and extracted.")


if __name__ == "__main__":
    main()
