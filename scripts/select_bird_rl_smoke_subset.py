#!/usr/bin/env python3
"""Select deterministic BIRD-RL rows by instance ID from a parquet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", required=True)
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    by_id = {row["extra_info"]["instance_id"]: row for row in rows}
    missing = [instance_id for instance_id in args.instance_id if instance_id not in by_id]
    if missing:
        raise SystemExit(f"Missing instance IDs: {missing}")
    selected = [by_id[instance_id] for instance_id in args.instance_id]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), args.output)
    print(f"Saved {len(selected)} rows to {args.output}: {args.instance_id}")


if __name__ == "__main__":
    main()
