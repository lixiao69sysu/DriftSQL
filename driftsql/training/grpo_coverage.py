"""Coverage planning helpers for deterministic GRPO curricula."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CoveragePlan:
    train_rows: int
    train_batch_size: int
    steps_per_epoch: int
    covered_rows_per_epoch: int
    dropped_rows_per_epoch: int
    planned_steps: int
    planned_prompt_slots: int
    full_row_coverage: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def coverage_plan(
    train_rows: int,
    train_batch_size: int,
    planned_steps: int,
) -> CoveragePlan:
    """Describe the exact prompt coverage of VERL's drop-last dataloader."""

    if train_rows <= 0:
        raise ValueError("train_rows must be positive")
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if planned_steps <= 0:
        raise ValueError("planned_steps must be positive")
    steps_per_epoch = train_rows // train_batch_size
    covered_rows = steps_per_epoch * train_batch_size
    dropped_rows = train_rows - covered_rows
    planned_slots = planned_steps * train_batch_size
    return CoveragePlan(
        train_rows=train_rows,
        train_batch_size=train_batch_size,
        steps_per_epoch=steps_per_epoch,
        covered_rows_per_epoch=covered_rows,
        dropped_rows_per_epoch=dropped_rows,
        planned_steps=planned_steps,
        planned_prompt_slots=planned_slots,
        full_row_coverage=bool(dropped_rows == 0 and planned_steps >= steps_per_epoch),
    )


def unique_first_order(instance_ids: Iterable[str], *, seed: int) -> list[int]:
    """Place one row per task before repeats, shuffling both sections stably.

    With VERL's sequential sampler, every checkpoint before the repeat section
    has the widest possible independent-task coverage. All original rows still
    appear exactly once in the resulting epoch.
    """

    grouped: dict[str, list[int]] = {}
    for index, raw_id in enumerate(instance_ids):
        instance_id = str(raw_id)
        if not instance_id:
            raise ValueError(f"empty instance_id at row {index}")
        grouped.setdefault(instance_id, []).append(index)
    if not grouped:
        raise ValueError("instance_ids must not be empty")

    primaries = [indices[0] for indices in grouped.values()]
    repeats = [index for indices in grouped.values() for index in indices[1:]]
    rng = random.Random(seed)
    rng.shuffle(primaries)
    rng.shuffle(repeats)
    order = primaries + repeats
    if sorted(order) != list(range(len(order))):
        raise RuntimeError("coverage ordering lost or duplicated source rows")
    return order
