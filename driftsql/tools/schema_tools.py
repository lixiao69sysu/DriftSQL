"""Schema version and diff tools exposed to the agent."""

from __future__ import annotations

from driftsql.drift import SchemaDiff


def inspect_schema_diff(diff: SchemaDiff) -> dict[str, object]:
    """Return an auditable, JSON-serializable schema change observation."""

    return diff.to_observation()
