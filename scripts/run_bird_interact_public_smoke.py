#!/usr/bin/env python3
"""Run the public Mini-Interact SQLite environment and report scoring readiness."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = (
    PROJECT_ROOT
    / "third_party/BIRD-Interact/mini_interact/knowledge_based/mini_interact_agent"
)
sys.path.insert(0, str(AGENT_ROOT))

# Importing ``src.envs`` normally pulls every optional environment, including
# Gym. The Mini SQLite action handler only needs the nested test-case package.
src_package = types.ModuleType("src")
src_package.__path__ = [str(AGENT_ROOT / "src")]
envs_package = types.ModuleType("src.envs")
envs_package.__path__ = [str(AGENT_ROOT / "src/envs")]
sys.modules.setdefault("src", src_package)
sys.modules.setdefault("src.envs", envs_package)

from batch_run_bird_interact.action_handler_sqlite import (  # noqa: E402
    close_db_connection,
    execute_env_action,
    execute_submit_action,
)
from batch_run_bird_interact.sample_status import SampleStatus  # noqa: E402
from driftsql.data import audit_mini_interact  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/mini-interact",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/bird_interact_public_smoke.json",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.data_root / "mini_interact.jsonl")
    row = rows[args.index]
    status = SampleStatus(idx=args.index, original_data=row)

    schema, schema_ok = execute_env_action("get_schema()", status, str(args.data_root))
    knowledge_names_text, names_ok = execute_env_action(
        "get_all_external_knowledge_names()", status, str(args.data_root)
    )
    knowledge_names = ast.literal_eval(knowledge_names_text) if names_ok else []
    definition, definition_ok = ("", False)
    if knowledge_names:
        definition, definition_ok = execute_env_action(
            f"get_knowledge_definition({knowledge_names[0]!r})",
            status,
            str(args.data_root),
        )
    execution, execution_ok = execute_env_action("execute(SELECT 1)", status, str(args.data_root))
    submit_observation, submit_reward, phase1, phase2, finished = execute_submit_action(
        "SELECT 1", status, str(args.data_root)
    )

    db_id = row["selected_database"]
    db_path = args.data_root / db_id / f"{db_id}.sqlite"
    close_db_connection(str(db_path))

    audit = audit_mini_interact(args.data_root)
    report = {
        "dataset": {
            "rows": audit["rows"],
            "databases": audit["selected_databases"],
            "files_ok": not any(
                audit[key]
                for key in ("malformed_rows", "duplicate_ids", "missing_assets", "invalid_databases")
            ),
            "rows_with_solution_sql": audit["rows_with_solution_sql"],
            "rows_with_test_cases": audit["rows_with_test_cases"],
            "official_scoring_available": audit["ground_truth_complete"],
        },
        "official_environment_smoke": {
            "instance_id": row["instance_id"],
            "database": db_id,
            "get_schema_ok": schema_ok and bool(schema.strip()),
            "visible_knowledge_names": len(knowledge_names),
            "get_knowledge_definition_ok": definition_ok and bool(definition.strip()),
            "execute_sql_ok": execution_ok,
            "execute_sql_observation": execution,
            "public_submit_reward": submit_reward,
            "public_submit_observation": submit_observation,
            "phase1_completed": phase1,
            "phase2_completed": phase2,
            "task_finished": finished,
        },
        "conclusion": (
            "The public Mini-Interact environment is runnable, but official SR/reward cannot be "
            "computed because all public sol_sql and test_cases fields are empty."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
