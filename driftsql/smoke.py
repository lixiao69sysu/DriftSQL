"""Run the first executable schema-drift recovery episode."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

from driftsql.contracts import AgentStep, Task, Trajectory
from driftsql.drift import ColumnRename, SchemaDiff
from driftsql.environment import VersionedSQLite
from driftsql.rewards import compute_reward
from driftsql.tools import inspect_schema_diff


def run_smoke_episode() -> dict:
    temp_root = os.environ.get("DRIFTSQL_TMPDIR", "/tmp")
    with tempfile.TemporaryDirectory(prefix="driftsql-smoke-", dir=temp_root) as temp_dir:
        environment = VersionedSQLite(Path(temp_dir), "retail")
        environment.create(
            "v1",
            ddl=[
                """
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    annual_revenue REAL NOT NULL
                )
                """
            ],
            seed_sql=[
                "INSERT INTO customers VALUES (1, 'Ada', 12000.0)",
                "INSERT INTO customers VALUES (2, 'Lin', 8000.0)",
            ],
        )

        stale_sql = "SELECT customer_id, name FROM customers WHERE annual_revenue > 10000"
        v1_result = environment.execute_read_only("v1", stale_sql)

        environment.clone("v1", "v2")
        mutation = ColumnRename(table="customers", old_name="customer_id", new_name="client_id")
        with environment.connect("v2") as connection:
            mutation.apply(connection)

        stale_error = None
        try:
            environment.execute_read_only("v2", stale_sql)
        except sqlite3.OperationalError as error:
            stale_error = str(error)

        diff = SchemaDiff(
            db_id="retail",
            from_version="v1",
            to_version="v2",
            operations=(mutation.as_operation(),),
        )
        diff_observation = inspect_schema_diff(diff)
        repaired_sql = mutation.rewrite(stale_sql)
        v2_result = environment.execute_read_only("v2", repaired_sql)
        success = v1_result.rows == v2_result.rows and stale_error is not None

        reward = compute_reward(
            success=success,
            inspected_drift=True,
            validated_result=True,
            tool_cost=2.0,
        )
        trajectory = Trajectory(
            task=Task(
                task_id="retail-column-rename-001",
                db_id="retail",
                db_version="v2",
                question="List high-value customers.",
            ),
            steps=[
                AgentStep("execute_sql", {"sql": stale_sql}, {"error": stale_error}),
                AgentStep("inspect_schema_diff", {}, diff_observation),
                AgentStep("execute_sql", {"sql": repaired_sql}, {"rows": v2_result.rows}),
            ],
            final_sql=repaired_sql,
            success=success,
            total_reward=reward.total,
            total_cost=2.0,
        )

        return {
            "success": success,
            "stale_error": stale_error,
            "repaired_sql": repaired_sql,
            "result_rows": v2_result.rows,
            "reward": reward.to_dict(),
            "trajectory": trajectory.to_dict(),
        }


def main() -> None:
    result = run_smoke_episode()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
