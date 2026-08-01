from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from driftsql.data import INTERACTIVE_TOOL_NAMES, build_mini_interact_eval_record


def test_builds_public_interactive_record_without_inventing_labels() -> None:
    with tempfile.TemporaryDirectory(prefix="driftsql-interactive-data-", dir="/tmp") as directory:
        root = Path(directory)
        db_dir = root / "retail"
        db_dir.mkdir()
        with sqlite3.connect(db_dir / "retail.sqlite") as connection:
            connection.execute("CREATE TABLE orders (id INTEGER, amount REAL)")
        (db_dir / "retail_schema.txt").write_text(
            "CREATE TABLE orders (id INTEGER, amount REAL);", encoding="utf-8"
        )
        (db_dir / "retail_kb.jsonl").write_text(
            json.dumps({"knowledge": "Revenue", "definition": "SUM(amount)"}) + "\n",
            encoding="utf-8",
        )
        (db_dir / "retail_column_meaning_base.json").write_text("{}\n", encoding="utf-8")
        row = {
            "instance_id": "retail_1",
            "selected_database": "retail",
            "amb_user_query": "Show active-customer revenue.",
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
            "sol_sql": [],
            "test_cases": [],
        }

        record = build_mini_interact_eval_record(row, root, index=0)
        assert record["agent_name"] == "driftsql_tool_agent"
        assert record["reward_model"]["ground_truth"] == ""
        assert not record["extra_info"]["public_ground_truth_available"]
        assert tuple(record["extra_info"]["tool_selection"]) == INTERACTIVE_TOOL_NAMES
        assert set(record["extra_info"]["tools_kwargs"]) == set(INTERACTIVE_TOOL_NAMES)
        for tool in record["extra_info"]["tools_kwargs"].values():
            assert tool["create_kwargs"]["isolate_db"] is True
