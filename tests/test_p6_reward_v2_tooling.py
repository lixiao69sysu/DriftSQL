from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.replay_p6_reward_versions import canonical_trace, summarize as replay_summary
from scripts.summarize_p6_addcolumn72_checkpoints import summarize as checkpoint_summary


ROOT = Path(__file__).resolve().parents[1]


class P6RewardV2ToolingTest(unittest.TestCase):
    def test_focus1000_release_has_exact_quotas_and_prompt_hashes(self) -> None:
        output = ROOT / "data/processed/p6_focus1000_reward_ab"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["train_rows"], 1000)
        self.assertEqual(summary["tune_rows"], 432)
        self.assertEqual(
            summary["focus_categories"],
            {"add_column": 600, "compound": 160, "must_ask": 160, "other": 80},
        )
        self.assertEqual(summary["train_tune_database_overlap"], [])
        self.assertTrue(summary["prompt_bytes_unchanged"])
        self.assertFalse(summary["prompt_target_leakage"])
        self.assertFalse(summary["fresh_blind_rows_read"])

        rows = pq.read_table(output / "train.parquet").to_pylist()
        manifest = [
            json.loads(line)
            for line in (output / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), len(manifest))
        for row, metadata in zip(rows, manifest, strict=True):
            encoded = json.dumps(
                row["prompt"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(metadata["prompt_sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(metadata["task_id"], row["extra_info"]["instance_id"])

    def test_replay_trace_uses_canonical_tool_calls(self) -> None:
        trace = canonical_trace(
            [
                {"tool_name": "execute_sql", "arguments": {"sql": "SELECT 1"}},
                {"tool_name": "submit_solution", "arguments": {"sql": "SELECT 1"}},
            ]
        )
        self.assertIn('<tool_call>{"arguments": {"sql": "SELECT 1"}, "name": "execute_sql"}</tool_call>', trace)
        self.assertIn("submit_solution", trace)

    def test_replay_summary_enforces_reward_separation_gates(self) -> None:
        rows = [
            {
                "observed_task_success": True,
                "drift_type": "add_column",
                "unsafe": False,
                "v1": {"score": 1.0},
                "v2": {"score": 1.2},
            },
            {
                "observed_task_success": False,
                "drift_type": "add_column",
                "unsafe": False,
                "v1": {"score": 0.1},
                "v2": {"score": -0.2},
            },
        ]
        summary = replay_summary("candidate", rows)
        self.assertTrue(summary["passed"])
        self.assertGreaterEqual(summary["v2_success_failure_gap"], 0.8)

    def test_addcolumn_summary_counts_stale_shortcut_and_ordered_protocol(self) -> None:
        source = {
            "stale": {"extra_info": {"stale_sql": "SELECT * FROM items"}},
            "correct": {"extra_info": {"stale_sql": "SELECT * FROM items"}},
        }
        rows = [
            {
                "instance_id": "stale",
                "called_tools": ["execute_sql", "submit_solution"],
                "final_sql": "SELECT * FROM items",
                "termination_reason": "submitted",
                "task_success": False,
                "executable": True,
                "safety": {"unsafe": False, "timed_out": False},
            },
            {
                "instance_id": "correct",
                "called_tools": [
                    "get_schema_version",
                    "inspect_schema_diff",
                    "execute_sql",
                    "submit_solution",
                ],
                "final_sql": "SELECT id FROM items",
                "termination_reason": "submitted",
                "task_success": True,
                "executable": True,
                "safety": {"unsafe": False, "timed_out": False},
            },
        ]
        summary = checkpoint_summary("sft", rows, source)
        self.assertEqual(summary["stale_execute_submit_shortcut"], 1)
        self.assertEqual(summary["ordered_inspection"], 1)
        self.assertEqual(summary["inspect_before_execute"], 1)


if __name__ == "__main__":
    unittest.main()
