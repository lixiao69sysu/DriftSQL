from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from driftsql.drift import fingerprint_query
from driftsql.rewards.agentic import compute_score
from scripts.replay_p6_reward_versions import V1_WEIGHTS, V2_WEIGHTS, derive_v1_from_shared_metrics


def tool_trace(calls: list[tuple[str, dict]]) -> str:
    return "\n".join(
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
        for name, arguments in calls
    )


class RewardV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="driftsql-reward-v2-", dir="/tmp")
        self.database = Path(self.temporary.name) / "source.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE items (id INTEGER, name TEXT)")
            connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
        expected = fingerprint_query(self.database, "SELECT id, name FROM items ORDER BY id")
        self.extra_info = {
            "instance_id": "add-column-unit",
            "db_id": "unit",
            "drift_type": "add_column",
            "interaction_profile": "schema_only",
            "source_db": str(self.database),
            "schema_diff": {
                "db_id": "unit",
                "from_version": "v1",
                "to_version": "v2",
                "operations": [
                    {
                        "type": "add_column",
                        "table": "items",
                        "old_name": None,
                        "new_name": "audit_flag",
                        "declared_type": "INTEGER",
                        "default_sql": "0",
                    }
                ],
            },
            "result_fingerprint": {
                "row_count": expected.row_count,
                "value_hash": expected.value_hash,
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def score(self, calls: list[tuple[str, dict]], *, events: list[dict] | None = None, **overrides):
        weights = {
            "reward_version": "v2",
            "success_weight": 1.0,
            "valid_weight": 0.1,
            "efficient_weight": 0.05,
            "semantic_candidate_weight": 0.15,
            "terminal_weight": 0.15,
            "add_column_inspect_weight": 0.05,
            "add_column_weight": 0.15,
            "add_column_protocol_penalty": 0.2,
            "missing_submit_penalty": 0.5,
            "tool_call_cost": 0.01,
        }
        weights.update(overrides)
        extra_info = dict(self.extra_info)
        if events is not None:
            extra_info["environment_events"] = events
        return compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra_info,
            **weights,
        )

    def test_executable_stale_wildcard_shortcut_is_negative(self) -> None:
        stale = "SELECT * FROM items ORDER BY id"
        result = self.score(
            [("execute_sql", {"sql": stale}), ("submit_solution", {"sql": stale})]
        )
        self.assertTrue(result["execution_success"])
        self.assertTrue(result["terminal_validated"])
        self.assertFalse(result["task_success"])
        self.assertFalse(result["candidate_task_success"])
        self.assertEqual(result["reward_valid"], 0.0)
        self.assertEqual(result["reward_terminal"], 0.0)
        self.assertEqual(result["penalty_add_column_protocol"], 0.2)
        self.assertLess(result["score"], 0.0)

    def test_premature_stale_execute_has_explicit_policy_penalty(self) -> None:
        stale = "SELECT * FROM items ORDER BY id"
        extra = dict(self.extra_info)
        extra["stale_sql"] = stale
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(
                [
                    ("execute_sql", {"sql": stale}),
                    ("submit_solution", {"sql": stale}),
                ]
            ),
            ground_truth="",
            extra_info=extra,
            reward_version="v3",
            premature_stale_execute_penalty=0.5,
        )

        self.assertTrue(result["premature_stale_execute"])
        self.assertEqual(result["penalty_premature_stale_execute"], 0.5)

    def test_stale_execute_after_ordered_inspection_is_not_premature(self) -> None:
        stale = "SELECT * FROM items ORDER BY id"
        calls = [
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": stale}),
            ("submit_solution", {"sql": stale}),
        ]
        extra = dict(self.extra_info)
        extra.update(
            {
                "stale_sql": stale,
                "environment_events": [
                    {"tool": name, "arguments": arguments, "metrics": {}}
                    for name, arguments in calls
                ],
            }
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra,
            reward_version="v3",
            premature_stale_execute_penalty=0.5,
        )

        self.assertTrue(result["ordered_drift_inspection"])
        self.assertFalse(result["premature_stale_execute"])
        self.assertEqual(result["penalty_premature_stale_execute"], 0.0)

    def test_canonical_stale_probe_then_repair_is_not_premature(self) -> None:
        stale = "SELECT * FROM items ORDER BY id"
        repaired = "SELECT id, name FROM items ORDER BY id"
        calls = [
            ("execute_sql", {"sql": stale}),
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": repaired}),
            ("submit_solution", {"sql": repaired}),
        ]
        extra = dict(self.extra_info)
        extra.update(
            {
                "stale_sql": stale,
                "environment_events": [
                    {"tool": name, "arguments": arguments, "metrics": {}}
                    for name, arguments in calls
                ],
            }
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra,
            reward_version="v3",
            premature_stale_execute_penalty=0.5,
        )

        self.assertEqual(result["decision_action"], "execute_sql")
        self.assertFalse(result["premature_stale_execute"])
        self.assertEqual(result["penalty_premature_stale_execute"], 0.0)

    def test_correct_full_protocol_receives_semantic_and_terminal_rewards(self) -> None:
        sql = "SELECT id, name FROM items ORDER BY id"
        calls = [
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        events = [
            {"tool": name, "arguments": arguments, "metrics": {}}
            for name, arguments in calls
        ]
        result = self.score(calls, events=events)
        self.assertTrue(result["task_success"])
        self.assertTrue(result["ordered_drift_inspection"])
        self.assertTrue(result["candidate_task_success"])
        self.assertTrue(result["add_column_candidate_validated"])
        self.assertTrue(result["add_column_protocol_complete"])
        self.assertEqual(result["reward_semantic_candidate"], 0.15)
        self.assertEqual(result["reward_terminal"], 0.15)
        self.assertEqual(result["reward_add_column_inspect"], 0.05)
        self.assertEqual(result["reward_add_column"], 0.15)
        self.assertEqual(result["penalty_add_column_protocol"], 0.0)
        self.assertGreater(result["score"], 1.4)

    def test_inspection_must_be_ordered_and_unmasked(self) -> None:
        sql = "SELECT id, name FROM items ORDER BY id"
        calls = [
            ("inspect_schema_diff", {}),
            ("get_schema_version", {}),
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        reversed_result = self.score(calls)
        self.assertFalse(reversed_result["ordered_drift_inspection"])
        self.assertFalse(reversed_result["add_column_inspected"])
        self.assertEqual(reversed_result["reward_add_column_inspect"], 0.0)
        self.assertEqual(reversed_result["penalty_add_column_protocol"], 0.2)

        ordered_calls = [
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        events = [
            {"tool": "get_schema_version", "arguments": {}, "metrics": {}},
            {
                "tool": "inspect_schema_diff",
                "arguments": {},
                "metrics": {"action_masked": True},
            },
            {"tool": "execute_sql", "arguments": {"sql": sql}, "metrics": {}},
            {"tool": "submit_solution", "arguments": {"sql": sql}, "metrics": {}},
        ]
        masked_result = self.score(ordered_calls, events=events)
        self.assertFalse(masked_result["ordered_drift_inspection"])
        self.assertFalse(masked_result["add_column_protocol_complete"])

    def test_v2_never_rewards_clarification_attempt_alone(self) -> None:
        calls = [("ask_user", {"question": "Could you clarify?"})]
        extra = dict(self.extra_info)
        extra.update(
            {
                "drift_type": "rename_column",
                "interaction_profile": "must_ask",
                "environment_events": [
                    {"tool": "ask_user", "arguments": {}, "metrics": {}}
                ],
            }
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra,
            reward_version="v2",
            clarification_attempt_weight=0.2,
            unmatched_clarification_penalty=0.15,
            missing_submit_penalty=0.5,
        )
        self.assertTrue(result["clarification_attempted"])
        self.assertEqual(result["reward_clarification_attempt"], 0.0)
        self.assertEqual(result["penalty_unmatched_clarification"], 0.15)
        self.assertLess(result["score"], 0.0)

    def test_v2_rewards_only_matched_clarification_and_valid_followup(self) -> None:
        calls = [
            ("ask_user", {"question": "Which active definition should I use?"}),
            ("get_knowledge_definition", {"name": "active"}),
        ]
        extra = dict(self.extra_info)
        extra.update(
            {
                "drift_type": "rename_column",
                "interaction_profile": "must_ask",
                "environment_events": [
                    {
                        "tool": "ask_user",
                        "arguments": calls[0][1],
                        "metrics": {"clarification_matched": True},
                    },
                    {
                        "tool": "get_knowledge_definition",
                        "arguments": calls[1][1],
                        "metrics": {},
                    },
                ],
            }
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra,
            reward_version="v2",
            required_clarification_weight=0.1,
            clarification_attempt_weight=0.2,
            post_clarification_weight=0.05,
            missing_submit_penalty=0.5,
        )
        self.assertEqual(result["reward_required_clarification"], 0.1)
        self.assertEqual(result["reward_post_clarification"], 0.05)
        self.assertEqual(result["reward_clarification_attempt"], 0.0)

    def test_v2_unsafe_sql_cannot_receive_positive_reward(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(
                [
                    ("execute_sql", {"sql": "DROP TABLE items"}),
                    ("submit_solution", {"sql": "DROP TABLE items"}),
                ]
            ),
            ground_truth="",
            extra_info={},
            reward_version="v2",
            unsafe_penalty=1.0,
        )
        self.assertTrue(result["unsafe"])
        self.assertEqual(result["penalty_unsafe"], 1.0)
        self.assertLess(result["score"], 0.0)

    def test_unknown_reward_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported reward_version"):
            self.score([], reward_version="v4")

    def test_v3_requires_full_add_column_protocol_for_success_reward(self) -> None:
        sql = "SELECT id, name FROM items ORDER BY id"
        complete_calls = [
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        complete_events = [
            {"tool": name, "arguments": arguments, "metrics": {}}
            for name, arguments in complete_calls
        ]
        complete = self.score(
            complete_calls,
            events=complete_events,
            reward_version="v3",
            semantic_candidate_weight=0.35,
        )
        shortcut_calls = [
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        shortcut_events = [
            {"tool": name, "arguments": arguments, "metrics": {}}
            for name, arguments in shortcut_calls
        ]
        shortcut = self.score(
            shortcut_calls,
            events=shortcut_events,
            reward_version="v3",
            semantic_candidate_weight=0.35,
        )

        self.assertTrue(complete["protocol_success"])
        self.assertEqual(complete["reward_success"], 1.0)
        self.assertTrue(shortcut["task_success"])
        self.assertFalse(shortcut["protocol_success"])
        self.assertEqual(shortcut["reward_success"], 0.0)
        self.assertEqual(shortcut["reward_semantic_candidate"], 0.35)

    def test_v3_correct_candidate_without_submit_outranks_wrong_candidate(self) -> None:
        correct_sql = "SELECT id, name FROM items ORDER BY id"
        wrong_sql = "SELECT id FROM items ORDER BY id"
        common = {
            "reward_version": "v3",
            "semantic_candidate_weight": 0.35,
            "missing_submit_penalty": 0.3,
            "add_column_protocol_penalty": 0.05,
        }
        correct = self.score(
            [
                ("get_schema_version", {}),
                ("inspect_schema_diff", {}),
                ("execute_sql", {"sql": correct_sql}),
            ],
            **common,
        )
        wrong = self.score(
            [
                ("get_schema_version", {}),
                ("inspect_schema_diff", {}),
                ("execute_sql", {"sql": wrong_sql}),
            ],
            **common,
        )

        self.assertTrue(correct["candidate_task_success"])
        self.assertFalse(wrong["candidate_task_success"])
        self.assertGreater(correct["score"], wrong["score"])

    def test_v3_must_ask_shortcut_does_not_receive_success_reward(self) -> None:
        sql = "SELECT id, name FROM items ORDER BY id"
        calls = [
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        events = [
            {"tool": name, "arguments": arguments, "metrics": {}}
            for name, arguments in calls
        ]
        extra = dict(self.extra_info)
        extra.update(
            {
                "drift_type": "clean",
                "interaction_profile": "must_ask",
                "environment_events": events,
            }
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=tool_trace(calls),
            ground_truth="",
            extra_info=extra,
            reward_version="v3",
            success_weight=1.0,
            semantic_candidate_weight=0.35,
            missing_required_clarification_penalty=0.2,
        )

        self.assertTrue(result["task_success"])
        self.assertFalse(result["protocol_success"])
        self.assertEqual(result["reward_success"], 0.0)
        self.assertEqual(result["reward_semantic_candidate"], 0.35)

    def test_offline_v1_derivation_matches_direct_v1_scoring(self) -> None:
        sql = "SELECT id, name FROM items ORDER BY id"
        calls = [
            ("get_schema_version", {}),
            ("inspect_schema_diff", {}),
            ("execute_sql", {"sql": sql}),
            ("submit_solution", {"sql": sql}),
        ]
        events = [
            {"tool": name, "arguments": arguments, "metrics": {}}
            for name, arguments in calls
        ]
        extra = dict(self.extra_info) | {"environment_events": events}
        common = {
            "data_source": "driftsql/test",
            "solution_str": tool_trace(calls),
            "ground_truth": "",
            "extra_info": extra,
        }
        direct = compute_score(**common, **V1_WEIGHTS)
        v2 = compute_score(**common, **V2_WEIGHTS)
        derived = derive_v1_from_shared_metrics(
            dict(v2) | {"drift_type": "add_column"},
            event_execution_success=False,
        )
        reward_keys = [key for key in direct if key.startswith(("reward_", "penalty_"))]
        self.assertEqual({key: direct[key] for key in reward_keys}, {key: derived[key] for key in reward_keys})
        self.assertEqual(direct["score"], derived["score"])


if __name__ == "__main__":
    unittest.main()
