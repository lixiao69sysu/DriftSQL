from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from driftsql.data import build_rl_record, build_sft_record
from driftsql.drift import build_column_rename_example
from driftsql.rewards.agentic import compute_score


def _tool_trace(calls: list[tuple[str, dict]]) -> str:
    return "\n".join(
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
        for name, arguments in calls
    )


class TrainingPipelineTest(unittest.TestCase):
    def test_manifest_converts_and_reward_executes_on_v2(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="driftsql-pipeline-",
            dir="/tmp",
        ) as temp_dir:
            database = Path(temp_dir) / "retail.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE orders "
                    "(order_id INTEGER PRIMARY KEY, total_amount REAL)"
                )
                connection.executemany(
                    "INSERT INTO orders VALUES (?, ?)",
                    [(1, 12.5), (2, 7.0)],
                )
            example = build_column_rename_example(
                source="unit_test",
                source_index=0,
                db_id="retail",
                question="Show order totals.",
                evidence="",
                sql="SELECT total_amount FROM orders ORDER BY order_id",
                database=database,
            )
            manifest = example.to_dict()
            rl_record = build_rl_record(
                manifest,
                index=0,
                split="train",
            )
            sft_record = build_sft_record(manifest)

            self.assertEqual(
                rl_record["agent_name"],
                "driftsql_tool_agent",
            )
            self.assertEqual(
                len(rl_record["extra_info"]["tools_kwargs"]),
                4,
            )
            self.assertEqual(
                sum(
                    message["role"] == "assistant"
                    for message in sft_record["messages"]
                ),
                5,
            )

            repaired = example.repaired_sql
            correct_trace = _tool_trace(
                [
                    ("execute_sql", {"sql": example.stale_sql}),
                    ("get_schema_version", {}),
                    ("inspect_schema_diff", {}),
                    ("execute_sql", {"sql": repaired}),
                    ("submit_solution", {"sql": repaired}),
                ]
            )
            correct = compute_score(
                data_source=rl_record["data_source"],
                solution_str=correct_trace,
                ground_truth=rl_record["reward_model"]["ground_truth"],
                extra_info=rl_record["extra_info"],
            )
            stale = compute_score(
                data_source=rl_record["data_source"],
                solution_str=_tool_trace(
                    [
                        (
                            "submit_solution",
                            {"sql": example.stale_sql},
                        )
                    ]
                ),
                ground_truth=rl_record["reward_model"]["ground_truth"],
                extra_info=rl_record["extra_info"],
            )

            self.assertTrue(correct["task_success"])
            self.assertEqual(
                correct["instance_id"],
                rl_record["extra_info"]["instance_id"],
            )
            self.assertTrue(correct["inspected_drift"])
            self.assertTrue(correct["tested_solution"])
            self.assertGreater(correct["score"], 1.0)
            self.assertFalse(stale["task_success"])
            self.assertLess(stale["score"], 0.0)

    def test_five_tool_metrics_shape_reward_and_penalise_repetition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driftsql-shaped-", dir="/tmp") as temp_dir:
            database = Path(temp_dir) / "retail.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE orders (order_id INTEGER, total_amount REAL)")
                connection.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 12.5), (2, 7.0)])
            example = build_column_rename_example(
                source="unit_test",
                source_index=0,
                db_id="retail",
                question="Show order totals.",
                evidence="",
                sql="SELECT total_amount FROM orders ORDER BY order_id",
                database=database,
            )
            record = build_rl_record(example.to_dict(), index=0, split="train")
            repaired = example.repaired_sql
            trace = _tool_trace(
                [
                    ("get_schema", {"query": "orders"}),
                    ("ask_user", {"question": "What does total_amount mean?"}),
                    ("get_knowledge_definition", {"name": "total_amount"}),
                    ("execute_sql", {"sql": repaired}),
                    ("execute_sql", {"sql": repaired}),
                    ("submit_solution", {"sql": repaired}),
                ]
            )
            events = [
                {"tool": "get_schema", "metrics": {"schema_retrieved": True}},
                {
                    "tool": "ask_user",
                    "metrics": {"clarification_matched": True, "duplicate_question": False},
                },
                {
                    "tool": "get_knowledge_definition",
                    "metrics": {"knowledge_retrieved": True},
                },
                {"tool": "execute_sql", "metrics": {"execution_success": True}},
                {"tool": "execute_sql", "metrics": {"execution_success": True}},
                {"tool": "submit_solution", "metrics": {"submitted": True}},
            ]
            score = compute_score(
                data_source=record["data_source"],
                solution_str=trace,
                ground_truth=record["reward_model"]["ground_truth"],
                extra_info=record["extra_info"]
                | {"environment_events": events, "response_len": 1000},
            )

            self.assertTrue(score["task_success"])
            self.assertTrue(score["clarification_matched"])
            self.assertTrue(score["schema_retrieved"])
            self.assertTrue(score["knowledge_retrieved"])
            self.assertEqual(score["duplicate_executions"], 1)
            self.assertEqual(score["reward_clarify"], 0.2)
            self.assertEqual(score["penalty_duplicate"], 0.05)
            self.assertEqual(score["penalty_token_cost"], 0.01)
            self.assertFalse(score["efficient"])

    def test_turn_limit_and_repeated_tool_classes_receive_terminal_penalties(self) -> None:
        trace = _tool_trace(
            [
                ("get_schema", {"query": "orders"}),
                ("ask_user", {"question": "What is active?"}),
                ("get_knowledge_definition", {"name": "active"}),
                ("get_schema", {"query": "orders again"}),
                ("ask_user", {"question": "Can you clarify active again?"}),
                ("get_knowledge_definition", {"name": "active again"}),
                ("execute_sql", {"sql": "SELECT 1"}),
            ]
        )
        result = compute_score(
            data_source="driftsql/test",
            solution_str=trace,
            ground_truth="",
            extra_info={"trajectory_turn_limit": True},
        )
        self.assertTrue(result["turn_limit"])
        self.assertEqual(result["excess_clarifications"], 1)
        self.assertEqual(result["excess_retrievals"], 2)
        self.assertEqual(result["penalty_repeated_tool"], 0.15)
        self.assertEqual(result["penalty_turn_limit"], 0.3)
        self.assertEqual(result["penalty_missing_submit"], 0.2)
        self.assertEqual(result["reward_clarify"], 0.0)
        self.assertLess(result["score"], -0.7)

    def test_early_invalid_termination_cannot_avoid_missing_submit_penalty(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str="<think>done</think>",
            ground_truth="",
            extra_info={"trajectory_turn_limit": False},
        )
        self.assertTrue(result["missing_submit"])
        self.assertFalse(result["turn_limit"])
        self.assertEqual(result["penalty_missing_submit"], 0.2)
        self.assertEqual(result["score"], -0.2)

    def test_unsafe_sql_receives_explicit_negative_reward(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str=_tool_trace(
                [
                    ("execute_sql", {"sql": "DROP TABLE orders"}),
                    ("submit_solution", {"sql": "DROP TABLE orders"}),
                ]
            ),
            ground_truth="",
            extra_info={},
        )
        self.assertTrue(result["unsafe"])
        self.assertEqual(result["penalty_unsafe"], 1.0)
        self.assertLess(result["score"], -1.0)

    def test_read_only_explain_is_not_treated_as_unsafe(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str=_tool_trace(
                [
                    ("execute_sql", {"sql": "EXPLAIN ANALYZE SELECT * FROM orders"}),
                ]
            ),
            ground_truth="",
            extra_info={},
        )
        self.assertFalse(result["unsafe"])
        self.assertEqual(result["penalty_unsafe"], 0.0)

    def test_targeted_reward_credits_required_clarification_and_terminal(self) -> None:
        sql = "SELECT 1"
        calls = _tool_trace(
            [
                ("ask_user", {"question": "Which active definition should I use?"}),
                ("get_knowledge_definition", {"name": "active"}),
                ("execute_sql", {"sql": sql}),
                ("submit_solution", {"sql": sql}),
            ]
        )
        events = [
            {
                "tool": "ask_user",
                "metrics": {"clarification_matched": True},
            },
            {"tool": "get_knowledge_definition", "metrics": {}},
            {"tool": "execute_sql", "metrics": {"execution_success": True}},
            {"tool": "submit_solution", "metrics": {}},
        ]
        result = compute_score(
            data_source="driftsql/test",
            solution_str=calls,
            ground_truth="",
            extra_info={
                "interaction_profile": "must_ask",
                "environment_events": events,
                "key_action_mask_tokens": 24,
                "advantage_scope": "episode",
                "episode_response_mask_tokens": 120,
                "advantage_mask_tokens": 120,
            },
            required_clarification_weight=0.3,
            post_clarification_weight=0.1,
            terminal_weight=0.2,
            missing_required_clarification_penalty=0.3,
        )
        self.assertTrue(result["clarification_required"])
        self.assertTrue(result["clarification_attempted"])
        self.assertTrue(result["post_clarification_valid"])
        self.assertTrue(result["terminal_validated"])
        self.assertEqual(result["reward_required_clarification"], 0.3)
        self.assertEqual(result["reward_post_clarification"], 0.1)
        self.assertEqual(result["reward_terminal"], 0.2)
        self.assertEqual(result["penalty_missing_required_clarification"], 0.0)
        self.assertEqual(result["key_action_mask_tokens"], 24)
        self.assertEqual(result["advantage_scope"], "episode")
        self.assertEqual(result["episode_response_mask_tokens"], 120)
        self.assertEqual(result["advantage_mask_tokens"], 120)

    def test_targeted_reward_separates_attempt_from_matched_clarification(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str=_tool_trace(
                [
                    ("ask_user", {"question": "Could you clarify?"}),
                    ("execute_sql", {"sql": "SELECT 1"}),
                ]
            ),
            ground_truth="",
            extra_info={"interaction_profile": "must_ask"},
            clarification_attempt_weight=0.2,
            missing_required_clarification_penalty=0.4,
            unmatched_clarification_penalty=0.15,
        )
        self.assertTrue(result["clarification_attempted"])
        self.assertFalse(result["clarification_matched"])
        self.assertEqual(result["reward_clarification_attempt"], 0.2)
        self.assertEqual(result["penalty_missing_required_clarification"], 0.0)
        self.assertEqual(result["penalty_unmatched_clarification"], 0.15)

    def test_matched_clarification_is_rewarded_before_terminal_submission(self) -> None:
        result = compute_score(
            data_source="driftsql/test",
            solution_str=_tool_trace(
                [
                    ("ask_user", {"question": "Which active definition should I use?"}),
                    ("get_knowledge_definition", {"name": "active"}),
                ]
            ),
            ground_truth="",
            extra_info={
                "interaction_profile": "must_ask",
                "environment_events": [
                    {"tool": "ask_user", "metrics": {"clarification_matched": True}},
                    {"tool": "get_knowledge_definition", "metrics": {}},
                ],
            },
            required_clarification_weight=0.6,
        )
        self.assertTrue(result["clarification_matched"])
        self.assertFalse(result["format_valid"])
        self.assertEqual(result["reward_required_clarification"], 0.6)

    def test_add_column_inspection_reward_requires_unmasked_version_and_diff(self) -> None:
        calls = _tool_trace(
            [
                ("get_schema_version", {}),
                ("inspect_schema_diff", {}),
            ]
        )
        valid = compute_score(
            data_source="driftsql/test",
            solution_str=calls,
            ground_truth="",
            extra_info={
                "drift_type": "add_column",
                "environment_events": [
                    {"tool": "get_schema_version", "metrics": {}},
                    {"tool": "inspect_schema_diff", "metrics": {"schema_diff_inspected": True}},
                ],
            },
            add_column_inspect_weight=0.4,
        )
        masked = compute_score(
            data_source="driftsql/test",
            solution_str=calls,
            ground_truth="",
            extra_info={
                "drift_type": "add_column",
                "environment_events": [
                    {"tool": "get_schema_version", "metrics": {}},
                    {"tool": "inspect_schema_diff", "metrics": {"action_masked": True}},
                ],
            },
            add_column_inspect_weight=0.4,
        )
        self.assertTrue(valid["add_column_inspected"])
        self.assertEqual(valid["reward_add_column_inspect"], 0.4)
        self.assertFalse(masked["add_column_inspected"])
        self.assertEqual(masked["reward_add_column_inspect"], 0.0)

    def test_targeted_reward_penalises_missing_ask_and_add_column_protocol(self) -> None:
        sql = "SELECT 1"
        result = compute_score(
            data_source="driftsql/test",
            solution_str=_tool_trace([("submit_solution", {"sql": sql})]),
            ground_truth="",
            extra_info={
                "interaction_profile": "must_ask",
                "drift_type": "add_column",
            },
            missing_required_clarification_penalty=0.3,
            add_column_protocol_penalty=0.2,
        )
        self.assertTrue(result["clarification_required"])
        self.assertFalse(result["add_column_protocol_complete"])
        self.assertEqual(result["penalty_missing_required_clarification"], 0.3)
        self.assertEqual(result["penalty_add_column_protocol"], 0.2)

    def test_decision_reward_scores_first_unmasked_action(self) -> None:
        correct = compute_score(
            data_source="driftsql/p6/decision/inspect_schema_diff",
            solution_str=_tool_trace([("inspect_schema_diff", {})]),
            ground_truth="",
            extra_info={"decision_target_action": "inspect_schema_diff"},
            decision_action_weight=1.0,
            decision_action_mismatch_penalty=0.75,
            missing_submit_penalty=0.0,
        )
        wrong = compute_score(
            data_source="driftsql/p6/decision/inspect_schema_diff",
            solution_str=_tool_trace([("execute_sql", {"sql": "SELECT 1"})]),
            ground_truth="",
            extra_info={"decision_target_action": "inspect_schema_diff"},
            decision_action_weight=1.0,
            decision_action_mismatch_penalty=0.75,
            missing_submit_penalty=0.0,
        )
        self.assertTrue(correct["decision_action_correct"])
        self.assertEqual(correct["reward_decision_action"], 1.0)
        self.assertEqual(correct["penalty_decision_action"], 0.0)
        self.assertFalse(wrong["decision_action_correct"])
        self.assertEqual(wrong["reward_decision_action"], 0.0)
        self.assertEqual(wrong["penalty_decision_action"], 0.75)

    def test_masked_decision_action_cannot_earn_reward(self) -> None:
        result = compute_score(
            data_source="driftsql/p6/decision/ask_user",
            solution_str=_tool_trace([("ask_user", {"question": "Which definition?"})]),
            ground_truth="",
            extra_info={
                "decision_target_action": "ask_user",
                "environment_events": [
                    {"tool": "ask_user", "metrics": {"action_masked": True}}
                ],
            },
            decision_action_weight=1.0,
            decision_action_mismatch_penalty=0.5,
            missing_submit_penalty=0.0,
        )
        self.assertEqual(result["decision_action"], "")
        self.assertFalse(result["decision_action_correct"])
        self.assertEqual(result["reward_decision_action"], 0.0)
        self.assertEqual(result["penalty_decision_action"], 0.5)


if __name__ == "__main__":
    unittest.main()
