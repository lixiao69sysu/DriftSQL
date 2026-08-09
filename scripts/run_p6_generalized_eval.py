#!/usr/bin/env python3
"""Run the Stage-6 evaluator with the P6 generalized seven-tool contract."""

from __future__ import annotations

import run_stage6_eval as evaluator


evaluator.TOOL_NAMES = (
    "get_schema_version",
    "inspect_schema_diff",
    "get_schema",
    "ask_user",
    "get_knowledge_definition",
    "execute_sql",
    "submit_solution",
)


if __name__ == "__main__":
    evaluator.main()
