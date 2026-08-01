from __future__ import annotations

import asyncio
import json

import pytest

from driftsql.integrations.agent_loop import DriftJsonToolParser
from driftsql.rewards.agentic import extract_tool_calls
from driftsql.tool_calls import extract_tool_call_dicts


# Exact assistant outputs captured in retry4/rollouts/1.jsonl. These are kept
# here so parser regressions do not depend on an ephemeral training artifact.
RETRY4_OUTPUTS = (
    """<think>
To recover the correct query for the active database, I will follow these steps:

1. **Get the current schema version** to understand if any table names or column names have changed.
2. **Inspect schema differences** between the cached schema and the active schema to identify any changes.
3. **Test the SQL query** against the active database to ensure it works correctly.
4. **Submit the final SQL query** that is valid for the active database.

</think>
{"name": "get_schema_version", "arguments": {}}""",
    """<think>Let's start by checking the current schema version and inspecting any schema changes that might affect the previous query. This will help us ensure that the query remains valid and accurate in the active database.</think>

```json
{
  "name": "get_schema_version",
  "arguments": {}
}
```""",
    """<think>
The previous query is still valid based on the cached schema snapshot, and there are no indications of schema changes that would affect its execution. Since the query is valid and does not rely on any specific changes to the database structure, we can submit it as is.
</think>
{
  "name": "submit_solution",
  "arguments": {
    "sql": "SELECT DISTINCT T1.medal_name FROM medal AS T1 INNER JOIN competitor_event AS T2 ON T1.id = T2.medal_id INNER JOIN games_competitor AS T3 ON T2.competitor_id = T3.id INNER JOIN person AS T4 ON T3.person_id = T4.id WHERE T4.full_name = 'Lee Chong Wei' AND T2.medal_id <> 4"
  }
}""",
    """<think>First, I need to verify if the schema has changed since the cached snapshot. If the schema has changed, I will inspect the schema differences to understand the changes. If the schema has not changed, I will validate the user's previous SQL query against the active database and submit it as the solution.</think>

{
  "name": "get_schema_version",
  "arguments": {}
}""",
)


@pytest.mark.parametrize(
    ("output", "expected_name"),
    zip(
        RETRY4_OUTPUTS,
        (
            "get_schema_version",
            "get_schema_version",
            "submit_solution",
            "get_schema_version",
        ),
        strict=True,
    ),
)
def test_retry4_real_outputs_are_parsed(output: str, expected_name: str) -> None:
    calls = extract_tool_call_dicts(output)
    assert [call["name"] for call in calls] == [expected_name]


def test_tagged_json_remains_supported() -> None:
    output = (
        '<think>Inspect the audited change.</think><tool_call>{"name": '
        '"inspect_schema_diff", "arguments": {}}</tool_call>'
    )
    assert extract_tool_calls(output) == [
        {"name": "inspect_schema_diff", "arguments": {}}
    ]


def test_reward_scanner_ignores_tool_observations() -> None:
    trajectory = "\n".join(
        (
            RETRY4_OUTPUTS[0],
            '{"db_id": "flight_2", "db_version": "v2"}',
            '<tool_call>{"name":"inspect_schema_diff","arguments":{}}</tool_call>',
            '{"operations":[{"kind":"rename_column"}]}',
            RETRY4_OUTPUTS[2],
        )
    )
    assert [call["name"] for call in extract_tool_calls(trajectory)] == [
        "get_schema_version",
        "inspect_schema_diff",
        "submit_solution",
    ]


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, _token_ids: list[int]) -> str:
        return self.text


@pytest.mark.parametrize("output", RETRY4_OUTPUTS)
def test_agent_parser_accepts_retry4_real_outputs(output: str) -> None:
    parser = DriftJsonToolParser(_Tokenizer(output))
    content, calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))
    expected = extract_tool_call_dicts(output)[0]

    assert len(calls) == 1
    assert calls[0].name == expected["name"]
    assert json.loads(calls[0].arguments) == expected["arguments"]
    assert expected["name"] not in content
