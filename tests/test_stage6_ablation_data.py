from __future__ import annotations

from scripts.prepare_stage6_ablation_data import B0_TOOLS, B1_TOOLS, build_b0, build_b1


def _record() -> dict:
    state = {"db_id": "db", "schema_diff": {"operations": []}}
    return {
        "prompt": [{"role": "system", "content": "old"}, {"role": "user", "content": "q"}],
        "extra_info": {
            "tool_selection": list(B0_TOOLS),
            "tools_kwargs": {
                name: {"create_kwargs": dict(state)} for name in B0_TOOLS
            },
        },
    }


def test_b1_only_adds_version_diff_tools_and_prompt() -> None:
    source = _record()
    result = build_b1(source)
    assert tuple(result["extra_info"]["tool_selection"]) == B1_TOOLS
    assert "get_schema_version" in result["extra_info"]["tools_kwargs"]
    assert "inspect_schema_diff" in result["extra_info"]["tools_kwargs"]
    assert "audited schema diff" in result["prompt"][0]["content"]
    assert source["extra_info"]["tool_selection"] == list(B0_TOOLS)


def test_b0_preserves_stage5_tool_selection() -> None:
    result = build_b0(_record())
    assert tuple(result["extra_info"]["tool_selection"]) == B0_TOOLS
    assert result["extra_info"]["stage6_variant"] == "b0_stage5_selected_policy"
