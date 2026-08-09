"""Conservative SQL identifier rewriting shared by drift and planning."""

from __future__ import annotations

import re


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def rewrite_sql_identifier(sql: str, old_name: str, new_name: str) -> str:
    """Rewrite one SQL identifier without touching string literals."""

    if not _IDENTIFIER.fullmatch(old_name) or not _IDENTIFIER.fullmatch(new_name):
        raise ValueError(f"Unsafe SQL identifier mapping: {old_name!r} -> {new_name!r}")
    output: list[str] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'":
            start = index
            index += 1
            while index < len(sql):
                if sql[index] == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                    index += 2
                    continue
                if sql[index] == "'":
                    index += 1
                    break
                index += 1
            output.append(sql[start:index])
            continue
        if char in ('"', "`", "["):
            closing = "]" if char == "[" else char
            start = index
            index += 1
            while index < len(sql) and sql[index] != closing:
                index += 1
            index = min(index + 1, len(sql))
            quoted = sql[start:index]
            content = quoted[1:-1]
            output.append(char + new_name + closing if content.casefold() == old_name.casefold() else quoted)
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(sql) and (sql[index].isalnum() or sql[index] == "_"):
                index += 1
            token = sql[start:index]
            output.append(new_name if token.casefold() == old_name.casefold() else token)
            continue
        output.append(char)
        index += 1
    return "".join(output)
