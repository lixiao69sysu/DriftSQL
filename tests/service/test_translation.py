from __future__ import annotations

import asyncio
from pathlib import Path

from driftsql.service.inference.backend import ScriptedModelBackend
from driftsql.service.translation import (
    QwenChineseEnglishTranslator,
    TranslationResult,
    TranslationUnavailableError,
)

from .test_product_service import service_client


class FakeTranslator:
    model_id = "fake-zh-en"
    available = True
    loaded = True

    async def translate(self, text: str, source_locale: str) -> TranslationResult:
        return TranslationResult(
            original_text=text,
            translated_text="Count active customers by region using @book_publishing_company/authors/au_id.",
            applied=True,
            model_id=self.model_id,
            source_locale=source_locale,
            elapsed_ms=12.5,
        )


class StubQwenTranslator(QwenChineseEnglishTranslator):
    def __init__(self) -> None:
        super().__init__(Path("."))
        self._tokenizer = object()
        self._model = object()
        self._torch = object()

    def _load(self) -> None:
        return

    def _generate_translation(self, protected_text: str) -> str:
        assert protected_text == (
            "检查 <DRIFTSQL_0> 并保留 <DRIFTSQL_1>，统计 <DRIFTSQL_2> 行。"
        )
        return "Check <DRIFTSQL_0> and preserve <DRIFTSQL_1> while counting <DRIFTSQL_2> rows."


class DroppingStubQwenTranslator(StubQwenTranslator):
    def _generate_translation(self, protected_text: str) -> str:
        return "Check <DRIFTSQL_0>."


class DomainStubQwenTranslator(StubQwenTranslator):
    def _generate_translation(self, protected_text: str) -> str:
        assert protected_text == "检查column drift并统计active customers。"
        return "Check for column drift and count active customers."


class LeadingPathStubQwenTranslator(StubQwenTranslator):
    def _generate_translation(self, protected_text: str) -> str:
        assert protected_text == (
            "数据库路径是 <DRIFTSQL_0>。检查当前 Schema 是否发生column drift，"
            "统计出版社总数。再执行 SQL 验证。"
        )
        return (
            "The database path is <DRIFTSQL_0>. Check whether the current Schema has column drift, "
            "count publishers, and then execute SQL for validation."
        )


def test_translation_preserves_schema_paths_code_and_numbers() -> None:
    translator = StubQwenTranslator()
    translated = translator._translate_sync(
        "检查 @book_publishing_company/authors/au_id 并保留 `customer_id`，统计 10 行。"
    )
    assert translated == (
        "Check @book_publishing_company/authors/au_id and preserve `customer_id` while counting 10 rows."
    )


def test_translation_fails_closed_when_model_drops_protected_tokens() -> None:
    translator = DroppingStubQwenTranslator()
    try:
        translator._translate_sync(
            "检查 @book_publishing_company/authors/au_id 并保留 `customer_id`，统计 10 行。"
        )
    except TranslationUnavailableError as error:
        assert "protected database token" in str(error)
    else:
        raise AssertionError("unsafe translation was accepted")


def test_english_input_bypasses_translation() -> None:
    async def run() -> None:
        original = "Count active customers."
        result = await StubQwenTranslator().translate(original, "en-US")
        assert result.translated_text == original
        assert result.applied is False

    asyncio.run(run())


def test_domain_terms_are_normalized_without_becoming_hard_placeholders() -> None:
    translated = DomainStubQwenTranslator()._translate_sync("检查字段漂移并统计活跃客户数。")
    assert translated == "Check for column drift and count active customers."


def test_leading_database_path_is_anchored_while_schema_and_sql_are_not_placeholders() -> None:
    translated = LeadingPathStubQwenTranslator()._translate_sync(
        "@book_publishing_company 检查当前 Schema 是否发生字段漂移，统计出版社总数。再执行 SQL 验证。"
    )
    assert translated.startswith("The database path is @book_publishing_company.")


def test_query_session_records_original_and_english_agent_input(tmp_path: Path) -> None:
    async def run() -> None:
        async with service_client(tmp_path, ScriptedModelBackend(), translator=FakeTranslator()) as (app, client):
            databases = (await client.get("/api/databases")).json()
            db_id = next(item["db_id"] for item in databases if item["db_id"] == "book_publishing_company")
            original = "按地区统计活跃客户数。"
            response = await client.post(
                "/api/query-sessions",
                json={"db_id": db_id, "question": original, "locale": "zh-CN"},
            )
            assert response.status_code == 201
            session = response.json()
            assert session["question"] == original
            assert session["labels"]["translation_applied"] == "true"
            assert session["labels"]["agent_locale"] == "en-US"
            translation = session["result"]["translation"]
            assert translation["translated_question"].startswith("Count active customers")
            assert 'column "au_id" in table "authors"' in translation["agent_question"]
            active = app.state.orchestrator._active[session["session_id"]]
            prompt = "\n".join(message["content"] for message in active.messages)
            assert original not in prompt
            assert "Count active customers by region" in prompt
            assert "@book_publishing_company" not in prompt
            event = app.state.repository.list_events(session["session_id"])[0]
            translation_event = event.payload["input_translation"]
            assert translation_event["original"] == original
            assert translation_event["translated"].startswith("Count active customers by region")
            assert 'column "au_id" in table "authors"' in translation_event["agent_input"]

    asyncio.run(run())
