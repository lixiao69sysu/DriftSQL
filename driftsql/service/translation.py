"""Lazy CPU Chinese-to-English translation for the English-trained Agent."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PROTECTED_PATTERN = (
    r"```[\s\S]*?```|`[^`\n]+`|"
    r"\b(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b.*?(?=[，。！？；\n]|$)|"
    r"@[A-Za-z0-9_./-]+|"
    r"\b[A-Za-z][A-Za-z0-9]*(?:[_.$/-][A-Za-z0-9_.$/-]+)+\b|"
    r"\b\d+(?:\.\d+)?\b"
)
QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DOMAIN_GLOSSARY = (
    ("字段重命名", "column rename"),
    ("表重命名", "table rename"),
    ("字段类型变更", "column type change"),
    ("复合漂移", "compound drift"),
    ("字段漂移", "column drift"),
    ("模式漂移", "schema drift"),
    ("业务知识", "business knowledge"),
    ("活跃客户数", "active customers"),
    ("活跃客户", "active customers"),
    ("只读", "read-only"),
)
QWEN_TOKEN_RE = re.compile(PROTECTED_PATTERN, re.IGNORECASE)


class TranslationUnavailableError(RuntimeError):
    """Raised when a required translation cannot be produced safely."""


@dataclass(frozen=True)
class TranslationResult:
    original_text: str
    translated_text: str
    applied: bool
    model_id: str
    source_locale: str
    target_locale: str = "en-US"
    elapsed_ms: float = 0.0


class TranslationService(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def loaded(self) -> bool: ...

    @property
    def model_id(self) -> str: ...

    async def translate(self, text: str, source_locale: str) -> TranslationResult: ...


class PassthroughTranslationService:
    """No-op implementation used when translation is explicitly disabled."""

    model_id = "disabled"
    available = False
    loaded = False

    async def translate(self, text: str, source_locale: str) -> TranslationResult:
        return TranslationResult(
            original_text=text,
            translated_text=text,
            applied=False,
            model_id=self.model_id,
            source_locale=source_locale,
            elapsed_ms=0.0,
        )


class QwenChineseEnglishTranslator:
    """Instruction-constrained CPU translator with exact identifier preservation."""

    SYSTEM_PROMPT = (
        "Translate Chinese to English without omitting any meaning. Output only the translation. "
        "CRITICAL: placeholders are mandatory immutable database terms. The output is invalid unless it "
        "contains every <DRIFTSQL_N> placeholder found in the source exactly once and unchanged. "
        "Before answering, silently verify that none are missing."
    )

    def __init__(
        self,
        model_path: Path,
        *,
        max_input_tokens: int = 512,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"{QWEN_MODEL_ID}@{QWEN_MODEL_REVISION[:12]}"

    @property
    def available(self) -> bool:
        required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
        return all((self.model_path / name).is_file() for name in required)

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    async def translate(self, text: str, source_locale: str) -> TranslationResult:
        if source_locale != "zh-CN" or not CHINESE_RE.search(text):
            return TranslationResult(
                original_text=text,
                translated_text=text,
                applied=False,
                model_id=self.model_id,
                source_locale=source_locale,
            )
        started = time.perf_counter()
        translated = await asyncio.to_thread(self._translate_sync, text)
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            applied=True,
            model_id=self.model_id,
            source_locale=source_locale,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _load(self) -> None:
        if self.loaded:
            return
        if not self.available:
            raise TranslationUnavailableError(
                f"Translation model is missing or incomplete: {self.model_path}. "
                "Run scripts/download_translation_model.py."
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise TranslationUnavailableError(
                "Translation dependencies are unavailable; install the translation optional dependencies."
            ) from error
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        self._model.to("cpu")
        self._model.eval()
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None

    def _translate_sync(self, text: str) -> str:
        with self._lock:
            self._load()
            protected_values: list[str] = []

            def protect(match: re.Match[str]) -> str:
                placeholder = f"<DRIFTSQL_{len(protected_values)}>"
                protected_values.append(match.group(0))
                return placeholder

            protected_text = QWEN_TOKEN_RE.sub(protect, text)
            for source_term, english_term in DOMAIN_GLOSSARY:
                protected_text = protected_text.replace(source_term, english_term)
            if protected_values and protected_values[0].startswith("@"):
                leading_path = "<DRIFTSQL_0>"
                if protected_text.startswith(leading_path):
                    remainder = protected_text[len(leading_path) :].lstrip()
                    protected_text = f"数据库路径是 {leading_path}。{remainder}"
            translated = self._generate_translation(protected_text)
            placeholders = [f"<DRIFTSQL_{index}>" for index in range(len(protected_values))]
            if any(translated.count(placeholder) != 1 for placeholder in placeholders):
                raise TranslationUnavailableError(
                    "Translation omitted, changed, or duplicated a protected database token."
                )
            for placeholder, value in zip(placeholders, protected_values, strict=True):
                translated = translated.replace(placeholder, value)
            translated = translated.strip().strip('"').strip()
            if not translated or CHINESE_RE.search(translated):
                raise TranslationUnavailableError("Translation did not produce a complete English instruction.")
            return translated

    def _generate_translation(self, protected_text: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": protected_text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
