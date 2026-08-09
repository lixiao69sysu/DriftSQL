"""Typed service configuration with safe project-local defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRIFTSQL_SERVICE_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "DriftSQL Agent API"
    service_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8001, ge=1, le=65535)
    auth_enabled: bool = False
    api_key: SecretStr | None = None
    auth_session_ttl_seconds: int = Field(default=28800, ge=60, le=604800)
    auth_cookie_secure: bool = False

    model_backend: Literal["vllm", "scripted"] = "vllm"
    base_model_path: Path = PROJECT_ROOT / "models/Qwen2.5-Coder-7B-Instruct"
    default_model_id: str = "grpo-step25-seed20260810"
    adapter_path: Path = (
        PROJECT_ROOT / "checkpoints/p6_contract_observation_grpo_arm_c_7b/global_step_25/merged/lora_adapter"
    )
    experiment_catalog_path: Path = PROJECT_ROOT / "configs/service/experiments.json"
    model_registry_path: Path = PROJECT_ROOT / "configs/service/models.yaml"
    tensor_parallel_size: int = Field(default=2, ge=1, le=8)
    gpu_memory_utilization: float = Field(default=0.82, gt=0.0, lt=1.0)
    max_model_len: int = Field(default=8192, ge=1024)
    max_lora_rank: int = Field(default=32, ge=1)
    max_concurrent_sessions: int = Field(default=2, ge=1, le=16)
    executor_max_rows: int = Field(default=5, ge=1, le=100)
    schema_max_chars: int = Field(default=3500, ge=256, le=20000)
    knowledge_max_results: int = Field(default=1, ge=1, le=10)

    translation_enabled: bool = True
    translation_model_path: Path = PROJECT_ROOT / "models/Qwen2.5-0.5B-Instruct"
    translation_max_input_tokens: int = Field(default=512, ge=32, le=2048)
    translation_max_new_tokens: int = Field(default=256, ge=16, le=1024)

    default_max_turns: int = Field(default=7, ge=1, le=32)
    maximum_max_turns: int = Field(default=12, ge=1, le=64)
    default_timeout_seconds: float = Field(default=120.0, gt=0)
    maximum_timeout_seconds: float = Field(default=600.0, gt=0)
    default_max_tool_calls: int = Field(default=7, ge=1, le=64)
    default_max_new_tokens: int = Field(default=512, ge=16, le=4096)
    default_max_total_tokens: int = Field(default=32768, ge=64, le=65536)

    scenario_path: Path = PROJECT_ROOT / "data/processed/stage8_fresh_sft/tune_agent_eval.jsonl"
    tool_config_path: Path = PROJECT_ROOT / "configs/tools/drift_tools.yaml"
    repository_path: Path = PROJECT_ROOT / "data/service/driftsql_service.sqlite"
    temporary_root: Path = PROJECT_ROOT / "data/tmp/service"
    replay_review_dir: Path = PROJECT_ROOT / "data/processed/p4_replay_review"
    wandb_enabled: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "driftsql-rl"
    wandb_api_key: SecretStr | None = Field(default=None, validation_alias="WANDB_API_KEY")
    wandb_timeout_seconds: int = Field(default=10, ge=1, le=60)
    wandb_max_runs: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def require_api_key_when_auth_is_enabled(self) -> ServiceSettings:
        if self.auth_enabled and (self.api_key is None or not self.api_key.get_secret_value().strip()):
            raise ValueError("DRIFTSQL_SERVICE_API_KEY is required when authentication is enabled")
        return self

    def ensure_directories(self) -> None:
        self.repository_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
