"""
config/settings.py
==================
Centralised environment configuration via Pydantic BaseSettings.

ALL environment variable reads in the application must go through
`get_settings()`. No module may call `os.environ` directly.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.llm_model_name)
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Priority (highest to lowest):
      1. OS environment variables
      2. .env file
      3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="development | staging | production")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # ── API Security ──────────────────────────────────────────────────────
    api_key: str = Field(..., description="Secret key for X-API-Key header authentication.")
    cors_origins: list[str] = Field(
        default=["http://localhost:8501"],
        description="Allowed CORS origins.",
    )
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)

    # ── LLM Provider ─────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model_name: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_request_timeout: int = Field(default=60, ge=5)

    # ── Embedding Model ───────────────────────────────────────────────────
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu", description="cpu | cuda | mps")

    # ── Job Store ────────────────────────────────────────────────────────
    job_ttl_seconds: int = Field(default=86400, ge=60)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL. Leave blank or unset for in-memory store (dev).",
    )

    @field_validator("redis_url", mode="before")
    @classmethod
    def normalize_redis_url(cls, v: object) -> Optional[str]:
        """Coerce empty string to None so that REDIS_URL= in .env means 'no Redis'."""
        if v == "" or v is None:
            return None
        return str(v)

    # ── Chunker ──────────────────────────────────────────────────────────
    default_max_chunk_lines: int = Field(default=80, ge=20, le=500)
    default_similarity_top_k: int = Field(default=3, ge=1, le=10)

    # ── API Server ────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)
    api_base_url: str = Field(default="http://localhost:8000")

    # ── Streamlit UI ──────────────────────────────────────────────────────
    streamlit_api_base_url: str = Field(default="http://localhost:8000")
    streamlit_port: int = Field(default=8501)

    # ── Reports ──────────────────────────────────────────────────────────
    reports_output_dir: str = Field(default="./reports/output")

    # ── Validators ───────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Allow JSON string or list for CORS_ORIGINS env var."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [v]
        return v

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings singleton.
    LRU cache ensures the .env file is read only once per process.
    """
    return Settings()  # type: ignore[call-arg]
