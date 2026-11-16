"""Minimal configuration and result contracts; no engine-specific objects."""

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(StrictModel):
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str = "RSI_API_KEY"
    max_tokens: int = Field(default=1024, ge=32, le=8192)
    timeout: float = Field(default=90, gt=0, le=600)
    temperature: float = Field(default=0, ge=0, le=2)
    extra_body: dict = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def safe_url(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Credentials and query strings are not allowed in base_url")
        return value.rstrip("/")


class MetaSettings(StrictModel):
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str = "RSI_API_KEY"
    max_tokens: int = Field(default=2048, ge=32, le=8192)
    timeout: float = Field(default=180, gt=0, le=600)
    command: str = "claude"
    state_dir: str = ".local/claude"

    @field_validator("base_url")
    @classmethod
    def safe_url(cls, value):
        return ModelSettings.safe_url(value)


class AgentSpec(StrictModel):
    system_prompt: str = Field(min_length=1, max_length=8000)


class Proposal(AgentSpec):
    reason: str = Field(min_length=1, max_length=4000)


class Limits(StrictModel):
    task_concurrency: int = Field(default=1, ge=1, le=32)
    rounds: int = Field(default=3, ge=0, le=100)
    max_task_calls: int = Field(default=200, ge=1)
    max_meta_calls: int = Field(default=3, ge=0)
    wall_seconds: float = Field(default=1800, gt=0)


class RunConfig(StrictModel):
    evaluator: Literal["numeric", "math_numeric"] = "numeric"
    task: ModelSettings
    meta: MetaSettings
    agent: AgentSpec
    limits: Limits = Field(default_factory=Limits)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_config(path):
    return RunConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
