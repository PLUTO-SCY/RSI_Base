"""Contracts shared by the experiment loop and its components."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, model_validator

from .storage import Ledger
from .types import AgentSpec, StrictModel

STRATEGY_NAME = r"^[a-z][a-z0-9_]{0,47}$"


class PipelineProposal(StrictModel):
    action: str = Field(pattern=STRATEGY_NAME)
    data_source: str | None = Field(default=None, pattern=STRATEGY_NAME)
    system_prompt: str | None = Field(default=None, min_length=1, max_length=8000)
    reason: str = Field(min_length=1, max_length=4000)
    parameters: dict | None = None

    @model_validator(mode="after")
    def valid_update(self):
        if self.action in {"offline_sft", "online_sft"}:
            raise ValueError("Use action sft with a data_source for built-in SFT")
        if self.action == "harness":
            if not self.system_prompt or not self.system_prompt.strip() or self.data_source:
                raise ValueError("A harness update needs a prompt and no training data")
        elif self.action in {"sft", "grpo"}:
            sources = {"online"} if self.action == "grpo" else {"offline", "online"}
            if self.system_prompt or self.data_source not in sources:
                raise ValueError("A weight update needs a supported data source and no prompt")
        if self.action in {"harness", "sft", "grpo"} and self.parameters:
            raise ValueError("Built-in strategies use the configured settings")
        return self

    @property
    def strategy(self):
        return f"{self.data_source}_sft" if self.action == "sft" else self.action


class ModelState(StrictModel):
    name: str
    sha256: str
    path: str | None = None


class TaskState(StrictModel):
    agent: AgentSpec
    model: ModelState


@dataclass(frozen=True)
class Step:
    directory: Path
    parent: TaskState
    index: int
    ledger: Ledger
    evaluate: Callable


@dataclass(frozen=True)
class Strategy:
    description: str
    apply: Callable[[Step, PipelineProposal], tuple[TaskState | None, dict | None]]
    scheduled: PipelineProposal | None = None
    task_calls: int = 0
    evidence: dict = field(default_factory=dict)
