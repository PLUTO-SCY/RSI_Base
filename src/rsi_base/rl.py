"""Fixed-harness LoRA GRPO inputs, model selection and Task evaluation."""

import random
from pathlib import Path

from pydantic import Field

from .agent import TaskAgent
from .benchmark import check_disjoint, load_problems
from .loop import evaluate, file_hash
from .math_scoring import math_score
from .memory_audit import cluster_interval
from .sft import AdapterConfig, load_adapter, local_key, locked
from .storage import Ledger, read_json, write_json
from .types import AgentSpec, Limits, ModelSettings, digest


class GRPOConfig(AdapterConfig):
    system_prompt: str = Field(
        default=(
            "Solve the math problem. Explain your reasoning, then end your answer with "
            "#### followed by a single number."
        ),
        min_length=1,
        max_length=8000,
    )
    seed: int = 42
    steps: int = Field(default=20, ge=2, le=10000)
    train_samples: int = Field(default=240, ge=2)
    prompts_per_batch: int = Field(default=8, ge=2, le=128)
    samples_per_prompt: int = Field(default=4, ge=2, le=32)
    max_prompt_tokens: int = Field(default=512, ge=32, le=8192)
    max_answer_tokens: int = Field(default=512, ge=32, le=8192)
    temperature: float = Field(default=0.8, gt=0, le=2)
    learning_rate: float = Field(default=1e-5, gt=0, le=0.001)
    save_every: int = Field(default=5, ge=1)
    training_timeout: int = Field(default=7200, ge=60)
    evaluation_timeout: int = Field(default=7200, ge=60)
    evaluation_concurrency: int = Field(default=16, ge=1, le=32)


def training_problems(config, data):
    rows = load_problems(Path(data) / "train.jsonl")
    if config.train_samples > len(rows) or config.train_samples < config.prompts_per_batch:
        raise ValueError("train_samples must fit the training split and cover a whole batch")
    return random.Random(config.seed).sample(rows, config.train_samples)


def freeze_inputs(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"GRPO inputs changed. Use a new output directory: {path}")
    else:
        write_json(path, value)


def parent_artifact(config, root):
    if not config.parent_adapter:
        return None
    return load_adapter(Path(root) / config.parent_adapter)


def evaluate_grpo(config, data, output, root, base_url, *, split):
    """Score a serviced parent and child with the same harness and decode settings."""
    if split not in {"val", "test"}:
        raise ValueError("Only development and frozen test can be evaluated")
    local_key()
    with locked(output) as output:
        data, root = Path(data), Path(root)
        artifact = load_adapter(output / "adapter")
        identity = artifact["identity"]
        if identity["config"] != config.model_dump():
            raise ValueError("Evaluation config differs from the trained adapter")
        if identity["train_split_sha256"] != file_hash(data / "train.jsonl"):
            raise ValueError("Training split changed")
        parent = parent_artifact(config, root)
        parent_sha = parent["artifact_sha256"] if parent else None
        if identity["parent_adapter_sha256"] != parent_sha:
            raise ValueError("Training parent adapter changed")
        if split == "test":
            if not (output / "evaluation/val/report.json").is_file():
                raise ValueError("Complete development evaluation before opening test data")
            decision = read_json(output / "decision.json")
            development = read_json(output / "evaluation/val/report.json")
            if (
                decision != development["decision"]
                or decision["adapter_sha256"] != artifact["artifact_sha256"]
                or decision["val_sha256"] != file_hash(data / "val.jsonl")
            ):
                raise ValueError("Development decision or its inputs changed")
        training = load_problems(data / "train.jsonl")
        problems = load_problems(data / f"{split}.jsonl")
        check_disjoint(training, problems)
        if split == "test":
            check_disjoint(load_problems(data / "val.jsonl"), problems)
        parent_name = parent["served_model"] if parent else config.base_model
        names = {"parent": parent_name, "candidate": artifact["served_model"]}
        directory = output / "evaluation" / split
        settings = {
            side: ModelSettings(
                model=name,
                base_url=base_url,
                api_key_env="RSI_LOCAL_API_KEY",
                max_tokens=config.max_answer_tokens,
                temperature=0,
                extra_body={"top_p": 1.0},
            )
            for side, name in names.items()
        }
        freeze_inputs(
            directory / "identity.json",
            {
                "split_sha256": digest(problems),
                "candidate_sha256": artifact["artifact_sha256"],
                "parent_sha256": parent_sha,
                "harness": config.system_prompt,
                "settings": {side: model.model_dump() for side, model in settings.items()},
            },
        )
        ledger = Ledger(
            directory / "calls.jsonl",
            Limits(
                max_task_calls=len(problems) * 3,
                max_meta_calls=0,
                wall_seconds=config.evaluation_timeout,
            ),
        )
        records, metrics = {}, {}
        try:
            for side, model in settings.items():
                records[side], metrics[side] = evaluate(
                    directory / side,
                    problems,
                    AgentSpec(system_prompt=config.system_prompt),
                    TaskAgent(model, root),
                    ledger,
                    concurrency=config.evaluation_concurrency,
                    scorer=math_score,
                    expected_model=model.model,
                )
        finally:
            ledger.close()
        deltas = {
            baseline["id"]: child["score"] - baseline["score"]
            for baseline, child in zip(records["parent"], records["candidate"], strict=True)
        }
        if split == "val":
            accepted = sum(deltas.values()) > 0
            decision = {
                "accepted": accepted,
                "selected_model": names["candidate" if accepted else "parent"],
                "rule": "Higher paired development accuracy under the fixed harness",
                "adapter_sha256": artifact["artifact_sha256"],
                "val_sha256": file_hash(data / "val.jsonl"),
            }
            freeze_inputs(output / "decision.json", decision)
        report = {
            "split": split,
            "metrics": metrics,
            "paired_gain": sum(deltas.values()) / len(deltas),
            "paired_interval": cluster_interval(deltas, seed=config.seed),
            "decision": decision,
        }
        freeze_inputs(directory / "report.json", report)
        return report
