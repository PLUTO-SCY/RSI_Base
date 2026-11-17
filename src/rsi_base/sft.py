"""Train-only trajectory collection and frozen model/harness comparison for LoRA SFT."""

import fcntl
import json
import os
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field, field_validator

from .agent import TaskAgent
from .classification import MemoryHarness, MemoryPolicy, label_score, normalize
from .loop import file_hash
from .memory_audit import paired_records
from .memory_loop import MemoryConfig, evaluate_policy, load_rows, stable_improvement
from .storage import Ledger, read_json, write_json
from .types import Limits, StrictModel, digest


class SFTConfig(StrictModel):
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    base_directory: str = ".local/models/qwen3-4b-instruct-2507"
    parent_adapter: str | None = None
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    epochs: float = Field(default=1, gt=0, le=20)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    batch_size: int = Field(default=1, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=128)
    max_length: int = Field(default=4096, ge=128, le=16384)
    max_steps: int = Field(default=-1, ge=-1)
    save_steps: int = Field(default=20, ge=1)
    gradient_checkpointing: bool = True
    seed: int = 42
    min_samples: int = Field(default=16, ge=1)
    collection_repeats: int = Field(default=1, ge=1, le=5)
    evaluation_repeats: int = Field(default=3, ge=1, le=5)
    max_collection_calls: int = Field(default=2000, ge=1)
    max_evaluation_calls: int = Field(default=10000, ge=1)
    wall_seconds: float = Field(default=7200, gt=0)

    @field_validator("target_modules")
    @classmethod
    def modules(cls, value):
        if not value or len(set(value)) != len(value) or any(not item.strip() for item in value):
            raise ValueError("target_modules must contain distinct nonempty module names")
        return value

    @field_validator("max_steps")
    @classmethod
    def steps(cls, value):
        if value == 0:
            raise ValueError("max_steps must be -1 (epochs) or positive")
        return value


@contextmanager
def locked(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".sft.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output


def source_identity(root):
    return {
        "source_sha256": digest(
            {str(p.relative_to(root)): file_hash(p) for p in (root / "src").rglob("*.py")}
        ),
        "core_lock_sha256": file_hash(root / "uv.lock"),
        "train_lock_sha256": file_hash(root / "requirements/train.lock"),
    }


def pin(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Pinned SFT inputs changed: {path}")
    write_json(path, value)


def load_adapter(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    files = manifest["files"]
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise ValueError("Incomplete LoRA artifact")
    for name, expected in files.items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError("LoRA artifact changed")
    if manifest["artifact_sha256"] != digest(files):
        raise ValueError("LoRA artifact identity changed")
    return manifest


def successes(records, training, labels):
    """Supervise only the final verified call; never relabel an incorrect response."""
    bank = {row["id"]: row for row in training}
    samples, reasons, seen = [], Counter(), set()
    for record in records:
        problem = bank.get(record["problem_id"])
        if problem is None:
            raise ValueError("Non-training case in SFT collection")
        traces = record["traces"]
        if not traces or any(t.get("finish_reason") != "stop" for t in traces):
            reasons["unfinished_trace"] += 1
            continue
        final = traces[-1]
        if final["response"] != record["response"]:
            raise ValueError("Final trace differs from scored output")
        score = label_score(final["response"], problem["answer"], labels)
        if not score["score"]:
            reasons[score["status"]] += 1
            continue
        if record.get("score") != 1:
            raise ValueError("Collection score differs from fixed verifier")
        if any(identifier not in bank for identifier in record["memory_ids"]):
            raise ValueError("Non-training memory in SFT collection")
        used = [bank[identifier] for identifier in record["memory_ids"]]
        if any(normalize(row["question"]) == normalize(problem["question"]) for row in used):
            raise ValueError("Training query retrieved itself")
        messages = final["messages"]
        if [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("Unsupported Task message contract")
        if any(not isinstance(m["content"], str) for m in messages):
            raise ValueError("Task messages must contain text")
        messages = [*messages, {"role": "assistant", "content": final["response"]}]
        key = digest(messages)
        if key in seen:
            reasons["duplicate_messages"] += 1
            continue
        seen.add(key)
        samples.append(
            {
                "id": record["id"],
                "problem_id": problem["id"],
                "messages": messages,
                "trace_sha256": digest(final),
            }
        )
    return samples, dict(reasons)


def tokenize_sample(tokenizer, messages, max_length):
    """Keep the full chat template and mask every prompt token from the loss."""
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected one system/user/assistant training example")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[: len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template does not preserve the assistant boundary")
    if len(full) > max_length:
        return None
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt) + full[len(prompt) :],
    }


def collect(config, source_run, data, output, root, *, task=None, base_url=None):
    source_run, data, root = Path(source_run), Path(data), Path(root)
    frozen = read_json(source_run / "frozen.json")
    parent = read_json(source_run / "run.json")
    memory_config = MemoryConfig.model_validate(parent["config"])
    if (
        file_hash(data / "train.jsonl") != parent["split_sha256"]["train"]
        or file_hash(data / "labels.json") != parent["labels_sha256"]
    ):
        raise ValueError("Collection memory differs from frozen harness data")
    if base_url:
        task_settings = type(memory_config.task).model_validate(
            {**memory_config.task.model_dump(), "base_url": base_url}
        )
        memory_config = memory_config.model_copy(update={"task": task_settings})
    parent_model = config.base_model
    parent_hash = None
    if config.parent_adapter:
        adapter = load_adapter(root / config.parent_adapter)
        if adapter["identity"]["config"]["base_model"] != config.base_model:
            raise ValueError("Parent adapter belongs to another base model")
        parent_model, parent_hash = adapter["served_model"], adapter["artifact_sha256"]
    if memory_config.task.model != parent_model:
        raise ValueError("Generation model must match the base/parent adapter being trained")
    identity = {
        **source_identity(root),
        "config": config.model_dump(),
        "source_run": str(source_run.resolve()),
        "source_run_sha256": file_hash(source_run / "run.json"),
        "frozen_sha256": file_hash(source_run / "frozen.json"),
        "train_sha256": file_hash(data / "train.jsonl"),
        "labels_sha256": file_hash(data / "labels.json"),
        "parent_adapter_sha256": parent_hash,
        "split_sha256": parent["split_sha256"],
        "memory_config": memory_config.model_dump(),
        "initial_policy": frozen["baseline"]["policy"],
        "policy": frozen["selected"]["policy"],
        "leave_one_out": True,
        "supervision": "Final verified model response only; training split only",
    }
    with locked(output) as output:
        pin(output / "collection_identity.json", identity)
        if (output / "dataset.json").exists():
            return verify_dataset(output, config)
        training = load_rows(data / "train.jsonl")
        if not training or len({r["id"] for r in training}) != len(training):
            raise ValueError("Training rows must have distinct IDs")
        memory = random.Random(memory_config.seed).sample(training, len(training))
        labels = read_json(data / "labels.json")
        harness = MemoryHarness(memory, labels, context_chars=memory_config.context_chars)
        (output / "collection").mkdir(exist_ok=True)
        ledger = Ledger(
            output / "collection/calls.jsonl",
            Limits(
                max_task_calls=config.max_collection_calls,
                max_meta_calls=0,
                wall_seconds=config.wall_seconds,
            ),
        )
        try:
            records, summary = evaluate_policy(
                output / "collection/evaluation",
                training,
                MemoryPolicy.model_validate(identity["policy"]),
                harness,
                task or TaskAgent(memory_config.task, root),
                ledger,
                repeats=config.collection_repeats,
                seed=config.seed,
                leave_one_out=True,
            )
        finally:
            ledger.close()
        samples, excluded = successes(records, training, labels)
        if len(samples) < config.min_samples:
            raise ValueError(f"Only {len(samples)} verified samples; need {config.min_samples}")
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples)
        (output / "train.jsonl").write_text(payload)
        report = {
            "identity_sha256": digest(identity),
            "train_sha256": file_hash(output / "train.jsonl"),
            "samples": len(samples),
            "excluded": excluded,
            "collection_summary": summary,
            "development_opened": False,
            "test_opened": False,
        }
        write_json(output / "dataset.json", report)
        return report


def verify_dataset(output, config):
    output = Path(output)
    identity, report = (
        read_json(output / "collection_identity.json"),
        read_json(output / "dataset.json"),
    )
    if (
        identity["config"] != config.model_dump()
        or report["identity_sha256"] != digest(identity)
        or report["train_sha256"] != file_hash(output / "train.jsonl")
    ):
        raise ValueError("SFT dataset or configuration changed")
    rows = load_rows(output / "train.jsonl")
    if len(rows) != report["samples"] or len(rows) < config.min_samples:
        raise ValueError("SFT sample count changed")
    return report


def compare(config, data, output, root, *, base_url, split="val", tasks=None):
    """Cross model and harness on development; test only after the decision is sealed."""
    data, output, root = Path(data), Path(output), Path(root)
    verify_dataset(output, config)
    collection = read_json(output / "collection_identity.json")
    adapter = load_adapter(output / "adapter")
    if (
        config.parent_adapter
        and load_adapter(root / config.parent_adapter)["artifact_sha256"]
        != collection["parent_adapter_sha256"]
    ):
        raise ValueError("Parent adapter changed after collection")
    if (
        adapter["identity"]["dataset_sha256"] != file_hash(output / "train.jsonl")
        or adapter["identity"]["config"] != config.model_dump()
        or adapter["identity"]["collection_sha256"]
        != file_hash(output / "collection_identity.json")
    ):
        raise ValueError("Adapter was trained on a different dataset")
    if any(collection[key] != value for key, value in source_identity(root).items()):
        raise ValueError("SFT source or dependency locks changed after collection")
    if split not in {"val", "test"}:
        raise ValueError("Only development or frozen test may be evaluated")
    if split == "test" and not (output / "decision.json").exists():
        raise ValueError("Freeze the development decision before opening test data")
    if split == "test":
        decision = read_json(output / "decision.json")
        development = read_json(output / "evaluation/val/report.json")
        development_identity = read_json(output / "evaluation/val/identity.json")
        if (
            decision != development["decision"]
            or decision["evaluation_identity_sha256"] != digest(development_identity)
            or decision["adapter_sha256"] != adapter["artifact_sha256"]
        ):
            raise ValueError("Frozen development decision changed")
    memory_config = MemoryConfig.model_validate(collection["memory_config"])
    settings = memory_config.task.model_copy(update={"base_url": base_url})
    settings = type(settings).model_validate(settings.model_dump())
    models = {
        "parent": settings,
        "candidate": settings.model_copy(update={"model": adapter["served_model"]}),
    }
    identity = {
        **source_identity(root),
        "collection_sha256": file_hash(output / "collection_identity.json"),
        "adapter_sha256": adapter["artifact_sha256"],
        "config": config.model_dump(),
        "train_sha256": file_hash(data / "train.jsonl"),
        "labels_sha256": file_hash(data / "labels.json"),
        "query_sha256": file_hash(data / f"{split}.jsonl"),
        "models": {name: value.model_dump() for name, value in models.items()},
        "split": split,
    }
    if identity["query_sha256"] != collection["split_sha256"][split]:
        raise ValueError("Evaluation split differs from the frozen harness data")
    if any(identity[key] != collection[key] for key in ["train_sha256", "labels_sha256"]):
        raise ValueError("Evaluation memory differs from collection")
    if split == "test":
        identity["decision_sha256"] = file_hash(output / "decision.json")
    with locked(output):
        directory = output / "evaluation" / split
        pin(directory / "identity.json", identity)
        if (directory / "report.json").exists():
            return read_json(directory / "report.json")
        training, queries = load_rows(data / "train.jsonl"), load_rows(data / f"{split}.jsonl")
        if not queries or len({r["id"] for r in queries}) != len(queries):
            raise ValueError("Evaluation queries must have distinct IDs")
        known = {normalize(row["question"]) for row in training}
        if any(normalize(row["question"]) in known for row in queries):
            raise ValueError("Evaluation query overlaps training memory")
        memory = random.Random(memory_config.seed).sample(training, len(training))
        labels = read_json(data / "labels.json")
        harness = MemoryHarness(memory, labels, context_chars=memory_config.context_chars)
        policies = {
            "initial": MemoryPolicy.model_validate(collection["initial_policy"]),
            "evolved": MemoryPolicy.model_validate(collection["policy"]),
        }
        records, metrics = {}, {}
        ledger = Ledger(
            directory / "calls.jsonl",
            Limits(
                max_task_calls=config.max_evaluation_calls,
                max_meta_calls=0,
                wall_seconds=config.wall_seconds,
            ),
        )
        try:
            for model_name, model in models.items():
                task = tasks[model_name] if tasks else TaskAgent(model, root)
                for policy_name, policy in policies.items():
                    key = f"{model_name}_{policy_name}"
                    records[key], metrics[key] = evaluate_policy(
                        directory / key,
                        queries,
                        policy,
                        harness,
                        task,
                        ledger,
                        repeats=config.evaluation_repeats,
                        seed=config.seed,
                    )
        finally:
            ledger.close()
        pair = paired_records(records["parent_evolved"], records["candidate_evolved"])
        report = {
            "split": split,
            "test_status": memory_config.test_status,
            "metrics": metrics,
            "lora_effect_with_evolved_harness": pair,
            "test_used_for_selection": False,
        }
        if split == "val":
            accepted = stable_improvement(metrics["candidate_evolved"], metrics["parent_evolved"])
            decision = {
                "accepted": accepted,
                "rule": "No repeat regresses; mean gain at least one question per repeat, with the evolved harness fixed",
                "evaluation_identity_sha256": digest(identity),
                "adapter_sha256": adapter["artifact_sha256"],
                "selected_model": models["candidate" if accepted else "parent"].model_dump(),
                "policy": collection["policy"],
                "test_opened": False,
            }
            pin(output / "decision.json", decision)
            next_config = memory_config.model_copy(
                update={
                    "task": models["candidate" if accepted else "parent"],
                    "prior_policy": policies["evolved"],
                    "initial_baseline": "prior_search_best",
                    "model_context": {
                        "origin": "lora_sft",
                        "training_config": config.model_dump(),
                        "training_metrics": adapter.get("metrics", {}),
                        "optimizer_steps": adapter.get("optimizer_steps"),
                        "train_samples": read_json(output / "dataset.json")["samples"],
                        "candidate_model": adapter["served_model"],
                        "candidate_adapter_sha256": adapter["artifact_sha256"],
                        "accepted": accepted,
                        "development_metrics": metrics,
                        "development_paired_gain": pair,
                        "test_opened": False,
                    },
                    "meta": memory_config.meta.model_copy(
                        update={"state_dir": f".local/claude-sft/{digest(identity)[:12]}"}
                    ),
                }
            )
            write_json(output / "next-search.json", next_config.model_dump())
            report["decision"] = decision
        else:
            report["decision"] = read_json(output / "decision.json")
        write_json(directory / "report.json", report)
        return report


def local_key():
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
