"""Train-only trajectory collection and frozen model/harness comparison for SFT."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .agent import TaskAgent
from .agent import local_key as local_key
from .classification import MemoryHarness, MemoryPolicy, label_score, normalize
from .loop import file_hash
from .memory_audit import cluster_interval, paired_records
from .memory_loop import MemoryConfig, evaluate_policy, load_rows, stable_improvement
from .storage import Ledger, lock_directory, read_json, write_json
from .types import Limits, digest
from .weights import AdapterConfig as AdapterConfig
from .weights import load_adapter as load_adapter
from .weights import load_artifact, load_parent, parent_endpoint


class SFTConfig(AdapterConfig):
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
    collection_temperature: float | None = Field(default=None, ge=0, le=2)
    training_target: Literal["verified", "reference"] = "verified"
    evaluation_repeats: int = Field(default=3, ge=1, le=5)
    max_collection_calls: int = Field(default=2000, ge=1)
    max_evaluation_calls: int = Field(default=10000, ge=1)
    wall_seconds: float = Field(default=7200, gt=0)

    @field_validator("max_steps")
    @classmethod
    def steps(cls, value):
        if value == 0:
            raise ValueError("max_steps must be -1 (epochs) or positive")
        return value


def locked(output):
    return lock_directory(output, ".sft.lock")


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


def successes(records, training, labels, *, training_target="verified"):
    """Export final calls with verifier-accepted or explicitly supervised targets."""
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
        if record.get("score") != score["score"]:
            raise ValueError("Collection score differs from fixed verifier")
        if not score["score"] and training_target == "verified":
            reasons[score["status"]] += 1
            continue
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
        target = (
            normalize(problem["answer"]) if training_target == "reference" else final["response"]
        )
        if normalize(target) not in labels:
            raise ValueError("Training target is absent from the label set")
        key = digest([messages, normalize(target)])
        if key in seen:
            reasons["duplicate_messages"] += 1
            continue
        seen.add(key)
        samples.append(
            {
                "id": record["id"],
                "problem_id": problem["id"],
                "messages": [*messages, {"role": "assistant", "content": target}],
                "trace_sha256": digest(final),
                "observed_status": score["status"],
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
    weights = load_parent(config, root)
    parent_model = weights["served_model"] if weights else config.base_model
    parent_hash = weights["artifact_sha256"] if weights else None
    if memory_config.task.model != parent_model:
        raise ValueError("Generation model must match the training parent")
    rollout_settings = memory_config.task
    if config.collection_temperature is not None:
        if "temperature" in rollout_settings.extra_body:
            raise ValueError("Set collection_temperature outside task.extra_body")
        rollout_settings = rollout_settings.model_copy(
            update={"temperature": config.collection_temperature}
        )
    identity = {
        **source_identity(root),
        "config": config.model_dump(),
        "source_run": str(source_run.resolve()),
        "source_run_sha256": file_hash(source_run / "run.json"),
        "frozen_sha256": file_hash(source_run / "frozen.json"),
        "train_sha256": file_hash(data / "train.jsonl"),
        "labels_sha256": file_hash(data / "labels.json"),
        config.parent_hash_key: parent_hash,
        "split_sha256": parent["split_sha256"],
        "memory_config": memory_config.model_dump(),
        "rollout_settings": rollout_settings.model_dump(),
        "initial_policy": frozen["baseline"]["policy"],
        "policy": frozen["selected"]["policy"],
        "leave_one_out": True,
        "supervision": config.training_target,
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
                task or TaskAgent(rollout_settings, root),
                ledger,
                repeats=config.collection_repeats,
                seed=config.seed,
                leave_one_out=True,
            )
        finally:
            ledger.close()
        if config.parent_path or config.training_mode == "full":
            check_response_models(records, parent_model)
        samples, excluded = successes(
            records, training, labels, training_target=config.training_target
        )
        if len(samples) < config.min_samples:
            raise ValueError(f"Only {len(samples)} verified samples; need {config.min_samples}")
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples)
        (output / "train.jsonl").write_text(payload)
        report = {
            "identity_sha256": digest(identity),
            "train_sha256": file_hash(output / "train.jsonl"),
            "samples": len(samples),
            "unique_training_problems": len({row["problem_id"] for row in samples}),
            "targets_from": config.training_target,
            "observed_statuses_in_samples": dict(
                Counter(row["observed_status"] for row in samples)
            ),
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
        SFTConfig.model_validate(identity["config"]) != config
        or report["identity_sha256"] != digest(identity)
        or report["train_sha256"] != file_hash(output / "train.jsonl")
    ):
        raise ValueError("SFT dataset or configuration changed")
    rows = load_rows(output / "train.jsonl")
    if len(rows) != report["samples"] or len(rows) < config.min_samples:
        raise ValueError("SFT sample count changed")
    return report


def check_response_models(records, expected):
    if any(
        trace.get("response_model") != expected for record in records for trace in record["traces"]
    ):
        raise ValueError(f"Task endpoint did not report the requested model: {expected}")


def compare(config, data, output, root, *, base_url, parent_base_url=None, split="val", tasks=None):
    """Cross model and harness on development; test only after the decision is sealed."""
    data, output, root = Path(data), Path(output), Path(root)
    verify_dataset(output, config)
    collection = read_json(output / "collection_identity.json")
    artifact = load_artifact(output / config.artifact_name, config.training_mode)
    parent = load_parent(config, root)
    if (parent["artifact_sha256"] if parent else None) != collection[config.parent_hash_key]:
        raise ValueError("Training parent changed after collection")
    if (
        artifact["identity"]["dataset_sha256"] != file_hash(output / "train.jsonl")
        or SFTConfig.model_validate(artifact["identity"]["config"]) != config
        or artifact["identity"]["collection_sha256"]
        != file_hash(output / "collection_identity.json")
    ):
        raise ValueError("Model was trained on a different dataset")
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
            or decision[config.artifact_hash_key] != artifact["artifact_sha256"]
        ):
            raise ValueError("Frozen development decision changed")
    memory_config = MemoryConfig.model_validate(collection["memory_config"])
    settings = memory_config.task.model_dump()
    model_settings = type(memory_config.task)
    models = {
        "parent": model_settings.model_validate(
            {**settings, "base_url": parent_endpoint(config, base_url, parent_base_url)}
        ),
        "candidate": model_settings.model_validate(
            {**settings, "base_url": base_url, "model": artifact["served_model"]}
        ),
    }
    identity = {
        **source_identity(root),
        "collection_sha256": file_hash(output / "collection_identity.json"),
        config.artifact_hash_key: artifact["artifact_sha256"],
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
                    if (
                        model_name == "candidate"
                        or config.parent_path
                        or config.training_mode == "full"
                    ):
                        check_response_models(records[key], model.model)
        finally:
            ledger.close()
        pair = paired_records(records["parent_evolved"], records["candidate_evolved"])
        initial_pair = paired_records(records["parent_initial"], records["candidate_initial"])
        report = {
            "split": split,
            "test_status": memory_config.test_status,
            "metrics": metrics,
            f"{config.training_mode}_effect_with_evolved_harness": pair,
            f"{config.training_mode}_effect_with_initial_harness": initial_pair,
            "paired_question_intervals": {
                "evolved_harness": cluster_interval(pair["question_deltas"], seed=config.seed),
                "initial_harness": cluster_interval(
                    initial_pair["question_deltas"], seed=config.seed
                ),
            },
            "test_used_for_selection": False,
        }
        if split == "val":
            accepted = stable_improvement(metrics["candidate_evolved"], metrics["parent_evolved"])
            decision = {
                "accepted": accepted,
                "rule": "No repeat regresses; mean gain at least one question per repeat, with the evolved harness fixed",
                "evaluation_identity_sha256": digest(identity),
                config.artifact_hash_key: artifact["artifact_sha256"],
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
                        "origin": f"{config.training_mode}_sft",
                        "training_config": config.model_dump(),
                        "training_metrics": artifact.get("metrics", {}),
                        "optimizer_steps": artifact.get("optimizer_steps"),
                        "train_samples": read_json(output / "dataset.json")["samples"],
                        "candidate_model": artifact["served_model"],
                        f"candidate_{config.artifact_hash_key}": artifact["artifact_sha256"],
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
