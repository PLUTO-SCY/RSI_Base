"""SFT contracts: train-only evidence, assistant masking, immutable artifacts and selection."""

import json
from collections import Counter
from pathlib import Path

import pytest

from rsi_base.classification import MemoryPolicy
from rsi_base.loop import file_hash
from rsi_base.memory_loop import MemoryConfig
from rsi_base.sft import (
    SFTConfig,
    collect,
    compare,
    load_adapter,
    successes,
    tokenize_sample,
    verify_dataset,
)
from rsi_base.storage import read_json, write_json
from rsi_base.types import MetaSettings, ModelSettings, digest


class Task:
    def __init__(self, settings, *, correct=True):
        self.settings, self.correct, self.calls = settings, correct, []

    def run(self, spec, question):
        self.calls.append(question)
        query = question.split("\nQuery:\n")[-1].split("\n\nChoose")[0]
        response = "fruit" if "apple" in query or "pear" in query else "machine"
        if not self.correct:
            response = "machine" if response == "fruit" else "fruit"
        return {
            "messages": [
                {"role": "system", "content": spec.system_prompt},
                {"role": "user", "content": question},
            ],
            "response": response,
            "finish_reason": "stop",
            "response_model": self.settings.model,
        }


@pytest.fixture
def case(tmp_path):
    root, data, source, output = [tmp_path / name for name in ["root", "data", "source", "sft"]]
    for directory in [root / "src", root / "requirements", data, source]:
        directory.mkdir(parents=True)
    (root / "uv.lock").write_text("core lock")
    (root / "requirements/train.lock").write_text("training lock")
    (root / "src/contract.py").write_text("VERSION = 1\n")
    training = [
        {"id": "t1", "question": "apple orchard", "answer": "fruit"},
        {"id": "t2", "question": "diesel engine", "answer": "machine"},
        {"id": "t3", "question": "pear garden", "answer": "fruit"},
    ]
    for split, rows in [
        ("train", training),
        ("val", [{"id": "v1", "question": "apple selection", "answer": "fruit"}]),
        ("test", [{"id": "e1", "question": "pear final", "answer": "fruit"}]),
    ]:
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(data / "labels.json", ["fruit", "machine"])
    memory_config = MemoryConfig(
        task=ModelSettings(model="fixture-base", base_url="http://127.0.0.1:18001/v1"),
        meta=MetaSettings(model="fixture-meta", base_url="https://example.com"),
        rounds=1,
        validation_repeats=2,
    )
    write_json(
        source / "run.json",
        {
            "config": memory_config.model_dump(),
            "split_sha256": {
                split: file_hash(data / f"{split}.jsonl") for split in ["train", "val", "test"]
            },
            "labels_sha256": file_hash(data / "labels.json"),
        },
    )
    policy = MemoryPolicy(examples=2)
    write_json(
        source / "frozen.json",
        {
            "baseline": {"policy": MemoryPolicy(retrieval="none", examples=0).model_dump()},
            "selected": {"policy": policy.model_dump()},
        },
    )
    config = SFTConfig(base_model="fixture-base", min_samples=1, evaluation_repeats=2)
    return {
        "root": root,
        "data": data,
        "source_run": source,
        "output": output,
        "config": config,
        "task": Task(memory_config.task),
        "training": training,
        "memory_config": memory_config,
    }


def collection(case):
    return collect(
        **{key: case[key] for key in ["config", "source_run", "data", "output", "root", "task"]}
    )


def artifact(case):
    output = case["output"]
    directory = output / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_bytes(b"fixture weights")
    files = {p.name: file_hash(p) for p in directory.iterdir()}
    value = {
        "identity": {
            "config": case["config"].model_dump(),
            "dataset_sha256": file_hash(output / "train.jsonl"),
            "collection_sha256": file_hash(output / "collection_identity.json"),
        },
        "files": files,
        "artifact_sha256": digest(files),
        "served_model": "fixture-lora",
    }
    write_json(directory / "manifest.json", value)
    return value


def tasks(case, *, improves=True):
    settings = case["memory_config"].task
    return {
        "parent": Task(settings, correct=not improves),
        "candidate": Task(settings.model_copy(update={"model": "fixture-lora"}), correct=improves),
    }


def evaluate(case, workers, split="val"):
    return compare(
        case["config"],
        case["data"],
        case["output"],
        case["root"],
        base_url="http://127.0.0.1:18001/v1",
        split=split,
        tasks=workers,
    )


def test_collection_uses_training_only_leave_one_out_and_reuses(case, monkeypatch):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.parent == case["data"] and path.name in {"val.jsonl", "test.jsonl"}:
            raise AssertionError("Selection/final-test content entered SFT collection")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    report = collection(case)
    assert report["samples"] == 3
    assert report["development_opened"] is report["test_opened"] is False
    assert len(case["task"].calls) == 3
    for prompt in case["task"].calls:
        query = prompt.split("\nQuery:\n")[-1].split("\n\nChoose")[0]
        assert f"Description: {query}\nLabel:" not in prompt
    assert collection(case) == report
    assert len(case["task"].calls) == 3
    with (case["output"] / "train.jsonl").open("a") as handle:
        handle.write("corruption\n")
    with pytest.raises(ValueError, match="changed"):
        verify_dataset(case["output"], case["config"])


def test_collection_refuses_changed_hyperparameters(case):
    collection(case)
    case["config"] = case["config"].model_copy(update={"learning_rate": 0.0001})
    with pytest.raises(ValueError, match="Pinned"):
        collection(case)


def test_export_keeps_only_correct_final_call_and_rejects_foreign_cases():
    training = [{"id": "t", "question": "apple", "answer": "fruit"}]
    messages = [{"role": "system", "content": "classify"}, {"role": "user", "content": "apple"}]
    final = {"messages": messages, "response": "fruit", "finish_reason": "stop"}
    correct = {
        "id": "0:t",
        "problem_id": "t",
        "traces": [{**final, "response": "machine"}, final],
        "response": "fruit",
        "score": 1,
        "memory_ids": [],
    }
    wrong = {
        **correct,
        "id": "1:t",
        "traces": [{**final, "response": "machine"}],
        "response": "machine",
        "score": 0,
    }
    truncated = {**correct, "id": "2:t", "traces": [{**final, "finish_reason": "length"}]}
    samples, rejected = successes([correct, wrong, truncated], training, ["fruit", "machine"])
    assert len(samples) == 1 and samples[0]["messages"][-1] == {
        "role": "assistant",
        "content": "fruit",
    }
    assert len(samples[0]["messages"]) == 3
    assert rejected == {"wrong": 1, "unfinished_trace": 1}
    with pytest.raises(ValueError, match="Non-training"):
        successes([{**correct, "problem_id": "test-case"}], training, ["fruit", "machine"])
    with pytest.raises(ValueError, match="itself"):
        successes([{**correct, "memory_ids": ["t"]}], training, ["fruit", "machine"])


def test_assistant_boundary_mask_and_overlength_are_not_truncated():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [11, 12, 13] if kwargs["add_generation_prompt"] else [11, 12, 13, 91, 92]

    messages = [{"role": role, "content": "text"} for role in ["system", "user", "assistant"]]
    sample = tokenize_sample(Tokenizer(), messages, 5)
    assert sample["input_ids"] == [11, 12, 13, 91, 92]
    assert sample["labels"] == [-100, -100, -100, 91, 92]
    assert tokenize_sample(Tokenizer(), messages, 4) is None


@pytest.mark.parametrize("improves", [True, False])
def test_crossed_evaluation_freezes_selection_and_recovery(case, improves):
    collection(case)
    artifact(case)
    workers = tasks(case, improves=improves)
    with pytest.raises(ValueError, match="Freeze"):
        evaluate(case, workers, "test")
    report = evaluate(case, workers)
    assert set(report["metrics"]) == {
        "parent_initial",
        "parent_evolved",
        "candidate_initial",
        "candidate_evolved",
    }
    assert report["decision"]["accepted"] is improves
    assert report["decision"]["test_opened"] is False
    next_config = MemoryConfig.model_validate(read_json(case["output"] / "next-search.json"))
    assert next_config.task.model == ("fixture-lora" if improves else "fixture-base")
    assert next_config.initial_baseline == "prior_search_best"
    assert next_config.model_context["accepted"] is improves
    assert next_config.model_context["test_opened"] is False
    before = {name: len(worker.calls) for name, worker in workers.items()}
    assert evaluate(case, workers) == report
    assert before == {name: len(worker.calls) for name, worker in workers.items()}
    final = evaluate(case, workers, "test")
    assert final["decision"] == report["decision"]
    assert final["test_used_for_selection"] is False
    events = [
        json.loads(line)
        for line in (case["output"] / "evaluation/val/calls.jsonl").read_text().splitlines()
    ]
    assert (
        Counter(row["event"] for row in events)["started"]
        == Counter(row["event"] for row in events)["finished"]
    )


def test_test_access_refuses_modified_decision_or_adapter(case):
    collection(case)
    artifact(case)
    workers = tasks(case)
    evaluate(case, workers)
    path = case["output"] / "decision.json"
    decision = read_json(path)
    write_json(path, {**decision, "accepted": False})
    with pytest.raises(ValueError, match="decision changed"):
        evaluate(case, workers, "test")
    (case["output"] / "adapter/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_adapter(case["output"] / "adapter")


def test_distributed_launcher_separates_worker_arguments(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    launcher = tmp_path / "scripts/train_lora.sh"
    shutil.copyfile(root / "scripts/train_lora.sh", launcher)
    interpreter = tmp_path / ".venv-train/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import argparse,json,os,sys\n"
        "parser=argparse.ArgumentParser()\n"
        "parser.add_argument('--standalone',action='store_true')\n"
        "parser.add_argument('--nnodes')\n"
        "parser.add_argument('--nproc_per_node')\n"
        "parser.add_argument('--module',action='store_true')\n"
        "parser.add_argument('--run-path')\n"
        "parser.add_argument('training_script')\n"
        "parser.add_argument('training_args',nargs=argparse.REMAINDER)\n"
        "args=parser.parse_args(sys.argv[3:])\n"
        "print(json.dumps({'module':args.training_script,'args':args.training_args,'gpus':os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    interpreter.chmod(0o755)
    monkeypatch.setenv("RSI_TRAIN_GPUS", "1,3")
    monkeypatch.setenv("RSI_TRAIN_PROCESSES", "2")
    result = subprocess.run(
        ["sh", str(launcher), "--run", "runs/example"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "module": "rsi_base.lora_train",
        "args": ["--run", "runs/example"],
        "gpus": "1,3",
    }
