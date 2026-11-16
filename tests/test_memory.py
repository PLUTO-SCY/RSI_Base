import json

import pytest
from pydantic import ValidationError

from rsi_base.classification import MemoryBatch, MemoryHarness, MemoryPolicy, label_score
from rsi_base.improve import parse_proposal
from rsi_base.memory_loop import evaluate_policy, stable_improvement
from rsi_base.storage import Ledger
from rsi_base.types import Limits


@pytest.fixture
def training():
    return [
        {"id": "train-a", "question": "red apple orchard", "answer": "fruit"},
        {"id": "train-b", "question": "yellow banana orchard", "answer": "fruit"},
        {"id": "train-c", "question": "blue car engine", "answer": "machine"},
    ]


def test_label_contract_rejects_ambiguity_and_unknown_labels():
    labels = ["fruit", "machine"]
    assert label_score("[LABEL] FRUIT [/LABEL]", "fruit", labels)["score"] == 1
    for response in ["fruit", "[LABEL]other[/LABEL]", "[LABEL]fruit[/LABEL][LABEL]machine[/LABEL]"]:
        assert label_score(response, "fruit", labels)["status"] == "format_error"


def test_retrieval_is_training_only_and_leave_one_out(training):
    harness = MemoryHarness(training, ["fruit", "machine"])
    policy = MemoryPolicy(examples=3)
    prompt, used = harness.context("red apple orchard", policy, exclude_id="train-a")
    assert "train-a" not in used
    assert "red apple orchard\nLabel:" not in prompt
    assert set(used) <= {"train-b", "train-c"}
    _, limited = harness.context("apple", MemoryPolicy(examples=3, max_per_label=1))
    assert len(limited) == 2


@pytest.mark.parametrize("workflow,expected", [("direct", 1), ("verify", 2), ("rewrite", 2)])
def test_harness_excludes_query_reference_and_bounds_calls(training, workflow, expected):
    captured = []

    def call(stage, spec, question):
        captured.append((stage, question))
        return {"response": "[LABEL]fruit[/LABEL]"}

    harness = MemoryHarness(training, ["fruit", "machine"])
    row = {"id": "heldout", "question": "apple on tree", "answer": "PRIVATE_REFERENCE"}
    result = harness.run(MemoryPolicy(workflow=workflow), row, call)
    assert result["model_calls"] == expected
    assert "PRIVATE_REFERENCE" not in json.dumps(captured)
    assert all(identifier.startswith("train-") for identifier in result["memory_ids"])


def test_nested_policy_schema_cannot_change_model_or_evaluator():
    with pytest.raises(ValidationError):
        MemoryPolicy.model_validate({"model": "different"})
    value = {
        "candidates": [
            {"policy": MemoryPolicy().model_dump(), "reason": "a"},
            {"policy": MemoryPolicy(examples=16).model_dump(), "reason": "b"},
        ]
    }
    assert len(parse_proposal(json.dumps(value), MemoryBatch).candidates) == 2
    with pytest.raises(ValidationError):
        MemoryBatch.model_validate({"candidates": value["candidates"][:1]})


def test_selector_requires_gain_without_repeat_regression():
    parent = {"correct_by_repeat": [2, 2, 2]}
    assert stable_improvement({"correct_by_repeat": [3, 3, 3]}, parent)
    assert not stable_improvement({"correct_by_repeat": [4, 1, 4]}, parent)
    assert not stable_improvement({"correct_by_repeat": [2, 2, 3]}, parent)


def test_two_step_resume_keeps_finished_first_call(training, tmp_path):
    class Flaky:
        calls = []
        fail = True

        def run(self, spec, question):
            self.calls.append(question)
            if "Draft label" in question and self.fail:
                self.fail = False
                raise ConnectionError("interrupted second stage")
            return {"response": "[LABEL]fruit[/LABEL]"}

    task = Flaky()
    harness = MemoryHarness(training, ["fruit", "machine"])
    problems = [{"id": "q", "question": "apple on a tree", "answer": "fruit"}]
    ledger = Ledger(tmp_path / "calls.jsonl", Limits(max_task_calls=10))
    policy = MemoryPolicy(workflow="verify")
    with pytest.raises(ConnectionError):
        evaluate_policy(tmp_path / "eval", problems, policy, harness, task, ledger, repeats=1)
    _, summary = evaluate_policy(
        tmp_path / "eval", problems, policy, harness, task, ledger, repeats=1
    )
    assert summary["accuracy"] == 1
    assert len(task.calls) == 3
    assert "Draft label" not in task.calls[0]
    assert all("Draft label" in q for q in task.calls[1:])
    _, again = evaluate_policy(
        tmp_path / "eval", problems, policy, harness, task, ledger, repeats=1
    )
    assert again == summary and len(task.calls) == 3
    with pytest.raises(ValueError, match="inputs changed"):
        evaluate_policy(
            tmp_path / "eval", problems, MemoryPolicy(), harness, task, ledger, repeats=1
        )
