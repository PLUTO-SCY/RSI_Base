import json

import pytest
from pydantic import ValidationError

from rsi_base.classification import (
    MemoryBatch,
    MemoryHarness,
    MemoryPolicy,
    SingleMemoryBatch,
    effective_policy,
    label_score,
)
from rsi_base.improve import parse_proposal
from rsi_base.memory_audit import cluster_interval, finalize, paired_records
from rsi_base.memory_loop import MemoryConfig, evaluate_policy, stable_improvement
from rsi_base.memory_search import local_neighbor, search, training_feedback
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
    assert label_score(" FRUIT ", "fruit", labels)["score"] == 1
    for response in ["[LABEL]fruit[/LABEL]", "other", "fruit or machine"]:
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
        return {"response": "fruit"}

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
            return {"response": "fruit"}

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


def test_finalization_refuses_unfrozen_search_before_loading_test(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        finalize([tmp_path / "unfinished"], tmp_path / "missing-data", tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


def test_paired_uncertainty_clusters_repeats_by_question():
    baseline = [
        {"id": f"{r}:{p}", "problem_id": p, "repeat": r, "score": 0}
        for r in range(3)
        for p in ["a", "b"]
    ]
    selected = [{**row, "score": int(row["problem_id"] == "a")} for row in baseline]
    paired = paired_records(baseline, selected)
    assert paired["question_deltas"] == {"a": 1, "b": 0}
    assert paired["mean_delta"] == 0.5
    interval = cluster_interval(paired["question_deltas"], samples=1000)
    assert interval["mean"] == 0.5
    assert interval["low"] == 0 and interval["high"] == 1


@pytest.mark.parametrize(
    ("strategy", "fail_meta"), [("meta", False), ("mixed", False), ("mixed", True)]
)
def test_memory_search_feedback_boundary_selection_and_frozen_resume(
    tmp_path, monkeypatch, strategy, fail_meta
):
    from rsi_base.storage import write_json

    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "uv.lock").write_text("fixture")
    write_json(data / "labels.json", ["fruit", "machine"])
    write_json(data / "manifest.json", {"fixture": True})
    for split, size in [("train", 4), ("val", 2), ("test", 1)]:
        rows = [
            {"id": f"{split}-{i}", "question": f"{split.upper()}_PRIVATE_{i}", "answer": "fruit"}
            for i in range(size)
        ]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    class Task:
        calls = 0

        def run(self, spec, question):
            self.calls += 1
            return {"response": "fruit" if "improved" in spec.system_prompt else "machine"}

    task = Task()
    monkeypatch.setattr("rsi_base.memory_loop.TaskAgent", lambda *_: task)

    class Meta:
        calls = 0

        def propose(self, current, evidence, output):
            self.calls += 1
            text = json.dumps(evidence)
            assert "VAL_PRIVATE" not in text and "TEST_PRIVATE" not in text
            assert "TRAIN_PRIVATE" in text
            if fail_meta:
                raise ValueError("Meta proposal exceeded its output token budget")
            items = [
                {
                    "policy": current.model_copy(update={"examples": 9}).model_dump(),
                    "reason": "fixture",
                },
                {
                    "policy": {"system_prompt": "improved generic classification"},
                    "reason": "fixture",
                },
            ]
            cls = SingleMemoryBatch if strategy == "mixed" else MemoryBatch
            return cls.model_validate(
                {"candidates": items[-1:] if strategy == "mixed" else items}
            ), {}

    config = MemoryConfig(
        task={"model": "fixture", "base_url": "http://localhost:8000"},
        meta={"model": "fixture", "base_url": "http://localhost:8000"},
        rounds=1,
        validation_repeats=2,
        diagnostic_queries=4,
        max_task_calls=200,
        proposal_strategy=strategy,
    )
    meta = Meta()
    first = search(config, data, tmp_path / "run", tmp_path, meta=meta)
    assert first["selected"]["summary"]["accuracy"] == (0 if fail_meta else 1)
    if fail_meta:
        assert (
            len(
                json.loads((tmp_path / "run/evolution/001/proposal.json").read_text())["candidates"]
            )
            == 1
        )
        assert (tmp_path / "run/evolution/001/invalid.json").exists()
    assert not first["test_opened"] and not first["weights_updated"]
    before = task.calls
    again = search(config, data, tmp_path / "run", tmp_path, meta=meta)
    assert again == first and task.calls == before and meta.calls == 1


def test_draft_label_changes_second_retrieval_and_preserves_leave_one_out(training):
    harness = MemoryHarness(training, ["fruit", "machine"])
    calls = []

    def call(stage, spec, question):
        calls.append((stage, question))
        return {"response": "fruit"}

    result = harness.run(
        MemoryPolicy(examples=1, workflow="verify", review_retrieval="draft_contrast"),
        training[0],
        call,
        leave_one_out=True,
    )
    assert result["model_calls"] == 2
    assert len(result["memory_by_stage"]["predict"]) == 1
    assert set(result["memory_by_stage"]["verify"]) == {"train-b", "train-c"}
    assert "train-a" not in result["memory_ids"]
    assert "DIFFERENT labels" in calls[1][1]


@pytest.mark.parametrize("layout", ["coverage", "contrast_pairs"])
def test_coverage_and_comparison_contexts_stay_within_budget(training, layout):
    harness = MemoryHarness(training, ["fruit", "machine"], context_chars=2000)
    context, used = harness.context("apple machine", MemoryPolicy(examples=2, layout=layout))
    assert len(context) <= 2000
    assert "Label: fruit" in context and "Label: machine" in context
    assert set(used) <= {row["id"] for row in training}
    if layout == "contrast_pairs":
        assert "Contrast:" in context


def test_effective_policy_ignores_unused_review_knobs():
    first = MemoryPolicy(
        workflow="direct", confirmers=3, challengers=16, review_retrieval="draft_contrast"
    )
    second = MemoryPolicy(workflow="direct")
    assert effective_policy(first) == effective_policy(second)
    assert effective_policy(
        MemoryPolicy(workflow="verify", review_retrieval="draft_contrast", confirmers=3)
    ) != effective_policy(
        MemoryPolicy(workflow="verify", review_retrieval="draft_contrast", confirmers=5)
    )


def test_local_neighbor_is_bounded_structural_and_label_independent():
    parent = MemoryPolicy(examples=32, max_per_label=4, workflow="verify")
    candidate = local_neighbor(parent, 1, {})
    assert candidate.policy.max_per_label == 3
    assert candidate.policy.system_prompt == parent.system_prompt
    assert (
        sum(a != parent.model_dump()[key] for key, a in candidate.policy.model_dump().items()) == 1
    )


def test_training_feedback_preserves_both_predictions_and_shared_memory(training):
    records = [
        {
            "problem_id": "train-a",
            "score": 0,
            "response": "machine",
            "memory_ids": ["train-b", "train-c"],
            "memory_by_stage": {"predict": ["train-b"], "verify": ["train-c", "train-b"]},
            "traces": [{"response": "fruit"}, {"response": "machine"}],
        }
    ]
    feedback = training_feedback(records, {r["id"]: r for r in training}, training, "verify")
    assert feedback["examples"][0]["steps"] == [
        {"stage": "predict", "response": "fruit", "finish_reason": None},
        {"stage": "verify", "response": "machine", "finish_reason": None},
    ]
    assert feedback["examples"][0]["memory_by_stage"]["verify"] == ["train-c", "train-b"]
    assert set(feedback["shared_training_memory"]) == {"train-b", "train-c"}
