import json
from types import SimpleNamespace

import pytest

from rsi_base.agent import TaskAgent
from rsi_base.benchmark import check_disjoint, load_problems, parse_number, score
from rsi_base.improve import ClaudeMeta, parse_proposal
from rsi_base.loop import build_evidence, evaluate, run
from rsi_base.storage import BudgetExceeded, Ledger
from rsi_base.types import AgentSpec, Limits, Proposal, RunConfig


def configuration(rounds=3):
    return RunConfig.model_validate(
        {
            "task": {"model": "fixture", "base_url": "http://localhost:8000/v1"},
            "meta": {"model": "fixture", "base_url": "http://localhost:8000"},
            "agent": {"system_prompt": "seed"},
            "limits": {
                "rounds": rounds,
                "max_task_calls": 20,
                "max_meta_calls": 10,
                "wall_seconds": 600,
            },
        }
    )


def datasets(tmp_path):
    for name, question in (("dev", "DEVELOPMENT task"), ("test", "HELDOUT task")):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps({"id": name, "question": question, "answer": "2"}) + "\n"
        )
    return tmp_path / "dev.jsonl", tmp_path / "test.jsonl"


class FakeTask:
    def __init__(self):
        self.calls = []

    def run(self, spec, question):
        self.calls.append((spec.system_prompt, question))
        return {
            "response": "#### 2" if spec.system_prompt == "improved" else "#### 1",
            "usage": {"completion_tokens": 3},
            "elapsed_seconds": 0,
        }


class FakeMeta:
    def __init__(self):
        self.calls = []

    def propose(self, current, evidence, output):
        self.calls.append((current, evidence))
        number = len(self.calls)
        if number == 3:
            return SimpleNamespace(
                model_dump=lambda: {
                    "system_prompt": "improved",
                    "reason": "invalid",
                    "model": "changed",
                }
            ), {}
        return Proposal(system_prompt="improved" if number == 1 else "worse", reason="fixture"), {}


@pytest.mark.parametrize(
    ("response", "target"),
    [
        ("work\n#### 1,234", "1234"),
        ("#### -0.5", "-1/2"),
        ("\\boxed{2}", "2"),
        ("$5", "5"),
        ("#### 6.", "6"),
    ],
)
def test_numeric_score(response, target):
    assert score(response, target)["score"] == 1


def test_score_requires_explicit_numeric_answer():
    assert score("I think it is 3", "3")["status"] == "format_error"
    assert score("#### 3", "4")["score"] == 0
    with pytest.raises(ValueError):
        parse_number("__import__('os').system('false')")


def test_proposal_cannot_modify_fixed_model():
    with pytest.raises(ValueError):
        parse_proposal('{"system_prompt":"x","reason":"y","model":"changed"}')


def test_meta_rejects_unsupported_sampling_settings():
    config = configuration().model_dump()
    config["meta"]["temperature"] = 0.5
    with pytest.raises(ValueError):
        RunConfig.model_validate(config)


def test_overlap_detected_by_question():
    with pytest.raises(ValueError):
        check_disjoint(
            [{"id": "a", "question": "Same  question"}], [{"id": "b", "question": "same question"}]
        )


def test_math_jsonl_preserves_unicode_line_separators(tmp_path):
    path = tmp_path / "numbers.jsonl"
    rows = [{"id": "case", "question": "Line one\u2028Line two", "answer": "2"}]
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_problems(path) == rows


def test_evaluation_never_caches_misserved_adapter(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", Limits())
    problems = [{"id": "case", "question": "How many?", "answer": "2"}]
    try:
        with pytest.raises(ValueError, match="did not serve requested model"):
            evaluate(
                tmp_path / "evaluation",
                problems,
                AgentSpec(system_prompt="fixed prompt"),
                FakeTask(),
                ledger,
                expected_model="rsi-grpo-example",
            )
    finally:
        ledger.close()
    assert not (tmp_path / "evaluation/results.jsonl").exists()


def test_loop_selection_rollback_isolation_and_resume(tmp_path):
    dev, test = datasets(tmp_path)
    task, meta = FakeTask(), FakeMeta()
    out = tmp_path / "run"
    first = run(configuration(), dev, test, out, tmp_path, task=task, meta=meta)
    assert first["selected_round"] == 1
    assert [s["decision"] for s in first["history"]] == [
        "baseline",
        "accepted",
        "retained_parent",
        "rejected_invalid_proposal",
    ]
    assert first["held_out"]["accuracy"] == 1
    assert first["weights_updated"] is False
    assert first["application_calls"] == {"task": 4, "meta": 3}
    assert all("HELDOUT" not in json.dumps(evidence) for _, evidence in meta.calls)
    before = (len(task.calls), len(meta.calls))
    resumed = run(configuration(), dev, test, out, tmp_path, resume=True, task=task, meta=meta)
    assert resumed == first
    assert before == (len(task.calls), len(meta.calls))


def test_resume_refuses_changed_configuration_and_tampering(tmp_path):
    dev, test = datasets(tmp_path)
    out = tmp_path / "run"
    run(configuration(0), dev, test, out, tmp_path, task=FakeTask(), meta=FakeMeta())
    with pytest.raises(ValueError, match="identical"):
        run(
            configuration(1),
            dev,
            test,
            out,
            tmp_path,
            resume=True,
            task=FakeTask(),
            meta=FakeMeta(),
        )
    result = out / "rounds/000/results.jsonl"
    result.write_text(result.read_text().replace("#### 1", "#### 2"))
    with pytest.raises(ValueError, match="checksum"):
        run(
            configuration(0),
            dev,
            test,
            out,
            tmp_path,
            resume=True,
            task=FakeTask(),
            meta=FakeMeta(),
        )


def test_interrupted_evaluation_resumes_without_repeating_completed_tasks(tmp_path):
    dev, test = datasets(tmp_path)

    class Flaky(FakeTask):
        fail = True

        def run(self, spec, question):
            if spec.system_prompt == "improved" and self.fail:
                self.fail = False
                raise ConnectionError("fixture interruption")
            return super().run(spec, question)

    task, meta = Flaky(), FakeMeta()
    out = tmp_path / "run"
    with pytest.raises(ConnectionError):
        run(configuration(1), dev, test, out, tmp_path, task=task, meta=meta)
    result = run(configuration(1), dev, test, out, tmp_path, resume=True, task=task, meta=meta)
    assert len(meta.calls) == 1
    assert task.calls.count(("seed", "DEVELOPMENT task")) == 1
    assert result["application_calls"] == {"task": 4, "meta": 1}


def test_failed_calls_consume_budget_across_resume(tmp_path):
    limits = Limits(max_task_calls=1)
    ledger = Ledger(tmp_path / "calls.jsonl", limits)
    ledger.start("task")
    again = Ledger(tmp_path / "calls.jsonl", limits)
    with pytest.raises(BudgetExceeded):
        again.start("task")


def test_wall_deadline_survives_hard_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("rsi_base.storage.time.time", lambda: 1000)
    limits = Limits(wall_seconds=10)
    Ledger(tmp_path / "calls.jsonl", limits).start("task")
    monkeypatch.setattr("rsi_base.storage.time.time", lambda: 1011)
    with pytest.raises(BudgetExceeded, match="wall-time"):
        Ledger(tmp_path / "calls.jsonl", limits).start("task")


def test_development_cannot_spend_final_evaluation_reservation(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl", Limits(max_task_calls=3))
    ledger.reserved_task_calls = 2
    ledger.start("task")
    with pytest.raises(BudgetExceeded):
        ledger.start("task")
    ledger.reserved_task_calls = 0
    ledger.start("task")
    ledger.start("task")


def test_feedback_prioritizes_late_errors_and_remembers_rejected_proposal(tmp_path):
    rows = [
        {
            "id": str(i),
            "question": f"dev {i}",
            "response": "#### 1",
            "score": int(i != 10),
            "status": "correct" if i != 10 else "wrong",
        }
        for i in range(12)
    ]
    problems = [{"id": r["id"], "answer": "2"} for r in rows]
    candidate = tmp_path / "001"
    candidate.mkdir()
    (candidate / "proposal.json").write_text(json.dumps({"system_prompt": "tried", "reason": "x"}))
    (candidate / "decision.json").write_text(
        json.dumps({"decision": "retained_parent", "candidate_accuracy": 0.5})
    )
    (candidate / "results.jsonl").write_text(json.dumps(rows[10]) + "\n")
    evidence = build_evidence(rows, problems, [], tmp_path)
    assert evidence["examples"][0]["id"] == "10"
    assert evidence["examples"][0]["reference_answer"] == "2"
    assert evidence["previous_candidates"][0]["proposal"]["system_prompt"] == "tried"
    assert evidence["previous_candidates"][0]["decision"] == "retained_parent"


def test_concurrent_batch_settles_successes_and_resumes_missing_rows(tmp_path):
    problems = [{"id": str(i), "question": str(i), "answer": "2"} for i in range(4)]

    class Partial:
        fail = True

        def run(self, spec, question):
            if question == "1" and self.fail:
                self.fail = False
                raise ConnectionError("one failed call")
            return {"response": "#### 2"}

    ledger = Ledger(tmp_path / "calls.jsonl", Limits(max_task_calls=10))
    task = Partial()
    with pytest.raises(ConnectionError):
        evaluate(tmp_path / "eval", problems, AgentSpec(system_prompt="x"), task, ledger, 3)
    assert [
        json.loads(line)["id"]
        for line in (tmp_path / "eval/results.jsonl").read_text().splitlines()
    ] == ["0", "2"]
    records, summary = evaluate(
        tmp_path / "eval", problems, AgentSpec(system_prompt="x"), task, ledger, 3
    )
    assert [r["id"] for r in records] == ["0", "1", "2", "3"]
    assert summary["accuracy"] == 1
    assert ledger.used["task"] == 5
    events = ledger.records()
    assert sum(r["event"] == "started" for r in events) == sum(
        r["event"] == "finished" for r in events
    )


def test_math_evaluator_is_used_by_both_development_and_final(tmp_path):
    class BoxedTask:
        def run(self, spec, question):
            return {"response": r"Work. \boxed{2}"}

    dev, test = datasets(tmp_path)
    config = configuration(0).model_copy(update={"evaluator": "math_numeric"})
    result = run(config, dev, test, tmp_path / "out", tmp_path, task=BoxedTask(), meta=FakeMeta())
    assert result["development_accuracy"] == 1
    assert result["held_out"]["accuracy"] == 1
    assert result["held_out"]["strict_format_compliant"] == 0
    with pytest.raises(ValueError, match="identical"):
        run(
            configuration(0),
            dev,
            test,
            tmp_path / "out",
            tmp_path,
            resume=True,
            task=BoxedTask(),
            meta=FakeMeta(),
        )


def test_fixed_output_contract_survives_strategy_changes(tmp_path, monkeypatch):
    captured = []

    def create(**kwargs):
        captured.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=r"\boxed{2}"), finish_reason="stop")
            ],
            usage=None,
            model="fixture",
        )

    monkeypatch.setenv("RSI_API_KEY", "fixture")
    monkeypatch.setattr(
        "rsi_base.agent.OpenAI",
        lambda **_: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    settings = configuration().task.model_copy(update={"user_suffix": "Put the answer in a box."})
    task = TaskAgent(settings, tmp_path)
    for strategy in ["Solve carefully.", "Check all cases."]:
        task.run(AgentSpec(system_prompt=strategy), "The question")
    assert captured[0][0] != captured[1][0]
    assert captured[0][1] == captured[1][1]
    assert captured[0][1]["content"] == "The question\n\nPut the answer in a box."
    with pytest.raises(ValueError):
        parse_proposal('{"system_prompt":"x","reason":"x","user_suffix":"change contract"}')


def test_loop_rejects_strategy_over_configured_character_budget(tmp_path):
    class TooLong:
        def propose(self, current, evidence, output):
            return Proposal(system_prompt="x" * 129, reason="too long"), {}

    dev, test = datasets(tmp_path)
    config = configuration(1)
    config = config.model_copy(
        update={"meta": config.meta.model_copy(update={"prompt_char_budget": 128})}
    )
    result = run(config, dev, test, tmp_path / "run", tmp_path, task=FakeTask(), meta=TooLong())
    assert result["history"][1]["decision"] == "rejected_invalid_proposal"
    assert result["application_calls"] == {"task": 2, "meta": 1}


def test_claude_uses_project_state_and_process_local_provider(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text("RSI_API_KEY=fixture-key\n")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unrelated.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-token")
    monkeypatch.setattr("rsi_base.improve.shutil.which", lambda _: "/fake/claude")
    captured = {}

    def fake_run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="fixture-version")
        captured.update(kwargs)
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "type": "result",
                    "is_error": False,
                    "num_turns": 1,
                    "result": '{"system_prompt":"improved","reason":"fixture"}',
                }
            ),
        )

    monkeypatch.setattr("rsi_base.improve.subprocess.run", fake_run)
    proposal, metadata = ClaudeMeta(configuration().meta, tmp_path).propose(
        AgentSpec(system_prompt="seed"), {"examples": []}, tmp_path / "result"
    )
    assert proposal.system_prompt == "improved"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8000"
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]
    assert captured["env"]["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path))
    assert "--bare" in captured["command"]
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
    assert metadata["global_configuration_unchanged"]
    import os

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://unrelated.example"


def test_claude_state_cannot_escape_project(tmp_path, monkeypatch):
    monkeypatch.setattr("rsi_base.improve.shutil.which", lambda _: "/fake/claude")
    config = configuration()
    settings = config.meta.model_copy(update={"state_dir": "../elsewhere"})
    (tmp_path / ".env.local").write_text("RSI_API_KEY=fixture-key\n")
    with pytest.raises(ValueError, match="inside"):
        ClaudeMeta(settings, tmp_path).propose(config.agent, [], tmp_path)


@pytest.mark.parametrize(
    ("message", "exception"),
    [
        ("Claude's response exceeded the 8192 output token maximum.", ValueError),
        ("Provider authorization failed", RuntimeError),
    ],
)
def test_meta_overflow_is_rejectable_and_other_provider_errors_still_stop(
    tmp_path, monkeypatch, message, exception
):
    (tmp_path / ".env.local").write_text("RSI_API_KEY=fixture-key\n")
    monkeypatch.setattr("rsi_base.improve.shutil.which", lambda _: "/fake/claude")
    monkeypatch.setattr(
        "rsi_base.improve.subprocess.run",
        lambda *_, **__: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"type": "result", "is_error": True, "result": message}),
            stderr="",
        ),
    )
    with pytest.raises(exception):
        ClaudeMeta(configuration().meta, tmp_path).propose(
            configuration().agent, {}, tmp_path / "out"
        )
    assert (tmp_path / "out/meta_result.json").exists()
