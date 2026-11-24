import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rsi_base.loop import file_hash
from rsi_base.pipeline import Pipeline, PipelineConfig, PipelineProposal
from rsi_base.storage import read_json, read_jsonl, write_json
from rsi_base.types import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "proposal",
    [
        {"action": "sft"},
        {"action": "sft", "data_source": "online", "system_prompt": "new"},
        {"action": "grpo", "data_source": "offline"},
        {"action": "harness", "data_source": "offline", "system_prompt": "new"},
        {"action": "harness", "system_prompt": " "},
    ],
)
def test_proposal_cannot_change_both_components(proposal):
    with pytest.raises(ValidationError):
        PipelineProposal(reason="Try an update", **proposal)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def numeric_data(data):
    data.mkdir()
    for split, prefix, count in (("train", "train", 6), ("val", "dev", 2), ("test", "heldout", 2)):
        write_rows(
            data / f"{split}.jsonl",
            [
                {
                    "id": f"{prefix}-{i}",
                    "question": f"{prefix} question {i}\u2028Solve for x.",
                    "answer": "2",
                }
                for i in range(count)
            ],
        )
    write_rows(
        data / "references.jsonl",
        [
            {"id": f"train-{i}", "response": "Reference steps\u2028Therefore\n#### 2"}
            for i in range(6)
        ],
    )


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_pipeline_routes_supervision_and_continues_selected_harness(tmp_path, monkeypatch, mode):
    data, base, output = (tmp_path / name for name in ("data", "base", "run"))
    numeric_data(data)
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"fake base")
    shared = {"base_directory": str(base), "training_mode": mode}
    config = PipelineConfig.model_validate(
        {
            "meta": {"model": "fake-meta", "base_url": "http://localhost"},
            "agent": {"system_prompt": "initial"},
            "sft": {**shared, "min_samples": 1, "collection_repeats": 2},
            "sft_overrides": {"online": {"learning_rate": 0.0003}},
            "grpo": {**shared, "train_samples": 2, "prompts_per_batch": 2},
            "sft_train_samples": 3,
            "min_gain_questions": 0,
            "limits": {"rounds": 4, "max_meta_calls": 4, "max_task_calls": 50},
        }
    )

    task_calls, meta_calls, train_calls = [], [], []
    proposals = iter(
        [
            PipelineProposal(action="harness", system_prompt="revised", reason="Fix formatting"),
            PipelineProposal(action="sft", data_source="offline", reason="Use supervision"),
            PipelineProposal(action="grpo", data_source="online", reason="Use reward"),
            PipelineProposal(
                action="sft", data_source="online", reason="Imitate verified rollouts"
            ),
        ]
    )

    class Meta:
        def propose(self, current, evidence, directory):
            meta_calls.append(evidence)
            assert "heldout" not in json.dumps(evidence)
            assert evidence["available_strategies"] == [
                "harness",
                "offline_sft",
                "online_sft",
                "grpo",
            ]
            return next(proposals), {}

    class Task:
        def __init__(self, model, temperature):
            self.model, self.temperature = model, temperature

        def run(self, spec, question):
            task_calls.append((self.model.name, spec.system_prompt, question))
            wrong = question.startswith("dev question 0") and (
                spec.system_prompt == "initial" or self.model.name == "online_candidate"
            )
            unfinished = (
                question.startswith("train")
                and self.temperature > 0
                and not (
                    question.startswith("train question 0")
                    or question.startswith("train question 1")
                    or question.startswith("train question 2")
                )
            )
            return {
                "response": f"Task solution\u2028Final:\n#### {3 if wrong else 2}",
                "response_model": self.model.name,
                "finish_reason": "length" if unfinished else "stop",
                "messages": [
                    {"role": "system", "content": spec.system_prompt},
                    {"role": "user", "content": question},
                ],
            }

    def trainer(root, method, config, training_data, destination, deadline, **resources):
        train_calls.append((method, config, destination))
        name = "grpo_candidate"
        identity = {
            "config": config.model_dump(),
            "base_files": pipeline.components.base_files,
            config.parent_hash_key: (
                read_json(Path(config.parent_path) / "manifest.json")["artifact_sha256"]
                if config.parent_path
                else None
            ),
        }
        if method == "sft":
            collection = destination / "collection_identity.json"
            name = f"{read_json(collection)['data_source']}_candidate"
            identity.update(
                collection_sha256=file_hash(collection),
                dataset_sha256=file_hash(destination / "train.jsonl"),
            )
        else:
            identity["train_split_sha256"] = file_hash(training_data / "train.jsonl")
            write_json(destination / "rollout_stats.json", {"steps": [], "mixed_reward_groups": 1})
        artifact = destination / config.artifact_name
        artifact.mkdir()
        names = (
            ["model.safetensors", "config.json", "tokenizer_config.json"]
            if mode == "full"
            else ["adapter_model.safetensors", "adapter_config.json"]
        )
        for filename in names:
            (artifact / filename).write_text(name)
        files = {filename: file_hash(artifact / filename) for filename in names}
        write_json(
            artifact / "manifest.json",
            {
                "identity": identity,
                "files": files,
                "artifact_sha256": digest(files),
                "served_model": name,
                "optimizer_steps": 1,
            },
        )
        return 0, destination / "fake-trainer.log"

    monkeypatch.setattr("rsi_base.pipeline_setup.ClaudeMeta", lambda *args, **kwargs: Meta())
    monkeypatch.setattr("rsi_base.pipeline_strategies.run_training", trainer)
    pipeline = Pipeline(config, data, output, ROOT)
    monkeypatch.setattr(
        pipeline.server, "agent", lambda model, temperature, deadline: Task(model, temperature)
    )
    with pytest.raises(ValueError, match="Complete development"):
        pipeline.test()
    frozen = pipeline.run()
    assert [step["strategy"] for step in frozen["rounds"]] == [
        "harness",
        "offline_sft",
        "grpo",
        "online_sft",
    ]
    assert [step["accepted"] for step in frozen["rounds"]] == [True, True, True, False]
    assert frozen["selected"]["model"]["name"] == "grpo_candidate"
    assert frozen["selected"]["agent"]["system_prompt"] == "revised"
    assert all(not question.startswith("heldout") for _, _, question in task_calls)
    (_, offline, first_dir), (_, grpo, second_dir), (_, online, third_dir) = train_calls
    assert offline.parent_path is None
    assert grpo.parent_path == str(first_dir / offline.artifact_name)
    assert online.parent_path == str(second_dir / grpo.artifact_name)
    assert offline.learning_rate == 0.0002
    assert online.learning_rate == 0.0003
    assert grpo.system_prompt == "revised"
    assert len(read_jsonl(first_dir / "train.jsonl")) == 3
    assert all(
        row["messages"][-1]["content"].startswith("Reference steps")
        for row in read_jsonl(first_dir / "train.jsonl")
    )
    online_rows = read_jsonl(third_dir / "train.jsonl")
    assert online_rows and len(online_rows) == len({row["problem_id"] for row in online_rows})
    assert all(row["messages"][-1]["content"].startswith("Task solution") for row in online_rows)
    assert all(row["messages"][0]["content"] == "revised" for row in online_rows)
    assert all(
        name == "grpo_candidate" for name, _, question in task_calls if question.startswith("train")
    )
    training_calls = len(task_calls)
    assert pipeline.run(resume=True) == frozen
    assert len(train_calls) == 3 and len(meta_calls) == 4 and len(task_calls) == training_calls
    report = pipeline.test()
    assert report["selected_round"] == 3 and report["comparison"]["questions"] == 2
    assert all(question.startswith("heldout") for _, _, question in task_calls[training_calls:])
    frozen["selected"] = frozen["initial"]
    write_json(output / "frozen.json", frozen)
    with pytest.raises(ValueError, match="inputs changed"):
        pipeline.test()


def test_pipeline_can_load_components_without_trainers_or_local_weights(tmp_path, monkeypatch):
    module = tmp_path / "configured_rsi_test.py"
    module.write_text(
        """
from rsi_base.pipeline_types import Strategy, TaskState
from rsi_base.storage import read_jsonl
from rsi_base.types import AgentSpec

def loader(path):
    return read_jsonl(path)

def score(response, answer):
    return {"score": int(response == answer), "status": "scored"}

class Task:
    def agent(self, model, *, temperature, deadline):
        self.name = model.name
        return self

    def run(self, spec, question):
        return {
            "response": spec.system_prompt, "response_model": self.name, "finish_reason": "stop"
        }

    def release(self, devices):
        raise AssertionError("Prompt-only experiments need no trainer resources")

    def stop(self):
        pass

def make_task(root, output, runtime, model):
    return Task()

def make_strategy(root, data, prompt):
    assert not data.train and not data.references

    def apply(step, proposal):
        agent = AgentSpec(system_prompt=prompt)
        return TaskState(agent=agent, model=step.parent.model), {"custom_update": True}

    return Strategy('Use action "revise".', apply)

class Planner:
    def propose(self, agent, evidence, directory):
        assert evidence["available_strategies"] == ["revise"]
        assert "heldout" not in str(evidence)
        return {"action": "revise", "reason": "Try configured prompt", "parameters": {}}, {}

def make_planner(root):
    return Planner()

def make_selection(gain):
    return lambda comparison: comparison["gain_questions"] == gain
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    data, output = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    for split, identifier in (("val", "dev"), ("test", "heldout")):
        write_rows(
            data / f"{split}.jsonl",
            [{"id": identifier, "question": identifier, "answer": "positive"}],
        )
    (data / "references.jsonl").write_text("Invalid unused references")
    config = PipelineConfig.model_validate(
        {
            "agent": {"system_prompt": "negative"},
            "model": {"base_directory": str(tmp_path / "missing")},
            "runtime": {"task_base_url": "http://localhost:9999/v1"},
            "benchmark": {
                "loader": "configured_rsi_test:loader",
                "scorer": "configured_rsi_test:score",
            },
            "task": {"factory": "configured_rsi_test:make_task"},
            "planner": {"factory": "configured_rsi_test:make_planner"},
            "selection": {"factory": "configured_rsi_test:make_selection", "options": {"gain": 1}},
            "strategies": [
                {
                    "name": "revise",
                    "factory": "configured_rsi_test:make_strategy",
                    "options": {"prompt": "positive"},
                }
            ],
            "sft": {"max_steps": "unused"},
            "grpo": {"train_samples": "unused"},
            "limits": {"rounds": 1, "max_meta_calls": 1, "max_task_calls": 2},
        }
    )
    pipeline = Pipeline(config, data, output, ROOT)
    frozen = pipeline.run()
    assert frozen["selected_round"] == 1
    assert frozen["rounds"][0]["training"] == {"custom_update": True}
    assert pipeline.test()["comparison"]["gain_questions"] == 1
    assert read_json(output / "run.json")["source"]["components"]["strategy:revise"] == file_hash(
        module
    )
    with module.open("a") as stream:
        stream.write("\n# New revision\n")
    with pytest.raises(ValueError, match="inputs changed"):
        Pipeline(config, data, output, ROOT).run(resume=True)
