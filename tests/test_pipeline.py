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
            "grpo": {**shared, "train_samples": 2, "prompts_per_batch": 2},
            "sft_train_samples": 3,
            "min_gain_questions": 0,
            "limits": {"rounds": 4, "max_meta_calls": 4, "max_task_calls": 50},
        }
    )
    pipeline = Pipeline(config, data, output, ROOT)

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

    def trainer(root, server, method, config, training_data, destination, deadline):
        train_calls.append((method, config, destination))
        name = "grpo_candidate"
        identity = {
            "config": config.model_dump(),
            "base_files": pipeline.base_files,
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

    monkeypatch.setattr("rsi_base.pipeline.ClaudeMeta", lambda *args, **kwargs: Meta())
    monkeypatch.setattr("rsi_base.pipeline.train", trainer)
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
