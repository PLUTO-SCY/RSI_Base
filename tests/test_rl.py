import json

import pytest

from rsi_base import rl
from rsi_base.loop import file_hash
from rsi_base.storage import write_json
from rsi_base.types import digest


def fixture_artifact(directory, names, model, identity):
    directory.mkdir(parents=True)
    files = {}
    for name in names:
        (directory / name).write_text("fixture")
        files[name] = file_hash(directory / name)
    manifest = {
        "files": files,
        "artifact_sha256": digest(files),
        "served_model": model,
        "identity": identity,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_fixed_harness_evaluation_and_frozen_test_gate(tmp_path, monkeypatch, mode):
    parent_path = str(tmp_path / "previous" / "model") if mode == "full" else None
    config = rl.GRPOConfig(training_mode=mode, parent_checkpoint=parent_path)
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        row = {"id": split, "question": f"Numeric problem {split}", "answer": "2"}
        (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    run = tmp_path / "run"
    candidate = f"fixture-{mode}"
    training_config = config.model_dump()
    parent_sha, parent_model = None, config.base_model
    if mode == "full":
        names = ("config.json", "tokenizer_config.json", "model.safetensors")
        parent_model = "fixture-parent"
        parent_sha = fixture_artifact(
            tmp_path / "previous" / "model",
            names,
            parent_model,
            {"config": {"base_model": config.base_model}},
        )["artifact_sha256"]
    else:
        names = ("adapter_config.json", "adapter_model.safetensors")
        training_config.pop("training_mode")
        training_config.pop("parent_checkpoint")
    fixture_artifact(
        run / config.artifact_name,
        names,
        candidate,
        {
            "config": training_config,
            "train_split_sha256": file_hash(data / "train.jsonl"),
            config.parent_hash_key: parent_sha,
        },
    )
    requests = []

    class Task:
        def __init__(self, settings, root):
            self.settings = settings

        def run(self, spec, question):
            requests.append(
                (spec.system_prompt, question, self.settings.model, self.settings.base_url)
            )
            return {
                "response_model": self.settings.model,
                "response": "#### 2" if self.settings.model == candidate else "#### 1",
            }

    monkeypatch.setattr(rl, "TaskAgent", Task)
    arguments = (config, data, run, tmp_path, "http://localhost:18001/v1")
    with pytest.raises(ValueError, match="development evaluation"):
        rl.evaluate_grpo(*arguments, split="test")
    assert requests == []
    selection = rl.evaluate_grpo(*arguments, split="val")
    assert selection["decision"]["selected_model"] == candidate
    parent_url = "http://127.0.0.1:18000/v1" if mode == "full" else arguments[-1]
    assert requests == [
        (config.system_prompt, "Numeric problem val", parent_model, parent_url),
        (config.system_prompt, "Numeric problem val", candidate, arguments[-1]),
    ]
    held_out = rl.evaluate_grpo(*arguments, split="test")
    assert held_out["decision"] == selection["decision"]
    assert held_out["paired_gain"] == 1.0
