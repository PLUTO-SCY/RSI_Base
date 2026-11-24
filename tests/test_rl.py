import json

import pytest

from rsi_base import rl
from rsi_base.loop import file_hash
from rsi_base.storage import write_json
from rsi_base.types import digest


def test_fixed_harness_evaluation_and_frozen_test_gate(tmp_path, monkeypatch):
    config = rl.GRPOConfig()
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val", "test"):
        row = {"id": split, "question": f"Numeric problem {split}", "answer": "2"}
        (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    run = tmp_path / "run"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    files = {}
    for file in ("adapter_config.json", "adapter_model.safetensors"):
        (adapter / file).write_text("fixture")
        files[file] = file_hash(adapter / file)
    candidate = "fixture-adapter"
    write_json(
        adapter / "manifest.json",
        {
            "files": files,
            "artifact_sha256": digest(files),
            "served_model": candidate,
            "identity": {
                "config": config.model_dump(),
                "train_split_sha256": file_hash(data / "train.jsonl"),
                "parent_adapter_sha256": None,
            },
        },
    )
    requests = []

    class Task:
        def __init__(self, settings, root):
            self.model = settings.model

        def run(self, spec, question):
            requests.append((spec.system_prompt, question))
            return {
                "response_model": self.model,
                "response": "#### 2" if self.model == candidate else "#### 1",
            }

    monkeypatch.setattr(rl, "TaskAgent", Task)
    arguments = (config, data, run, tmp_path, "http://localhost:18001/v1")
    with pytest.raises(ValueError, match="development evaluation"):
        rl.evaluate_grpo(*arguments, split="test")
    assert requests == []
    selection = rl.evaluate_grpo(*arguments, split="val")
    assert selection["decision"]["selected_model"] == candidate
    assert requests == [(config.system_prompt, "Numeric problem val")] * 2
    held_out = rl.evaluate_grpo(*arguments, split="test")
    assert held_out["decision"] == selection["decision"]
    assert held_out["paired_gain"] == 1.0
