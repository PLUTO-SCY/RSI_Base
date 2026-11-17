import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from rsi_base.classification import MemoryHarness, MemoryPolicy, effective_policy
from rsi_base.memory_code import run_code
from rsi_base.memory_loop import MemoryConfig, evaluate_policy
from rsi_base.memory_search import search
from rsi_base.storage import Ledger, read_json, write_json
from rsi_base.types import Limits


class RecordingTask:
    def __init__(self):
        self.calls = 0

    def run(self, spec, question):
        self.calls += 1
        return {
            "response": "fruit" if "improved" in spec.system_prompt else "machine",
            "messages": [
                {"role": "system", "content": spec.system_prompt},
                {"role": "user", "content": question},
            ],
        }


def test_execution_cache_preserves_repeats_and_reuses_equivalent_layouts(tmp_path):
    training = [
        {"id": "a", "question": "apple orchard", "answer": "fruit"},
        {"id": "b", "question": "car engine", "answer": "machine"},
    ]
    harness = MemoryHarness(training, ["fruit", "machine"])
    task = RecordingTask()
    ledger = Ledger(tmp_path / "calls.jsonl", Limits(max_task_calls=20))
    cases = [{"id": "v", "question": "new apple", "answer": "fruit"}]
    first = MemoryPolicy(layout="coverage", examples=1, selection="diverse")
    _, a = evaluate_policy(tmp_path / "a", cases, first, harness, task, ledger, repeats=3)
    _, b = evaluate_policy(
        tmp_path / "b",
        cases,
        first.model_copy(update={"examples": 2}),
        harness,
        task,
        ledger,
        repeats=3,
    )
    assert task.calls == 3 and a["correct_by_repeat"] == b["correct_by_repeat"]
    assert len(list((tmp_path / "request_cache").rglob("*.json"))) == 3
    _, c = evaluate_policy(
        tmp_path / "c",
        cases,
        first.model_copy(update={"system_prompt": "improved"}),
        harness,
        task,
        ledger,
        repeats=3,
    )
    assert task.calls == 6 and c["accuracy"] == 1


def test_shared_cache_rejects_changed_response(tmp_path):
    bank = [{"id": "a", "question": "apple", "answer": "fruit"}]
    harness = MemoryHarness(bank, ["fruit"])
    ledger = Ledger(tmp_path / "calls.jsonl", Limits(max_task_calls=20))
    cases = [{"id": "v", "question": "new apple", "answer": "fruit"}]
    task = RecordingTask()
    evaluate_policy(tmp_path / "a", cases, MemoryPolicy(), harness, task, ledger, repeats=1)
    path = next((tmp_path / "request_cache").rglob("*.json"))
    value = read_json(path)
    value["result"]["response"] = "changed"
    write_json(path, value)
    with pytest.raises(ValueError, match="Shared request cache"):
        evaluate_policy(
            tmp_path / "b", cases, MemoryPolicy(examples=9), harness, task, ledger, repeats=1
        )


def test_contrast_pair_count_and_selection_aliases_canonicalize():
    a = effective_policy(MemoryPolicy(examples=7, selection="diverse", layout="contrast_pairs"))
    b = effective_policy(MemoryPolicy(examples=8, selection="top", layout="contrast_pairs"))
    assert a == b
    assert effective_policy(MemoryPolicy(retrieval_code="unused")) == effective_policy(
        MemoryPolicy()
    )


def test_tool_agent_reads_development_and_debugs_before_submission(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_json(data / "labels.json", ["fruit", "machine"])
    write_json(data / "manifest.json", {"fixture": True})
    (tmp_path / "uv.lock").write_text("fixture")
    source = tmp_path / "src/rsi_base"
    source.mkdir(parents=True)
    (source / "classification.py").write_text("visible harness source")
    (tmp_path / ".env.local").write_text("RSI_API_KEY=HIDDEN_CREDENTIAL")
    for split in ["train", "val"]:
        (data / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"{split}-{i}",
                        "question": f"{split.upper()}_PRIVATE_{i}",
                        "answer": "fruit",
                    }
                )
                + "\n"
                for i in range(4)
            )
        )
    (data / "test.jsonl").write_bytes(b"\xffTEST_MUST_NEVER_BE_OPENED")
    task = RecordingTask()
    monkeypatch.setattr("rsi_base.memory_loop.TaskAgent", lambda *_: task)
    monkeypatch.setattr("rsi_base.memory_agent.agent_random_policy", lambda _: MemoryPolicy())

    class Meta:
        calls = 0

        def propose(self, current, evidence, output, *, tool_config, tool_names, tool_env):
            self.calls += 1
            url = tool_config["mcpServers"]["rsi"]["args"][-1]

            def call(name, arguments):
                request = urllib.request.Request(
                    url,
                    data=json.dumps(
                        {"method": "tools/call", "params": {"name": name, "arguments": arguments}}
                    ).encode(),
                    headers={"Authorization": "Bearer " + tool_env["RSI_AGENT_TOKEN"]},
                )
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                return result

            catalog = call("list_artifacts", {})
            assert "HIDDEN_CREDENTIAL" not in json.dumps(
                catalog
            ) and "test.jsonl" not in json.dumps(catalog)
            assert "VAL_PRIVATE" in json.dumps(
                call("read_artifact", {"artifact": "development/cases.json"})
            )
            assert "visible harness source" in json.dumps(
                call("read_artifact", {"artifact": "source/classification.py"})
            )
            assert call("read_artifact", {"artifact": "../../.env.local"})["isError"]
            policy = current.model_copy(update={"system_prompt": "improved reusable classifier"})
            result = call(
                "evaluate_candidate", {"policy": policy.model_dump(), "reason": "fixture"}
            )
            identifier = json.loads(result["content"][0]["text"])["candidate_id"]
            full = call("get_case", {"candidate_id": identifier, "case_id": "val-0"})
            assert "messages" in json.dumps(full) and "reference" not in json.dumps(full)
            call("submit_candidates", {"candidate_ids": [identifier]})
            return "submitted", {}

    config = MemoryConfig(
        task={"model": "fixture", "base_url": "http://localhost:1"},
        meta={"model": "fixture", "base_url": "http://localhost:1"},
        proposal_strategy="agent",
        rounds=1,
        probe_queries=4,
        max_probes=2,
        validation_repeats=2,
        max_task_calls=300,
    )
    meta = Meta()
    first = search(config, data, tmp_path / "run", tmp_path, meta=meta)
    assert first["selected"]["summary"]["accuracy"] == 1
    before = task.calls
    assert search(config, data, tmp_path / "run", tmp_path, meta=meta) == first
    assert meta.calls == 1 and task.calls == before
    assert not first["test_opened"]


HAS_SANDBOX = (
    shutil.which("bwrap")
    or (Path(__file__).resolve().parents[1] / ".local/sandbox-tools/usr/bin/bwrap").exists()
)


@pytest.mark.skipif(not HAS_SANDBOX, reason="Optional namespace launcher is not installed")
def test_candidate_code_is_namespaced_and_cannot_read_host_file(tmp_path, monkeypatch):
    secret = tmp_path / "host-secret"
    secret.write_text("HIDDEN")
    monkeypatch.setenv("RSI_TEST_SECRET", "HIDDEN")
    code = f"""import os, socket
def rank(query, memory):
    assert not os.environ.get("RSI_TEST_SECRET")
    assert not os.path.exists({str(secret)!r})
    assert not os.path.exists("/proc/1/environ")
    try:
        socket.create_connection(("1.1.1.1",443),timeout=0.1)
        raise AssertionError("network escaped")
    except OSError:
        pass
    return [row["id"] for row in memory]
"""
    result = run_code(
        code, [{"query": "a", "memory": [{"id": "a", "question": "apple", "answer": "fruit"}]}]
    )
    assert result == [[{"id": "a", "score": 1}]]


@pytest.mark.skipif(not HAS_SANDBOX, reason="Optional namespace launcher is not installed")
def test_code_ranking_rejects_out_of_memory_answers():
    with pytest.raises(ValueError, match="memory-only contract"):
        run_code('def rank(query,memory): return ["invented"]', [{"query": "a", "memory": []}])
