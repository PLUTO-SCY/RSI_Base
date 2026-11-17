"""Contracts for a clean setup: data integrity, bounded smoke calls and secret-free checks."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from rsi_base.classification import MemoryCandidate, SingleMemoryBatch
from rsi_base.loop import file_hash
from rsi_base.memory_loop import MemoryConfig
from rsi_base.types import digest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


@pytest.mark.parametrize(
    "with_meta,response", [(False, "fruit"), (True, "fruit"), (True, "invalid-label")]
)
def test_smoke_has_bounded_calls_and_never_opens_final_test(
    tmp_path, monkeypatch, with_meta, response
):
    monkeypatch.setenv("RSI_LOCAL_API_KEY", "EMPTY")
    data = tmp_path / "data"
    data.mkdir()
    train = [{"id": "train", "question": "apple orchard", "answer": "fruit"}]
    val = [{"id": str(i), "question": f"selection-only-{i}", "answer": "fruit"} for i in range(3)]
    for split, rows in [("train", train), ("val", val)]:
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data / "labels.json").write_text('["fruit"]')
    # Any attempt to deserialize the final-test file would fail immediately.
    (data / "test.jsonl").write_bytes(b"\xffPRIVATE_FINAL_TEST")
    captured = []

    class Task:
        def run(self, spec, question):
            captured.append(question)
            return {"response": response}

    class Meta:
        def propose(self, current, evidence, output):
            assert "selection-only" not in json.dumps(evidence)
            return SingleMemoryBatch(
                candidates=[MemoryCandidate(policy=current, reason="probe")]
            ), {}

    config = MemoryConfig.model_validate_json((ROOT / "configs/default.json").read_text())
    output = tmp_path / "smoke"
    run = script("smoke_harness.py")["smoke"]
    if response == "invalid-label":
        with pytest.raises(ValueError, match="output contract"):
            run(config, data, output, tmp_path, task=Task(), meta=Meta(), with_meta=with_meta)
        events = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
        assert not any(row["event"] == "started" and row["kind"] == "meta" for row in events)
        return
    report = run(config, data, output, tmp_path, task=Task(), meta=Meta(), with_meta=with_meta)
    assert report["application_calls"] == {"task": 2, "meta": int(with_meta)}
    assert report["test_opened"] is False
    assert len(captured) == 2 and all("PRIVATE_FINAL_TEST" not in q for q in captured)
    with pytest.raises(ValueError, match="already exists"):
        run(config, data, output, tmp_path, task=Task())
    assert len(captured) == 2


def test_idempotent_data_verification_detects_corruption(tmp_path):
    (tmp_path / "val.jsonl").write_text('{"id":"v"}\n')
    (tmp_path / "labels.json").write_text('["fruit"]')
    manifest = {
        "labels": ["fruit"],
        "labels_sha256": file_hash(tmp_path / "labels.json"),
        "splits": {"val": {"count": 1, "sha256": file_hash(tmp_path / "val.jsonl")}},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({**manifest, "lock_sha256": digest(manifest)})
    )
    expected = manifest
    verify = script("prepare_data.py")["verify_asset"]
    assert verify(tmp_path, expected) == {"val": 1}
    (tmp_path / "val.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="changed data"):
        verify(tmp_path, expected)


def test_rate_limit_fallback_keeps_blob_integrity():
    payload = b'{"question":"public"}\n'
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    requests = []

    def get(url, **kwargs):
        requests.append(url)
        return httpx.Response(
            403 if len(requests) == 1 else 200, content=payload, request=httpx.Request("GET", url)
        )

    downloader = script("prepare_data.py")
    assert downloader["download_blob"]("train", blob, get=get) == payload
    assert downloader["RAW_REVISION"] in requests[1]
    requests.clear()
    with pytest.raises(ValueError, match="fingerprint"):
        downloader["download_blob"]("train", "0" * 40, get=get)


def test_doctor_config_errors_do_not_echo_secret_values(tmp_path):
    config = json.loads((ROOT / "configs/default.json").read_text())
    secret = "DUMMY-SECRET-MUST-NOT-BE-PRINTED"
    config["meta"]["api_key"] = secret
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/doctor.py"), "--config", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "api_key" in result.stderr
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("version,expected", [("2.1.169", True), ("2.0.0", False)])
def test_doctor_hidden_help_flag_uses_only_verified_pinned_contract(
    tmp_path, monkeypatch, capsys, version, expected
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import prepare_data

    monkeypatch.setattr(prepare_data, "verify_asset", lambda *_: {"train": 200})
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/claude")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **_: (
            f"{version} (Claude Code)"
            if command[-1] == "--version"
            else "--bare --no-session-persistence --effort"
        ),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(
            200, json={"data": [{"id": "fixture"}]}, request=httpx.Request("GET", url)
        ),
    )
    monkeypatch.setenv("RSI_API_KEY", "DUMMY")
    monkeypatch.setenv("RSI_LOCAL_API_KEY", "EMPTY")
    config = json.loads((ROOT / "configs/default.json").read_text())
    config["task"]["model"] = "fixture"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["doctor", "--ready", "--config", str(path)])
    with pytest.raises(SystemExit):
        script("doctor.py")["main"]()
    report = json.loads(capsys.readouterr().out)
    cli = next(row for row in report["checks"] if row["check"] == "claude_code")
    assert cli["ok"] is expected
    if expected:
        assert cli["detail"]["flags_not_listed_in_help"] == ["--strict-mcp-config"]
    assert report["model_generations"] == report["meta_api_calls"] == 0


def test_scanner_distinguishes_runtime_names_from_tokens_without_echoing(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    directory = tmp_path / "scripts"
    directory.mkdir()
    scanner = directory / "scan_secrets.py"
    shutil.copyfile(ROOT / "scripts/scan_secrets.py", scanner)
    example = tmp_path / "README.md"
    example.write_text("task-runtime-expanded-123.json")

    def scan():
        return subprocess.run([sys.executable, str(scanner)], capture_output=True, text=True)

    assert scan().returncode == 0
    # Construct a nonfunctional synthetic token at runtime, never store it in repository sources.
    token = "sk-" + "x" * 32
    example.write_text("api_key='" + token + "'")
    result = scan()
    assert result.returncode != 0 and "README.md" in result.stdout
    assert token not in result.stdout + result.stderr
