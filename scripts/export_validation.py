#!/usr/bin/env python3
"""Publish an aggregate validation artifact, excluding prompts, questions and credentials."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

from rsi_base.loop import file_hash, run
from rsi_base.storage import read_json, write_json
from rsi_base.types import RunConfig, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/deepseek-v0.1"))
    parser.add_argument("--dev", type=Path, default=Path("data/gsm8k/dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/gsm8k/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/v0.1.json"))
    args = parser.parse_args()
    root = Path.cwd()
    initial = read_json(args.run / "summary.json")
    manifest = read_json(args.run / "run.json")
    source_dir = root / "src/rsi_base"
    source_hash = digest({p.name: file_hash(p) for p in source_dir.glob("*.py")})
    if source_hash != manifest["implementation_sha256"]:
        raise SystemExit("Implementation has changed since this run; do not relabel old evidence.")
    if file_hash(root / "uv.lock") != manifest["dependency_lock_sha256"]:
        raise SystemExit("Dependency lock changed since this run.")

    def calls():
        return [json.loads(line) for line in (args.run / "calls.jsonl").read_text().splitlines()]

    before = sum(row["event"] == "started" for row in calls())
    resumed = run(
        RunConfig.model_validate(manifest["config"]),
        args.dev,
        args.test,
        args.run,
        root,
        resume=True,
    )
    after = sum(row["event"] == "started" for row in calls())
    if before != after or initial != resumed:
        raise SystemExit("Completed-run resume did not preserve result/call counts.")
    totals = {
        "task": {"input_tokens": 0, "output_tokens": 0},
        "meta": {"input_tokens": 0, "output_tokens": 0},
    }
    meta_completed = []
    for row in calls():
        if row["event"] != "finished":
            continue
        usage = row.get("usage") or {}
        totals[row["kind"]]["input_tokens"] += usage.get(
            "prompt_tokens", usage.get("input_tokens", 0)
        )
        totals[row["kind"]]["output_tokens"] += usage.get(
            "completion_tokens", usage.get("output_tokens", 0)
        )
        if row["kind"] == "meta":
            meta_completed.append(row)
    observed_models = set()
    for path in args.run.rglob("results.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("response_model"):
                observed_models.add(row["response_model"])
    report = {
        "scope": "live API software pilot; no weight training or gain claim",
        "environment": {
            "python": platform.python_version(),
            "openai_sdk": version("openai"),
            "claude_cli": sorted({r["cli_version"] for r in meta_completed}),
        },
        "implementation_sha256": source_hash,
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "dataset": read_json(args.data_manifest),
        "rounds": initial["rounds"],
        "selected_round": initial["selected_round"],
        "development_accuracy": initial["development_accuracy"],
        "held_out": initial["held_out"],
        "decisions": [h["decision"] for h in initial["history"]],
        "application_calls": initial["application_calls"],
        "loop_token_usage": totals,
        "requested_task_model": initial["model"],
        "observed_task_models": sorted(observed_models),
        "requested_meta_model": initial["meta_model"],
        "global_configurations_unchanged": bool(meta_completed)
        and all(r.get("global_configuration_unchanged", False) for r in meta_completed),
        "resume": {"same_summary": True, "new_application_calls": after - before},
        "weights_updated": initial["weights_updated"],
        "provider_dollar_bill": "not verified; CLI estimates are not gateway invoices",
        "outside_loop_probes": "excluded from these loop counters; recorded locally",
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
