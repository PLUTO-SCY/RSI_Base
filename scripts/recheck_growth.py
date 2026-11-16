#!/usr/bin/env python3
"""Repeat the frozen baseline/selected pair on dev; never feed results back or reselect."""

import argparse
import fcntl
import json
from pathlib import Path

from run_growth import paired

from rsi_base.agent import TaskAgent
from rsi_base.benchmark import load_problems, score
from rsi_base.loop import completed, evaluate, file_hash
from rsi_base.math_scoring import math_score
from rsi_base.storage import Ledger, read_json, write_json
from rsi_base.types import AgentSpec, Limits, RunConfig, digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/qwen-1_7b-boxed-v3")
    parser.add_argument("--repeats", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    search = args.run / "search"
    manifest = read_json(search / "run.json")
    source = digest({p.name: file_hash(p) for p in (root / "src/rsi_base").glob("*.py")})
    if manifest["implementation_sha256"] != source or manifest[
        "dependency_lock_sha256"
    ] != file_hash(root / "uv.lock"):
        raise ValueError("Repeat requires the same implementation and dependencies")
    result = read_json(search / "summary.json")
    config = RunConfig.model_validate(manifest["config"])
    dev_path = root / "data/math-growth/dev.jsonl"
    if file_hash(dev_path) != manifest["dev_sha256"]:
        raise ValueError("Development data changed")
    for directory in (search / "rounds").iterdir():
        if completed(directory) is None:
            raise ValueError("Search is not complete")
    frozen = AgentSpec.model_validate(result["history"][-1]["agent"])
    directory = args.run / "recheck"
    directory.mkdir(exist_ok=True)
    identity = {
        "search_sha256": file_hash(search / "summary.json"),
        "script_sha256": file_hash(__file__),
        "paired_analysis_sha256": file_hash(root / "scripts/run_growth.py"),
        "repeats": args.repeats,
        "selection_frozen": True,
    }
    with (args.run / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "manifest.json"
        if path.exists() and read_json(path) != identity:
            raise ValueError("Recheck inputs changed")
        write_json(path, identity)
        dev = load_problems(dev_path)
        limits = Limits(
            rounds=0,
            max_meta_calls=0,
            max_task_calls=2 * len(dev) * args.repeats + 16,
            wall_seconds=3600,
            task_concurrency=config.limits.task_concurrency,
        )
        ledger = Ledger(directory / "calls.jsonl", limits)
        task = TaskAgent(config.task, root)
        scorer = math_score if config.evaluator == "math_numeric" else score
        rows = []
        try:
            for index in range(args.repeats):
                baseline, base_summary = evaluate(
                    directory / str(index) / "baseline",
                    dev,
                    config.agent,
                    task,
                    ledger,
                    limits.task_concurrency,
                    scorer,
                )
                if frozen == config.agent:
                    selected, selected_summary = baseline, base_summary
                else:
                    selected, selected_summary = evaluate(
                        directory / str(index) / "selected",
                        dev,
                        frozen,
                        task,
                        ledger,
                        limits.task_concurrency,
                        scorer,
                    )
                rows.append(
                    {
                        "repeat": index + 1,
                        "baseline": base_summary,
                        "selected": selected_summary,
                        "paired": paired(baseline, selected),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            write_json(
                directory / "report.json",
                {
                    "selection_frozen": True,
                    "split": "dev",
                    "purpose": "Check repeatability only; not independent generalization or model selection",
                    "selected_round": result["selected_round"],
                    "repeats": rows,
                    "application_calls": ledger.used,
                    "identity": identity,
                },
            )
        finally:
            ledger.close()


if __name__ == "__main__":
    main()
