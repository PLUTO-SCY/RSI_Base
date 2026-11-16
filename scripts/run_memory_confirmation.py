#!/usr/bin/env python3
"""Predeclare and evaluate an additional sealed cohort; never alter frozen choices."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.classification import MemoryPolicy
from rsi_base.loop import file_hash
from rsi_base.memory_audit import cluster_interval, paired_records
from rsi_base.memory_loop import MemoryConfig, evaluate_policy, experiment, load_rows
from rsi_base.storage import Ledger, read_json, write_json
from rsi_base.types import Limits, digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "evaluate", "summarize"])
    parser.add_argument("--suite", type=Path, default=root / "runs/memory-s2d-suite-v4")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026])
    args = parser.parse_args()
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
    data = root / "data/memory-confirmation"
    original = root / "data/memory-s2d"
    seeds = [42, 123, 2026]
    path = args.suite / "confirmation_protocol.json"
    identity = {
        "seeds": seeds,
        "repeats": 3,
        "variants": ["strong_fixed_baseline", "frozen_selected"],
        "script_sha256": file_hash(__file__),
        "data_manifest_sha256": file_hash(data / "manifest.json"),
        "cohorts_reported_separately": True,
        "extra_stability_gate": "Positive paired delta for each search in BOTH cohorts and pooled question-cluster 95% lower bound >0; original 211-query gate also reported unchanged",
        "no_model_or_meta_calls_in_planning": True,
    }
    if args.phase == "plan":
        if (args.suite / "final_protocol.json").exists():
            raise ValueError("Confirmation plan must be locked before original test scores")
        if path.exists() and read_json(path) != identity:
            raise ValueError("Confirmation plan changed")
        write_json(path, identity)
        print(
            json.dumps(
                {
                    "planned_before_test": True,
                    "examples": read_json(data / "manifest.json")["retained_count"],
                }
            )
        )
        return
    if not path.exists() or read_json(path) != identity:
        raise ValueError("Requires identical predeclared confirmation plan")
    directories = [args.suite / f"seed{seed}" for seed in seeds]
    if any(not (directory / "frozen.json").exists() for directory in directories):
        raise ValueError("All searches must be frozen before any test evaluation")
    if args.phase == "evaluate":
        for directory in directories:
            frozen = read_json(directory / "frozen.json")
            if args.seed is not None and frozen["seed"] != args.seed:
                continue
            config = MemoryConfig.model_validate(read_json(directory / "run.json")["config"])
            with experiment(config, original, directory, root) as (harness, _, _, _, _, task):
                problems = load_rows(data / "test.jsonl")
                folder = directory / "confirmation"
                folder.mkdir(exist_ok=True)
                ledger = Ledger(
                    folder / "calls.jsonl",
                    Limits(
                        max_task_calls=12 * len(problems) + 32, max_meta_calls=0, wall_seconds=7200
                    ),
                )
                try:
                    values, rows = {}, {}
                    for name in ["baseline", "selected"]:
                        policy = MemoryPolicy.model_validate(frozen[name]["policy"])
                        rows[name], values[name] = evaluate_policy(
                            folder / digest(policy.model_dump())[:16],
                            problems,
                            policy,
                            harness,
                            task,
                            ledger,
                            repeats=3,
                            seed=config.seed,
                        )
                    report = {
                        "seed": config.seed,
                        **values,
                        "paired": paired_records(rows["baseline"], rows["selected"]),
                        "frozen_sha256": file_hash(directory / "frozen.json"),
                        "protocol_sha256": file_hash(path),
                        "application_calls": dict(ledger.used),
                        "data_manifest": read_json(data / "manifest.json"),
                    }
                    write_json(folder / "report.json", report)
                    print(
                        json.dumps(
                            {
                                "seed": config.seed,
                                "baseline": values["baseline"]["correct_by_repeat"],
                                "selected": values["selected"]["correct_by_repeat"],
                            }
                        ),
                        flush=True,
                    )
                finally:
                    ledger.close()
        return
    primary = read_json(args.suite / "report.json")
    confirmation = [read_json(directory / "confirmation/report.json") for directory in directories]
    primary_by_seed = {row["seed"]: row for row in primary["runs"]}
    combined, extra = {}, {}
    for row in confirmation:
        for identifier, value in row["paired"]["question_deltas"].items():
            combined.setdefault(identifier, []).append(value)
            extra.setdefault(identifier, []).append(value)
        for identifier, value in primary_by_seed[row["seed"]]["paired"]["question_deltas"].items():
            combined.setdefault(identifier, []).append(value)
    interval = cluster_interval({k: sum(v) / len(v) for k, v in combined.items()})
    result = {
        "protocol": identity,
        "runs": confirmation,
        "confirmation_only_interval": cluster_interval(
            {k: sum(v) / len(v) for k, v in extra.items()}
        ),
        "pooled_question_interval": interval,
        "original_gate_passed": primary["stability_gate_passed"],
        "extra_gate_passed": all(
            row["paired"]["mean_delta"] > 0
            and primary_by_seed[row["seed"]]["paired"]["mean_delta"] > 0
            for row in confirmation
        )
        and interval["low"] > 0,
    }
    write_json(args.suite / "confirmation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"runs", "protocol"}}))


if __name__ == "__main__":
    main()
