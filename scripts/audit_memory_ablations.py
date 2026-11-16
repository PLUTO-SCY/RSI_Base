#!/usr/bin/env python3
"""Frozen diagnostic ablations; never use their scores to reselect or resume evolution."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.classification import MemoryPolicy
from rsi_base.loop import file_hash
from rsi_base.memory_audit import paired_records
from rsi_base.memory_loop import MemoryConfig, evaluate_policy, experiment, load_rows
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "runs/memory-s2d-expanded-v1")
    parser.add_argument("--data", type=Path, default=root / "data/memory-s2d-expanded")
    args = parser.parse_args()
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
    protocol = read_json(args.suite / "suite.json")
    directories = [args.suite / f"seed{seed}" for seed in protocol["seeds"]]
    if any(not (p / "frozen.json").exists() for p in directories):
        raise ValueError("All searches must be frozen before diagnostic test access")
    plan = {
        "definitions": {
            "structure_only": "Selected retrieval/context/workflow with baseline system_prompt",
            "prompt_only": "Baseline retrieval/context/workflow with selected system_prompt",
        },
        "selection_frozen": True,
        "script_sha256": file_hash(__file__),
        "frozen_sha256": {p.name: file_hash(p / "frozen.json") for p in directories},
        "purpose": "Mechanism diagnostics only; no candidate selection or additional evolution",
    }
    path = args.suite / "ablation_protocol.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Ablation protocol changed")
    write_json(path, plan)
    reports = []
    for directory in directories:
        config = MemoryConfig.model_validate(read_json(directory / "run.json")["config"])
        frozen = read_json(directory / "frozen.json")
        baseline = MemoryPolicy.model_validate(frozen["baseline"]["policy"])
        selected = MemoryPolicy.model_validate(frozen["selected"]["policy"])
        with experiment(config, args.data, directory, root) as (harness, _, _, _, ledger, task):
            test = load_rows(args.data / "test.jsonl")
            variants = {
                "baseline": baseline,
                "selected": selected,
                "structure_only": selected.model_copy(
                    update={"system_prompt": baseline.system_prompt}
                ),
                "prompt_only": baseline.model_copy(
                    update={"system_prompt": selected.system_prompt}
                ),
            }
            rows, summaries = {}, {}
            for name, policy in variants.items():
                # Reuse already frozen test evaluations for identical policies.
                location = directory / "test" / digest(policy.model_dump())[:16]
                rows[name], summaries[name] = evaluate_policy(
                    location,
                    test,
                    policy,
                    harness,
                    task,
                    ledger,
                    repeats=config.validation_repeats,
                    seed=config.seed,
                )
            reports.append(
                {
                    "seed": config.seed,
                    "summaries": summaries,
                    "paired_vs_baseline": {
                        name: paired_records(rows["baseline"], records)
                        for name, records in rows.items()
                        if name != "baseline"
                    },
                    "changed_fields": [
                        key
                        for key, value in baseline.model_dump().items()
                        if selected.model_dump()[key] != value
                    ],
                    "memory_changed_fraction": sum(
                        a["memory_ids"] != b["memory_ids"]
                        for a, b in zip(rows["baseline"], rows["selected"], strict=True)
                    )
                    / len(test)
                    / config.validation_repeats,
                    "application_calls_after_ablations": dict(ledger.used),
                }
            )
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "ablations": {k: v["correct_by_repeat"] for k, v in summaries.items()},
                    }
                ),
                flush=True,
            )
    write_json(args.suite / "ablations.json", {"protocol": plan, "runs": reports})


if __name__ == "__main__":
    main()
