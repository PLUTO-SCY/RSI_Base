#!/usr/bin/env python3
"""Post-hoc development-only audit; never change the frozen official selection."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.classification import MemoryPolicy
from rsi_base.loop import file_hash
from rsi_base.memory_loop import MemoryConfig, evaluate_policy, experiment, stable_improvement
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
    if not all((args.suite / f"seed{seed}/frozen.json").exists() for seed in (42, 123, 2026)):
        raise ValueError("All official searches must be frozen first")
    directory = args.suite / f"seed{args.seed}"
    original_hash = file_hash(directory / "frozen.json")
    frozen, registry = (
        read_json(directory / "frozen.json"),
        read_json(directory / "agent_registry.json"),
    )
    config = MemoryConfig.model_validate(read_json(directory / "run.json")["config"])
    results = []
    with experiment(config, args.data, directory, root) as (
        harness,
        _,
        development,
        _,
        ledger,
        task,
    ):
        for identifier, item in registry.items():
            number = int(identifier.split("-", 1)[0][1:])
            location = item.get("validation", f"probe_selection_audit/{identifier}")
            try:
                _, summary = evaluate_policy(
                    directory / location,
                    development,
                    MemoryPolicy.model_validate(item["policy"]),
                    harness,
                    task,
                    ledger,
                    repeats=config.validation_repeats,
                    seed=config.seed,
                )
            except ValueError as exc:
                write_json(
                    directory / location / "audit_invalid.json",
                    {"error_type": type(exc).__name__, "detail": str(exc)[:1500]},
                )
                results.append(
                    {
                        "candidate_id": identifier,
                        "round": number,
                        "full_development_error": type(exc).__name__,
                        "eligible_vs_actual_round_parent": False,
                    }
                )
                continue
            result = {
                "candidate_id": identifier,
                "round": number,
                "officially_submitted": "validation" in item,
                "retrieval": item["policy"]["retrieval"],
                "policy_sha256": digest(item["policy"]),
                "probe_correct": item["summary"]["correct_by_repeat"],
                "development_correct": summary["correct_by_repeat"],
                "eligible_vs_actual_round_parent": stable_improvement(
                    summary, frozen["history"][number - 1]["summary"]
                ),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        result = {
            "seed": config.seed,
            "kind": "Post-hoc development selection audit; not a new evolution run or independent confirmation",
            "source_sha256": read_json(directory / "run.json")["source_sha256"],
            "frozen_sha256": original_hash,
            "script_sha256": file_hash(__file__),
            "official_selection_changed": False,
            "test_used": False,
            "candidates": results,
            "physical_application_calls_after_audit": dict(ledger.used),
        }
        if file_hash(directory / "frozen.json") != original_hash:
            raise ValueError("Frozen official selection changed")
        write_json(args.suite / f"probe-selection-audit-seed{config.seed}.json", result)


if __name__ == "__main__":
    main()
