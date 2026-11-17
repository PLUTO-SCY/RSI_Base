#!/usr/bin/env python3
"""Audit frozen search records, without model calls or final-test records."""

import argparse
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rsi_base.storage import read_json, write_json


@lru_cache(maxsize=4)
def first_request_signatures(path):
    signatures = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        messages = row["traces"][0]["messages"]
        signatures[row["id"]] = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()
        ).hexdigest()
    return signatures


def audit(suite):
    results, noops = [], []
    for directory in sorted(suite.glob("seed*")):
        frozen = read_json(directory / "frozen.json")
        events = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
        starts = Counter(
            row.get("evaluation", "")
            for row in events
            if row["event"] == "started" and row["kind"] == "task"
        )
        proposals = frozen["proposals"]
        direct_pairs = 0
        for child in proposals:
            number = int(re.search(r"round(\d+)", child["id"])[1])
            parent = frozen["history"][number - 1]
            if parent["policy"]["workflow"] != "direct" or child["policy"]["workflow"] != "direct":
                continue
            direct_pairs += 1
            before = first_request_signatures(directory / parent["evaluation"] / "results.jsonl")
            after = first_request_signatures(directory / child["evaluation"] / "results.jsonl")
            if not before or before.keys() != after.keys():
                raise ValueError("Selection request IDs do not match")
            if before == after:
                relative = f"evolution/{number:03}/{child['id'].split('-')[-1]}"
                physical_calls = (
                    sum(count for path, count in starts.items() if path.endswith("/" + relative))
                    if child["evaluation"] == relative
                    else 0
                )
                noops.append(
                    {
                        "seed": frozen["seed"],
                        "candidate": child["id"],
                        "identical_requests": len(before),
                        "new_task_call_admissions": physical_calls,
                        "eligible": child["eligible"],
                        "parent_validation": parent["summary"]["correct_by_repeat"],
                        "child_validation": child["summary"]["correct_by_repeat"],
                    }
                )
        inputs = [
            read_json(path)["evidence"]["training_diagnostics"]
            for path in sorted((directory / "evolution").glob("*/meta_input.json"))
        ]
        runtimes = [
            read_json(path)["arguments"]
            for path in (directory / "evolution").glob("*/meta_runtime.json")
        ]
        meta = [child for child in proposals if child["id"].endswith("candidate2")]
        local = [child for child in proposals if child["id"].endswith("candidate1")]
        results.append(
            {
                "seed": frozen["seed"],
                "rounds": frozen["rounds"],
                "meta_candidates_evaluated": len(meta),
                "local_candidates_evaluated": len(local),
                "meta_retrieval_methods": dict(
                    Counter(child["policy"]["retrieval"] for child in meta)
                ),
                "distinct_training_feedback_payloads": len(
                    {json.dumps(value, sort_keys=True) for value in inputs}
                ),
                "feedback_examples_per_round": [len(value["examples"]) for value in inputs],
                "all_meta_tool_sets_empty": bool(runtimes)
                and all(args[args.index("--tools") + 1] == "" for args in runtimes),
                "direct_parent_child_pairs_audited": direct_pairs,
            }
        )
    if len(results) != 3:
        raise ValueError("Expected a complete three-seed frozen suite")
    return {
        "suite": suite.name,
        "runs": results,
        "identical_input_candidates": noops,
        "new_task_call_admissions_for_identical_candidates": sum(
            row["new_task_call_admissions"] for row in noops
        ),
        "any_identical_input_candidate_eligible": any(row["eligible"] for row in noops),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=root / "runs/memory-search-diagnosis.json")
    args = parser.parse_args()
    suites = args.suite or [
        root / "runs/memory-s2d-starter-v1",
        root / "runs/memory-s2d-expanded-v2",
    ]
    result = {
        "scope": "Post-hoc engineering audit of stored training feedback and selection requests; not a causal performance ablation",
        "new_model_calls": 0,
        "test_records_accessed": False,
        "suites": [audit(suite) for suite in suites],
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "suites": [
                    {
                        "suite": row["suite"],
                        "identical_input_candidates": len(row["identical_input_candidates"]),
                        "task_admissions": row["new_task_call_admissions_for_identical_candidates"],
                    }
                    for row in result["suites"]
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
