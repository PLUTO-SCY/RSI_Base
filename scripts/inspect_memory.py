#!/usr/bin/env python3
"""Print compact progress and call counts without revealing credentials or task text."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, nargs="?", default=root / "runs/memory-s2d-agent-demo")
    args = parser.parse_args()
    reports = []
    directories = (
        [args.suite] if (args.suite / "calls.jsonl").exists() else sorted(args.suite.glob("seed*"))
    )
    for directory in directories:
        path = directory / "calls.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        starts = [row for row in rows if row["event"] == "started"]
        finished = {row["call_id"] for row in rows if row["event"] == "finished"}
        pending = [row for row in starts if row["call_id"] not in finished]
        states = [
            json.loads(path.read_text())
            for path in sorted((directory / "evolution").glob("*/decision.json"))
        ]
        proposals = []
        for path in sorted((directory / "evolution").glob("*/candidate*/summary.json")):
            summary = json.loads(path.read_text())
            proposals.append(
                {
                    "round": path.parent.parent.name,
                    "candidate": path.parent.name,
                    "validation": summary["correct_by_repeat"],
                }
            )
        report = {
            "run": directory.name,
            "frozen": (directory / "frozen.json").exists(),
            "test_evaluated": (directory / "audit.json").exists(),
            "application_calls": dict(Counter(row["kind"] for row in starts)),
            "unsettled_calls": [
                {
                    "kind": row["kind"],
                    "round": row.get("round"),
                    "age_seconds": round(time.time() - row["time"]),
                }
                for row in pending
            ],
            "selected_validation_curve": [
                state["summary"]["correct_by_repeat"] for state in states
            ],
            "random_candidates_completed": len(list((directory / "random").glob("*/summary.json"))),
            "candidates": proposals,
        }
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
