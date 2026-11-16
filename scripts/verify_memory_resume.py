#!/usr/bin/env python3
"""Verify a fully frozen suite resumes without new API admissions or changed choices."""

import argparse
import json
import subprocess
from pathlib import Path

from rsi_base.loop import file_hash
from rsi_base.storage import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(args.suite.glob("seed*"))
    if len(directories) != 3 or any(not (d / "frozen.json").exists() for d in directories):
        raise ValueError("Requires three completed searches")

    def state():
        return {
            d.name: {
                "frozen_sha256": file_hash(d / "frozen.json"),
                "started_calls": [
                    r
                    for r in map(json.loads, (d / "calls.jsonl").read_text().splitlines())
                    if r["event"] == "started"
                ],
            }
            for d in directories
        }

    before = state()
    subprocess.run(
        [
            str(root / ".venv/bin/python"),
            str(root / "scripts/run_memory.py"),
            "suite",
            "--output",
            str(args.suite),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if state() != before:
        raise ValueError("Frozen choices changed or new calls were admitted")
    report = {
        "identical_frozen_choices": True,
        "new_task_calls": 0,
        "new_meta_calls": 0,
        "frozen_sha256": {name: row["frozen_sha256"] for name, row in before.items()},
    }
    write_json(args.suite / "resume_verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
