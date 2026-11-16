#!/usr/bin/env python3
"""Read progress without displaying task text, prompts, provider responses or credentials."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run / "search" if (args.run / "search").is_dir() else args.run
    for directory in sorted((root / "rounds").iterdir()):
        value = {"round": int(directory.name)}
        records = directory / "results.jsonl"
        value["completed"] = len(records.read_text().splitlines()) if records.exists() else 0
        for name in ["summary", "decision"]:
            path = directory / f"{name}.json"
            if path.exists():
                value.update(
                    {
                        k: v
                        for k, v in json.loads(path.read_text()).items()
                        if k
                        in {
                            "correct",
                            "assigned",
                            "accuracy",
                            "format_errors",
                            "truncated",
                            "decision",
                            "selected_round",
                            "dev_accuracy",
                        }
                    }
                )
        value["meta_proposal_ready"] = (directory / "proposal.json").exists()
        print(json.dumps(value))


if __name__ == "__main__":
    main()
