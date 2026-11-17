#!/usr/bin/env python3
"""Resolve a historical commit ID after the metadata-only authorship correction."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def resolve(identifier, records):
    value = identifier.lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", value):
        raise ValueError("Supply a commit SHA or a prefix of at least seven hex characters")
    matches = {
        row["new"]
        for row in records
        if row["old"].startswith(value) or row["new"].startswith(value)
    }
    if len(matches) > 1:
        raise ValueError("Ambiguous commit prefix; supply the full SHA")
    return next(iter(matches)) if matches else None


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    args = parser.parse_args()
    records = json.loads((root / "docs/history/identity-rewrite-map.json").read_text())["commits"]
    try:
        result = resolve(args.commit, records)
    except ValueError as exc:
        parser.error(str(exc))
    if result is None:
        result = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=root, text=True
        ).strip()
    print(result)


if __name__ == "__main__":
    main()
