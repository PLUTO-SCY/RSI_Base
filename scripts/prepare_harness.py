#!/usr/bin/env python3
"""Prepare or verify the published data, without historical runs or model calls."""

import json
import subprocess
import sys
from pathlib import Path

from rsi_base.loop import file_hash
from rsi_base.storage import read_json


def verify_asset(directory, expected_manifest_sha256):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or file_hash(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"Incomplete or changed manifest: {directory}; no files were overwritten")
    manifest = read_json(manifest_path)
    splits = manifest.get("splits") or {
        "test": {"sha256": manifest["test_sha256"], "count": manifest["retained_count"]}
    }
    for split, expected in splits.items():
        path = directory / f"{split}.jsonl"
        if not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Incomplete or changed data: {path}; no files were overwritten")
        if len(path.read_text().splitlines()) != expected["count"]:
            raise ValueError(f"Unexpected row count: {path}")
    if "labels" in manifest and read_json(directory / "labels.json") != manifest["labels"]:
        raise ValueError(f"Changed label set: {directory}")
    return {name: row["count"] for name, row in splits.items()}


def main():
    root = Path(__file__).resolve().parents[1]
    published = root / "docs/validation/memory-expanded-manifest.json"
    expected = read_json(published)
    stages = [
        (
            "memory-s2d",
            expected["source_reserve"]["prior_data_manifest_sha256"],
            ["prepare_memory_data.py"],
        ),
        (
            "memory-confirmation",
            expected["source_reserve_manifest_sha256"],
            ["prepare_memory_confirmation.py"],
        ),
        (
            "memory-s2d-expanded",
            file_hash(published),
            ["prepare_memory_expanded.py", "--rebuild-published-split"],
        ),
    ]
    for name, expected_sha, command in stages:
        directory = root / "data" / name
        if not directory.exists():
            subprocess.run(
                [sys.executable, str(root / "scripts" / command[0]), *command[1:]], check=True
            )
        print(json.dumps({"dataset": name, "verified": verify_asset(directory, expected_sha)}))
    print("Published 200/300/291 split ready; no API generation and no new-held-out claim.")


if __name__ == "__main__":
    main()
