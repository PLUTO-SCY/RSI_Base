#!/usr/bin/env python3
"""Reuse control measurements when task requests, data, scoring and evaluation are unchanged."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from rsi_base.loop import completed, file_hash
from rsi_base.memory_loop import MemoryConfig
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    config = MemoryConfig.model_validate_json(args.config.read_text())
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.source_commit, "--", "src/rsi_base"], text=True
    ).splitlines()
    blobs = {
        path: subprocess.check_output(["git", "show", f"{args.source_commit}:{path}"])
        for path in paths
        if path.endswith(".py")
    }
    source_digest = digest(
        {
            str(Path(path).relative_to("src")): hashlib.sha256(value).hexdigest()
            for path, value in blobs.items()
        }
    )
    for name in ["agent.py", "classification.py", "types.py", "storage.py", "loop.py"]:
        if (root / "src/rsi_base" / name).read_bytes() != blobs[f"src/rsi_base/{name}"]:
            raise ValueError("Task, policy, scorer or checkpoint implementation changed")

    def evaluation_ast(text):
        return ast.dump(
            next(
                node
                for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "evaluate_policy"
            ),
            include_attributes=False,
        )

    if evaluation_ast((root / "src/rsi_base/memory_loop.py").read_text()) != evaluation_ast(
        blobs["src/rsi_base/memory_loop.py"].decode()
    ):
        raise ValueError("Evaluation logic changed")
    endpoints = [
        "http://127.0.0.1:18000/v1",
        "http://127.0.0.1:18001/v1",
        "http://127.0.0.1:18002/v1",
    ]
    report = {
        "source_commit": args.source_commit,
        "policy_and_evaluation_code_identical": True,
        "data_identical": True,
        "cached_controls_only": True,
        "seeds": {},
    }
    if args.output.exists():
        raise ValueError("Refusing to overwrite imported measurements")
    for seed, endpoint in zip([42, 123, 2026], endpoints, strict=True):
        source = args.source / f"seed{seed}"
        manifest = read_json(source / "run.json")
        expected_task = (
            type(config.task)
            .model_validate({**config.task.model_dump(), "base_url": endpoint})
            .model_dump()
        )
        if (
            manifest["source_sha256"] != source_digest
            or manifest["config"]["task"] != expected_task
            or manifest["config"]["seed"] != seed
        ):
            raise ValueError("Source implementation or task/seed configuration differs")
        if (
            manifest["config"]["validation_repeats"] != config.validation_repeats
            or manifest["config"]["context_chars"] != config.context_chars
        ):
            raise ValueError("Evaluation budget differs")
        if manifest["data_manifest_sha256"] != file_hash(args.data / "manifest.json") or manifest[
            "lock_sha256"
        ] != file_hash(root / "uv.lock"):
            raise ValueError("Data/dependencies differ")
        baselines = source / "baselines"
        for directory in baselines.iterdir():
            if completed(directory) is None:
                raise ValueError("Control measurement incomplete")
        destination = args.output / f"seed{seed}"
        destination.mkdir(parents=True)
        shutil.copytree(baselines, destination / "baselines")
        events = map(json.loads, (source / "calls.jsonl").read_text().splitlines())
        calls = sum(
            row["event"] == "started"
            and row.get("kind") == "task"
            and "/baselines/" in row.get("evaluation", "")
            for row in events
        )
        note = {
            "source_run": str(source),
            "source_run_manifest_sha256": file_hash(source / "run.json"),
            "original_control_api_calls": calls,
            "new_api_calls_for_import": 0,
            "selection_labels_not_exposed_to_meta": True,
        }
        write_json(destination / "baseline_import.json", note)
        report["seeds"][seed] = note
    write_json(args.output / "baseline_import.json", report)
    print(
        json.dumps(
            {
                "imported_seeds": 3,
                "new_api_calls": 0,
                "independence": "Same observed controls shared across optimizer comparisons, not new independent samples",
            }
        )
    )


if __name__ == "__main__":
    main()
