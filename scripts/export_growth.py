#!/usr/bin/env python3
"""Export aggregate evidence and figures; keep questions, responses and secrets local."""

import argparse
import json
import shutil
from pathlib import Path

from rsi_base.loop import completed, file_hash
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/qwen-math-growth-8k")
    parser.add_argument("--runtime", type=Path, default=root / ".local/task-runtime.json")
    parser.add_argument("--name", choices=["growth", "growth-v2", "growth-small"], default="growth")
    parser.add_argument("--implementation", type=Path, default=root / "src/rsi_base")
    args = parser.parse_args()
    search = args.run / "search"
    provenance = read_json(search / "run.json")
    expected = digest({p.name: file_hash(p) for p in args.implementation.glob("*.py")})
    if provenance["implementation_sha256"] != expected:
        raise ValueError("Implementation changed since this run")
    if provenance["dependency_lock_sha256"] != file_hash(root / "uv.lock"):
        raise ValueError("Dependencies changed since this run")
    report = read_json(args.run / "audit/report.json")
    for path in (search / "rounds").iterdir():
        if path.is_dir() and completed(path) is None:
            raise ValueError("Incomplete search round")
    history = read_json(search / "summary.json")["history"]
    for index, row in enumerate(report["curve"]):
        if not index or row["candidate_dev"] is None:
            continue
        parent = search / "rounds" / f"{history[index - 1]['selected_round']:03}/results.jsonl"
        previous = {r["id"]: r["score"] for r in map(json.loads, parent.read_text().splitlines())}
        current = [
            json.loads(line)
            for line in (search / f"rounds/{index:03}/results.jsonl").read_text().splitlines()
        ]
        row["gained_vs_parent"] = sum(r["score"] > previous[r["id"]] for r in current)
        row["lost_vs_parent"] = sum(r["score"] < previous[r["id"]] for r in current)
    if completed(search / "final") is None:
        raise ValueError("Incomplete monitor evaluation")
    report["provenance"] = provenance
    report["audit_manifest"] = read_json(args.run / "audit/manifest.json")
    report["calibration_8k"] = read_json(args.run / "calibration/summary.json")
    previous_calibration = root / "runs/qwen-math-growth/calibration"
    calibration_directories = [args.run / "calibration"]
    if (previous_calibration / "config.json").exists() and read_json(
        previous_calibration / "config.json"
    )["task"]["model"] == provenance["config"]["task"]["model"]:
        report["calibration_4k"] = read_json(previous_calibration / "summary.json")
        calibration_directories.append(previous_calibration)
    calibration_calls = {"task": 0, "meta": 0}
    for directory in calibration_directories:
        path = directory / "calls.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["event"] == "started":
                    calibration_calls[row["kind"]] += 1
    report["calibration_application_calls"] = calibration_calls
    report["task_runtime"] = read_json(args.runtime)
    if report["task_runtime"]["model"] != provenance["config"]["task"]["model"]:
        raise ValueError("Runtime metadata belongs to a different task model")
    old = root / "runs/deepseek-v0.1"
    if (old / "summary.json").exists():
        report["previous_smoke"] = {
            "baseline": read_json(old / "rounds/000/summary.json"),
            "candidates": [read_json(old / f"rounds/{r:03}/summary.json") for r in range(1, 4)],
            "conclusion": "All 6/6: no headroom for strict dev improvement",
        }
    destination = root / "docs/validation"
    destination.mkdir(exist_ok=True)
    write_json(destination / f"{args.name}.json", report)
    for extension in ["png", "svg"]:
        shutil.copyfile(
            args.run / f"audit/growth.{extension}", destination / f"{args.name}.{extension}"
        )
    print(
        json.dumps(
            {
                "selected_round": report["selected_round"],
                "final": report["final"],
                "artifacts": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
