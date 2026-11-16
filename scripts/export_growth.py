#!/usr/bin/env python3
"""Export aggregate evidence and figures; keep questions, responses and secrets local."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from rsi_base.loop import completed, file_hash, run
from rsi_base.storage import read_json, write_json
from rsi_base.types import RunConfig, digest


def matching_commit(root, expected):
    commits = subprocess.check_output(
        ["git", "log", "--format=%H", "--", "src/rsi_base"], cwd=root, text=True
    ).splitlines()
    for commit in commits:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", "src/rsi_base"],
            cwd=root,
            text=True,
        ).splitlines()
        hashes = {}
        for path in paths:
            if path.endswith(".py"):
                import hashlib

                data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
                hashes[Path(path).name] = hashlib.sha256(data).hexdigest()
        if digest(hashes) == expected:
            return commit
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/qwen-math-growth-8k")
    parser.add_argument("--runtime", type=Path, default=root / ".local/task-runtime.json")
    parser.add_argument("--name", choices=["growth", "growth-v2", "growth-small"], default="growth")
    parser.add_argument("--implementation", type=Path, default=root / "src/rsi_base")
    parser.add_argument("--verify-resume", action="store_true")
    args = parser.parse_args()
    search = args.run / "search"
    provenance = read_json(search / "run.json")
    expected = digest({p.name: file_hash(p) for p in args.implementation.glob("*.py")})
    if provenance["implementation_sha256"] != expected:
        raise ValueError("Implementation changed since this run")
    if provenance["dependency_lock_sha256"] != file_hash(root / "uv.lock"):
        raise ValueError("Dependencies changed since this run")
    report = read_json(args.run / "audit/report.json")
    if args.verify_resume:
        original = read_json(search / "summary.json")

        def started():
            return [
                row
                for row in map(json.loads, (search / "calls.jsonl").read_text().splitlines())
                if row["event"] == "started"
            ]

        before = started()
        resumed = run(
            RunConfig.model_validate(provenance["config"]),
            root / "data/math-growth/dev.jsonl",
            root / "data/math-growth/monitor.jsonl",
            search,
            root,
            resume=True,
        )
        if resumed != original or started() != before:
            raise ValueError("Completed resume changed the summary or admitted a new call")
        write_json(
            args.run / "verification.json",
            {
                "resume_identical": True,
                "new_application_calls": {"task": 0, "meta": 0},
                "summary_sha256": file_hash(search / "summary.json"),
                "implementation_sha256": provenance["implementation_sha256"],
            },
        )
    if (args.run / "verification.json").exists():
        report["resume_verification"] = read_json(args.run / "verification.json")
    for path in (search / "rounds").iterdir():
        if path.is_dir() and completed(path) is None:
            raise ValueError("Incomplete search round")
    search_summary = read_json(search / "summary.json")
    history = search_summary["history"]
    report["first_selected_monitor_evaluation"] = search_summary["held_out"]
    report["monitor_repeat_note"] = (
        "Core evaluates the frozen selected prompt once; the displayed audit curve then evaluates each distinct incumbent. Both passes are retained."
    )
    report["selected_agent"] = history[-1]["agent"]
    report["proposals"] = []
    for path in sorted((search / "rounds").glob("*/proposal.json")):
        report["proposals"].append({"round": int(path.parent.name), **read_json(path)})
    # Frozen policies make curve replay possible without re-querying Meta.
    dev_ids = set(report["data_manifest"]["splits"]["dev"]["ids"])
    source_rows = search / "rounds/000/results.jsonl"
    questions = [
        r["question"]
        for r in map(json.loads, source_rows.read_text().splitlines())
        if r["id"] in dev_ids
    ]
    policy_text = " ".join(
        (
            report["selected_agent"]["system_prompt"],
            *[p["system_prompt"] + " " + p["reason"] for p in report["proposals"]],
        )
    )
    policy_text = " ".join(policy_text.split()).casefold()
    if any(len(q) > 40 and " ".join(q.split()).casefold() in policy_text for q in questions):
        raise ValueError(
            "A proposal contains a development question; keep it private and audit leakage"
        )
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
    report["implementation_git_commit"] = matching_commit(root, provenance["implementation_sha256"])
    report["audit_manifest"] = read_json(args.run / "audit/manifest.json")
    if (args.run / "recheck/report.json").exists():
        report["frozen_development_rechecks"] = read_json(args.run / "recheck/report.json")
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
        source = args.run / f"audit/growth.{extension}"
        target = destination / f"{args.name}.{extension}"
        if extension == "svg":
            # Matplotlib emits path lines with trailing spaces; normalize public text only.
            target.write_text(
                "\n".join(line.rstrip() for line in source.read_text().splitlines()) + "\n"
            )
        else:
            shutil.copyfile(source, target)
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
