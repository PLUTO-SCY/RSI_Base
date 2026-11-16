#!/usr/bin/env python3
"""Export all calibration attempts, including failed protocol fits and diagnostic rescoring."""

import json
from pathlib import Path

from rsi_base.loop import file_hash
from rsi_base.math_scoring import math_score
from rsi_base.storage import read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    attempts = [
        ("qwen-math-growth", "calibration/qwen3_4b_4k.json"),
        ("qwen-math-growth-8k", "qwen_math_growth.json"),
        ("qwen-small-math-growth", "calibration/qwen2_5_1_5b_system.json"),
        ("qwen-small-math-growth-format", "calibration/qwen2_5_1_5b_system_example.json"),
        ("qwen-small-math-growth-user", "calibration/qwen2_5_1_5b_user.json"),
        ("qwen-1_7b-math-growth", "calibration/qwen3_1_7b_strict.json"),
        ("qwen-1_7b-math-v2", "qwen_1_7b_math_growth.json"),
    ]
    data = root / "data/math-growth/calibration.jsonl"
    references = {r["id"]: r["answer"] for r in map(json.loads, data.read_text().splitlines())}
    rows, total_calls = [], {"task": 0, "meta": 0}
    for name, config_path in attempts:
        directory = root / "runs" / name / "calibration"
        config = read_json(root / "configs" / config_path)
        records = list(map(json.loads, (directory / "results.jsonl").read_text().splitlines()))
        alternative = [math_score(r["response"], references[r["id"]]) for r in records]
        calls = {"task": 0, "meta": 0}
        for event in map(json.loads, (directory / "calls.jsonl").read_text().splitlines()):
            if event["event"] == "started":
                calls[event["kind"]] += 1
        for kind, count in calls.items():
            total_calls[kind] += count
        rows.append(
            {
                "run": name,
                "config": f"configs/{config_path}",
                "config_sha256": file_hash(root / "configs" / config_path),
                "task_model": config["task"]["model"],
                "evaluator": config.get("evaluator", "numeric"),
                "summary": read_json(directory / "summary.json"),
                "application_calls": calls,
                "math_numeric_rescore_diagnostic_only": {
                    "correct": sum(r["score"] for r in alternative),
                    "format_errors": sum(r["status"] == "format_error" for r in alternative),
                },
            }
        )
    native = read_json(root / ".local/native-task-probe.json")
    write_json(
        root / "docs/validation/calibrations.json",
        {
            "calibration_sha256": file_hash(data),
            "attempts": rows,
            "application_calls": total_calls,
            "interpretation": "Protocol/model calibration and same-response rescoring are not RSI improvements",
            "native_vs_served_probe": [
                {"native": r["native"], "served": r["served"]} for r in native["results"]
            ],
        },
    )
    print(json.dumps({"attempts": len(rows), "calibration_calls": total_calls}))


if __name__ == "__main__":
    main()
