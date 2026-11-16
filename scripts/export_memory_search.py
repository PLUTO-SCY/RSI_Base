#!/usr/bin/env python3
"""Archive frozen searches independently of final-test access, including failed proposals."""

import argparse
import json
from pathlib import Path

from rsi_base.storage import read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "runs/memory-s2d-suite-v2")
    parser.add_argument("--name", default="memory-search-v2")
    args = parser.parse_args()
    directories = sorted(args.suite.glob("seed*"))
    if len(directories) != 3 or any(not (d / "frozen.json").exists() for d in directories):
        raise ValueError("All three searches must be frozen")
    reports = [
        {
            "source": read_json(d / "run.json"),
            "calibration": read_json(d / "calibration.json"),
            "search": read_json(d / "frozen.json"),
        }
        for d in directories
    ]
    for directory, report in zip(directories, reports, strict=True):
        report["invalid_rounds"] = [
            {
                "round": path.parent.name,
                "rejection": read_json(path),
                "raw_usage": read_json(path.parent / "meta_result.json").get("usage")
                if (path.parent / "meta_result.json").exists()
                else None,
            }
            for path in sorted((directory / "evolution").glob("*/invalid.json"))
        ]
    failures = []
    path = root / "runs/memory-s2d-suite-v1/seed42/evolution/001/meta_result.json"
    if path.exists():
        result = read_json(path)
        failures.append(
            {
                "run": "memory-s2d-suite-v1",
                "kind": "meta_output_overflow",
                "is_error": result["is_error"],
                "message": result.get("result"),
                "usage": result.get("usage"),
                "candidates_produced": 0,
                "test_accessed": False,
            }
        )
    calibrations = {}
    for name in ["memory-s2d-preflight-v1", "memory-s2d-preflight-v2"]:
        path = root / "runs" / name / "calibration.json"
        if path.exists():
            calibrations[name] = read_json(path)
    result = {
        "suite_protocol": read_json(args.suite / "suite.json"),
        "runs": reports,
        "prior_failed_meta": failures,
        "preflight_calibrations": calibrations,
        "data_manifest": read_json(root / "data/memory-s2d/manifest.json"),
        "weights_updated": False,
        "test_accessed": any((d / "audit.json").exists() for d in directories),
        "scope": "Frozen search evidence only; no unperformed test results inferred",
    }
    if (args.suite / "resume_verification.json").exists():
        result["resume_verification"] = read_json(args.suite / "resume_verification.json")
    write_json(root / "docs/validation" / f"{args.name}.json", result)
    print(json.dumps({"searches": len(reports), "test_accessed": result["test_accessed"]}))


if __name__ == "__main__":
    main()
