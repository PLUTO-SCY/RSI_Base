#!/usr/bin/env python3
"""Run the independent fixed-weight memory-harness experiment."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.memory_audit import finalize
from rsi_base.memory_loop import MemoryConfig, calibrate
from rsi_base.memory_search import search
from rsi_base.storage import read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "search", "suite", "finalize"])
    parser.add_argument("--config", type=Path, default=root / "configs/memory_s2d.json")
    parser.add_argument(
        "--seed", type=int, help="Independent calibration/search seed; suite seeds are predeclared"
    )
    parser.add_argument("--data", type=Path, default=root / "data/memory-s2d")
    parser.add_argument("--output", type=Path, default=root / "runs/memory-s2d-seed42")
    args = parser.parse_args()
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
    config = MemoryConfig.model_validate_json(args.config.read_text())
    if args.seed is not None:
        if args.phase not in {"calibrate", "search"}:
            raise ValueError("--seed applies only to independent calibration/search")
        config = config.model_copy(
            update={
                "seed": args.seed,
                "meta": config.meta.model_copy(
                    update={"state_dir": f".local/claude-memory/seed{args.seed}"}
                ),
            }
        )
    if args.phase == "calibrate":
        result = calibrate(config, args.data, args.output, root)
    elif args.phase == "search":
        result = search(config, args.data, args.output, root)
    else:
        seeds = [42, 123, 2026]
        protocol = {
            "config": config.model_dump(),
            "seeds": seeds,
            "test_gate": "Freeze all three searches before any final evaluation",
            "stability_gate": "Positive test delta in every search seed and question-cluster bootstrap lower bound > 0; repeats averaged within each question",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "suite.json"
        if path.exists() and read_json(path) != protocol:
            raise ValueError("Predeclared suite protocol changed")
        write_json(path, protocol)
        directories = [args.output / f"seed{seed}" for seed in seeds]
        if args.phase == "suite":
            for seed, directory in zip(seeds, directories, strict=True):
                local = config.model_copy(
                    update={
                        "seed": seed,
                        "meta": config.meta.model_copy(
                            update={"state_dir": f".local/claude-memory/seed{seed}"}
                        ),
                    }
                )
                search(local, args.data, directory, root)
            result = {"searches_frozen": True, "seeds": seeds, "test_evaluated": False}
        else:
            result = finalize(directories, args.data, args.output, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
