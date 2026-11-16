#!/usr/bin/env python3
"""Run the independent fixed-weight memory-harness experiment."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.memory_loop import MemoryConfig, calibrate


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate"])
    parser.add_argument("--config", type=Path, default=root / "configs/memory_s2d.json")
    parser.add_argument("--data", type=Path, default=root / "data/memory-s2d")
    parser.add_argument("--output", type=Path, default=root / "runs/memory-s2d-seed42")
    args = parser.parse_args()
    os.environ.setdefault("RSI_LOCAL_API_KEY", "EMPTY")
    config = MemoryConfig.model_validate_json(args.config.read_text())
    result = calibrate(config, args.data, args.output, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
