#!/usr/bin/env python3
"""Train a LoRA adapter with verl or compare it through the Task endpoint."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.rl import GRPOConfig, evaluate_grpo


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate", "test"])
    parser.add_argument("--config", type=Path, default=root / "configs/lora_grpo.json")
    parser.add_argument("--data", type=Path, default=root / "data/gsm8k-grpo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    args = parser.parse_args()
    if args.phase == "train":
        interpreter = root / ".venvs/rl/bin/python"
        if not interpreter.is_file():
            parser.error("Run sh scripts/harness.sh setup --rl first.")
        environment = {**os.environ, "PYTHONPATH": str(root / "src")}
        os.execve(
            interpreter,
            [
                str(interpreter),
                str(root / "scripts/train_grpo.py"),
                "--config",
                str(args.config),
                "--data",
                str(args.data),
                "--output",
                str(args.output),
            ],
            environment,
        )
    config = GRPOConfig.model_validate_json(args.config.read_text())
    report = evaluate_grpo(
        config,
        args.data,
        args.output,
        root,
        args.base_url,
        split="val" if args.phase == "evaluate" else "test",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
