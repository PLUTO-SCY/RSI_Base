#!/usr/bin/env python3
"""Collect verified training trajectories, train LoRA, compare and freeze model selection."""

import argparse
import json
import os
from pathlib import Path

from rsi_base.sft import SFTConfig, collect, compare, local_key


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "train", "evaluate", "test"])
    parser.add_argument("--config", type=Path, default=root / "configs/lora_sft.json")
    parser.add_argument("--source-run", type=Path, default=root / "runs/demo")
    parser.add_argument("--data", type=Path, default=root / "data/s2d")
    parser.add_argument("--output", type=Path, default=root / "runs/lora-demo")
    parser.add_argument(
        "--base-url", help="Collection endpoint override; comparison default: local port 18001"
    )
    args = parser.parse_args()
    local_key()
    config = SFTConfig.model_validate_json(args.config.read_text())
    if args.phase == "train":
        os.execv(
            "/bin/sh",
            [
                "sh",
                str(root / "scripts/train_lora.sh"),
                "--config",
                str(args.config.resolve()),
                "--run",
                str(args.output.resolve()),
            ],
        )
    if args.phase == "collect":
        result = collect(
            config, args.source_run, args.data, args.output, root, base_url=args.base_url
        )
        print(
            json.dumps(
                {"samples": result["samples"], "excluded": result["excluded"], "test_opened": False}
            )
        )
    else:
        result = compare(
            config,
            args.data,
            args.output,
            root,
            base_url=args.base_url or "http://127.0.0.1:18001/v1",
            split="val" if args.phase == "evaluate" else "test",
        )
        print(
            json.dumps(
                {
                    "split": result["split"],
                    "accuracy": {
                        name: value["accuracy"] for name, value in result["metrics"].items()
                    },
                    "model_update_accepted": result["decision"]["accepted"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
