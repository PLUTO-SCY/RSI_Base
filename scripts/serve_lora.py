#!/usr/bin/env python3
"""Serve a verified full checkpoint or LoRA artifact."""

import argparse
import os
from pathlib import Path

from rsi_base.weights import AdapterConfig, load_artifact, load_parent, model_files


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/lora-demo")
    parser.add_argument(
        "--parent", action="store_true", help="Serve the training parent on port 18000 by default"
    )
    args, extra = parser.parse_known_args()
    full = (args.run / "model").is_dir()
    path = (args.run / ("model" if full else "adapter")).resolve()
    artifact = load_artifact(path, "full" if full else "lora")
    training_config = artifact["identity"]["config"]
    config = AdapterConfig.model_validate(
        {key: training_config[key] for key in AdapterConfig.model_fields if key in training_config}
    )
    if (config.training_mode == "full") != full:
        raise ValueError("Serving artifact layout differs from its training mode")
    model_dir = (root / config.base_directory).resolve()
    model_name = config.base_model
    modules = []
    if full and not args.parent:
        model_dir, model_name = path, artifact["served_model"]
    else:
        parent = load_parent(config, root)
        if parent and parent["artifact_sha256"] != artifact["identity"][config.parent_hash_key]:
            raise ValueError("Serving parent differs from training")
        if full and parent:
            model_dir = (root / config.parent_checkpoint).resolve()
            model_name = parent["served_model"]
        else:
            if model_files(model_dir) != artifact["identity"]["base_files"]:
                raise ValueError("Serving base checkpoint differs from training")
            if not args.parent:
                modules.append(f"{artifact['served_model']}={path}")
            if parent:
                parent_dir = (root / config.parent_adapter).resolve()
                modules.append(f"{parent['served_model']}={parent_dir}")
    environment = {
        **os.environ,
        "RSI_TASK_MODEL_DIR": str(model_dir),
        "RSI_TASK_MODEL_NAME": model_name,
        "RSI_TASK_PORT": os.environ.get("RSI_TASK_PORT", "18000" if args.parent else "18001"),
    }
    command = ["sh", str(root / "scripts/serve_task.sh")]
    if modules:
        rank = max(8, 1 << (config.rank - 1).bit_length())
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(rank),
                "--max-cpu-loras",
                str(len(modules)),
                "--lora-modules",
                *modules,
            ]
        )
    os.execve("/bin/sh", [*command, *extra], environment)


if __name__ == "__main__":
    main()
