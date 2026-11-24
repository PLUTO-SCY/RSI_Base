#!/usr/bin/env python3
"""Serve an immutable LoRA artifact alongside its base and optional parent adapter."""

import argparse
import os
from pathlib import Path

from rsi_base.lora_train import model_files
from rsi_base.sft import AdapterConfig, load_adapter


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "runs/lora-demo")
    args, extra = parser.parse_known_args()
    path = (args.run / "adapter").resolve()
    artifact = load_adapter(path)
    training_config = artifact["identity"]["config"]
    config = AdapterConfig.model_validate(
        {key: training_config[key] for key in AdapterConfig.model_fields if key in training_config}
    )
    if model_files(root / config.base_directory) != artifact["identity"]["base_files"]:
        raise ValueError("Serving base checkpoint differs from training")
    modules = [f"{artifact['served_model']}={path}"]
    if config.parent_adapter:
        parent = (root / config.parent_adapter).resolve()
        parent_artifact = load_adapter(parent)
        expected_parent = artifact["identity"].get("parent_adapter_sha256")
        if expected_parent and expected_parent != parent_artifact["artifact_sha256"]:
            raise ValueError("Serving parent adapter differs from training")
        modules.append(f"{parent_artifact['served_model']}={parent}")
    environment = {
        **os.environ,
        "RSI_TASK_MODEL_DIR": str((root / config.base_directory).resolve()),
        "RSI_TASK_MODEL_NAME": config.base_model,
        "RSI_TASK_PORT": os.environ.get("RSI_TASK_PORT", "18001"),
    }
    rank = max(8, 1 << (config.rank - 1).bit_length())
    command = [
        "sh",
        str(root / "scripts/serve_task.sh"),
        "--enable-lora",
        "--max-lora-rank",
        str(rank),
        "--max-cpu-loras",
        str(len(modules)),
        "--lora-modules",
        *modules,
        *extra,
    ]
    os.execve("/bin/sh", command, environment)


if __name__ == "__main__":
    main()
