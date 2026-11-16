#!/usr/bin/env python3
"""Download a pinned public task checkpoint into project-local storage."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument(
        "--directory", type=Path, default=root / ".local/models/qwen3-4b-instruct-2507"
    )
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise ValueError("Use an immutable full commit revision")
    print(
        snapshot_download(
            args.model,
            revision=args.revision,
            local_dir=args.directory,
            ignore_patterns=["*.bin", "*.gguf", "*.onnx"],
            token=False,
        )
    )


if __name__ == "__main__":
    main()
