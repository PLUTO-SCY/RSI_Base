#!/usr/bin/env python3
"""Record the local task backend and checkpoint fingerprints, without environment dumps."""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument(
        "--directory", type=Path, default=Path(".local/models/qwen3-4b-instruct-2507")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--output", type=Path, default=Path(".local/task-runtime.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = {}
    for path in sorted(args.directory.iterdir()):
        if path.suffix in {".json", ".safetensors"}:
            with path.open("rb") as handle:
                files[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    value = {
        "model": args.model,
        "revision": args.revision,
        "gpu_index": args.gpu,
        "port": args.port,
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ["vllm", "torch", "transformers", "huggingface-hub"]
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "files": files,
        "server_script": (root / "scripts/serve_task.sh").read_text(),
        "task_requirements_sha256": hashlib.sha256(
            (root / "requirements/task.lock").read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Recorded {args.model} checkpoint and runtime fingerprints")


if __name__ == "__main__":
    main()
