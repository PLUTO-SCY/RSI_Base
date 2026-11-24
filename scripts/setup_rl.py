#!/usr/bin/env python3
"""Install the pinned verl stack in a project-local environment."""

import fcntl
import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

from rsi_base.storage import read_json


def main():
    root = Path(__file__).resolve().parents[1]
    pin = read_json(root / "requirements/verl.json")
    local = root / ".local/verl"
    local.mkdir(parents=True, exist_ok=True)
    temporary = local / "tmp"
    temporary.mkdir(exist_ok=True)
    revision = pin["revision"]
    archive = local / f"{revision}.tar.gz"
    source = local / f"verl-{revision}"
    with (local / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not archive.exists():
            pending = archive.with_suffix(".partial")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "120",
                    "--output",
                    str(pending),
                    f"https://codeload.github.com/verl-project/verl/tar.gz/{revision}",
                ],
                check=True,
            )
            if hashlib.sha256(pending.read_bytes()).hexdigest() != pin["archive_sha256"]:
                raise ValueError("Unexpected verl source archive hash")
            pending.rename(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["archive_sha256"]:
            raise ValueError("The cached verl source archive changed")
        with tarfile.open(archive) as bundle:
            bundle.extractall(local, filter="data")
        if hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest() != pin["lock_sha256"]:
            raise ValueError("Unexpected verl dependency lock hash")
        environment = {
            **os.environ,
            "TMPDIR": str(temporary),
            "UV_CACHE_DIR": str(root / ".local/uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(root / ".local/python"),
            "UV_PROJECT_ENVIRONMENT": str(root / ".venvs/rl"),
            "UV_LINK_MODE": "copy",
        }
        # Use SDPA in the trainer because this tag's external flash-attn wheel is unavailable.
        subprocess.run(
            [
                "uv",
                "sync",
                "--frozen",
                "--no-dev",
                "--no-editable",
                "--extra",
                "fsdp",
                "--extra",
                "vllm",
                "--no-install-package",
                "flash-attn",
                "--no-install-package",
                "trl",
                "--python",
                pin["python"],
            ],
            cwd=source,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
