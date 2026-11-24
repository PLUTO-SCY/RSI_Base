#!/usr/bin/env python3
"""Check tracked and nonignored sources without printing matching secret values."""

import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
known = [
    v
    for k, v in dotenv_values(root / ".env.local").items()
    if v and k.endswith(("API_KEY", "TOKEN", "SECRET"))
]
# Require a token boundary to avoid matching "sk-" inside ordinary task-runtime filenames.
# Known project secret values are also checked at every position independently below.
pattern = re.compile(
    rb"(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{25,})"
)
paths = subprocess.run(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    cwd=root,
    capture_output=True,
    check=True,
).stdout.split(b"\0")
failures = set()
for encoded in paths:
    if not encoded:
        continue
    name = encoded.decode()
    path = root / name
    if name == ".env.local" or name.startswith(
        (
            ".local/",
            ".venv/",
            ".venv-task/",
            ".venv-train/",
            ".venvs/",
            "runs/",
            "data/",
            "reports/",
            "results/",
            "archives/",
            "docs/validation/",
        )
    ):
        failures.add(name)
        continue
    if not path.is_file():
        continue
    data = path.read_bytes()
    if pattern.search(data) or any(value.encode() in data for value in known):
        failures.add(name)
    staged = subprocess.run(["git", "show", ":" + name], cwd=root, capture_output=True, check=False)
    if staged.returncode == 0 and (
        pattern.search(staged.stdout) or any(v.encode() in staged.stdout for v in known)
    ):
        failures.add(name)
if failures:
    print("Potential secret or private artifact in: " + ", ".join(sorted(failures)))
    raise SystemExit(1)
print("Source and staged secret scan passed; private artifacts excluded.")
