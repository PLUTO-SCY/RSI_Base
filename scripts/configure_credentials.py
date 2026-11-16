#!/usr/bin/env python3
"""Store a supplied provider credential locally without echoing it."""

import getpass
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / ".env.local"
if target.exists():
    raise SystemExit(".env.local already exists; edit it locally rather than overwrite it.")
secret = getpass.getpass("Project API key (hidden): ").strip()
if not secret or any(c in secret for c in "\r\n'"):
    raise SystemExit("Invalid credential.")
fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(fd, "w") as handle:
    handle.write("RSI_API_BASE_URL=https://zgc.apihy.com/v1\n")
    handle.write("RSI_API_KEY='" + secret + "'\n")
print("Saved project-local credential to .env.local (mode 0600).")
