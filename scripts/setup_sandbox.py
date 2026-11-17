#!/usr/bin/env python3
"""Extract a pinned namespace launcher locally; never install a system package."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
pin = json.loads((root / "requirements/sandbox.json").read_text())
if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
    raise SystemExit("Provide a compatible bwrap on PATH on this platform")
package = root / ".local/installers/bubblewrap.deb"
package.parent.mkdir(parents=True, exist_ok=True)
if not package.exists() or hashlib.sha256(package.read_bytes()).hexdigest() != pin["sha256"]:
    response = httpx.get(pin["url"], timeout=60, follow_redirects=True)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != pin["sha256"]:
        raise ValueError("Namespace launcher package checksum mismatch")
    package.write_bytes(response.content)
destination = root / ".local/sandbox-tools"
subprocess.run(["dpkg-deb", "--extract", str(package), str(destination)], check=True)
print(subprocess.check_output([str(destination / "usr/bin/bwrap"), "--version"], text=True).strip())
