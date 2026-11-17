"""Execute candidate retrieval code with no host home, credentials, devices or network."""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def launcher():
    root = Path(__file__).resolve().parents[2]
    path = os.environ.get("RSI_BWRAP_BIN") or shutil.which("bwrap")
    if not path:
        path = str(root / ".local/sandbox-tools/usr/bin/bwrap")
    if not Path(path).is_file():
        raise ValueError("Code retrieval needs bwrap; run scripts/setup_sandbox.py")
    return path


def run_code(code, jobs, *, metrics=None):
    started = time.monotonic()
    command = [
        launcher(),
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for directory in ["/usr", "/lib", "/lib64"]:
        if Path(directory).exists():
            command += ["--ro-bind", directory, directory]
    command += [
        "--ro-bind",
        str(Path(sys.base_prefix).resolve()),
        "/python",
        "--ro-bind",
        str(Path(__file__).with_name("sandbox_worker.py")),
        "/runner.py",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        "/python/bin/python3.11",
        "-P",
        "-S",
        "/runner.py",
    ]
    try:
        result = subprocess.run(
            command,
            input=json.dumps({"code": code, "jobs": jobs}),
            capture_output=True,
            text=True,
            timeout=45,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Candidate sandbox exceeded its wall-time allowance") from exc
    if result.returncode:
        raise ValueError("Candidate sandbox failed: " + result.stderr[-1500:])
    try:
        envelope = json.loads(result.stdout)
        outputs = envelope["rankings"]
        if not isinstance(outputs, list) or len(outputs) != len(jobs):
            raise ValueError("Wrong code output batch size")
        checked = []
        for job, ranking in zip(jobs, outputs, strict=True):
            allowed = {row["id"] for row in job["memory"]}
            if not isinstance(ranking, list) or len(ranking) > len(allowed):
                raise ValueError("rank must return a list no longer than the supplied memory")
            seen, normalized = set(), []
            for offset, item in enumerate(ranking):
                identifier = item if isinstance(item, str) else item["id"]
                score = len(ranking) - offset if isinstance(item, str) else float(item["score"])
                if identifier not in allowed or identifier in seen or not math.isfinite(score):
                    raise ValueError("rank returned a duplicate/unknown memory ID or invalid score")
                seen.add(identifier)
                normalized.append({"id": identifier, "score": score})
            checked.append(normalized)
        if metrics is not None:
            metrics.append(
                {
                    "jobs": len(jobs),
                    "wall_seconds": time.monotonic() - started,
                    "worker_cpu_seconds": envelope["cpu_seconds"],
                    "worker_peak_rss_kib": envelope["peak_rss_kib"],
                }
            )
        return checked
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Candidate rank output violates the memory-only contract") from exc
