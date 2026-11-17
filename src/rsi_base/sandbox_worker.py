"""Stdlib-only worker, run inside a filesystem/network/PID namespace."""

import contextlib
import hashlib
import io
import json
import random
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
payload = json.load(sys.stdin)
space = {"__name__": "candidate_retriever"}
# Candidate stdout is not a transport channel; only validated JSON is returned.
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(payload["code"], "candidate.py", "exec"), space)
    results = []
    for job in payload["jobs"]:
        random.seed(int(hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(), 16))
        results.append(space["rank"](job["query"], job["memory"]))
usage = resource.getrusage(resource.RUSAGE_SELF)
json.dump(
    {
        "rankings": results,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
    },
    sys.stdout,
    allow_nan=False,
)
