"""Small file records and append-only application call accounting."""

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def redact(text):
    return re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", str(text))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(redact(json.dumps(value, ensure_ascii=False, indent=2)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pin_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Pinned inputs changed. Use a new output directory: {path}")
    else:
        write_json(path, value)


@contextmanager
def lock_directory(output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / name).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(redact(json.dumps(value, ensure_ascii=False)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Counts task API calls and Claude invocations separately, including failed starts."""

    def __init__(self, path, limits):
        self.path, self.limits = Path(path), limits
        self.used = {"task": 0, "meta": 0}
        self.reserved_task_calls = 0
        rows = self.records()
        for row in rows:
            if row["event"] == "started":
                self.used[row["kind"]] += 1
        # A persisted deadline survives hard process termination, including pauses.
        starts = [row["time"] for row in rows if row["event"] in ("run_started", "started")]
        created = min(starts) if starts else time.time()
        self.deadline = created + limits.wall_seconds
        if not any(row["event"] == "run_started" for row in rows):
            append_json(
                self.path, {"event": "run_started", "time": created, "deadline": self.deadline}
            )
        self.started_at = time.monotonic()

    def records(self):
        if not self.path.exists():
            return []
        return read_jsonl(self.path)

    def start(self, kind, **metadata):
        maximum = self.limits.max_task_calls if kind == "task" else self.limits.max_meta_calls
        if kind == "task":
            maximum -= self.reserved_task_calls
        if self.used[kind] >= maximum:
            raise BudgetExceeded(f"{kind} call limit exhausted")
        if time.time() >= self.deadline:
            raise BudgetExceeded("run wall-time allowance exhausted")
        identifier = uuid.uuid4().hex
        append_json(
            self.path,
            {
                "event": "started",
                "kind": kind,
                "call_id": identifier,
                "time": time.time(),
                **metadata,
            },
        )
        self.used[kind] += 1
        return identifier

    def finish(self, identifier, kind, **result):
        append_json(
            self.path,
            {
                "event": "finished",
                "kind": kind,
                "call_id": identifier,
                "time": time.time(),
                **result,
            },
        )

    def close(self):
        append_json(
            self.path, {"event": "segment", "elapsed_seconds": time.monotonic() - self.started_at}
        )
