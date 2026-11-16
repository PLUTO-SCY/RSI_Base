#!/usr/bin/env python3
"""Fixed numeric-only MATH-500 research split, not its official evaluation protocol."""

import hashlib
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import httpx

from rsi_base.benchmark import check_disjoint, load_problems, parse_number
from rsi_base.storage import write_json

REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
URL = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{REVISION}/test.jsonl"
SIZES = {"calibration": 16, "dev": 48, "monitor": 32, "final": 64}


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/math-growth"
    if output.exists():
        raise SystemExit("Output exists; keep the frozen split or explicitly choose a new protocol")
    response = httpx.get(URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    raw = response.content
    source = [json.loads(line) for line in raw.splitlines()]
    eligible = []
    for index, row in enumerate(source):
        if row["level"] < 3 or "[asy]" in row["problem"]:
            continue
        answer = re.sub(r"\\(?:d?frac)\{(-?\d+)\}\{(\d+)\}", r"\1/\2", row["answer"])
        try:
            answer = str(parse_number(answer))
        except (ValueError, ZeroDivisionError):
            continue
        eligible.append((index, row, answer))
    random.Random(42).shuffle(eligible)
    if len(eligible) < sum(SIZES.values()):
        raise ValueError("Not enough eligible problems")
    output.mkdir(parents=True)
    (output / "source.jsonl").write_bytes(raw)
    manifest = {
        "source": URL,
        "revision": REVISION,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": 42,
        "source_count": len(source),
        "eligible_count": len(eligible),
        "filter": "level >= 3; no [asy]; numeric or integer fraction answer",
        "protocol": "repartitioned numeric MATH-500; not an official benchmark score",
        "splits": {},
    }
    offset = 0
    for name, count in SIZES.items():
        selected = eligible[offset : offset + count]
        offset += count
        rows = [
            {"id": f"math500-{index:03}", "question": row["problem"], "answer": answer}
            for index, row, answer in selected
        ]
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (output / f"{name}.jsonl").write_bytes(content)
        manifest["splits"][name] = {
            "count": count,
            "sha256": hashlib.sha256(content).hexdigest(),
            "ids": [r["id"] for r in rows],
            "levels": dict(Counter(row["level"] for _, row, _ in selected)),
        }
    for a, b in itertools.combinations(SIZES, 2):
        check_disjoint(load_problems(output / f"{a}.jsonl"), load_problems(output / f"{b}.jsonl"))
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"eligible": len(eligible), "sizes": SIZES, "directory": str(output)}))


if __name__ == "__main__":
    main()
