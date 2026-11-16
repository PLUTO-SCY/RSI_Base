#!/usr/bin/env python3
"""Fetch official GSM8K train data and create disjoint, deterministic small splits."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Full official repository commit SHA")
    parser.add_argument("--output", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--dev-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("revision must be a complete lowercase commit SHA")
    if min(args.dev_size, args.test_size) < 1:
        parser.error("split sizes must be positive")
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{args.revision}/grade_school_math/data/train.jsonl"
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    items, seen = [], set()
    for index, row in enumerate(rows):
        question = row["question"].strip()
        normalized = " ".join(question.split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(
            {
                "id": f"gsm8k-train-{index}",
                "question": question,
                "answer": row["answer"].split("####")[-1].strip(),
            }
        )
    random.Random(args.seed).shuffle(items)
    if args.dev_size + args.test_size >= len(items):
        parser.error("not enough unique examples for requested splits")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("dev.jsonl", "test.jsonl", "manifest.json")):
        raise SystemExit("Output already contains split files; choose a new output directory.")
    for name, subset in (
        ("dev", items[: args.dev_size]),
        ("test", items[args.dev_size : args.dev_size + args.test_size]),
    ):
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
            encoding="utf-8",
        )
    manifest = {
        "source": url,
        "revision": args.revision,
        "source_sha256": hashlib.sha256(response.content).hexdigest(),
        "source_split": "train",
        "seed": args.seed,
        "dev_size": args.dev_size,
        "test_size": args.test_size,
        "evaluation_scope": "held-out training-pool pilot, not official test",
        "split_hashes": {
            n: hashlib.sha256((args.output / f"{n}.jsonl").read_bytes()).hexdigest()
            for n in ("dev", "test")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
