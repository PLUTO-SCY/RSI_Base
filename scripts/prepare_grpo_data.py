#!/usr/bin/env python3
"""Build the numeric benchmark from the pinned GSM8K release."""

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import httpx

from rsi_base.benchmark import check_disjoint, load_problems
from rsi_base.loop import file_hash
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def get_split(lock, split, cache):
    expected = lock["sources"][split]
    path = cache / f"{expected['sha256']}.jsonl"
    if not path.exists():
        url = (
            f"https://raw.githubusercontent.com/{lock['repository']}/{lock['revision']}/"
            f"grade_school_math/data/{split}.jsonl"
        )
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Unexpected GSM8K {split} source hash")
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            staged = Path(temporary) / "source.jsonl"
            staged.write_bytes(response.content)
            staged.rename(path)
    if file_hash(path) != expected["sha256"]:
        raise ValueError(f"Cached GSM8K {split} source changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected["count"]:
        raise ValueError(f"Unexpected GSM8K {split} size")
    return [
        {
            "id": f"gsm8k-{split}-{i}",
            "question": row["question"],
            "answer": row["answer"].rsplit("####", 1)[1].strip(),
        }
        for i, row in enumerate(rows)
    ]


def prepare(output, root):
    lock = read_json(root / "requirements/data/gsm8k.json")
    manifest = output / "manifest.json"
    if output.exists():
        prior = read_json(manifest)
        if prior["lock_sha256"] != digest(lock) or any(
            file_hash(output / f"{split}.jsonl") != metadata["sha256"]
            for split, metadata in prior["splits"].items()
        ):
            raise ValueError("Existing GSM8K data changed. Use a new output directory")
        return prior
    cache = root / ".local/gsm8k-sources"
    cache.mkdir(parents=True, exist_ok=True)
    rows = get_split(lock, "train", cache)
    test = get_split(lock, "test", cache)
    indices = set(
        random.Random(lock["validation_seed"]).sample(range(len(rows)), lock["validation_count"])
    )
    splits = {"train": [], "val": [], "test": test}
    for index, row in enumerate(rows):
        splits["val" if index in indices else "train"].append(row)
    check_disjoint(splits["train"], splits["val"])
    check_disjoint(rows, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gsm8k-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        result = {"lock_sha256": digest(lock), "splits": {}}
        for split, examples in splits.items():
            path = staged / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
            load_problems(path)
            result["splits"][split] = {"count": len(examples), "sha256": file_hash(path)}
        write_json(staged / "manifest.json", result)
        staged.rename(output)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data/gsm8k-grpo")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, root), indent=2))


if __name__ == "__main__":
    main()
