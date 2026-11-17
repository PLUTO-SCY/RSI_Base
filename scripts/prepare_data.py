#!/usr/bin/env python3
"""Build or verify the default dataset from pinned public sources; no model calls."""

import argparse
import base64
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from rsi_base.classification import normalize
from rsi_base.loop import file_hash
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest

ROOT = Path(__file__).resolve().parents[1]


def load_lock():
    return read_json(ROOT / "requirements/data/s2d.json")


RAW_REVISION = load_lock()["sources"]["reference"]["revision"]


def download_blob(split, blob, *, get=httpx.get):
    source = load_lock()["sources"]["reference"]
    url = f"https://api.github.com/repos/{source['repository']}/git/blobs/{blob}"
    try:
        response = get(url, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError:
        # Public API quotas must not require users to configure a GitHub token.
        raw = (
            f"https://raw.githubusercontent.com/{source['repository']}/"
            f"{source['revision']}/{source['path']}/{split}.jsonl"
        )
        response = get(raw, timeout=60, follow_redirects=True)
        response.raise_for_status()
        data = response.content
    else:
        data = base64.b64decode(response.json()["content"])
    if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob:
        raise ValueError("Git data blob fingerprint mismatch")
    return data


def verify_asset(directory, lock):
    manifest = directory / "manifest.json"
    if not manifest.is_file() or read_json(manifest) != {**lock, "lock_sha256": digest(lock)}:
        raise ValueError(f"Incomplete or changed manifest: {directory}; no files were overwritten")
    for split, expected in lock["splits"].items():
        path = directory / f"{split}.jsonl"
        if not path.is_file() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Incomplete or changed data: {path}; no files were overwritten")
        if len(path.read_text().splitlines()) != expected["count"]:
            raise ValueError(f"Unexpected row count: {path}")
    labels = directory / "labels.json"
    if (
        not labels.is_file()
        or file_hash(labels) != lock["labels_sha256"]
        or read_json(labels) != lock["labels"]
    ):
        raise ValueError(f"Changed label set: {directory}")
    return {name: row["count"] for name, row in lock["splits"].items()}


def trigrams(text):
    return {text[i : i + 3] for i in range(max(0, len(text) - 2))}


def build(lock):
    assembled, seen = {}, set()
    for split, source in lock["sources"]["reference"]["splits"].items():
        payload = download_blob(split, source["git_blob"])
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if len(rows) != source["count"]:
            raise ValueError(f"Unexpected source size: {split}")
        converted = []
        for index, row in enumerate(rows):
            question, answer = row["question"], row["answer"].strip().casefold()
            normalized = normalize(question)
            if not normalized or answer not in lock["labels"]:
                raise ValueError("Empty question or unknown label")
            if normalized in seen:
                if split != "test":
                    raise ValueError("Unexpected training/validation duplicate")
                continue
            seen.add(normalized)
            converted.append(
                {"id": f"s2d-{split}-{index:03}", "question": question, "answer": answer}
            )
        assembled[split] = converted

    additional = lock["sources"]["additional"]
    response = httpx.get(additional["url"], follow_redirects=True, timeout=60)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != additional["sha256"]:
        raise ValueError("Additional source fingerprint mismatch")
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if len(rows) != additional["count"]:
        raise ValueError("Unexpected additional source size")
    known_grams = [trigrams(text) for text in seen]
    groups = defaultdict(list)
    threshold = lock["deduplication"]["character_trigram_jaccard_threshold"]
    for index, row in enumerate(rows):
        question, answer = row["input_text"], normalize(row["output_text"])
        normalized = normalize(question)
        if answer not in lock["labels"] or normalized in seen:
            continue
        chars = trigrams(normalized)
        if any(
            len(chars & other) / max(1, len(chars | other)) >= threshold for other in known_grams
        ):
            continue
        groups[answer].append(
            {"id": f"s2d-confirm-{index:04}", "question": question, "answer": answer}
        )
        seen.add(normalized)
        known_grams.append(chars)
    total = sum(map(len, groups.values()))
    if total != lock["retained_additional_rows"]:
        raise ValueError("Unexpected deduplicated source size")
    size = lock["development_size"]
    allocations = {label: len(rows) * size // total for label, rows in groups.items()}
    remainder = sorted(groups, key=lambda label: (-(len(groups[label]) * size % total), label))
    for label in remainder[: size - sum(allocations.values())]:
        allocations[label] += 1
    rng, validation, test = random.Random(lock["split_seed"]), [], []
    for label in sorted(groups):
        rows = rng.sample(groups[label], len(groups[label]))
        validation.extend(rows[: allocations[label]])
        test.extend(rows[allocations[label] :])
    result = {"train": assembled["train"]}
    for split, rows in [("val", validation), ("test", test)]:
        result[split] = [
            {**row, "id": row["id"].replace("s2d-confirm-", f"s2d-expanded-{split}-")}
            for row in sorted(rows, key=lambda row: row["id"])
        ]
    payloads = {}
    for split, rows in result.items():
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        expected = lock["splits"][split]
        if len(rows) != expected["count"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Rebuilt split differs from data lock: {split}")
        payloads[f"{split}.jsonl"] = data
    labels = (json.dumps(lock["labels"], ensure_ascii=False, indent=2) + "\n").encode()
    if hashlib.sha256(labels).hexdigest() != lock["labels_sha256"]:
        raise ValueError("Rebuilt labels differ from data lock")
    payloads["labels.json"] = labels
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/s2d")
    args = parser.parse_args()
    lock = load_lock()
    if not args.output.exists():
        payloads = build(lock)
        args.output.mkdir(parents=True)
        for name, data in payloads.items():
            (args.output / name).write_bytes(data)
        write_json(args.output / "manifest.json", {**lock, "lock_sha256": digest(lock)})
    counts = verify_asset(args.output, lock)
    print(json.dumps({"dataset": str(args.output), "verified": counts, "model_calls": 0}))


if __name__ == "__main__":
    main()
