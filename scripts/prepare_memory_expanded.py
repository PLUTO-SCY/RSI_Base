#!/usr/bin/env python3
"""Repartition the still-unqueried reserve into 300 selection / 291 fresh-test queries."""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from rsi_base.classification import normalize
from rsi_base.loop import file_hash
from rsi_base.memory_loop import load_rows
from rsi_base.storage import read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data/memory-s2d-expanded")
    parser.add_argument(
        "--rebuild-published-split",
        action="store_true",
        help="Reconstruct and verify the published split without private historical run files",
    )
    args = parser.parse_args()
    old = root / "runs/memory-s2d-suite-v4"
    if not args.rebuild_published_split:
        status = read_json(old / "confirmation.json")
        if status["status"] != "cancelled_unevaluated" or list(
            old.glob("seed*/confirmation/calls.jsonl")
        ):
            raise ValueError("Reserve must have zero prior model evaluations")
    destination = args.output
    if destination.exists():
        raise ValueError("Refusing to overwrite a locked partition")
    reserve = root / "data/memory-confirmation"
    source = load_rows(reserve / "test.jsonl")
    if len(source) != 591:
        raise ValueError("Unexpected reserve size")
    groups = defaultdict(list)
    for row in source:
        groups[row["answer"]].append(row)
    allocations = {label: len(rows) * 300 // len(source) for label, rows in groups.items()}
    remainder = sorted(groups, key=lambda label: (-(len(groups[label]) * 300 % len(source)), label))
    for label in remainder[: 300 - sum(allocations.values())]:
        allocations[label] += 1
    rng, validation, test = random.Random(20261116), [], []
    for label in sorted(groups):
        rows = rng.sample(groups[label], len(groups[label]))
        validation.extend(rows[: allocations[label]])
        test.extend(rows[allocations[label] :])
    destination.mkdir(parents=True)
    original = root / "data/memory-s2d"
    shutil.copyfile(original / "train.jsonl", destination / "train.jsonl")
    shutil.copyfile(original / "labels.json", destination / "labels.json")
    for split, rows in [("val", validation), ("test", test)]:
        converted = [
            {**row, "id": row["id"].replace("s2d-confirm-", f"s2d-expanded-{split}-")}
            for row in sorted(rows, key=lambda row: row["id"])
        ]
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted)
        )
    seen, splits = set(), {}
    for split in ["train", "val", "test"]:
        rows = load_rows(destination / f"{split}.jsonl")
        for row in rows:
            question = normalize(row["question"])
            if question in seen:
                raise ValueError("Cross-split overlap")
            seen.add(question)
        splits[split] = {
            "count": len(rows),
            "sha256": file_hash(destination / f"{split}.jsonl"),
            "ids": [row["id"] for row in rows],
        }
    write_json(
        destination / "manifest.json",
        {
            "benchmark": "Symptom2Disease expanded selector; independently repartitioned reserve, not the paper's official score",
            "splits": splits,
            "labels": read_json(destination / "labels.json"),
            "seed": 20261116,
            "selection_size": 300,
            "fresh_test_size": 291,
            "source_reserve_manifest_sha256": file_hash(reserve / "manifest.json"),
            "source_reserve": read_json(reserve / "manifest.json"),
            "previous_211_test_role": "Retired failure diagnostic; not a final-test criterion for this protocol",
            "prior_reserve_model_calls": 0,
            "new_test_scores_consulted": False,
            "partition_rule": "Label-stratified largest-remainder allocation to 300 validation, seeded shuffle within label; all remaining 291 become test",
            "train_unchanged_sha256": hashlib.sha256(
                (original / "train.jsonl").read_bytes()
            ).hexdigest(),
        },
    )
    if args.rebuild_published_split:
        expected = read_json(root / "docs/validation/memory-expanded-manifest.json")
        if read_json(destination / "manifest.json") != expected:
            raise ValueError("Reconstructed split differs from the published manifest")
    print(
        json.dumps(
            {
                "train": 200,
                "validation": len(validation),
                "fresh_test": len(test),
                "reserve_previously_evaluated": None if args.rebuild_published_split else False,
                "published_rebuild": args.rebuild_published_split,
                "new_unseen_holdout_claim": not args.rebuild_published_split,
            }
        )
    )


if __name__ == "__main__":
    main()
