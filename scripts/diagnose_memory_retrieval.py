#!/usr/bin/env python3
"""Training/validation-only retrieval diagnostics; no model calls or search feedback."""

import json
from collections import Counter
from pathlib import Path

from rsi_base.classification import MemoryIndex, MemoryPolicy
from rsi_base.memory_loop import load_rows
from rsi_base.storage import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/memory-s2d"
    index = MemoryIndex(load_rows(data / "train.jsonl"))
    validation = load_rows(data / "val.jsonl")
    reports = {}
    for retrieval in ["bm25", "tfidf", "hybrid"]:
        metrics = Counter()
        for problem in validation:
            rows, _ = index.retrieve(
                problem["question"], MemoryPolicy(retrieval=retrieval, examples=32)
            )
            metrics["nearest_label_correct"] += rows[0]["answer"] == problem["answer"]
            for k in [3, 8, 16, 32]:
                metrics[f"gold_label_in_top{k}"] += problem["answer"] in {
                    r["answer"] for r in rows[:k]
                }
                votes = Counter()
                for row in rows[:k]:
                    votes[row["answer"]] += max(0, row["retrieval_score"])
                metrics[f"weighted_vote_top{k}_correct"] += (
                    votes.most_common(1)[0][0] == problem["answer"]
                )
        reports[retrieval] = dict(metrics)
    result = {
        "split": "validation",
        "examples": len(validation),
        "test_accessed": False,
        "not_fed_to_active_search": True,
        "retrieval": reports,
    }
    write_json(root / "runs/memory-retrieval-diagnosis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
