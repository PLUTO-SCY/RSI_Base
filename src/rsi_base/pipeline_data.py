"""Numeric splits and the two supervised data sources in the joint pipeline."""

import json
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from .agent import task_messages
from .benchmark import check_disjoint, load_problems
from .loop import file_hash
from .math_scoring import math_score
from .memory_loop import load_rows
from .sft import verify_dataset
from .storage import write_json
from .types import digest


@dataclass(frozen=True)
class PipelineData:
    train: list[dict]
    development: list[dict]
    references: dict[str, str]
    hashes: dict[str, str | None]
    test_count: int


def load_data(directory):
    files = {split: directory / f"{split}.jsonl" for split in ("train", "val", "test")}
    splits = {split: load_problems(path) for split, path in files.items()}
    for a, b in combinations(splits.values(), 2):
        check_disjoint(a, b)
    references = {}
    path = directory / "references.jsonl"
    if path.exists():
        answers = {row["id"]: row["answer"] for row in splits["train"]}
        for row in load_rows(path):
            if (
                set(row) != {"id", "response"}
                or not isinstance(row["id"], str)
                or not isinstance(row["response"], str)
            ):
                raise ValueError("Expected string id and response in references.jsonl")
            identifier, response = row["id"], row["response"]
            if identifier not in answers or identifier in references:
                raise ValueError(f"Duplicate or non-training reference ID: {identifier}")
            if not math_score(response, answers[identifier])["score"]:
                raise ValueError(f"Incorrect supervised reference for {identifier}")
            references[identifier] = response
    return PipelineData(
        splits["train"],
        splits["val"],
        references,
        {
            **{split: file_hash(path) for split, path in files.items()},
            "references": file_hash(path) if path.exists() else None,
        },
        len(splits["test"]),
    )


def select_sft_problems(data, source, count, seed):
    pool = data.train
    if source == "offline":
        pool = [row for row in pool if row["id"] in data.references]
    return random.Random(seed).sample(pool, min(count, len(pool)))


def offline_examples(problems, agent, references):
    return [
        {
            "id": row["id"],
            "problem_id": row["id"],
            "messages": [
                *task_messages(agent.system_prompt, row["question"]),
                {"role": "assistant", "content": references[row["id"]]},
            ],
        }
        for row in problems
    ]


def online_examples(batches):
    examples, excluded, seen = [], Counter(), set()
    for repeat, batch in enumerate(batches):
        for row in batch:
            if row["finish_reason"] != "stop":
                excluded["unfinished_response"] += 1
                continue
            if not row["score"]:
                excluded[row["status"]] += 1
                continue
            messages = [*row["messages"], {"role": "assistant", "content": row["response"]}]
            key = digest(messages)
            if key in seen:
                excluded["duplicate_messages"] += 1
                continue
            seen.add(key)
            examples.append(
                {
                    "id": f"{repeat}:{row['id']}",
                    "problem_id": row["id"],
                    "messages": messages,
                    "trace_sha256": digest(row),
                }
            )
    return examples, dict(excluded)


def write_sft_dataset(directory, config, identity, examples, excluded):
    train_path = directory / "train.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
    write_json(
        directory / "dataset.json",
        {
            "identity_sha256": digest(identity),
            "train_sha256": file_hash(train_path),
            "samples": len(examples),
            "unique_training_problems": len({row["problem_id"] for row in examples}),
            "data_source": identity["data_source"],
            "excluded": excluded,
        },
    )
    return verify_dataset(directory, config)
