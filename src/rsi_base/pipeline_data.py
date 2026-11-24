"""Numeric splits and the two supervised data sources in the joint pipeline."""

import json
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from .agent import task_messages
from .benchmark import check_disjoint
from .components import checked_score
from .loop import file_hash
from .storage import read_jsonl, write_json
from .types import digest


@dataclass(frozen=True)
class PipelineData:
    train: list[dict]
    development: list[dict]
    references: dict[str, str]
    hashes: dict[str, str | None]


def load_data(directory, loader, scorer, *, load_references=True):
    files = {split: directory / f"{split}.jsonl" for split in ("train", "val", "test")}
    splits = {
        split: [] if split == "train" and not path.exists() else loader(path)
        for split, path in files.items()
    }
    if not splits["val"] or not splits["test"]:
        raise ValueError("Development and test splits must be nonempty")
    for a, b in combinations(splits.values(), 2):
        check_disjoint(a, b)
    references = {}
    path = directory / "references.jsonl"
    reference_file = load_references and path.is_file()
    if reference_file:
        answers = {row["id"]: row["answer"] for row in splits["train"]}
        for row in read_jsonl(path):
            if (
                set(row) != {"id", "response"}
                or not isinstance(row["id"], str)
                or not isinstance(row["response"], str)
            ):
                raise ValueError("Expected string id and response in references.jsonl")
            identifier, response = row["id"], row["response"]
            if identifier not in answers or identifier in references:
                raise ValueError(f"Duplicate or non-training reference ID: {identifier}")
            if not checked_score(scorer, response, answers[identifier])["score"]:
                raise ValueError(f"Incorrect supervised reference for {identifier}")
            references[identifier] = response
    return PipelineData(
        splits["train"],
        splits["val"],
        references,
        {
            **{split: file_hash(path) if path.exists() else None for split, path in files.items()},
            "references": file_hash(path) if reference_file else None,
        },
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


def write_sft_dataset(directory, identity, examples, excluded):
    train_path = directory / "train.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))
    report = {
        "identity_sha256": digest(identity),
        "train_sha256": file_hash(train_path),
        "samples": len(examples),
        "unique_training_problems": len({row["problem_id"] for row in examples}),
        "data_source": identity["data_source"],
        "excluded": excluded,
    }
    write_json(directory / "dataset.json", report)
    return report
