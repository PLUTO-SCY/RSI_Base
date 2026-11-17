#!/usr/bin/env python3
"""Exercise production Task plumbing on two selection queries, with an optional Meta call."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

from rsi_base.agent import TaskAgent, credentials
from rsi_base.classification import MemoryHarness, MemoryPolicy, SingleMemoryBatch
from rsi_base.improve import ClaudeMeta
from rsi_base.loop import file_hash
from rsi_base.memory_loop import MemoryConfig, evaluate_policy, load_rows, memory_meta_system
from rsi_base.storage import Ledger, read_json, write_json
from rsi_base.types import Limits


def smoke(config, data, output, root, *, with_meta=False, task=None, meta=None):
    """Keep final-test data entirely outside this inexpensive wiring check."""
    training, queries = load_rows(data / "train.jsonl"), load_rows(data / "val.jsonl")[:2]
    if len(queries) != 2:
        raise ValueError("Smoke needs at least two selection queries")
    if output.exists():
        raise ValueError("Smoke output already exists; use --output runs/another-smoke")
    output.mkdir(parents=True)
    policy = MemoryPolicy()
    write_json(
        output / "smoke.json",
        {
            "purpose": "Connectivity only; not an evolution result or final-test estimate",
            "config": config.model_dump(),
            "policy": policy.model_dump(),
            "split_sha256": {s: file_hash(data / f"{s}.jsonl") for s in ["train", "val"]},
            "test_opened": False,
            "meta_requested": with_meta,
        },
    )
    local = urlsplit(config.task.base_url).hostname in {"127.0.0.1", "localhost", "::1"}
    if local and not (
        os.environ.get(config.task.api_key_env) or credentials(root).get(config.task.api_key_env)
    ):
        os.environ[config.task.api_key_env] = "EMPTY"
    harness = MemoryHarness(
        training, read_json(data / "labels.json"), context_chars=config.context_chars
    )
    ledger = Ledger(
        output / "calls.jsonl",
        Limits(max_task_calls=2, max_meta_calls=int(with_meta), wall_seconds=600),
    )
    try:
        _, summary = evaluate_policy(
            output / "task",
            queries,
            policy,
            harness,
            task or TaskAgent(config.task, root),
            ledger,
            repeats=1,
            seed=config.seed,
            concurrency=1,
        )
        if summary["format_errors"] or summary["truncated"]:
            raise ValueError(
                "Task smoke replies violate the fixed label/output contract; inspect the local task records"
            )
        if with_meta:
            proposer = meta or ClaudeMeta(
                config.meta,
                root,
                system_prompt=memory_meta_system("mixed"),
                proposal_type=SingleMemoryBatch,
            )
            identifier = ledger.start("meta", purpose="explicit smoke connectivity")
            try:
                proposal, usage = proposer.propose(
                    policy,
                    {
                        "purpose": "Connectivity check; propose one general memory-policy candidate",
                        "validation_summary": summary,
                    },
                    output,
                )
            except Exception as exc:
                ledger.finish(
                    identifier, "meta", error_type=type(exc).__name__, usage_unconfirmed=True
                )
                raise
            ledger.finish(identifier, "meta", **usage)
            write_json(output / "proposal.json", proposal.model_dump())
        result = {
            "task_queries": 2,
            "application_calls": dict(ledger.used),
            "test_opened": False,
            "accuracy_is_not_a_research_result": True,
        }
        write_json(output / "summary.json", result)
        return result
    finally:
        ledger.close()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/default.json")
    parser.add_argument("--data", type=Path, default=root / "data/s2d")
    parser.add_argument("--output", type=Path, default=root / "runs/harness-smoke")
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Explicitly consume ONE Claude/Meta call in addition to the two Task calls",
    )
    args = parser.parse_args()
    result = smoke(
        MemoryConfig.model_validate_json(args.config.read_text()),
        args.data,
        args.output,
        root,
        with_meta=args.meta,
    )
    import json

    print(json.dumps(result))


if __name__ == "__main__":
    main()
