"""Meta-guided and random-control memory evolution, with a permanent search freeze."""

import json
import random
from pathlib import Path

from .classification import MemoryBatch, MemoryPolicy, normalize
from .improve import ClaudeMeta
from .memory_loop import (
    META_MEMORY,
    calibrate,
    evaluate_policy,
    experiment,
    rank_policy,
    stable_improvement,
)
from .storage import read_json, write_json
from .types import digest


def random_policy(rng):
    return MemoryPolicy(
        retrieval=rng.choice(["bm25", "tfidf", "hybrid", "fixed"]),
        examples=rng.choice([4, 8, 16, 24, 32]),
        selection=rng.choice(["top", "diverse"]),
        max_per_label=rng.choice([1, 2, 4, 32]),
        label_hints=rng.choice([False, True]),
        order=rng.choice(["best_first", "best_last"]),
        workflow=rng.choice(["direct", "verify", "rewrite"]),
    )


def search(config, data, output, root, *, meta=None):
    output = Path(output)
    if not (output / "calibration.json").exists():
        calibrate(config, data, output, root)
    with experiment(config, data, output, root) as (
        harness,
        training,
        validation,
        labels,
        ledger,
        task,
    ):
        if (output / "frozen.json").exists():
            return read_json(output / "frozen.json")
        calibration = read_json(output / "calibration.json")
        seed_name = calibration["selected_baseline"]
        seed = calibration["baselines"][seed_name]
        initial_policy = MemoryPolicy.model_validate(seed["policy"])
        current, current_summary = initial_policy, seed["summary"]
        current_path = f"baselines/{seed_name}"
        history = [
            {
                "round": 0,
                "policy": current.model_dump(),
                "summary": current_summary,
                "evaluation": current_path,
            }
        ]
        tried = []
        by_policy = {
            digest(row["policy"]): f"baselines/{name}"
            for name, row in calibration["baselines"].items()
        }
        diagnostics = random.Random(config.seed).sample(
            training, min(len(training), config.diagnostic_queries)
        )
        meta = meta or ClaudeMeta(
            config.meta, root, system_prompt=META_MEMORY, proposal_type=MemoryBatch
        )
        rng = random.Random(config.seed + 10000)

        def evaluate(policy, directory):
            key = digest(policy.model_dump())
            if key in by_policy:
                location = by_policy[key]
            else:
                location = str(directory.relative_to(output))
            _, summary = evaluate_policy(
                output / location,
                validation,
                policy,
                harness,
                task,
                ledger,
                repeats=config.validation_repeats,
                seed=config.seed,
            )
            by_policy[key] = location
            return summary, location

        for index in range(1, config.rounds + 1):
            directory = output / "evolution" / f"{index:03}"
            directory.mkdir(parents=True, exist_ok=True)
            diag_dir = output / "diagnostics" / digest(current.model_dump())[:16]
            records, diagnostic_summary = evaluate_policy(
                diag_dir,
                diagnostics,
                current,
                harness,
                task,
                ledger,
                repeats=1,
                seed=config.seed,
                leave_one_out=True,
            )
            references = {r["id"]: r for r in diagnostics}
            failures = [r for r in records if not r["score"]][:8]
            successes = [r for r in records if r["score"]][:2]
            evidence = {
                "proposal_schema": MemoryBatch.model_json_schema(),
                "training_diagnostics": [
                    {
                        "question": references[r["problem_id"]]["question"],
                        "reference": references[r["problem_id"]]["answer"],
                        "response": r["response"],
                        "memory": [
                            {"id": m["id"], "question": m["question"], "label": m["answer"]}
                            for m in training
                            if m["id"] in r["memory_ids"]
                        ],
                        "score": r["score"],
                    }
                    for r in failures + successes
                ],
                "diagnostic_summary": diagnostic_summary,
                "validation_incumbent": current_summary,
                "baseline_summaries": {
                    k: v["summary"] for k, v in calibration["baselines"].items()
                },
                "previous_candidates": tried,
                "labels": labels,
                "fixed_constraints": {
                    "context_chars": config.context_chars,
                    "output": "one exact allowed label; no tags or explanation",
                    "max_calls_per_query": 2,
                    "max_output_tokens_per_call": config.task.max_tokens,
                },
                "selection": "Each repetition must not regress; average gain must be >= one validation example. Best valid improvement is retained.",
                "validation_examples_exposed": False,
                "test_opened": False,
            }
            path = directory / "proposal.json"
            invalid = directory / "invalid.json"
            if not path.exists() and not invalid.exists():
                identifier = ledger.start("meta", round=index)
                write_json(
                    directory / "meta_input.json",
                    {"current": current.model_dump(), "evidence": evidence},
                )
                try:
                    proposal, metadata = meta.propose(current, evidence, directory)
                    proposal = MemoryBatch.model_validate(proposal.model_dump())
                    for item in proposal.candidates:
                        text = normalize(item.policy.system_prompt)
                        if any(normalize(row["question"]) in text for row in training + validation):
                            raise ValueError("Candidate embeds a dataset query")
                        if any(row["id"] in text for row in training + validation):
                            raise ValueError("Candidate embeds dataset identifiers")
                    write_json(path, proposal.model_dump())
                    ledger.finish(identifier, "meta", **metadata)
                except ValueError as exc:
                    write_json(invalid, {"error_type": type(exc).__name__})
                    ledger.finish(identifier, "meta", error_type=type(exc).__name__)
                except Exception as exc:
                    ledger.finish(
                        identifier, "meta", error_type=type(exc).__name__, usage_unconfirmed=True
                    )
                    raise
            eligible = []
            if path.exists():
                proposal = MemoryBatch.model_validate(read_json(path))
                for offset, candidate in enumerate(proposal.candidates):
                    name = f"round{index}-candidate{offset + 1}"
                    summary, location = evaluate(
                        candidate.policy, directory / f"candidate{offset + 1}"
                    )
                    accepted = stable_improvement(summary, current_summary)
                    row = {
                        "id": name,
                        **candidate.model_dump(),
                        "summary": summary,
                        "eligible": accepted,
                        "evaluation": location,
                    }
                    tried.append(row)
                    if accepted:
                        eligible.append((name, candidate.policy, summary, location))
            if eligible:
                _, current, current_summary, current_path = max(
                    eligible, key=lambda r: rank_policy(r[:3])
                )
            state = {
                "round": index,
                "policy": current.model_dump(),
                "summary": current_summary,
                "evaluation": current_path,
                "decision": "accepted" if eligible else "retained_parent",
            }
            history.append(state)
            write_json(directory / "decision.json", state)
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "round": index,
                        "validation": current_summary["correct_by_repeat"],
                        "decision": state["decision"],
                    }
                ),
                flush=True,
            )

        random_current, random_summary = initial_policy, seed["summary"]
        random_path, controls = f"baselines/{seed_name}", []
        for index in range(config.rounds):
            eligible = []
            for offset in range(2):
                policy = random_policy(rng)
                summary, location = evaluate(
                    policy, output / "random" / f"{index + 1:03}-{offset + 1}"
                )
                controls.append(
                    {"policy": policy.model_dump(), "summary": summary, "evaluation": location}
                )
                if stable_improvement(summary, random_summary):
                    eligible.append((str(offset), policy, summary, location))
            if eligible:
                _, random_current, random_summary, random_path = max(
                    eligible, key=lambda r: rank_policy(r[:3])
                )
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "random_round": index + 1,
                        "validation": random_summary["correct_by_repeat"],
                    }
                ),
                flush=True,
            )
        report = {
            "seed": config.seed,
            "rounds": config.rounds,
            "selected_baseline": seed_name,
            "baseline": seed,
            "selected": {
                "policy": current.model_dump(),
                "summary": current_summary,
                "evaluation": current_path,
            },
            "history": history,
            "proposals": tried,
            "random_control": {
                "policy": random_current.model_dump(),
                "summary": random_summary,
                "evaluation": random_path,
                "candidates": controls,
            },
            "application_calls_at_freeze": dict(ledger.used),
            "test_opened": False,
            "weights_updated": False,
            "controls": "Equal candidate count, validation repetitions and per-query call/token caps; actual calls and tokens are reported separately",
        }
        write_json(output / "frozen.json", report)
        return report
