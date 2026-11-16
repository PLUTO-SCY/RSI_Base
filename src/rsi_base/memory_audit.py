"""Frozen held-out evaluation and paired, question-clustered uncertainty estimates."""

import json
import random
from functools import partial
from pathlib import Path

from .classification import MemoryPolicy
from .loop import file_hash
from .memory_loop import MemoryConfig, evaluate_policy, experiment, load_rows
from .storage import read_json, write_json
from .types import digest


def paired_records(baseline, selected):
    base = {row["id"]: row for row in baseline}
    child = {row["id"]: row for row in selected}
    if set(base) != set(child):
        raise ValueError("Paired evaluation IDs differ")
    differences = {}
    repeats = sorted({row["repeat"] for row in baseline})
    details = []
    for repeat in repeats:
        keys = [key for key, row in base.items() if row["repeat"] == repeat]
        gains = sum(child[key]["score"] > base[key]["score"] for key in keys)
        losses = sum(child[key]["score"] < base[key]["score"] for key in keys)
        details.append({"repeat": repeat, "gains": gains, "losses": losses, "net": gains - losses})
        for key in keys:
            identifier = base[key]["problem_id"]
            differences.setdefault(identifier, []).append(child[key]["score"] - base[key]["score"])
    return {
        "by_repeat": details,
        "mean_delta": sum(sum(v) for v in differences.values()) / len(base),
        "question_deltas": {k: sum(v) / len(v) for k, v in differences.items()},
    }


def cluster_interval(question_deltas, *, seed=42, samples=20000):
    values = list(question_deltas.values())
    if not values:
        raise ValueError("No paired questions")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples)
    )
    return {
        "mean": sum(values) / len(values),
        "low": estimates[int(0.025 * samples)],
        "high": estimates[int(0.975 * samples)],
        "samples": samples,
        "seed": seed,
        "unit": "Question, averaging repeats/search seeds before resampling; conditional on this fixed model and dataset",
    }


def evaluate_test(policy, *, directory, test, harness, task, ledger, config):
    policy = MemoryPolicy.model_validate(policy)
    key = digest(policy.model_dump())
    return evaluate_policy(
        directory / "test" / key[:16],
        test,
        policy,
        harness,
        task,
        ledger,
        repeats=config.validation_repeats,
        seed=config.seed,
    )


def finalize(run_directories, data, output, root):
    directories = [Path(path) for path in run_directories]
    output = Path(output)
    if not directories or any(not (p / "frozen.json").exists() for p in directories):
        raise ValueError("Every search run must be frozen before opening any test labels")
    identity = {
        "runs": {str(p.resolve()): file_hash(p / "frozen.json") for p in directories},
        "test_sha256": file_hash(Path(data) / "test.jsonl"),
        "selection_frozen": True,
        "test_repeats": "same predeclared count as validation",
        "comparison": "Selected vs strongest fixed validation baseline; random control separate",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final_protocol.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Final protocol changed after test access")
    write_json(path, identity)
    reports, combined, random_combined = [], {}, {}
    for directory in directories:
        frozen = read_json(directory / "frozen.json")
        config = MemoryConfig.model_validate(read_json(directory / "run.json")["config"])
        with experiment(config, data, directory, root) as (harness, _, _, _, ledger, task):
            # This is the first point in the runner that deserializes test questions/labels.
            test = load_rows(Path(data) / "test.jsonl")
            evaluate = partial(
                evaluate_test,
                directory=directory,
                test=test,
                harness=harness,
                task=task,
                ledger=ledger,
                config=config,
            )

            baseline_rows, baseline_summary = evaluate(frozen["baseline"]["policy"])
            selected_rows, selected_summary = evaluate(frozen["selected"]["policy"])
            calibration = read_json(directory / "calibration.json")
            strong_name = calibration.get("strongest_fixed_baseline", frozen["selected_baseline"])
            strong_rows, strong_summary = evaluate(calibration["baselines"][strong_name]["policy"])
            random_rows, random_summary = evaluate(frozen["random_control"]["policy"])
            no_memory_policy = read_json(directory / "calibration.json")["baselines"]["no_memory"][
                "policy"
            ]
            _, no_memory_summary = evaluate(no_memory_policy)
            curve = []
            for state in frozen["history"]:
                _, summary = evaluate(state["policy"])
                curve.append(
                    {
                        "round": state["round"],
                        "validation": state["summary"],
                        "test": summary,
                        "policy_sha256": digest(state["policy"]),
                    }
                )
            pair = paired_records(baseline_rows, selected_rows)
            random_pair = paired_records(baseline_rows, random_rows)
            for key, value in pair["question_deltas"].items():
                combined.setdefault(key, []).append(value)
            for key, value in random_pair["question_deltas"].items():
                random_combined.setdefault(key, []).append(value)
            report = {
                "seed": config.seed,
                "baseline_name": frozen["selected_baseline"],
                "baseline": baseline_summary,
                "selected": selected_summary,
                "strong_fixed_baseline_name": strong_name,
                "strong_fixed_baseline": strong_summary,
                "selected_vs_strong_fixed": paired_records(strong_rows, selected_rows),
                "random_control": random_summary,
                "no_memory": no_memory_summary,
                "paired": pair,
                "random_paired": random_pair,
                "curve": curve,
                "selected_policy": frozen["selected"]["policy"],
                "source": read_json(directory / "run.json"),
                "frozen_sha256": file_hash(directory / "frozen.json"),
                "application_calls": dict(ledger.used),
            }
            write_json(directory / "audit.json", report)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "test_baseline": baseline_summary["correct_by_repeat"],
                        "test_selected": selected_summary["correct_by_repeat"],
                        "test_random": random_summary["correct_by_repeat"],
                    }
                ),
                flush=True,
            )
    pooled = {key: sum(values) / len(values) for key, values in combined.items()}
    interval = cluster_interval(pooled)
    random_interval = cluster_interval(
        {key: sum(values) / len(values) for key, values in random_combined.items()}
    )
    result = {
        "runs": reports,
        "selection_frozen": True,
        "weights_updated": False,
        "paired_question_cluster_bootstrap": interval,
        "random_question_cluster_bootstrap": random_interval,
        "positive_on_every_search_seed": all(r["paired"]["mean_delta"] > 0 for r in reports),
        "stability_gate_passed": len(reports) >= 3
        and all(r["paired"]["mean_delta"] > 0 for r in reports)
        and interval["low"] > 0,
        "scope": "Stability on one public benchmark with fixed weights; not evidence of universal task or model generalization",
        "final_protocol": identity,
    }
    write_json(output / "report.json", result)
    return result
