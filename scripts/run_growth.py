#!/usr/bin/env python3
"""Calibrate, search, then audit frozen prompts without returning holdouts to Meta."""

import argparse
import fcntl
import json
from pathlib import Path

from rsi_base.agent import TaskAgent
from rsi_base.benchmark import check_disjoint, load_problems
from rsi_base.loop import completed, evaluate, file_hash, run
from rsi_base.storage import Ledger, read_json, write_json
from rsi_base.types import AgentSpec, RunConfig, digest

ROOT = Path(__file__).resolve().parents[1]


def plot(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["curve"]
    x = [r["round"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, split, label in zip(
        axes, ["dev", "monitor"], ["Search set (48)", "Monitor set (32)"], strict=True
    ):
        if split == "dev":
            ax.plot(
                x,
                [
                    100 * r["candidate_dev"] if r["candidate_dev"] is not None else None
                    for r in rows
                ],
                "o--",
                alpha=0.6,
                label="Raw candidate",
            )
        ax.plot(x, [100 * r[f"selected_{split}"] for r in rows], "o-", label="Selected incumbent")
        ax.set(xlabel="Meta iteration", title=label, ylim=(0, 100), xticks=x)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Exact numeric accuracy (%)")
    final = report["final"]
    fig.suptitle(f"{report['task_model'].split('/')[-1]} fixed weights: prompt-only RSI")
    fig.text(
        0.5,
        0.01,
        f"Search selection is monotone by construction. Final sealed set (64): "
        f"{final['baseline']['correct']}/64 → {final['selected']['correct']}/64. "
        "Monitor and final never enter Meta feedback.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for extension in ["png", "svg"]:
        fig.savefig(destination / f"growth.{extension}", dpi=170)
    plt.close(fig)


def paired(baseline, selected):
    from collections import Counter

    import numpy as np

    a = {r["id"]: r["score"] for r in baseline}
    old_status = {r["id"]: r["status"] for r in baseline}
    differences = np.array([r["score"] - a[r["id"]] for r in selected])
    means = np.random.default_rng(42).choice(differences, (20000, len(differences))).mean(axis=1)
    return {
        "gained": int((differences > 0).sum()),
        "lost": int((differences < 0).sum()),
        "delta": float(differences.mean()),
        "gained_from_baseline_status": dict(
            Counter(old_status[r["id"]] for r in selected if r["score"] > a[r["id"]])
        ),
        "paired_bootstrap_95_percent_interval": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def audit(config, output, data):
    # Revalidate immutable search artifacts without making additional requests.
    search = run(
        config, data / "dev.jsonl", data / "monitor.jsonl", output / "search", ROOT, resume=True
    )
    directory = output / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config.model_dump(),
        "search_sha256": file_hash(output / "search/summary.json"),
        "script_sha256": file_hash(__file__),
        "split_manifest": read_json(data / "manifest.json"),
    }
    manifest = directory / "manifest.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Audit inputs changed")
    write_json(manifest, identity)
    monitor, final = load_problems(data / "monitor.jsonl"), load_problems(data / "final.jsonl")
    check_disjoint(load_problems(data / "dev.jsonl"), final)
    check_disjoint(monitor, final)
    ledger = Ledger(
        directory / "calls.jsonl", config.limits.model_copy(update={"max_task_calls": 500})
    )
    task = TaskAgent(config.task, ROOT)
    cache, curve = {}, []
    try:
        for index, state in enumerate(search["history"]):
            round_dir = output / "search/rounds" / f"{index:03}"
            completed(round_dir)
            selected_spec = digest(state["agent"])
            if index and not (round_dir / "proposal.json").exists():
                curve.append(
                    {
                        "round": index,
                        "decision": state["decision"],
                        "selected_round": state["selected_round"],
                        "candidate_dev": None,
                        "selected_dev": state["dev_accuracy"],
                        "candidate_monitor": None,
                        "selected_monitor": cache[selected_spec]["accuracy"],
                        "dev_summary": None,
                        "monitor_summary": cache[selected_spec],
                    }
                )
                print(json.dumps(curve[-1]), flush=True)
                continue
            spec = (
                config.agent
                if index == 0
                else AgentSpec.model_validate(
                    {"system_prompt": read_json(round_dir / "proposal.json")["system_prompt"]}
                )
            )
            key = digest(spec.model_dump())
            if selected_spec not in cache:
                _, cache[selected_spec] = evaluate(
                    directory / "monitor" / f"{index:03}",
                    monitor,
                    AgentSpec.model_validate(state["agent"]),
                    task,
                    ledger,
                    config.limits.task_concurrency,
                )
            dev_summary = read_json(round_dir / "summary.json")
            curve.append(
                {
                    "round": index,
                    "decision": state["decision"],
                    "selected_round": state["selected_round"],
                    "candidate_dev": dev_summary["accuracy"],
                    "selected_dev": state["dev_accuracy"],
                    "candidate_monitor": cache[key]["accuracy"] if key in cache else None,
                    "selected_monitor": cache[selected_spec]["accuracy"],
                    "dev_summary": dev_summary,
                    "monitor_summary": cache[selected_spec],
                }
            )
            print(json.dumps(curve[-1]), flush=True)
        # Both prompts are frozen by dev selection before either sees final questions.
        frozen = AgentSpec.model_validate(search["history"][-1]["agent"])
        baseline_rows, baseline_summary = evaluate(
            directory / "final/baseline",
            final,
            config.agent,
            task,
            ledger,
            config.limits.task_concurrency,
        )
        identical_prompt = frozen == config.agent
        if identical_prompt:
            selected_rows, selected_summary = baseline_rows, baseline_summary
        else:
            selected_rows, selected_summary = evaluate(
                directory / "final/selected",
                final,
                frozen,
                task,
                ledger,
                config.limits.task_concurrency,
            )
        report = {
            "protocol": "numeric MATH-500 research split; not official benchmark accuracy",
            "task_model": config.task.model,
            "weights_updated": False,
            "selection": "strict dev accuracy gain; no monitor/final feedback to Meta",
            "monitor_protocol": "Evaluate each unique selected incumbent once after search; cache unchanged prompts",
            "curve": curve,
            "selected_round": search["selected_round"],
            "final": {
                "identical_prompt": identical_prompt,
                "baseline": baseline_summary,
                "selected": selected_summary,
                "paired": paired(baseline_rows, selected_rows),
            },
            "search_application_calls": search["application_calls"],
            "audit_application_calls": ledger.used,
            "data_manifest": read_json(data / "manifest.json"),
        }
        write_json(directory / "report.json", report)
        plot(report, directory)
        print(json.dumps(report["final"], indent=2), flush=True)
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "search", "audit"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_math_growth.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/qwen-math-growth-8k")
    args = parser.parse_args()
    config = RunConfig.model_validate_json(args.config.read_text())
    data = ROOT / "data/math-growth"
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "calibrate":
            directory = args.output / "calibration"
            directory.mkdir(exist_ok=True)
            identity = {
                "task": config.task.model_dump(),
                "agent": config.agent.model_dump(),
                "data_sha256": file_hash(data / "calibration.jsonl"),
            }
            manifest = directory / "inputs.json"
            if manifest.exists() and read_json(manifest) != identity:
                raise ValueError("Calibration inputs changed; choose a new output directory")
            write_json(manifest, identity)
            ledger = Ledger(directory / "calls.jsonl", config.limits)
            try:
                _, summary = evaluate(
                    directory,
                    load_problems(data / "calibration.jsonl"),
                    config.agent,
                    TaskAgent(config.task, ROOT),
                    ledger,
                    config.limits.task_concurrency,
                )
                print(json.dumps(summary, indent=2))
            finally:
                ledger.close()
        elif args.stage == "search":
            print(
                json.dumps(
                    run(
                        config,
                        data / "dev.jsonl",
                        data / "monitor.jsonl",
                        args.output / "search",
                        ROOT,
                        resume=(args.output / "search/run.json").exists(),
                    ),
                    indent=2,
                )
            )
        else:
            audit(config, args.output, data)


if __name__ == "__main__":
    main()
