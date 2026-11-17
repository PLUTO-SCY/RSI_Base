#!/usr/bin/env python3
"""Publish aggregate curves, frozen policies and provenance; keep task traces local."""

import argparse
import json
import subprocess
from pathlib import Path

from rsi_base.loop import file_hash
from rsi_base.memory_audit import cluster_interval
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def plot(report, destination, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for index, row in enumerate(report["runs"]):
        color = f"C{index}"
        curve = row["curve"]
        x = [r["round"] for r in curve]
        for ax, split in zip(axes[:2], ["validation", "test"], strict=True):
            ax.plot(
                x,
                [100 * r[split]["accuracy"] for r in curve],
                "o-",
                color=color,
                label=f"Search {row['seed']}",
            )
        axes[2].plot(
            [0, 1],
            [100 * row["baseline"]["accuracy"], 100 * row["selected"]["accuracy"]],
            "o-",
            color=color,
            label=f"Search {row['seed']}",
        )
        axes[2].scatter(1.12, 100 * row["random_control"]["accuracy"], marker="x", color=color)
        if "strong_fixed_baseline" in row:
            axes[2].scatter(
                1.24,
                100 * row["strong_fixed_baseline"]["accuracy"],
                marker="D",
                facecolors="none",
                edgecolors=color,
            )
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_ylabel("Exact-label accuracy (%)")
    validation_size = report["runs"][0]["curve"][0]["validation"]["examples"]
    test_size = report["runs"][0]["baseline"]["examples"]
    axes[0].set_title(f"Search validation ({validation_size} queries × 3)")
    axes[1].set_title(f"Frozen test replay ({test_size} queries × 3)")
    axes[2].set_title("Final test; × random, ◇ strongest fixed")
    axes[0].set_xlabel("Evolution iteration")
    axes[1].set_xlabel("Evolution iteration")
    axes[2].set_xticks([0, 1], ["Seed harness", "Evolved harness"])
    fig.suptitle("S2D memory-harness evolution — Qwen3-4B, weights fixed")
    interval = report["paired_question_cluster_bootstrap"]
    fig.text(
        0.5,
        0.015,
        f"All searches frozen before test. Paired Δ = {100 * interval['mean']:+.2f} pp; question-cluster 95% CI [{100 * interval['low']:+.2f}, {100 * interval['high']:+.2f}] pp.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    for extension in ["png", "svg"]:
        target = destination / f"{name}.{extension}"
        fig.savefig(target, dpi=180)
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "runs/memory-s2d-starter-v1")
    parser.add_argument("--data", type=Path, default=root / "data/memory-s2d-expanded")
    parser.add_argument(
        "--runtime", type=Path, help="Runtime JSON captured for this experiment's Task service"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "docs/validation")
    parser.add_argument(
        "--tooling-commit", help="Commit containing the export and scheduling scripts"
    )
    parser.add_argument(
        "--name",
        choices=["memory-evolution", "memory-evolution-expanded"],
        default="memory-evolution",
    )
    args = parser.parse_args()
    report = read_json(args.suite / "report.json")
    for row in report["runs"]:
        if "strong_fixed_baseline" not in row:
            if row["source"]["config"].get("initial_baseline") is not None:
                raise ValueError("Explicit seed report requires its own strong-control audit")
            # Older strong-start audits already use the strongest fixed policy as baseline.
            row["strong_fixed_baseline_name"] = row["baseline_name"]
            row["strong_fixed_baseline"] = row["baseline"]
            row["selected_vs_strong_fixed"] = row["paired"]
    report["suite_protocol"] = read_json(args.suite / "suite.json")
    report["comparison_definition"] = {
        "primary": "Selected vs configured initial baseline, identified by each run's baseline_name",
        "strong_fixed": "Selected vs strongest fixed validation baseline, reported separately",
        "legacy_protocol_label": "The sealed final_protocol.comparison text predates explicit starter selection; actual baseline policies and paired records determine the comparison",
    }
    # Descriptive control comparisons; these do not alter the predeclared acceptance gate.
    for comparison in ["strong_fixed", "random"]:
        combined = {}
        for row in report["runs"]:
            selected = row["paired"]["question_deltas"]
            deltas = (
                row["selected_vs_strong_fixed"]["question_deltas"]
                if comparison == "strong_fixed"
                else {
                    key: value - row["random_paired"]["question_deltas"][key]
                    for key, value in selected.items()
                }
            )
            for key, value in deltas.items():
                combined.setdefault(key, []).append(value)
        report[f"selected_vs_{comparison}_question_cluster_bootstrap"] = cluster_interval(
            {key: sum(values) / len(values) for key, values in combined.items()}
        )
    report["data_manifest"] = read_json(args.data / "manifest.json")
    runtime_path = args.runtime or root / ".local/task-runtime-memory.json"
    report["task_runtime"] = read_json(runtime_path)
    if any(
        report["task_runtime"]["model"] != row["source"]["config"]["task"]["model"]
        for row in report["runs"]
    ):
        raise ValueError("Runtime fingerprint belongs to a different Task model")
    runtimes = {
        row["seed"]: runtime_path.parent / f"task-runtime-expanded-{row['seed']}.json"
        for row in report["runs"]
    }
    if all(path.exists() for path in runtimes.values()):
        report["task_runtime_by_seed"] = {seed: read_json(path) for seed, path in runtimes.items()}
        if any(
            r["files"] != report["task_runtime"]["files"]
            for r in report["task_runtime_by_seed"].values()
        ):
            raise ValueError("Worker checkpoints differ")
    report["export_script_sha256"] = file_hash(__file__)
    report["evaluation_scheduler_sha256"] = file_hash(root / "scripts/run_memory.py")
    if (args.suite / "audit_scheduler_transition.json").exists():
        report["evaluation_scheduler_transition"] = read_json(
            args.suite / "audit_scheduler_transition.json"
        )
    source = {
        str(p.relative_to(root / "src")): file_hash(p)
        for p in (root / "src/rsi_base").rglob("*.py")
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report["verified_source_git_commit"] = commit
    tooling_commit = args.tooling_commit or commit
    report["verified_tooling_git_commit"] = subprocess.check_output(
        ["git", "rev-parse", tooling_commit], cwd=root, text=True
    ).strip()
    for path in ["scripts/export_memory.py", "scripts/run_memory.py"]:
        committed_script = subprocess.check_output(
            ["git", "show", f"{tooling_commit}:{path}"], cwd=root
        )
        if committed_script != (root / path).read_bytes():
            raise ValueError(f"Uncommitted tooling: {path}")
    # Verify current worktree and committed package bytes match the source used in each run.
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", "src/rsi_base"], cwd=root, text=True
    ).splitlines()
    import hashlib

    committed = {
        str(Path(p).relative_to("src")): hashlib.sha256(
            subprocess.check_output(["git", "show", f"{commit}:{p}"], cwd=root)
        ).hexdigest()
        for p in paths
        if p.endswith(".py")
    }
    if digest(source) != digest(committed):
        raise ValueError("Current core differs from committed core")
    histories = []
    for row in report["runs"]:
        if row["source"]["source_sha256"] != digest(source):
            raise ValueError("Experiment source differs from current source")
        directory = args.suite / f"seed{row['seed']}"
        frozen = read_json(directory / "frozen.json")
        histories.append(frozen)
        if (directory / "baseline_import.json").exists():
            row["imported_controls"] = read_json(directory / "baseline_import.json")
    report["frozen_searches"] = histories
    if (args.suite / "ablations.json").exists():
        report["diagnostic_ablations"] = read_json(args.suite / "ablations.json")
    if (args.suite / "resume_verification.json").exists():
        report["resume_verification"] = read_json(args.suite / "resume_verification.json")
    if (args.suite / "confirmation.json").exists():
        report["additional_confirmation"] = read_json(args.suite / "confirmation.json")
    # Exact dataset text must not appear in any publicly exported policy/rationale.
    texts = [
        json.loads(line)["question"]
        for split in ["train", "val"]
        for line in (args.data / f"{split}.jsonl").read_text().splitlines()
    ]
    serialized = " ".join(json.dumps(report, ensure_ascii=False).split()).casefold()
    if any(" ".join(text.split()).casefold() in serialized for text in texts):
        raise ValueError("Dataset query in public output; keep raw evidence private")
    destination = args.output_dir
    write_json(destination / f"{args.name}.json", report)
    plot(report, destination, args.name)
    print(
        json.dumps(
            {
                "stability_gate_passed": report["stability_gate_passed"],
                "paired_delta": report["paired_question_cluster_bootstrap"],
            }
        )
    )


if __name__ == "__main__":
    main()
