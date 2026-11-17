#!/usr/bin/env python3
"""Export source/tool visibility and physical call costs without dataset text."""

import argparse
import json
from collections import Counter
from pathlib import Path

from rsi_base.loop import file_hash
from rsi_base.storage import read_json, write_json
from rsi_base.types import digest


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(directory):
    frozen = read_json(directory / "frozen.json")
    config = read_json(directory / "run.json")["config"]
    initial_records = rows(directory / "baselines" / frozen["selected_baseline"] / "results.jsonl")
    calls = rows(directory / "calls.jsonl")
    starts = [r for r in calls if r["event"] == "started"]
    finishes = {r["call_id"]: r for r in calls if r["event"] == "finished"}
    tools = [
        row
        for path in sorted((directory / "evolution").glob("*/tool_calls.jsonl"))
        for row in rows(path)
    ]
    inspected, inspection_history = set(), []
    for path in sorted((directory / "evolution").glob("*/tool_calls.jsonl")):
        cases = {
            r["arguments"]["case_id"]
            for r in rows(path)
            if r["tool"] == "get_case"
            and not r.get("error")
            and r["arguments"]["candidate_id"] == "parent"
        }
        inspection_history.append(
            {
                "round": int(path.parent.name),
                "parent_cases": len(cases),
                "new_parent_cases": len(cases - inspected),
            }
        )
        inspected.update(cases)
    registry = (
        read_json(directory / "agent_registry.json")
        if (directory / "agent_registry.json").exists()
        else {}
    )
    code = [r for r in registry.values() if r["policy"]["retrieval"] == "python"]
    batches = frozen["sandbox_completed_batches_this_process"]
    task_usage = Counter()
    phase_calls, phase_usage = Counter(), {}
    for r in starts:
        if r["kind"] == "task":
            usage = {
                k: v
                for k, v in (finishes.get(r["call_id"], {}).get("usage") or {}).items()
                if isinstance(v, (int, float))
            }
            task_usage.update(usage)
            parts = Path(r.get("evaluation", "")).parts
            phase = (
                "development_selection_audit"
                if "probe_selection_audit" in parts
                else "historical_test_replay"
                if "test" in parts
                else "official_search"
            )
            phase_calls[phase] += 1
            phase_usage.setdefault(phase, Counter()).update(usage)
    meta_results = [
        read_json(p) for p in sorted((directory / "evolution").glob("*/meta_result.json"))
    ]
    configuration_checks = [
        finishes.get(r["call_id"], {}).get("global_configuration_unchanged")
        for r in starts
        if r["kind"] == "meta"
    ]
    return {
        "seed": config["seed"],
        "source_sha256": read_json(directory / "run.json")["source_sha256"],
        "initial_harness": frozen["selected_baseline"],
        "initial_evaluation_records": len(initial_records),
        "initial_trace_counts": sorted({len(r["traces"]) for r in initial_records}),
        "initial_records_with_retrieved_memory": sum(
            bool(r["memory_ids"]) for r in initial_records
        ),
        "physical_application_calls": dict(Counter(r["kind"] for r in starts)),
        "official_search_application_calls_at_freeze": frozen["application_calls_at_freeze"],
        "physical_task_calls_by_phase": dict(phase_calls),
        "physical_task_usage_by_phase": {k: dict(v) for k, v in phase_usage.items()},
        "wall_seconds_between_first_and_last_call": max(r["time"] for r in calls if "time" in r)
        - min(r["time"] for r in starts),
        "unsettled_calls": sum(r["call_id"] not in finishes for r in starts),
        "physical_task_usage": dict(task_usage),
        "meta_cli_turns": sum(r.get("num_turns", 0) for r in meta_results),
        "meta_cli_reported_usage": [r.get("usage") for r in meta_results],
        "meta_cli_estimated_cost_usd_unverified": [r.get("total_cost_usd") for r in meta_results],
        "meta_cli_errors": sum(bool(r.get("is_error")) for r in meta_results),
        "provider_bill_verified": False,
        "global_configuration_unchanged": False
        if any(v is False for v in configuration_checks)
        else True
        if all(v is True for v in configuration_checks)
        else None,
        "meta_calls_with_confirmed_configuration": sum(v is True for v in configuration_checks),
        "meta_calls_with_unconfirmed_configuration": sum(v is None for v in configuration_checks),
        "tool_calls": dict(Counter(r["tool"] for r in tools)),
        "tool_errors": dict(Counter(r["error"] for r in tools if r.get("error"))),
        "duplicate_execution_probes_skipped": sum(
            bool(r.get("result", {}).get("duplicate_execution")) for r in tools
        ),
        "successful_probes_zero_new_task_calls": sum(
            r["tool"] == "evaluate_candidate"
            and r.get("result", {}).get("candidate_id") is not None
            and r["result"].get("new_task_calls") == 0
            for r in tools
        ),
        "read_artifacts": sorted(
            {
                r["arguments"]["artifact"]
                for r in tools
                if r["tool"] == "read_artifact" and not r.get("error")
            }
        ),
        "parent_cases_inspected": len(
            {
                r["arguments"]["case_id"]
                for r in tools
                if r["tool"] == "get_case"
                and r["arguments"]["candidate_id"] == "parent"
                and not r.get("error")
            }
        ),
        "parent_case_inspection_by_round": inspection_history,
        "candidate_cases_inspected": len(
            {
                (r["arguments"]["candidate_id"], r["arguments"]["case_id"])
                for r in tools
                if r["tool"] == "get_case"
                and r["arguments"]["candidate_id"] != "parent"
                and not r.get("error")
            }
        ),
        "successful_meta_probes": len(registry),
        "successful_python_probes": len(code),
        "unique_python_code_versions": len({digest(r["policy"]["retrieval_code"]) for r in code}),
        "full_validation_meta_candidates": len(frozen["proposals"]),
        "full_validation_random_candidates": len(frozen["random_control"]["candidates"]),
        "accepted_rounds": [
            r["round"] for r in frozen["history"][1:] if r["decision"] == "accepted"
        ],
        "development_correct_by_repeat": [
            r["summary"]["correct_by_repeat"] for r in frozen["history"]
        ],
        "random_development_correct_by_repeat": frozen["random_control"]["summary"][
            "correct_by_repeat"
        ],
        "sandbox_completed_batches_this_process": len(batches),
        "sandbox_cost_scope": "Completed batches on the persistent main retrieval index in this process; excludes failed/interrupted batches, previous process segments, and ephemeral confirm/challenge indexes",
        "sandbox_completed_worker_cpu_seconds_this_process": sum(
            r["worker_cpu_seconds"] for r in batches
        ),
        "sandbox_completed_wall_seconds_this_process": sum(r["wall_seconds"] for r in batches),
        "sandbox_peak_rss_kib_this_process": max(
            (r["worker_peak_rss_kib"] for r in batches), default=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "audit_script_sha256": file_hash(__file__),
        "runs": [audit(args.suite / f"seed{seed}") for seed in (42, 123, 2026)],
        "information_scope": "Full source/training/development available; availability does not imply every artifact was actually read",
        "configuration_confirmation_scope": "A null global_configuration_unchanged value indicates absent confirmation metadata for one or more Meta calls; see per-call confirmation counts",
        "test_status": "Historical replay, not fresh independent confirmation",
        "cost_scope": "Task usage counts physical requests. Meta CLI usage is not a verified provider bill. Sandbox completed batches exclude failures/interruptions and earlier process segments.",
        "control_scope": "Equal probe/full-validation caps; random code templates cover a narrower generator; actual costs differ",
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "seeds": [r["seed"] for r in result["runs"]]}))


if __name__ == "__main__":
    main()
