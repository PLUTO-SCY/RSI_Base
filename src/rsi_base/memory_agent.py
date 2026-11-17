"""Tool-driven, source-visible harness search; development data is explicitly accessible."""

import json
import random
from pathlib import Path

from .classification import MemoryPolicy
from .improve import ClaudeMeta, MetaCallFailure
from .memory_agent_tools import DevelopmentTools, bridge
from .memory_loop import calibrate, evaluate_policy, experiment, rank_policy, stable_improvement
from .memory_search import random_policy
from .storage import read_json, write_json
from .types import digest

AGENT_SYSTEM = """You improve a fixed-weight task harness as a coding and debugging agent.
You have host MCP tools: list_artifacts, read_artifact, get_case, evaluate_candidate,
submit_candidates. Inspect actual source/classification.py and representative full
development failures first. All training memory, development queries/gold, parent
and previous candidate inputs/outputs and control scores are accessible. Files are
paged, not hidden; use get_case to inspect any complete trace. Do not assume field
semantics: allowed labels are ALWAYS shown; label_hints are noisy retrieval votes.
Propose, run small development probes, compare parent-child gains/losses, and debug
the candidate. You may change configuration AND write a Python retrieval function.
For retrieval='python', retrieval_code defines rank(query, memory), taking a query
string and a list of {id,question,answer} training rows. Return ranked training IDs,
or [{id:..., score:...}] in descending relevance. Only stdlib Python3.11 is present.
The host excludes the query's own/identical memory rows, checks IDs, applies count,
diversity, order/layout, a fixed label contract and fixed scoring. Your function runs
in a CPU/memory-bounded namespace with no host home, credentials, GPU or network.
Implement real reusable retrieval/reranking logic rather than only changing prompts.
Do not embed case queries, IDs, hashes, case-to-answer lookups or remembered answers.
General label/symptom distinctions learned from the development data are permitted.
Final test is absent. Never request it or try to access external host files.
The evidence supplies probe_budget and tool_call_budget. Use at most probe_budget
distinct probes (up to twice that many attempts to fix invalid/no-op proposals),
then choose up to TWO probed IDs for full development validation.
Use at most eight reading calls before the first probe; read more after observing
candidate errors. Inspect parent failures using get_case rather than arbitrary byte
offsets into the large records file. Do not spend the whole round reading files.
Compare different retrieval mechanisms instead of repeatedly keeping BM25 with
minor wording/count edits. Keep system_prompt under 1200 characters. Call
submit_candidates before your final message. Submit [] if no useful candidates.
Only host validation accepts a new incumbent; probe accuracy is debugging feedback.
"""


def optimization_batch(development, parent_records, config, number):
    rng = random.Random(config.seed + number * 1000)
    # Identical, label-stratified probe cases for Meta and random search in each round.
    groups = {}
    for row in development:
        groups.setdefault(row["answer"], []).append(row)
    labels = rng.sample(sorted(groups), min(len(groups), config.probe_queries))
    hard = [rng.choice(groups[label]) for label in labels]
    used = {row["id"] for row in hard}
    rest = [row for row in development if row["id"] not in used]
    return hard + rng.sample(rest, min(config.probe_queries - len(hard), len(rest)))


def agent_random_policy(rng):
    policy = random_policy(rng)
    if rng.random() >= 0.25:
        return policy
    word_weight, char_weight = rng.uniform(0.2, 2), rng.uniform(0.2, 2)
    code = f"""import re, math
from collections import Counter
def rank(query, memory):
    def words(text): return set(re.findall(r"[a-z]+", text.lower()))
    def grams(text):
        text = " ".join(sorted(words(text)))
        return set(text[i:i+3] for i in range(max(0,len(text)-2)))
    q, g = words(query), grams(query)
    ranked = []
    for row in memory:
        w, h = words(row["question"]), grams(row["question"])
        score = {word_weight!r} * len(q&w)/max(1,len(q|w)) + {char_weight!r} * len(g&h)/max(1,len(g|h))
        ranked.append({{"id":row["id"],"score":score}})
    return sorted(ranked,key=lambda r:-r["score"])
"""
    return policy.model_copy(update={"retrieval": "python", "retrieval_code": code})


def agent_search(config, data, output, root, *, meta=None):
    output = Path(output)
    if not (output / "calibration.json").exists():
        calibrate(config, data, output, root)
    with experiment(config, data, output, root) as (
        harness,
        training,
        development,
        _,
        ledger,
        task,
    ):
        if (output / "frozen.json").exists():
            return read_json(output / "frozen.json")
        calibration = read_json(output / "calibration.json")
        name = calibration["selected_baseline"]
        initial = calibration["baselines"][name]
        current = MemoryPolicy.model_validate(initial["policy"])
        summary, current_path = initial["summary"], f"baselines/{name}"
        history = [
            {
                "round": 0,
                "policy": current.model_dump(),
                "summary": summary,
                "evaluation": current_path,
            }
        ]
        tried = []
        validation_cache = {
            digest(item["policy"]): f"baselines/{key}"
            for key, item in calibration["baselines"].items()
        }

        def validate(policy, location):
            key = digest(policy.model_dump())
            relative = validation_cache.get(key, str(location.relative_to(output)))
            records, metrics = evaluate_policy(
                output / relative,
                development,
                policy,
                harness,
                task,
                ledger,
                repeats=config.validation_repeats,
                seed=config.seed,
            )
            validation_cache[key] = relative
            return records, metrics, relative

        proposer = meta or ClaudeMeta(
            config.meta, root, system_prompt=AGENT_SYSTEM, proposal_type=None
        )
        for number in range(1, config.rounds + 1):
            parent_records, _, _ = validate(current, output / "parent")
            probes = optimization_batch(development, parent_records, config, number)
            tools = DevelopmentTools(
                config,
                root,
                output,
                number,
                current,
                parent_records,
                training,
                development,
                probes,
                harness,
                task,
                ledger,
            )
            submitted = tools.round_dir / "submitted.json"
            if not submitted.exists():
                call_id = ledger.start("meta", round=number)
                evidence = {
                    "schema": MemoryPolicy.model_json_schema(),
                    "current_validation": summary,
                    "protocol": "All development information accessible; no final test. Four optimization probes then two validation candidates.",
                    "tools": [
                        "list_artifacts",
                        "read_artifact",
                        "get_case",
                        "evaluate_candidate",
                        "submit_candidates",
                    ],
                    "round": number,
                    "probe_budget": config.max_probes,
                    "tool_call_budget": config.max_tool_calls,
                    "previous_candidates": tried,
                }
                write_json(
                    tools.round_dir / "meta_input.json",
                    {"current": current.model_dump(), "evidence": evidence},
                )
                try:
                    with bridge(tools) as (mcp, environment, names):
                        _, metadata = proposer.propose(
                            current,
                            evidence,
                            tools.round_dir,
                            tool_config=mcp,
                            tool_names=names,
                            tool_env=environment,
                        )
                    if not submitted.exists():
                        raise ValueError("Meta must submit candidates through the host tool")
                    ledger.finish(call_id, "meta", **metadata)
                except (ValueError, MetaCallFailure) as exc:
                    write_json(
                        tools.round_dir / "invalid.json",
                        {
                            "error_type": type(exc).__name__,
                            "detail": str(exc)[:800],
                            "submitted_before_error": submitted.exists(),
                        },
                    )
                    ledger.finish(
                        call_id,
                        "meta",
                        error_type=type(exc).__name__,
                        usage_unconfirmed=True,
                        submitted_before_error=submitted.exists(),
                    )
                    if not submitted.exists():
                        write_json(submitted, {"candidate_ids": []})
            choices = read_json(submitted)["candidate_ids"]
            registry = read_json(tools.registry_path) if tools.registry_path.exists() else {}
            eligible = []
            for offset, identifier in enumerate(choices):
                item = registry[identifier]
                policy = MemoryPolicy.model_validate(item["policy"])
                _, metrics, location = validate(policy, tools.round_dir / f"candidate{offset + 1}")
                accepted = stable_improvement(metrics, summary)
                row = {
                    "id": f"round{number}-candidate{offset + 1}",
                    "tool_candidate_id": identifier,
                    "policy": policy.model_dump(),
                    "reason": item["reason"],
                    "summary": metrics,
                    "eligible": accepted,
                    "evaluation": location,
                }
                tried.append(row)
                registry[identifier]["validation"] = location
                if accepted:
                    eligible.append((identifier, policy, metrics, location))
            write_json(tools.registry_path, registry)
            if eligible:
                _, current, summary, current_path = max(
                    eligible, key=lambda row: rank_policy(row[:3])
                )
            state = {
                "round": number,
                "policy": current.model_dump(),
                "summary": summary,
                "evaluation": current_path,
                "decision": "accepted" if eligible else "retained_parent",
            }
            history.append(state)
            write_json(tools.round_dir / "decision.json", state)
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "round": number,
                        "validation": summary["correct_by_repeat"],
                        "decision": state["decision"],
                    }
                ),
                flush=True,
            )

        rng = random.Random(config.seed + 10000)
        random_current, random_summary, random_path = (
            MemoryPolicy.model_validate(initial["policy"]),
            initial["summary"],
            f"baselines/{name}",
        )
        controls = []
        for number in range(1, config.rounds + 1):
            records, _, _ = validate(random_current, output / "random_parent")
            probes = optimization_batch(development, records, config, number)
            pool = []
            for offset in range(config.max_probes):
                policy = agent_random_policy(rng)
                try:
                    _, probe_summary = evaluate_policy(
                        output / "random_probes" / f"{number:03}-{offset}",
                        probes,
                        policy,
                        harness,
                        task,
                        ledger,
                        repeats=1,
                        seed=config.seed,
                    )
                    pool.append((str(offset), policy, probe_summary))
                except ValueError as exc:
                    write_json(
                        output / "random_probes" / f"{number:03}-{offset}" / "invalid.json",
                        {"error": str(exc)[:800]},
                    )
            eligible = []
            for offset, (_, policy, _) in enumerate(
                sorted(pool, key=rank_policy, reverse=True)[:2]
            ):
                _, metrics, location = validate(
                    policy, output / "random" / f"{number:03}-{offset + 1}"
                )
                controls.append(
                    {"policy": policy.model_dump(), "summary": metrics, "evaluation": location}
                )
                if stable_improvement(metrics, random_summary):
                    eligible.append((str(offset), policy, metrics, location))
            if eligible:
                _, random_current, random_summary, random_path = max(
                    eligible, key=lambda row: rank_policy(row[:3])
                )
            print(
                json.dumps(
                    {
                        "seed": config.seed,
                        "random_round": number,
                        "validation": random_summary["correct_by_repeat"],
                    }
                ),
                flush=True,
            )
        frozen = {
            "seed": config.seed,
            "rounds": config.rounds,
            "proposal_strategy": "agent",
            "selected_baseline": name,
            "baseline": initial,
            "selected": {
                "policy": current.model_dump(),
                "summary": summary,
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
            "controls": "Same probe and full-development candidate caps; random code templates included but a narrower generator than unrestricted Meta code; actual Task calls and Meta usage recorded",
            "sandbox_completed_batches_this_process": harness.index.code_metrics,
            "feedback_scope": "All training and development cases/source/traces; final test absent",
        }
        write_json(output / "frozen.json", frozen)
        return frozen
