"""A serial, single-child improvement loop with file checkpoints."""

import fcntl
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import TaskAgent
from .benchmark import check_disjoint, load_problems, score
from .improve import ClaudeMeta
from .math_scoring import math_score
from .storage import Ledger, append_json, read_json, read_jsonl, write_json
from .types import AgentSpec, Proposal, digest


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(directory, state):
    hashes = {
        p.name: file_hash(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "complete.json"
    }
    write_json(directory / "complete.json", {"files": hashes, "state": state})


def completed(directory):
    marker = directory / "complete.json"
    if not marker.exists():
        return None
    value = read_json(marker)
    for name, expected in value["files"].items():
        if Path(name).name != name or not (directory / name).is_file():
            raise ValueError("Checkpoint artifact missing")
        if file_hash(directory / name) != expected:
            raise ValueError("Checkpoint artifact checksum mismatch")
    return value["state"]


def evaluate(
    directory, problems, spec, task, ledger, concurrency=1, scorer=score, *, expected_model=None
):
    """Settle every admitted call; resume successes even across an interrupted batch."""
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "agent": spec.model_dump(),
        "tasks": digest(problems),
        "scorer": f"{scorer.__module__}.{scorer.__name__}",
    }
    manifest = directory / "evaluation.json"
    if manifest.exists():
        if read_json(manifest) != identity:
            raise ValueError("Evaluation inputs changed")
    else:
        write_json(manifest, identity)
    path = directory / "results.jsonl"
    records = read_jsonl(path) if path.exists() else []
    expected_ids = [x["id"] for x in problems]
    ids = [x["id"] for x in records]
    if len(set(ids)) != len(ids) or not set(ids).issubset(expected_ids):
        raise ValueError("Evaluation checkpoint contains duplicate or unknown IDs")
    if expected_model and any(row.get("response_model") != expected_model for row in records):
        raise ValueError(
            f"Evaluation checkpoint includes responses from a different model: {expected_model}"
        )
    done = {row["id"]: row for row in records}
    pending = [p for p in problems if p["id"] not in done]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for offset in range(0, len(pending), concurrency):
            jobs, first_error = [], None
            for problem in pending[offset : offset + concurrency]:
                try:
                    call_id = ledger.start("task", task_id=problem["id"], evaluation=str(directory))
                except Exception as exc:
                    first_error = exc
                    break
                jobs.append((problem, call_id, pool.submit(task.run, spec, problem["question"])))
            for problem, call_id, future in jobs:
                try:
                    result = future.result()
                    if expected_model and result.get("response_model") != expected_model:
                        raise ValueError(
                            f"Task endpoint did not serve requested model: {expected_model}"
                        )
                except Exception as exc:
                    ledger.finish(
                        call_id,
                        "task",
                        error_type=type(exc).__name__,
                        status=getattr(exc, "status_code", None),
                        usage_unconfirmed=True,
                    )
                    first_error = first_error or exc
                    continue
                row = {
                    "id": problem["id"],
                    "question": problem["question"],
                    **result,
                    **scorer(result["response"], problem["answer"]),
                }
                append_json(path, row)
                done[row["id"]] = row
                ledger.finish(
                    call_id,
                    "task",
                    usage=result.get("usage"),
                    elapsed_seconds=result.get("elapsed_seconds"),
                )
            if first_error:
                raise first_error
    records = [done[identifier] for identifier in expected_ids]
    summary = {
        "assigned": len(problems),
        "completed": len(records),
        "correct": sum(r["score"] for r in records),
        "accuracy": sum(r["score"] for r in records) / len(problems),
        "format_errors": sum(r["status"] == "format_error" for r in records),
        "answer_conflicts": sum(r["status"] == "conflicting_answers" for r in records),
        "strict_format_compliant": sum(
            r.get("strict_format_compliant", r["status"] != "format_error") for r in records
        ),
        "truncated": sum(r.get("finish_reason") == "length" for r in records),
        "output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records),
    }
    write_json(directory / "summary.json", summary)
    return records, summary


def build_evidence(parent_records, problems, history, rounds_dir):
    """Prioritize mistakes rather than the first dataset rows; retain rejected attempts."""
    references = {p["id"]: p["answer"] for p in problems}

    def examples(rows, failure_limit=8, success_limit=2):
        failures = [r for r in rows if not r["score"]]
        successes = [r for r in rows if r["score"]]
        selected = failures[:failure_limit] + successes[:success_limit]
        return [
            {
                "id": r["id"],
                "question": r["question"],
                "response": r["response"]
                if len(r["response"]) <= 5000
                else r["response"][:2800] + "\n[... middle omitted ...]\n" + r["response"][-2200:],
                "score": r["score"],
                "status": r["status"],
                "reference_answer": references[r["id"]],
                "finish_reason": r.get("finish_reason"),
                "parsed_answer": r.get("parsed_answer"),
            }
            for r in selected
        ]

    feedback = []
    for directory in sorted(rounds_dir.iterdir()):
        if (directory / "proposal.json").exists() and (directory / "decision.json").exists():
            decision = read_json(directory / "decision.json")
            rows_path = directory / "results.jsonl"
            rows = read_jsonl(rows_path) if rows_path.exists() else []
            feedback.append(
                {
                    "round": directory.name,
                    "proposal": read_json(directory / "proposal.json"),
                    "decision": decision["decision"],
                    "candidate_accuracy": decision.get("candidate_accuracy"),
                    "examples": examples(rows, 4, 1),
                }
            )
    return {
        "examples": examples(parent_records),
        "history": history,
        "previous_candidates": feedback[-3:],
        "development_reference_answers_provided": True,
    }


def run(
    config, dev_path, test_path, output, root, *, resume=False, baseline=False, task=None, meta=None
):
    output, root = Path(output).resolve(), Path(root).resolve()
    dev, test = load_problems(dev_path), load_problems(test_path)
    check_disjoint(dev, test)
    rounds = 0 if baseline else config.limits.rounds
    if config.limits.max_task_calls < (rounds + 1) * len(dev) + len(test):
        raise ValueError("Task call budget must cover all candidate and final evaluations")
    if config.limits.max_meta_calls < rounds:
        raise ValueError("Meta invocation budget is smaller than the round count")
    manifest = {
        "implementation_sha256": digest(
            {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        ),
        "dependency_lock_sha256": file_hash(root / "uv.lock")
        if (root / "uv.lock").is_file()
        else None,
        "config": config.model_dump(),
        "dev_sha256": file_hash(dev_path),
        "test_sha256": file_hash(test_path),
        "rounds": rounds,
        "scope": "harness-only; weights fixed",
        "selection": "strict development accuracy gain",
        "task_input_excludes_reference": True,
    }
    exists = output.exists() and any(output.iterdir())
    if exists and not resume:
        raise ValueError("Output exists; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process holds this run's lock") from exc
        if (output / "run.json").exists():
            if read_json(output / "run.json") != manifest:
                raise ValueError("Resume requires identical config and data hashes")
        elif exists:
            raise ValueError("Existing directory has no valid run manifest")
        else:
            write_json(output / "run.json", manifest)
        ledger = Ledger(output / "calls.jsonl", config.limits)
        ledger.reserved_task_calls = len(test)
        task = task or TaskAgent(config.task, root)
        meta = meta or ClaudeMeta(config.meta, root)
        try:
            return _run(config, dev, test, output, ledger, task, meta, rounds)
        finally:
            ledger.close()


def _run(config, dev, test, output, ledger, task, meta, rounds):
    scorer = math_score if config.evaluator == "math_numeric" else score
    current = config.agent
    parent_dir = output / "rounds" / "000"
    parent_dir.mkdir(parents=True, exist_ok=True)
    state = completed(parent_dir)
    if state is None:
        records, summary = evaluate(
            parent_dir, dev, current, task, ledger, config.limits.task_concurrency, scorer
        )
        state = {
            "agent": current.model_dump(),
            "selected_round": 0,
            "dev_accuracy": summary["accuracy"],
            "decision": "baseline",
        }
        seal(parent_dir, state)
    history = [state]
    for index in range(1, rounds + 1):
        directory = output / "rounds" / f"{index:03}"
        directory.mkdir(parents=True, exist_ok=True)
        cached = completed(directory)
        if cached is not None:
            state = cached
            history.append(state)
            continue
        current = AgentSpec.model_validate(state["agent"])
        parent_dir = output / "rounds" / f"{state['selected_round']:03}"
        parent_records = read_jsonl(parent_dir / "results.jsonl")
        evidence = build_evidence(parent_records, dev, history, output / "rounds")
        evidence["fixed_task_instructions"] = (
            config.task.user_suffix or "Preserve the parent's answer-format contract."
        )
        proposal_path = directory / "proposal.json"
        invalid_path = directory / "invalid_proposal.json"
        if not proposal_path.exists() and not invalid_path.exists():
            call_id = ledger.start("meta", round=index)
            write_json(
                directory / "meta_input.json",
                {"current": current.model_dump(), "development_evidence": evidence},
            )
            try:
                proposal, metadata = meta.propose(current, evidence, directory)
                proposal = Proposal.model_validate(proposal.model_dump())
                if len(proposal.system_prompt) > config.meta.prompt_char_budget:
                    raise ValueError("Candidate exceeds the configured prompt character budget")
                write_json(proposal_path, proposal.model_dump())
                ledger.finish(call_id, "meta", **metadata)
            except ValueError as exc:
                # Invalid edits are rejected, not applied or silently repaired.
                write_json(invalid_path, {"error_type": type(exc).__name__})
                ledger.finish(call_id, "meta", error_type=type(exc).__name__)
            except Exception as exc:
                ledger.finish(
                    call_id, "meta", error_type=type(exc).__name__, usage_unconfirmed=True
                )
                raise
        next_state = {**state, "round": index}
        if invalid_path.exists():
            next_state["decision"] = "rejected_invalid_proposal"
        else:
            proposal = Proposal.model_validate(read_json(proposal_path))
            child = AgentSpec(system_prompt=proposal.system_prompt)
            _records, summary = evaluate(
                directory, dev, child, task, ledger, config.limits.task_concurrency, scorer
            )
            next_state["candidate_accuracy"] = summary["accuracy"]
            if summary["accuracy"] > state["dev_accuracy"]:
                next_state.update(
                    agent=child.model_dump(),
                    selected_round=index,
                    dev_accuracy=summary["accuracy"],
                    decision="accepted",
                )
            else:
                next_state["decision"] = "retained_parent"
        write_json(directory / "decision.json", next_state)
        seal(directory, next_state)
        state = next_state
        history.append(state)
    # The selected prompt is frozen before final questions enter a task request.
    frozen = AgentSpec.model_validate(state["agent"])
    final_dir = output / "final"
    ledger.reserved_task_calls = 0
    final_dir.mkdir(parents=True, exist_ok=True)
    final_state = completed(final_dir)
    if final_state is None:
        _, final_summary = evaluate(
            final_dir, test, frozen, task, ledger, config.limits.task_concurrency, scorer
        )
        final_state = {"agent": frozen.model_dump(), "metrics": final_summary}
        seal(final_dir, final_state)
    result = {
        "rounds": rounds,
        "selected_round": state["selected_round"],
        "development_accuracy": state["dev_accuracy"],
        "held_out": final_state["metrics"],
        "application_calls": ledger.used,
        "history": history,
        "weights_updated": False,
        "model": config.task.model,
        "meta_model": config.meta.model,
        "evaluator": config.evaluator,
    }
    write_json(output / "summary.json", result)
    return result
