"""A serial, single-child improvement loop with file checkpoints."""

import fcntl
import hashlib
import json
from pathlib import Path

from .agent import TaskAgent
from .benchmark import check_disjoint, load_problems, score
from .improve import ClaudeMeta
from .storage import Ledger, append_json, read_json, write_json
from .types import AgentSpec, Proposal, digest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def evaluate(directory, problems, spec, task, ledger):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"agent": spec.model_dump(), "tasks": digest(problems)}
    manifest = directory / "evaluation.json"
    if manifest.exists():
        if read_json(manifest) != identity:
            raise ValueError("Evaluation inputs changed")
    else:
        write_json(manifest, identity)
    path = directory / "results.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected_ids = [x["id"] for x in problems]
    if [x["id"] for x in records] != expected_ids[: len(records)]:
        raise ValueError("Evaluation checkpoint order or IDs are invalid")
    for problem in problems[len(records) :]:
        call_id = ledger.start("task", task_id=problem["id"], evaluation=str(directory))
        try:
            # Only the question is passed to the task agent.
            result = task.run(spec, problem["question"])
        except Exception as exc:
            ledger.finish(
                call_id,
                "task",
                error_type=type(exc).__name__,
                status=getattr(exc, "status_code", None),
                usage_unconfirmed=True,
            )
            raise
        row = {
            "id": problem["id"],
            "question": problem["question"],
            **result,
            **score(result["response"], problem["answer"]),
        }
        append_json(path, row)
        records.append(row)
        ledger.finish(
            call_id,
            "task",
            usage=result.get("usage"),
            elapsed_seconds=result.get("elapsed_seconds"),
        )
    summary = {
        "assigned": len(problems),
        "completed": len(records),
        "correct": sum(r["score"] for r in records),
        "accuracy": sum(r["score"] for r in records) / len(problems),
        "format_errors": sum(r["status"] == "format_error" for r in records),
    }
    write_json(directory / "summary.json", summary)
    return records, summary


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
    current = config.agent
    parent_dir = output / "rounds" / "000"
    parent_dir.mkdir(parents=True, exist_ok=True)
    state = completed(parent_dir)
    if state is None:
        records, summary = evaluate(parent_dir, dev, current, task, ledger)
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
        parent_records = [
            json.loads(line) for line in (parent_dir / "results.jsonl").read_text().splitlines()
        ]
        public_records = [
            {k: row[k] for k in ("id", "question", "response", "score", "status")}
            for row in parent_records
        ]
        evidence = {"examples": public_records[:8], "history": history}
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
            _records, summary = evaluate(directory, dev, child, task, ledger)
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
        _, final_summary = evaluate(final_dir, test, frozen, task, ledger)
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
    }
    write_json(output / "summary.json", result)
    return result
