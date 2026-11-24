"""Run development, checkpoint selection and frozen tests over assembled modules."""

import sys
from pathlib import Path

from .improve import MetaCallFailure
from .loop import completed, evaluate, file_hash, seal
from .memory_audit import cluster_interval
from .pipeline_setup import PipelineConfig as PipelineConfig
from .pipeline_setup import compose
from .pipeline_types import PipelineProposal as PipelineProposal
from .pipeline_types import Step, TaskState
from .storage import Ledger, lock_directory, pin_json, read_json, read_jsonl, write_json


def compare_records(parent, candidate):
    before, after = ({row["id"]: row["score"] for row in rows} for rows in (parent, candidate))
    if before.keys() != after.keys() or not before:
        raise ValueError("Paired evaluations must cover the same questions")
    deltas = {key: after[key] - value for key, value in before.items()}
    return {
        "questions": len(deltas),
        "parent_correct": sum(before.values()),
        "candidate_correct": sum(after.values()),
        "gain_questions": sum(deltas.values()),
        "gain_accuracy": sum(deltas.values()) / len(deltas),
        "improved": [key for key, delta in deltas.items() if delta > 0],
        "regressed": [key for key, delta in deltas.items() if delta < 0],
    }, deltas


def examples(rows, problems, *, failures=8, successes=2):
    refs = {row["id"]: row["answer"] for row in problems}
    selected = [row for row in rows if not row["score"]][:failures]
    selected += [row for row in rows if row["score"]][:successes]
    return [
        {
            **{name: row[name] for name in ("id", "question", "score", "status", "finish_reason")},
            "response": row["response"][:1800] + "\n[...]\n" + row["response"][-600:]
            if len(row["response"]) > 2400
            else row["response"],
            "reference_answer": refs[row["id"]],
        }
        for row in selected
    ]


def recorded_evaluation(directory, summary):
    path = directory / "results.jsonl"
    if file_hash(path) != summary["results_sha256"]:
        raise ValueError(f"Development evaluation changed: {path}")
    return read_jsonl(path)


class Pipeline:
    def __init__(self, config, data, output, root, *, components=None):
        self.config = config
        self.data_dir, self.output, self.root = (Path(p).resolve() for p in (data, output, root))
        self.components = components or compose(config, self.data_dir, self.output, self.root)
        self.data = self.components.data
        self.server = self.components.task
        self.initial = self.components.initial
        self.manifest = {
            "config": config.model_dump(),
            "data": str(self.data_dir),
            "data_sha256": self.data.hashes,
            "source": self.components.sources,
            "base_files": self.components.base_files,
            "initial": self.initial.model_dump(),
            "available_strategies": list(self.components.strategies),
        }

    def validate_budget(self):
        config = self.config
        options = self.components.strategies
        if config.schedule is None:
            if config.limits.rounds and not options:
                raise ValueError("No enabled strategy has sufficient inputs")
            collection_calls = config.limits.rounds * max(
                (choice.task_calls for choice in options.values()), default=0
            )
            meta_calls = config.limits.rounds
        else:
            if set(config.schedule).difference(options):
                raise ValueError("A scheduled strategy is unavailable with the supplied inputs")
            collection_calls = sum(options[name].task_calls for name in config.schedule)
            meta_calls = sum(options[name].scheduled is None for name in config.schedule)
        total = (config.limits.rounds + 1) * len(self.data.development) + collection_calls
        if config.limits.max_task_calls < total:
            raise ValueError(
                "Task call budget must cover development evaluations and strategy rollouts"
            )
        if meta_calls and (
            self.components.planner is None or config.limits.max_meta_calls < meta_calls
        ):
            raise ValueError("Configure a planner and enough Meta calls for all proposal rounds")

    def evaluate(self, directory, problems, state, ledger, *, temperature=None):
        if temperature is None:
            temperature = self.config.runtime.evaluation_temperature
        # A finished evaluation can be resumed without loading another model.
        records = directory / "results.jsonl"
        if records.exists() and len(read_jsonl(records)) == len(problems):
            agent = None
        else:
            agent = self.server.agent(
                state.model, temperature=temperature, deadline=ledger.deadline
            )
        return evaluate(
            directory,
            problems,
            state.agent,
            agent,
            ledger,
            concurrency=self.config.limits.task_concurrency,
            scorer=self.components.scorer,
            expected_model=state.model.name,
        )

    def evaluate_dev(self, directory, state, ledger):
        _, metrics = self.evaluate(directory, self.data.development, state, ledger)
        return {**metrics, "results_sha256": file_hash(directory / "results.jsonl")}

    def propose(self, index, directory, state, metrics, records, history, ledger):
        config = self.config
        cached = directory / "plan.json"
        if cached.exists():
            return read_json(cached)
        scheduled = config.schedule[index - 1] if config.schedule is not None else None
        options = self.components.strategies
        allowed = [scheduled] if scheduled else list(options)
        preset = options[scheduled].scheduled if scheduled else None
        if preset is not None:
            plan = {"proposal": preset.model_dump(), "source": "schedule"}
        else:
            recent = []
            for step in history[-3:]:
                recent.append(
                    {
                        key: step[key]
                        for key in (
                            "round",
                            "parent_round",
                            "strategy",
                            "plan",
                            "training",
                            "comparison",
                            "status",
                        )
                    }
                )
                if step["comparison"] is None:
                    continue
                path = self.round_dir(step["round"]) / "development"
                step_rows = recorded_evaluation(path, step["development"])
                changed = set(step["comparison"]["improved"] + step["comparison"]["regressed"])
                recent[-1]["examples"] = examples(
                    [row for row in step_rows if row["id"] in changed],
                    self.data.development,
                    failures=2,
                    successes=1,
                )
            evidence = {
                "round": index,
                "available_strategies": allowed,
                "parent_model": state.model.name,
                "parent_weights_sha256": state.model.sha256,
                "development": metrics,
                "examples": examples(records, self.data.development),
                "previous_rounds": recent,
                "benchmark_instructions": config.benchmark.instructions,
                "train_questions": len(self.data.train),
                "worked_references": len(self.data.references),
                "strategy_options": {
                    name: {"description": options[name].description, **options[name].evidence}
                    for name in allowed
                },
                "required_development_gain_questions": config.min_gain_questions
                if config.selection is None
                else None,
            }
            write_json(directory / "evidence.json", evidence)
            call_id = ledger.start("meta", round=index)
            try:
                proposal, usage = self.components.planner.propose(state.agent, evidence, directory)
                proposal = PipelineProposal.model_validate(proposal)
            except (MetaCallFailure, ValueError) as error:
                ledger.finish(
                    call_id, "meta", error_type=type(error).__name__, usage_unconfirmed=True
                )
                plan = {"proposal": None, "source": "meta", "error": type(error).__name__}
            else:
                ledger.finish(call_id, "meta", **usage)
                if proposal.strategy not in allowed or (
                    config.meta
                    and len(proposal.system_prompt or "") > config.meta.prompt_char_budget
                ):
                    plan = {
                        "proposal": None,
                        "source": "meta",
                        "error": "unavailable_or_oversized_proposal",
                    }
                else:
                    plan = {"proposal": proposal.model_dump(), "source": "meta"}
        write_json(cached, plan)
        return plan

    def round_dir(self, index):
        return self.output / "rounds" / f"{index:03}"

    def run(self, *, resume=False):
        self.validate_budget()
        with lock_directory(self.output, ".pipeline.lock"):
            if not resume and any(p.name != ".pipeline.lock" for p in self.output.iterdir()):
                raise ValueError("Output already exists; use --resume or a new directory")
            pin_json(self.output / "run.json", self.manifest)
            ledger = Ledger(self.output / "calls.jsonl", self.config.limits)
            try:
                return self.develop(ledger)
            finally:
                self.server.stop()
                ledger.close()

    def develop(self, ledger):
        initial_directory = self.round_dir(0)
        initial_metrics = completed(initial_directory)
        if initial_metrics is None:
            if ledger is None:
                raise ValueError("Complete development before testing")
            initial_metrics = self.evaluate_dev(
                initial_directory / "development", self.initial, ledger
            )
            seal(initial_directory, initial_metrics)
        parent, parent_metrics = self.initial, initial_metrics
        records = recorded_evaluation(initial_directory / "development", initial_metrics)
        selected_round, history = 0, []
        for index in range(1, self.config.limits.rounds + 1):
            directory = self.round_dir(index)
            directory.mkdir(parents=True, exist_ok=True)
            decision = completed(directory)
            if decision is None:
                if ledger is None:
                    raise ValueError("Complete every development round before testing")
                plan = self.propose(
                    index, directory, parent, parent_metrics, records, history, ledger
                )
                decision = self.decide(
                    index, directory, plan, parent, records, selected_round, ledger
                )
                write_json(directory / "decision.json", decision)
                seal(directory, decision)
                print(
                    f"Round {index}: {decision['strategy']}, {decision['status']}",
                    file=sys.stderr,
                    flush=True,
                )
            if decision["parent_round"] != selected_round:
                raise ValueError("Round does not follow its recorded parent")
            if decision["accepted"]:
                parent = TaskState.model_validate(decision["candidate"])
                parent_metrics, selected_round = decision["development"], index
                records = recorded_evaluation(directory / "development", parent_metrics)
            history.append(decision)
        frozen = {
            "initial": self.initial.model_dump(),
            "selected": parent.model_dump(),
            "selected_round": selected_round,
            "initial_development": initial_metrics,
            "selected_development": parent_metrics,
            "rounds": history,
        }
        pin_json(self.output / "frozen.json", frozen)
        return frozen

    def decide(self, index, directory, plan, parent, records, selected_round, ledger):
        raw = plan["proposal"]
        proposal = PipelineProposal.model_validate(raw) if raw else None
        candidate, training = (None, None)
        if proposal:
            candidate, training = self.components.strategies[proposal.strategy].apply(
                Step(directory, parent, index, ledger, self.evaluate), proposal
            )
        metrics = comparison = None
        accepted = False
        if candidate:
            metrics = self.evaluate_dev(directory / "development", candidate, ledger)
            child = recorded_evaluation(directory / "development", metrics)
            comparison, _ = compare_records(records, child)
            accepted = self.components.accept(comparison)
            status = "accepted" if accepted else "insufficient_development_gain"
        else:
            status = training["reason"] if proposal else "meta_proposal_failed"
        return {
            "round": index,
            "parent_round": selected_round,
            "strategy": proposal.strategy if proposal else None,
            "plan": plan,
            "candidate": candidate.model_dump() if candidate else None,
            "training": training,
            "development": metrics,
            "comparison": comparison,
            "accepted": accepted,
            "status": status,
        }

    def test(self):
        with lock_directory(self.output, ".pipeline.lock"):
            path = self.output / "frozen.json"
            if not path.is_file() or read_json(self.output / "run.json") != self.manifest:
                raise ValueError("Complete development with identical inputs before testing")
            frozen = self.develop(ledger=None)
            directory = self.output / "test"
            directory.mkdir(exist_ok=True)
            pin_json(
                directory / "identity.json",
                {"frozen_sha256": file_hash(path), "test_sha256": self.data.hashes["test"]},
            )
            problems = self.components.load_problems(self.data_dir / "test.jsonl")
            selected = TaskState.model_validate(frozen["selected"])
            count = len(problems) * (1 if self.initial == selected else 2)
            if self.config.limits.max_task_calls < count:
                raise ValueError("Task call budget must cover the paired test")
            ledger = Ledger(directory / "calls.jsonl", self.config.limits)
            try:
                baseline, before = self.evaluate(
                    directory / "initial", problems, self.initial, ledger
                )
                child, after = (
                    (baseline, before)
                    if self.initial == selected
                    else self.evaluate(directory / "selected", problems, selected, ledger)
                )
            finally:
                self.server.stop()
                ledger.close()
            comparison, deltas = compare_records(baseline, child)
            report = {
                "selected_round": frozen["selected_round"],
                "initial": before,
                "selected": after,
                "comparison": comparison,
                "paired_interval": cluster_interval(
                    deltas, seed=self.config.benchmark.interval_seed
                ),
            }
            pin_json(directory / "report.json", report)
            return report
