"""Built-in update strategies and their training inputs."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .loop import file_hash
from .pipeline_data import (
    PipelineData,
    offline_examples,
    online_examples,
    select_sft_problems,
    write_sft_dataset,
)
from .pipeline_runtime import run_training
from .pipeline_types import PipelineProposal, Strategy, TaskState
from .pipeline_weights import ModelStore
from .storage import pin_json, read_json
from .types import AgentSpec


@dataclass(frozen=True)
class Training:
    root: Path
    data_dir: Path
    data: PipelineData
    models: ModelStore
    release: Callable[[list[int]], None]

    def execute(self, phase, config, output, step, devices):
        self.release(devices)
        gpus = ",".join(map(str, devices))
        if phase == "sft":
            environment = {"RSI_TRAIN_GPUS": gpus, "RSI_TRAIN_PROCESSES": str(len(devices))}
            max_seconds = config.wall_seconds
        else:
            environment = {"RSI_RL_GPUS": gpus}
            max_seconds = None
        return run_training(
            self.root,
            phase,
            config,
            self.data_dir,
            output,
            step.ledger.deadline,
            devices=devices,
            environment=environment,
            max_seconds=max_seconds,
        )

    def result(self, config, parent, target):
        model, artifact = self.models.artifact(target)
        identity = artifact["identity"]
        expected_parent = parent.model.sha256 if parent.model.path else None
        if (
            identity["config"] != config.model_dump()
            or identity[config.parent_hash_key] != expected_parent
        ):
            raise ValueError("Training artifact does not match the planned update and parent")
        return TaskState(agent=parent.agent, model=model), artifact


def training_config(schema, model, options):
    shared = model.model_dump()
    explicit = schema.model_validate({**shared, **options})
    fields = (
        "base_model",
        "base_revision",
        "base_directory",
        "training_mode",
        "parent_adapter",
        "parent_checkpoint",
    )
    if model.training_mode == "lora":
        fields += ("rank", "alpha", "dropout", "target_modules")
    if any(getattr(explicit, name) != shared[name] for name in fields):
        raise ValueError("Trainer options conflict with the pipeline model configuration")
    return explicit


def continue_from(config, model, **updates):
    field = "parent_checkpoint" if config.training_mode == "full" else "parent_adapter"
    return config.model_copy(update={field: model.path, **updates})


def harness_strategy():
    def apply(step, proposal):
        agent = AgentSpec(system_prompt=proposal.system_prompt.strip())
        if agent == step.parent.agent:
            return None, {"reason": "unchanged_harness"}
        return TaskState(agent=agent, model=step.parent.model), None

    return Strategy('Use action "harness" and a revised reusable system_prompt.', apply)


def collect_sft(training, output, source, config, sample_count, step):
    from .sft import source_identity, verify_dataset

    problems = select_sft_problems(training.data, source, sample_count, config.seed + step.index)
    parent = step.parent
    temperature = 0.7 if config.collection_temperature is None else config.collection_temperature
    identity = {
        **source_identity(training.root),
        "config": config.model_dump(),
        "parent": parent.model_dump(),
        config.parent_hash_key: parent.model.sha256 if parent.model.path else None,
        "data_source": source,
        "training_ids": [row["id"] for row in problems],
        "train_split_sha256": training.data.hashes["train"],
        "references_sha256": training.data.hashes["references"] if source == "offline" else None,
        "rollout_temperature": temperature if source == "online" else None,
    }
    pin_json(output / "collection_identity.json", identity)
    if (output / "dataset.json").exists():
        return verify_dataset(output, config)
    if source == "offline":
        samples, excluded = offline_examples(problems, parent.agent, training.data.references), {}
    else:
        batches = [
            step.evaluate(
                output / "rollouts" / f"{repeat:03}",
                problems,
                parent,
                step.ledger,
                temperature=temperature,
            )[0]
            for repeat in range(config.collection_repeats)
        ]
        samples, excluded = online_examples(batches)
    if len(samples) < config.min_samples:
        return {
            "samples": len(samples),
            "excluded": excluded,
            "reason": "insufficient_verified_sft_samples",
        }
    return write_sft_dataset(output, identity, samples, excluded)


def sft_strategy(training, source, options, sample_count, devices):
    from .sft import SFTConfig

    config = training_config(SFTConfig, training.models.config, options)
    offline = source == "offline"
    pool = len(training.data.references if offline else training.data.train)
    queries = min(sample_count, pool)
    samples = queries * (1 if offline else config.collection_repeats)
    calls = 0 if offline else samples
    if samples < config.min_samples:
        return None
    if calls > config.max_collection_calls:
        raise ValueError("Online SFT exceeds max_collection_calls")

    def apply(step, proposal):
        output = step.directory / "training"
        output.mkdir(exist_ok=True)
        config_ = continue_from(
            config, step.parent.model, training_target="reference" if offline else "verified"
        )
        dataset = collect_sft(training, output, source, config_, sample_count, step)
        if "reason" in dataset:
            return None, dataset
        target = output / config_.artifact_name
        if not target.exists():
            status, log = training.execute("sft", config_, output, step, devices)
            if status:
                raise RuntimeError(f"SFT failed. See {log}")
        state, artifact = training.result(config_, step.parent, target)
        identity = artifact["identity"]
        if identity["dataset_sha256"] != dataset["train_sha256"] or identity[
            "collection_sha256"
        ] != file_hash(output / "collection_identity.json"):
            raise ValueError("SFT artifact does not match the selected dataset")
        return state, {"dataset": dataset, "optimizer_steps": artifact["optimizer_steps"]}

    guidance = (
        "Train on worked references."
        if offline
        else "Train on correct, normally finished responses from the current model and harness."
    )
    return Strategy(
        f'Use action "sft" with data_source "{source}". {guidance}',
        apply,
        PipelineProposal(action="sft", data_source=source, reason="Configured fixed schedule"),
        calls,
        {"train_questions_per_update": queries, "minimum_samples": config.min_samples},
    )


def grpo_strategy(training, options, devices, benchmark):
    from .rl import GRPOConfig

    for key, expected in (("problem_loader", benchmark.loader), ("scorer", benchmark.scorer)):
        if key in options and options[key] != expected:
            raise ValueError(f"Configure {key} in the shared pipeline benchmark")
    config = training_config(GRPOConfig, training.models.config, options).model_copy(
        update={"problem_loader": benchmark.loader, "scorer": benchmark.scorer}
    )
    if not config.prompts_per_batch <= config.train_samples <= len(training.data.train):
        return None
    if config.prompts_per_batch % len(devices):
        raise ValueError("GRPO prompts_per_batch must divide evenly across its GPUs")
    if config.training_mode == "lora" and config.dropout != 0:
        raise ValueError("The verl GRPO backend requires zero LoRA dropout")

    def apply(step, proposal):
        output = step.directory / "training"
        output.mkdir(exist_ok=True)
        config_ = continue_from(
            config, step.parent.model, system_prompt=step.parent.agent.system_prompt
        )
        target = output / config_.artifact_name
        stats_path = output / "rollout_stats.json"
        if not target.exists():
            status, log = training.execute("rl", config_, output, step, devices)
            if status:
                if stats_path.exists():
                    stats = read_json(stats_path)
                    if len(stats["steps"]) == config_.steps and not stats["mixed_reward_groups"]:
                        return None, {"reason": "no_grpo_reward_variance", "rollouts": stats}
                raise RuntimeError(f"GRPO failed. See {log}")
        state, artifact = training.result(config_, step.parent, target)
        if artifact["identity"]["train_split_sha256"] != training.data.hashes["train"]:
            raise ValueError("GRPO artifact does not match the pinned training split")
        return state, {
            "rollouts": read_json(stats_path),
            "optimizer_steps": artifact["optimizer_steps"],
        }

    return Strategy(
        'Use action "grpo" with data_source "online". Use fresh grouped rollouts and the benchmark reward.',
        apply,
        PipelineProposal(action="grpo", data_source="online", reason="Configured fixed schedule"),
        evidence={"steps": config.steps, "samples_per_prompt": config.samples_per_prompt},
    )
