#!/usr/bin/env python3
"""Run the pinned verl FSDP2/vLLM backend and export trained weights."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

from rsi_base.agent import task_messages
from rsi_base.components import callable_hash, load_callable
from rsi_base.loop import file_hash
from rsi_base.rl import GRPOConfig, freeze_inputs, parent_artifact, training_problems
from rsi_base.sft import load_adapter, locked, source_identity
from rsi_base.storage import append_json, read_json, write_json
from rsi_base.types import digest
from rsi_base.weights import load_artifact, load_checkpoint, model_files, verify_updated_checkpoint

BACKEND_SHA256 = file_hash(__file__)


def build_parquet(problems, config, tokenizer, path):
    records = []
    for index, item in enumerate(problems):
        prompt = task_messages(config.system_prompt, item["question"])
        tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(tokens) > config.max_prompt_tokens:
            raise ValueError(f"Training prompt exceeds max_prompt_tokens: {item['id']}")
        records.append(
            {
                "data_source": "rsi_numeric",
                "prompt": prompt,
                "reward_model": {"style": "rule", "ground_truth": item["answer"]},
                "extra_info": {"index": index, "id": item["id"], "scorer": config.scorer},
            }
        )
    pq.write_table(pa.Table.from_pylist(records), path)


def verl_arguments(config, root, run, base_dir, parent_dir, gpu_count, ray_dir, attempt):
    full = config.training_mode == "full"
    sequence_length = config.max_prompt_tokens + config.max_answer_tokens
    data = run / "train.parquet"
    options = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(data),
        # verl requires a validation iterator. Validation is disabled in the trainer.
        "data.val_files": str(data),
        "data.train_batch_size": config.prompts_per_batch,
        "data.val_batch_size": config.prompts_per_batch,
        "data.max_prompt_length": config.max_prompt_tokens,
        "data.max_response_length": config.max_answer_tokens,
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.seed": config.seed,
        "actor_rollout_ref.model.path": str(base_dir),
        "actor_rollout_ref.model.lora_rank": 0 if full else config.rank,
        "actor_rollout_ref.model.use_remove_padding": False,
        "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32" if full else "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": config.learning_rate,
        "actor_rollout_ref.actor.optim.weight_decay": 0,
        "actor_rollout_ref.actor.ppo_mini_batch_size": config.prompts_per_batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
        "actor_rollout_ref.rollout.max_model_len": sequence_length,
        "actor_rollout_ref.rollout.max_num_seqs": 64,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.temperature": config.temperature,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.n": config.samples_per_prompt,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.agent.num_workers": gpu_count,
        "actor_rollout_ref.rollout.seed": config.seed,
        "reward.num_workers": gpu_count,
        "reward.custom_reward_function.path": str(root / "scripts/grpo_reward.py"),
        "trainer.balance_batch": False,
        "trainer.logger": ["console"],
        "trainer.n_gpus_per_node": gpu_count,
        "trainer.nnodes": 1,
        "trainer.default_local_dir": str(run / "checkpoints"),
        "trainer.rollout_data_dir": str(attempt / "rollout"),
        "trainer.save_freq": config.save_every,
        "trainer.total_epochs": math.ceil(
            config.steps / (config.train_samples // config.prompts_per_batch)
        ),
        "trainer.total_training_steps": config.steps,
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.include_dashboard": False,
        "+ray_kwargs.ray_init._temp_dir": ray_dir,
    }
    if full:
        options["trainer.max_actor_ckpt_to_keep"] = 2
    else:
        options.update(
            {
                "actor_rollout_ref.model.lora_alpha": config.alpha,
                "actor_rollout_ref.model.target_modules": config.target_modules,
                "+actor_rollout_ref.actor.checkpoint.save_lora_only": True,
            }
        )
        if parent_dir is not None:
            options["actor_rollout_ref.model.lora_adapter_path"] = str(parent_dir)
    return [f"{key}={json.dumps(value)}" for key, value in options.items()]


class AdapterMerger(FSDPModelMerger):
    def save_hf_model_and_tokenizer(self, state_dict):
        if not self.save_lora_adapter(state_dict):
            raise ValueError("verl checkpoint does not contain LoRA parameters")


def rollout_stats(run, config):
    by_step = {}
    for path in sorted((run / "attempts").glob("*/rollout/*.jsonl")):
        groups = defaultdict(list)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                # verl v1 appends rollout and output indices to the prompt's UID.
                prompt_uid, _, _ = row["uid"].rsplit("_", 2)
                groups[prompt_uid].append(row["score"])
        if len(groups) != config.prompts_per_batch or any(
            len(scores) != config.samples_per_prompt for scores in groups.values()
        ):
            raise ValueError(f"Incomplete rollout batch: {path}")
        by_step[int(path.stem)] = {
            "groups": len(groups),
            "mixed_reward_groups": sum(len(set(scores)) > 1 for scores in groups.values()),
            "mean_reward": sum(map(sum, groups.values()))
            / (len(groups) * config.samples_per_prompt),
        }
    steps = [{"step": step, **stats} for step, stats in sorted(by_step.items())]
    return {"steps": steps, "mixed_reward_groups": sum(row["mixed_reward_groups"] for row in steps)}


def final_checkpoint(run, config):
    step_file = run / "checkpoints/latest_checkpointed_iteration.txt"
    step = int(step_file.read_text())
    if step != config.steps:
        raise ValueError(f"Expected final training checkpoint at step {config.steps}, got {step}")
    metrics = rollout_stats(run, config)
    write_json(run / "rollout_stats.json", metrics)
    if not metrics["mixed_reward_groups"]:
        raise ValueError("Run has no within-prompt reward variance")
    return run / f"checkpoints/global_step_{step}/actor", step, metrics


def export_adapter(run, config, identity, parent_dir):
    checkpoint, step, metrics = final_checkpoint(run, config)
    with tempfile.TemporaryDirectory(prefix="adapter-", dir=run) as temporary:
        AdapterMerger(
            ModelMergerConfig(
                operation="merge",
                backend="fsdp",
                local_dir=str(checkpoint),
                hf_model_config_path=str(checkpoint / "huggingface"),
                target_dir=temporary,
            )
        ).merge_and_save()
        artifact = Path(temporary) / "lora_adapter"
        config_path = artifact / "adapter_config.json"
        peft = read_json(config_path)
        if (
            peft["r"] != config.rank
            or peft["lora_alpha"] != config.alpha
            or set(peft["target_modules"]) != set(config.target_modules)
        ):
            raise ValueError("Exported PEFT config differs from training")
        peft.update(
            base_model_name_or_path=config.base_model,
            revision=config.base_revision,
            target_modules=sorted(peft["target_modules"]),
        )
        write_json(config_path, peft)
        weights = load_file(artifact / "adapter_model.safetensors")
        if not weights or not all(
            torch.isfinite(weight).all().item() for weight in weights.values()
        ):
            raise ValueError("GRPO did not produce finite LoRA parameters")
        if parent_dir:
            parent = load_file(parent_dir / "adapter_model.safetensors")
            if weights.keys() != parent.keys():
                raise ValueError("RL adapter parameter keys differ from parent")
            changed = any(
                not torch.equal(weight, parent[name].to(weight.dtype))
                for name, weight in weights.items()
            )
        else:
            changed = any(
                weight.count_nonzero().item()
                for name, weight in weights.items()
                if ".lora_B." in name
            )
        if not changed:
            raise ValueError(
                "No LoRA change from the initialization. Check within-prompt reward variance."
            )
        files = {path.name: file_hash(path) for path in artifact.iterdir()}
        checksum = digest(files)
        manifest = {
            "identity": identity,
            "files": files,
            "artifact_sha256": checksum,
            "served_model": f"rsi-grpo-{checksum[:12]}",
            "optimizer_steps": step,
            "parameters_changed": changed,
            "base_weights_frozen": True,
            "metrics": metrics,
            "export_source_sha256": BACKEND_SHA256,
        }
        write_json(artifact / "manifest.json", manifest)
        artifact.rename(run / "adapter")
    return load_adapter(run / "adapter")


def export_full_model(run, config, identity, initial_directory):
    checkpoint, step, metrics = final_checkpoint(run, config)
    with tempfile.TemporaryDirectory(prefix=".model-", dir=run) as temporary:
        artifact = Path(temporary) / "model"
        artifact.mkdir()
        FSDPModelMerger(
            ModelMergerConfig(
                operation="merge",
                backend="fsdp",
                local_dir=str(checkpoint),
                hf_model_config_path=str(checkpoint / "huggingface"),
                target_dir=str(artifact),
            )
        ).merge_and_save()
        verify_updated_checkpoint(initial_directory, artifact)
        files = {path.name: file_hash(path) for path in artifact.iterdir() if path.is_file()}
        checksum = digest(files)
        write_json(
            artifact / "manifest.json",
            {
                "identity": identity,
                "files": files,
                "artifact_sha256": checksum,
                "served_model": f"rsi-full-grpo-{checksum[:12]}",
                "optimizer_steps": step,
                "parameters_changed": True,
                "base_weights_frozen": False,
                "metrics": metrics,
                "export_source_sha256": BACKEND_SHA256,
            },
        )
        manifest = load_checkpoint(artifact)
        artifact.rename(run / "model")
    return manifest


def train(config, data, run, root):
    devices = os.environ.get("RSI_RL_GPUS", "0,1").split(",")
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("RSI_RL_GPUS must be a comma-separated list of distinct device indices")
    if config.prompts_per_batch % len(devices):
        raise ValueError("prompts_per_batch must divide evenly across the selected GPUs")
    pin = read_json(root / "requirements/verl.json")
    if version("verl") != pin["version"]:
        raise ValueError("Run setup --rl to install the pinned verl version")
    full = config.training_mode == "full"
    if not full and config.dropout != 0:
        raise ValueError("This GRPO backend requires zero LoRA dropout")
    problems = training_problems(config, data)
    base_dir = (root / config.base_directory).resolve()
    base_files = model_files(base_dir)
    parent = parent_artifact(config, root)
    parent_dir = (root / config.parent_path).resolve() if parent else None
    if parent and parent["identity"]["base_files"] != base_files:
        raise ValueError("Training parent belongs to different base checkpoint bytes")
    start_dir = parent_dir if full and parent_dir is not None else base_dir
    if not full and parent_dir is not None:
        peft = read_json(parent_dir / "adapter_config.json")
        if (
            peft["r"] != config.rank
            or peft["lora_alpha"] != config.alpha
            or set(peft["target_modules"]) != set(config.target_modules)
            or peft.get("lora_dropout", 0) != 0
            or peft.get("bias", "none") != "none"
            or peft.get("use_rslora")
            or peft.get("use_dora")
            or peft.get("modules_to_save")
        ):
            raise ValueError(
                "Parent adapter does not match the GRPO base model and LoRA configuration"
            )
    identity = {
        **source_identity(root),
        "method": "grpo",
        "config": config.model_dump(),
        "base_files": base_files,
        config.parent_hash_key: parent["artifact_sha256"] if parent else None,
        "train_split_sha256": file_hash(data / "train.jsonl"),
        "selected_training_data_sha256": digest(problems),
        "verl": pin,
        "numpy": version("numpy"),
        "rl_numpy_lock_sha256": file_hash(root / "requirements/rl-numpy.lock"),
        "backend_files": {
            "train_grpo.py": BACKEND_SHA256,
            "grpo_reward.py": file_hash(root / "scripts/grpo_reward.py"),
            "problem_loader": callable_hash(load_callable(config.problem_loader)),
            "scorer": callable_hash(load_callable(config.scorer)),
        },
        "devices": devices,
    }
    with locked(run) as run:
        freeze_inputs(run / "train_identity.json", identity)
        target = run / config.artifact_name
        if target.exists():
            artifact = load_artifact(target, config.training_mode)
            if artifact["identity"] != identity:
                raise ValueError("Existing model does not match this training run")
            return artifact
        tokenizer = AutoTokenizer.from_pretrained(start_dir, local_files_only=True)
        build_parquet(problems, config, tokenizer, run / "train.parquet")
        attempts = run / "training_attempts.jsonl"
        started = time.time()
        append_json(attempts, {"event": "started", "time": started})
        try:
            latest = run / "checkpoints/latest_checkpointed_iteration.txt"
            if not latest.is_file() or int(latest.read_text()) < config.steps:
                launch_verl(config, root, run, start_dir, None if full else parent_dir, devices)
            result = (
                export_full_model(run, config, identity, start_dir)
                if full
                else export_adapter(run, config, identity, parent_dir)
            )
        except BaseException as error:
            append_json(
                attempts,
                {"event": "failed", "time": time.time(), "error_type": type(error).__name__},
            )
            raise
        append_json(
            attempts, {"event": "finished", "time": time.time(), "seconds": time.time() - started}
        )
        return result


def launch_verl(config, root, run, base_dir, parent_dir, devices):
    with tempfile.TemporaryDirectory(prefix="rsi-rl-", dir="/dev/shm") as ray_dir:
        attempt = run / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        arguments = verl_arguments(
            config, root, run, base_dir, parent_dir, len(devices), ray_dir, attempt
        )
        command = [sys.executable, "-m", "verl.trainer.main_ppo", *arguments]
        write_json(attempt / "command.json", command)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
            "HF_HUB_OFFLINE": "1",
            "HYDRA_FULL_ERROR": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": str(root / ".local/verl/triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / ".local/verl/inductor"),
            "VLLM_CACHE_ROOT": str(root / ".local/verl/vllm"),
            "VLLM_NO_USAGE_STATS": "1",
        }
        log = attempt / "verl.log"
        print(f"Starting verl training. Log: {log}", flush=True)
        with log.open("a") as stream:
            try:
                subprocess.run(
                    command,
                    cwd=run,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=config.training_timeout,
                )
            finally:
                ray_logs = Path(ray_dir) / "session_latest/logs"
                if ray_logs.is_dir():
                    shutil.copytree(ray_logs, attempt / "ray", symlinks=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/lora_grpo.json")
    parser.add_argument("--data", type=Path, default=root / "data/gsm8k-grpo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = GRPOConfig.model_validate_json(args.config.read_text())
    result = train(config, args.data.resolve(), args.output.resolve(), root)
    print(json.dumps({"model": result["served_model"], "steps": result["optimizer_steps"]}))


if __name__ == "__main__":
    main()
