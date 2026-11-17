"""Isolated Transformers/PEFT backend. Run through scripts/train_lora.sh."""

import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

from .loop import file_hash
from .memory_loop import load_rows
from .sft import (
    SFTConfig,
    load_adapter,
    locked,
    pin,
    source_identity,
    tokenize_sample,
    verify_dataset,
)
from .storage import append_json, read_json, write_json
from .types import digest


def model_files(directory):
    directory = Path(directory)
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.suffix in {".json", ".safetensors", ".txt"} and p.is_file()
    )
    if (
        not any(p.suffix == ".safetensors" for p in paths)
        or not (directory / "config.json").is_file()
    ):
        raise ValueError("Download the local base checkpoint first")
    return {p.name: file_hash(p) for p in paths}


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        width = max(len(row["input_ids"]) for row in features)
        return {
            key: torch.tensor(
                [row[key] + [padding] * (width - len(row[key])) for row in features],
                dtype=torch.long,
            )
            for key, padding in [
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }


def parameter_digest(model):
    import torch

    return digest(
        {
            name: hashlib.sha256(
                parameter.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    )


def train(config, output, root):
    output, root = Path(output).resolve(), Path(root).resolve()
    verify_dataset(output, config)
    collection = read_json(output / "collection_identity.json")
    if any(collection[key] != value for key, value in source_identity(root).items()):
        raise ValueError("SFT source or dependency locks changed after collection")
    base_directory = (root / config.base_directory).resolve()
    parent = load_adapter(root / config.parent_adapter) if config.parent_adapter else None
    identity = {
        **source_identity(root),
        "config": config.model_dump(),
        "dataset_sha256": file_hash(output / "train.jsonl"),
        "collection_sha256": file_hash(output / "collection_identity.json"),
        "base_files": model_files(base_directory),
        "parent_adapter_sha256": parent["artifact_sha256"] if parent else None,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "packages": {
            name: version(name)
            for name in ["torch", "transformers", "peft", "accelerate", "safetensors"]
        },
    }
    if identity["parent_adapter_sha256"] != collection["parent_adapter_sha256"]:
        raise ValueError("Parent adapter differs from the collection model")
    if parent and parent["identity"]["base_files"] != identity["base_files"]:
        raise ValueError("Parent adapter belongs to different base checkpoint bytes")
    rank = int(os.environ.get("RANK", "0"))
    with locked(output) if rank == 0 else nullcontext():
        if (output / "adapter").exists():
            artifact = load_adapter(output / "adapter")
            if artifact["identity"] != identity:
                raise ValueError("Completed adapter has different training inputs")
            return artifact
        if rank == 0:
            pin(output / "training/identity.json", identity)
        return train_backend(config, output, base_directory, identity, root, rank)


def train_backend(config, output, base_directory, identity, root, rank):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("The LoRA backend requires a CUDA GPU with BF16 support")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(base_directory, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    raw = load_rows(output / "train.jsonl")
    rows = [tokenize_sample(tokenizer, row["messages"], config.max_length) for row in raw]
    samples = [row for row in rows if row is not None]
    if len(samples) < config.min_samples:
        raise ValueError("Too few complete examples fit max_length; no examples are truncated")
    tokenization = {
        "retained": len(samples),
        "dropped_overlength": len(raw) - len(samples),
        "prompt_tokens": sum(row["labels"].count(-100) for row in samples),
        "supervised_tokens": sum(sum(t != -100 for t in row["labels"]) for row in samples),
        "max_sequence_length": max(len(row["input_ids"]) for row in samples),
    }
    arguments = TrainingArguments(
        output_dir=str(output / "training/checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=1,
        save_steps=config.save_steps,
        save_total_limit=2,
        save_safetensors=True,
        report_to=[],
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
    )
    # TrainingArguments initializes distributed state before model placement.
    model = AutoModelForCausalLM.from_pretrained(
        base_directory,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    if config.parent_adapter:
        model = PeftModel.from_pretrained(model, root / config.parent_adapter, is_trainable=True)
        previous = model.peft_config["default"]
        if (
            previous.r != config.rank
            or previous.lora_alpha != config.alpha
            or previous.lora_dropout != config.dropout
            or set(previous.target_modules) != set(config.target_modules)
        ):
            raise ValueError("Continuation must preserve the parent LoRA structure")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=config.target_modules,
                bias="none",
            ),
        )
    trainable, total = model.get_nb_trainable_parameters()
    if not 0 < trainable < total or any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("Only LoRA parameters may be trainable")
    before = parameter_digest(model) if rank == 0 else None
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=samples,
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    checkpoints = output / "training/checkpoints"
    previous_checkpoint = get_last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
    started = time.time()
    attempts = output / "training/attempts.jsonl"
    if rank == 0:
        append_json(
            attempts,
            {"event": "started", "time": started, "resume_from_checkpoint": previous_checkpoint},
        )
        write_json(output / "training/tokenization.json", tokenization)
    try:
        result = trainer.train(resume_from_checkpoint=previous_checkpoint)
        if rank == 0:
            after = parameter_digest(model)
            if before == after or not all(
                torch.isfinite(parameter).all().item()
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise ValueError("Training did not produce a finite changed adapter")
            staging = output / ".adapter-staging"
            if staging.exists():
                shutil.rmtree(staging)
            trainer.save_model(str(staging))
            trainer.save_state()
            files = {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}
            artifact_hash = digest(files)
            artifact = {
                "identity": identity,
                "files": files,
                "artifact_sha256": artifact_hash,
                "served_model": f"rsi-lora-{artifact_hash[:12]}",
                "trainable_parameters": trainable,
                "total_parameters": total,
                "parameters_changed": before != after,
                "trainable_sha256_before": before,
                "trainable_sha256_after": after,
                "tokenization": tokenization,
                "metrics": result.metrics,
                "optimizer_steps": trainer.state.global_step,
                "base_weights_frozen": True,
            }
            write_json(staging / "manifest.json", artifact)
            staging.rename(output / "adapter")
            append_json(
                attempts,
                {
                    "event": "finished",
                    "time": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "optimizer_steps": trainer.state.global_step,
                },
            )
        trainer.accelerator.wait_for_everyone()
        return load_adapter(output / "adapter")
    except BaseException as exc:
        if rank == 0:
            append_json(
                attempts, {"event": "failed", "time": time.time(), "error_type": type(exc).__name__}
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.json"))
    parser.add_argument("--run", type=Path, default=Path("runs/lora-demo"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = train(SFTConfig.model_validate_json(args.config.read_text()), args.run, root)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "adapter": str(args.run / "adapter"),
                    "model": result["served_model"],
                    "optimizer_steps": result["optimizer_steps"],
                }
            )
        )


if __name__ == "__main__":
    main()
