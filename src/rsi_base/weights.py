"""Validated trainable model artifacts shared by SFT, GRPO and Task serving."""

from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .loop import file_hash
from .storage import read_json
from .types import StrictModel, digest


class AdapterConfig(StrictModel):
    training_mode: Literal["lora", "full"] = "lora"
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    base_directory: str = ".local/models/qwen3-4b-instruct-2507"
    parent_adapter: str | None = None
    parent_checkpoint: str | None = None
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @field_validator("target_modules")
    @classmethod
    def modules(cls, value):
        if not value or len(set(value)) != len(value) or any(not item.strip() for item in value):
            raise ValueError("target_modules must contain distinct nonempty module names")
        return value

    @model_validator(mode="after")
    def parent_format(self):
        if self.training_mode == "full" and self.parent_adapter is not None:
            raise ValueError("Full training accepts parent_checkpoint for continuation")
        if self.training_mode == "lora" and self.parent_checkpoint is not None:
            raise ValueError(
                "LoRA uses parent_adapter; set base_directory to start on a full model"
            )
        return self

    @property
    def parent_path(self):
        return self.parent_checkpoint if self.training_mode == "full" else self.parent_adapter

    @property
    def artifact_name(self):
        return "model" if self.training_mode == "full" else "adapter"

    @property
    def artifact_hash_key(self):
        return "checkpoint_sha256" if self.training_mode == "full" else "adapter_sha256"

    @property
    def parent_hash_key(self):
        return f"parent_{self.artifact_hash_key}"


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


def load_manifest(directory, required):
    manifest = read_json(directory / "manifest.json")
    files = manifest["files"]
    if not required.issubset(files):
        raise ValueError("Incomplete training artifact")
    for name, expected in files.items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError("Training artifact changed")
    if manifest["artifact_sha256"] != digest(files):
        raise ValueError("Training artifact identity changed")
    return manifest


def load_adapter(directory):
    return load_manifest(Path(directory), {"adapter_config.json", "adapter_model.safetensors"})


def checkpoint_shards(directory):
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        by_tensor = read_json(index)["weight_map"]
        names = set(by_tensor.values())
        if not names or any(
            Path(name).name != name or not name.endswith(".safetensors") for name in names
        ):
            raise ValueError("Invalid Hugging Face checkpoint shard index")
        return names | {index.name}
    return {"model.safetensors"}


def load_checkpoint(directory):
    directory = Path(directory)
    return load_manifest(
        directory, {"config.json", "tokenizer_config.json"} | checkpoint_shards(directory)
    )


def load_artifact(directory, mode):
    return (load_checkpoint if mode == "full" else load_adapter)(directory)


def load_parent(config, root):
    if not config.parent_path:
        return None
    artifact = load_artifact(Path(root) / config.parent_path, config.training_mode)
    if artifact["identity"]["config"]["base_model"] != config.base_model:
        raise ValueError("Training parent belongs to another base model")
    return artifact


def parent_endpoint(config, candidate_url, override):
    if override:
        return override
    return "http://127.0.0.1:18000/v1" if config.training_mode == "full" else candidate_url


def verify_updated_checkpoint(initial, exported):
    """Compare actual tensors in their exported precision and reject nonfinite updates."""
    import torch
    from safetensors import safe_open

    def index(directory, stack):
        handles = [
            stack.enter_context(safe_open(directory / name, framework="pt"))
            for name in sorted(checkpoint_shards(directory))
            if name.endswith(".safetensors")
        ]
        return {name: handle for handle in handles for name in handle.keys()}

    changed = False
    with ExitStack() as stack:
        original = index(initial, stack)
        updated = index(exported, stack)
        if not original or original.keys() != updated.keys():
            raise ValueError("Exported full checkpoint has different parameter keys")
        for name, handle in updated.items():
            weights = handle.get_tensor(name)
            if not torch.isfinite(weights).all().item():
                raise ValueError(f"Nonfinite weight in full checkpoint: {name}")
            parent = original[name].get_tensor(name)
            if parent.shape != weights.shape:
                raise ValueError(f"Shape changed in full checkpoint: {name}")
            if not changed:
                changed = not torch.equal(weights, parent.to(weights.dtype))
    if not changed:
        raise ValueError("Full checkpoint contains no effective weight update")
