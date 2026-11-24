"""Validated trainable model artifacts shared by SFT, GRPO and Task serving."""

from contextlib import ExitStack
from pathlib import Path

from .loop import file_hash
from .storage import read_json
from .types import digest


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
