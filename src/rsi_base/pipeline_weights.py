"""Local checkpoints and the shared Task model identity."""

from .pipeline_types import ModelState
from .types import digest
from .weights import load_artifact, model_files


class ModelStore:
    def __init__(self, root, config, *, external=False):
        self.config = config
        if external:
            if config.parent_path:
                raise ValueError(
                    "The external endpoint adapter cannot activate a local initial checkpoint"
                )
            self.base_files = {}
            fingerprint = digest({"model": config.base_model, "revision": config.base_revision})
        else:
            self.base_files = model_files(root / config.base_directory)
            fingerprint = digest(self.base_files)
        self.initial = ModelState(name=config.base_model, sha256=fingerprint)
        if config.parent_path:
            self.initial, _ = self.artifact(root / config.parent_path)

    def artifact(self, path):
        config = self.config
        path = path.resolve()
        artifact = load_artifact(path, config.training_mode)
        identity = artifact["identity"]
        if identity["base_files"] != self.base_files or any(
            identity["config"][key] != getattr(config, key)
            for key in ("base_model", "base_revision", "training_mode")
        ):
            raise ValueError("Model artifact belongs to a different base or training mode")
        if config.training_mode == "lora" and any(
            identity["config"][key] != getattr(config, key)
            for key in ("rank", "alpha", "dropout", "target_modules")
        ):
            raise ValueError("LoRA artifact does not match the configured adapter structure")
        return ModelState(
            name=artifact["served_model"], sha256=artifact["artifact_sha256"], path=str(path)
        ), artifact
