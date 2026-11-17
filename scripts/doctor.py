#!/usr/bin/env python3
"""Check setup without model generation or exposing credential values."""

import argparse
import json
import os
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from prepare_harness import verify_asset
from pydantic import ValidationError

from rsi_base.agent import credentials
from rsi_base.memory_loop import MemoryConfig


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/memory_s2d_agent.json")
    parser.add_argument("--data", type=Path, default=root / "data/memory-s2d-expanded")
    parser.add_argument(
        "--ready", action="store_true", help="Require Meta/data/Task readiness; GET /models only"
    )
    parser.add_argument(
        "--local-gpu",
        action="store_true",
        help="Also inspect the local CUDA environment and weights",
    )
    args = parser.parse_args()
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"check": name, "ok": True, "detail": detail})
        except Exception as exc:
            checks.append({"check": name, "ok": False, "detail": str(exc)})

    def python_version():
        value = platform.python_version()
        if value != "3.11.15":
            raise ValueError(f"Expected Python 3.11.15, got {value}; rerun setup")
        return value

    check("python", python_version)
    check(
        "core_packages", lambda: {name: version(name) for name in ["openai", "pydantic", "httpx"]}
    )
    try:
        config = MemoryConfig.model_validate_json(args.config.read_text())
    except ValidationError as exc:
        parser.error(str(exc.errors(include_input=False, include_context=False, include_url=False)))
    check(
        "configuration", lambda: {"task_model": config.task.model, "meta_model": config.meta.model}
    )

    if args.ready:
        if config.proposal_strategy == "agent":

            def sandbox():
                from rsi_base.memory_code import run_code

                run_code("def rank(query, memory): return []", [{"query": "ready", "memory": []}])
                return "Namespace probe passed; host home/network absent"

            check("retrieval_sandbox", sandbox)

        def cli():
            executable = shutil.which(config.meta.command)
            if not executable:
                raise ValueError("Claude Code missing; run setup --claude or set meta.command")
            env = {
                **{
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith(("ANTHROPIC_", "CLAUDE_", "RSI_"))
                },
                "CLAUDE_CONFIG_DIR": str(root / ".local/claude-doctor"),
                "DISABLE_AUTOUPDATER": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            help_text = subprocess.check_output(
                [executable, "--help"], env=env, text=True, timeout=20
            )
            reported_version = subprocess.check_output(
                [executable, "--version"], env=env, text=True, timeout=20
            ).strip()
            required = ["--bare", "--no-session-persistence", "--strict-mcp-config", "--effort"]
            missing = [flag for flag in required if flag not in help_text]
            pinned = json.loads((root / "requirements/claude-code/package.json").read_text())[
                "dependencies"
            ]["@anthropic-ai/claude-code"]
            # The pinned runtime accepts these flags in real MCP runs, but can hide
            # --strict-mcp-config in help. Missing help text is not a parse failure.
            if missing and reported_version.split()[0] != pinned:
                raise ValueError(
                    "Claude CLI flags differ; install pinned 2.1.169 with setup --claude"
                )
            return {
                "version": reported_version,
                "flags_not_listed_in_help": missing,
                "compatibility": "Pinned runtime contract" if missing else "Required flags visible",
            }

        def credential():
            if not (
                os.environ.get(config.meta.api_key_env)
                or credentials(root).get(config.meta.api_key_env)
            ):
                raise ValueError(
                    f"Missing {config.meta.api_key_env}; run credentials (hidden input)"
                )
            return "Present; value never displayed"

        def endpoint():
            hostname = urlsplit(config.task.base_url).hostname
            key = os.environ.get(config.task.api_key_env) or credentials(root).get(
                config.task.api_key_env
            )
            if not key and hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError(f"Missing Task credential: {config.task.api_key_env}")
            try:
                response = httpx.get(
                    config.task.base_url + "/models",
                    headers={"Authorization": "Bearer " + (key or "EMPTY")},
                    timeout=10,
                    trust_env=hostname not in {"127.0.0.1", "localhost", "::1"},
                )
                response.raise_for_status()
                models = [row["id"] for row in response.json()["data"]]
            except httpx.HTTPError as exc:
                raise ValueError(
                    "Task endpoint unavailable; start serve first (no generation attempted)"
                ) from exc
            if config.task.model not in models:
                raise ValueError(f"Task model is not served: {config.task.model}")
            return models

        check("claude_code", cli)
        check("meta_credential", credential)
        from rsi_base.loop import file_hash

        check(
            "published_data",
            lambda: verify_asset(
                args.data, file_hash(root / "docs/validation/memory-expanded-manifest.json")
            ),
        )
        check("task_endpoint", endpoint)

    if args.local_gpu:

        def gpu():
            interpreter = root / ".venv-task/bin/python"
            if not interpreter.exists():
                raise ValueError("Local GPU environment missing; run setup --gpu")
            script = (
                "import json, torch; from importlib.metadata import version; "
                "assert torch.cuda.is_available(), 'CUDA unavailable: inspect NVIDIA driver'; "
                "assert torch.cuda.is_bf16_supported(), 'Default server requires BF16 support'; "
                "print(json.dumps({'torch':version('torch'),'vllm':version('vllm'),"
                "'cuda':torch.version.cuda,'gpu_count':torch.cuda.device_count(),"
                "'gpu0':torch.cuda.get_device_name(0)}))"
            )
            result = subprocess.run(
                [str(interpreter), "-c", script], capture_output=True, text=True, timeout=60
            )
            if result.returncode:
                raise ValueError("CUDA/BF16 probe failed; inspect driver or run setup --gpu")
            return json.loads(result.stdout)

        def weights():
            directory = Path(
                os.environ.get("RSI_TASK_MODEL_DIR", root / ".local/models/qwen3-4b-instruct-2507")
            )
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            if not shards or any(not (directory / name).is_file() for name in shards):
                raise ValueError("Checkpoint incomplete; run model to resume the pinned download")
            return {"directory": str(directory), "shards": len(shards)}

        check("local_gpu", gpu)
        check("local_weights", weights)
    print(
        json.dumps(
            {"checks": checks, "model_generations": 0, "meta_api_calls": 0},
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(int(any(not row["ok"] for row in checks)))


if __name__ == "__main__":
    main()
