"""Local Task serving and isolated trainer processes for the joint experiment."""

import contextlib
import os
import signal
import socket
import subprocess
import time

import httpx
from pydantic import Field, field_validator

from .agent import TaskAgent
from .sft import local_key
from .storage import write_json
from .types import ModelSettings, StrictModel
from .weights import load_artifact


class LocalRuntime(StrictModel):
    task_gpu: int = Field(default=0, ge=0)
    sft_gpus: list[int] = Field(default_factory=lambda: [1], min_length=1)
    grpo_gpus: list[int] = Field(default_factory=lambda: [2, 3], min_length=1)
    task_port: int = Field(default=18002, ge=1024, le=65535)
    max_model_len: int = Field(default=4096, ge=128, le=16384)
    max_tokens: int = Field(default=512, ge=32, le=8192)
    request_timeout: float = Field(default=120, gt=0, le=600)
    startup_timeout: float = Field(default=240, gt=0, le=600)

    @field_validator("sft_gpus", "grpo_gpus")
    @classmethod
    def devices(cls, value):
        if len(set(value)) != len(value) or min(value) < 0:
            raise ValueError("GPU indices must be distinct nonnegative integers")
        return value


class ModelState(StrictModel):
    name: str
    sha256: str
    path: str | None = None


def stop_child(process):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class TaskServer:
    def __init__(self, root, output, runtime, model_config):
        self.root, self.output = root, output
        self.runtime, self.model_config = runtime, model_config
        self.process = self.model = None

    def agent(self, model, *, temperature, deadline):
        if self.model != model or self.process.poll() is not None:
            self.stop()
            self.start(model, deadline)
        local_key()
        return TaskAgent(
            ModelSettings(
                model=model.name,
                base_url=f"http://127.0.0.1:{self.runtime.task_port}/v1",
                api_key_env="RSI_LOCAL_API_KEY",
                max_tokens=self.runtime.max_tokens,
                timeout=self.runtime.request_timeout,
                temperature=temperature,
                extra_body={"top_p": 1.0},
            ),
            self.root,
        )

    def start(self, model, deadline):
        port = self.runtime.task_port
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Task port {port} is busy; choose a free task_port") from error
        config = self.model_config
        model_dir = (self.root / config.base_directory).resolve()
        base_name = config.base_model
        command = [
            "sh",
            str(self.root / "scripts/serve_task.sh"),
            "--max-model-len",
            str(self.runtime.max_model_len),
            "--enforce-eager",
        ]
        if model.path:
            artifact = load_artifact(model.path, config.training_mode)
            if (artifact["artifact_sha256"], artifact["served_model"]) != (
                model.sha256,
                model.name,
            ):
                raise ValueError("Task checkpoint differs from the selected artifact")
            if config.training_mode == "full":
                model_dir, base_name = model.path, model.name
            else:
                rank = max(8, 1 << (config.rank - 1).bit_length())
                command += [
                    "--enable-lora",
                    "--max-lora-rank",
                    str(rank),
                    "--lora-modules",
                    f"{model.name}={model.path}",
                ]
        overrides = {
            "RSI_GPU": str(self.runtime.task_gpu),
            "RSI_TASK_PORT": str(port),
            "RSI_TASK_MODEL_DIR": str(model_dir),
            "RSI_TASK_MODEL_NAME": base_name,
        }
        directory = self.output / "task-servers" / str(time.time_ns())
        directory.mkdir(parents=True)
        write_json(directory / "command.json", {"command": command, "environment": overrides})
        log = directory / "server.log"
        with log.open("w") as stream:
            self.process = subprocess.Popen(
                command,
                cwd=self.root,
                env={**os.environ, **overrides},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        startup_deadline = min(deadline, time.time() + self.runtime.startup_timeout)
        with httpx.Client(trust_env=False, timeout=2) as client:
            while time.time() < startup_deadline and self.process.poll() is None:
                with contextlib.suppress(httpx.HTTPError):
                    response = client.get(f"http://127.0.0.1:{port}/v1/models")
                    response.raise_for_status()
                    if model.name in {item["id"] for item in response.json()["data"]}:
                        self.model = model
                        return
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"Task endpoint failed to serve {model.name}. See {log}")

    def stop(self):
        if self.process is not None:
            stop_child(self.process)
        self.process = self.model = None


def train(root, server, method, config, data, output, deadline):
    phase = "sft" if method == "sft" else "rl"
    devices = server.runtime.sft_gpus if method == "sft" else server.runtime.grpo_gpus
    if server.runtime.task_gpu in devices:
        server.stop()
    if deadline <= time.time():
        raise TimeoutError("Pipeline wall-time allowance exhausted before training")
    gpu_list = ",".join(map(str, devices))
    overrides = (
        {"RSI_TRAIN_GPUS": gpu_list, "RSI_TRAIN_PROCESSES": str(len(devices))}
        if method == "sft"
        else {"RSI_RL_GPUS": gpu_list}
    )
    config_path = output / "config.json"
    write_json(config_path, config.model_dump())
    command = [
        "sh",
        str(root / "scripts/harness.sh"),
        phase,
        "train",
        "--config",
        str(config_path),
        "--data",
        str(data),
        "--output",
        str(output),
    ]
    attempt = output / "pipeline-attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    write_json(attempt / "command.json", {"command": command, "environment": overrides})
    log = attempt / "trainer.log"
    started = time.monotonic()
    timeout = deadline - time.time()
    if method == "sft":
        timeout = min(timeout, config.wall_seconds)
    with log.open("w") as stream:
        child = subprocess.Popen(
            command,
            cwd=root,
            env={**os.environ, **overrides},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"{method} training timed out. See {log}") from error
        finally:
            stop_child(child)
            write_json(
                attempt / "result.json",
                {
                    "exit_code": child.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                    "gpus": devices,
                },
            )
    return status, log
