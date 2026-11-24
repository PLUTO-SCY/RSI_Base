# verl 环境排查

本文适用于本仓库的 verl v0.9.1、Python 3.12、PyTorch 2.11.0+cu130 和 vLLM 0.24.0 组合。GRPO 在 `.venvs/rl/` 中运行，最终评测通过 `.venv-task/` 中的 Task 服务完成。下面的命令在仓库根目录执行。 [上游发布版本](https://github.com/verl-project/verl/releases/tag/v0.9.1)

## 上游锁与项目修复

| 现象 | 原因 | 本仓库的处理 |
| --- | --- | --- |
| `uv pip check` 报 `mistral-common` 与 NumPy 不兼容 | v0.9.1 的 `tool.uv.override-dependencies` 用 `numpy>=2.0.0` 覆盖传递约束。原始锁在 Python 3.12 选择 NumPy 2.4.6，而 `mistral-common==1.11.3` 的发布元数据要求 `numpy>=1.25,<2.4` | 同步上游锁时跳过 NumPy，然后从带 wheel 哈希的 [NumPy 兼容锁](../requirements/rl-numpy.lock)安装 2.3.5。 [上游覆盖配置](https://github.com/verl-project/verl/blob/v0.9.1/pyproject.toml#L378-L383) · [包元数据](https://pypi.org/pypi/mistral-common/1.11.3/json) · [uv 覆盖规则](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides) |
| 下载 `flash-attn==2.8.3` wheel 返回 HTTP 404 | v0.9.1 的锁指向 wheelhouse 的发布资产。公开下载地址和对应发布 API 均返回 404，故障发生在下载阶段 | FSDP2 训练使用 PyTorch SDPA，并在安装时跳过独立的 `flash-attn` 包。 [上游锁](https://raw.githubusercontent.com/verl-project/verl/v0.9.1/uv.lock) · [wheelhouse 索引](https://verl-project.github.io/verl-wheelhouse/simple/flash-attn/) |

上游 `fsdp` extra 也会安装 TRL。项目的 GRPO 流程关闭 critic，安装时跳过 TRL。以上处理由 [安装脚本](../scripts/setup_rl.py)执行，上游源码与锁按 [版本文件](../requirements/verl.json)校验。

## 复现上游锁的选择

`setup --rl` 解包上游源码后，可以将安装计划定向到独立路径并查看版本。预演只列出将安装的包，下载还需单独验证。

```sh
rsi_root=$(pwd)
rsi_revision=$(.venv/bin/python -c 'import json; print(json.load(open("requirements/verl.json"))["revision"])')
rsi_diag="$rsi_root/.local/verl/diagnostics"
mkdir -p "$rsi_diag"
(
  cd "$rsi_root/.local/verl/verl-$rsi_revision" || exit 1
  PATH="$rsi_root/.local/bin:$PATH" \
  UV_CACHE_DIR="$rsi_root/.local/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$rsi_root/.local/python" \
  UV_PROJECT_ENVIRONMENT="$rsi_diag/preview-env" \
  uv sync --frozen --dry-run --no-install-project --no-dev \
    --extra fsdp --extra vllm --python 3.12.13
) > "$rsi_diag/upstream-plan.log" 2>&1
grep -E '^[[:space:]]*\+ (numpy|mistral-common|flash-attn|trl|torch|vllm)==' "$rsi_diag/upstream-plan.log"
```

用 Python 3.12 环境中已安装包的元数据核对实际生效的 NumPy 约束。`2.4.6` 在该范围外，兼容锁中的 `2.3.5` 在范围内。

```sh
.venvs/rl/bin/python - <<'PY'
from importlib.metadata import requires, version
from packaging.requirements import Requirement

for raw in requires("mistral-common") or []:
    req = Requirement(raw)
    if req.name == "numpy" and (req.marker is None or req.marker.evaluate()):
        print(f"mistral-common {version('mistral-common')} requires {req}")
        print("Upstream lock", "2.4.6" in req.specifier)
        print("Installed", version("numpy"), version("numpy") in req.specifier)
PY
```

从上游 `uv.lock` 找到与 Python 和平台对应的 wheel URL，再发起 HEAD 请求。这是当前锁中的 Linux x86_64 资产。

```sh
rsi_wheel='https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl'
curl --head --location --silent --show-error --max-time 20 \
  --output /dev/null --write-out 'HTTP %{http_code}\n' "$rsi_wheel"
```

## 检查宿主和修复后的运行环境

安装脚本在最后检查包元数据。安装或解包中断时，检查项目分区和临时目录的空间。训练使用 `/dev/shm` 存放本次 Ray 会话的临时文件。

```sh
mkdir -p .local/verl/diagnostics
sh scripts/harness.sh setup --rl > .local/verl/diagnostics/setup.log 2>&1
PATH="$(pwd)/.local/bin:$PATH" uv pip check --python .venvs/rl/bin/python
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
getconf GNU_LIBC_VERSION
df -h . /tmp /dev/shm
```

选择空闲 GPU 执行实际算子，确认 CUDA、NumPy 转换和 vLLM 扩展能够在同一解释器下工作。这里用 GPU 0 举例。

```sh
CUDA_VISIBLE_DEVICES=0 .venvs/rl/bin/python - <<'PY'
from importlib.metadata import version

import numpy as np
import torch
import vllm._custom_ops

print({name: version(name) for name in ("verl", "torch", "vllm", "numpy")})
print(torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
x = torch.from_numpy(np.ones((1, 2, 32, 32), dtype=np.float32)).to("cuda", torch.bfloat16)
assert (x @ x.transpose(-2, -1)).isfinite().all().item()
assert torch.nn.functional.scaled_dot_product_attention(x, x, x).isfinite().all().item()
print("CUDA BF16 matmul, SDPA and vLLM extension import passed")
PY
```

## 定位训练启动和导出故障

每次训练尝试在 `runs/<run>/attempts/<attempt>/` 保存实际命令 `command.json`、`verl.log` 和可用的 Ray 日志。如果 Hydra 提示无法覆盖配置项，可对实际命令追加 `--cfg job`，生成合成后的配置。把 `RSI_DEBUG_RUN` 设为需要排查的运行目录。

```sh
RSI_DEBUG_RUN=runs/grpo-demo .venv/bin/python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

run = Path(os.environ["RSI_DEBUG_RUN"]).resolve()
command_path = sorted(run.glob("attempts/*/command.json"))[-1]
command = json.loads(command_path.read_text())
with command_path.with_name("hydra-job.yaml").open("w") as output:
    subprocess.run([*command, "--cfg", "job"], cwd=run, stdout=output, check=True)
PY
```

verl v0.9.1 的 `save_lora_only` 在当前 YAML 中通过 `+actor_rollout_ref.actor.checkpoint.save_lora_only=true` 注入。verl v1 的采样日志通过 UID 标识同题组，导出时应与 `prompts_per_batch` 和 `samples_per_prompt` 核对。

更新版本锁后，可按 [LoRA GRPO 流程](lora-grpo.md)验证一次短训练、LoRA 导出、Task 回载和配对评测。保留验证命令、源码哈希及原始运行日志于本地忽略目录。
