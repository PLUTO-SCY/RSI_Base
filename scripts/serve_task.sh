#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
export CUDA_VISIBLE_DEVICES="${RSI_GPU:-0}"
export HF_HOME="$RSI_ROOT/.local/hf"
export XDG_CACHE_HOME="$RSI_ROOT/.local/cache"
export TRITON_CACHE_DIR="$RSI_ROOT/.local/triton"
export TORCHINDUCTOR_CACHE_DIR="$RSI_ROOT/.local/torch-inductor"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
exec .venv-task/bin/vllm serve "${RSI_TASK_MODEL_DIR:-.local/models/qwen3-4b-instruct-2507}" \
  --served-model-name "${RSI_TASK_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" \
  --host 127.0.0.1 --port "${RSI_TASK_PORT:-18000}" \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 32 \
  --gpu-memory-utilization 0.65 \
  --seed 42 --disable-log-requests "$@"
