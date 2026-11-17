#!/bin/sh
# A single GPU by default; torchrun supports explicit DDP workers on this machine.
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
[ -x .venv-train/bin/python ] || { echo "Run sh scripts/harness.sh setup --train first." >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${RSI_TRAIN_GPUS:-0}"
export HF_HOME="$RSI_ROOT/.local/hf"
export XDG_CACHE_HOME="$RSI_ROOT/.local/cache"
export TORCHINDUCTOR_CACHE_DIR="$RSI_ROOT/.local/torch-inductor"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [ "${RSI_TRAIN_PROCESSES:-1}" = 1 ]; then
  exec .venv-train/bin/python -m rsi_base.lora_train "$@"
fi
exec .venv-train/bin/python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node "$RSI_TRAIN_PROCESSES" --module -- rsi_base.lora_train "$@"
