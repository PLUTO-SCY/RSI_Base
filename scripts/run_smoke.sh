#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
exec .venv/bin/rsi run --config configs/deepseek_smoke.json \
  --dev data/gsm8k/dev.jsonl --test data/gsm8k/test.jsonl \
  --output "${1:-runs/deepseek-v0.1}" ${RSI_RESUME:+--resume}
