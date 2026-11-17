#!/bin/sh
# Keep the training stack separate from core and vLLM environments.
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
export PATH="$RSI_ROOT/.local/bin:$PATH"
export UV_CACHE_DIR="$RSI_ROOT/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RSI_ROOT/.local/python"
export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || { echo "Run sh scripts/harness.sh setup --train first." >&2; exit 1; }
if [ ! -x .venv-train/bin/python ]; then
  uv venv --python 3.11.15 .venv-train
fi
.venv-train/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), "Expected Python 3.11.15; use a separate environment"'
uv pip sync --python .venv-train/bin/python requirements/train.lock
