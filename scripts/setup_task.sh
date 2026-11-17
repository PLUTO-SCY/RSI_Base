#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
export PATH="$RSI_ROOT/.local/bin:$PATH"
export UV_CACHE_DIR="$RSI_ROOT/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RSI_ROOT/.local/python"
export UV_LINK_MODE=copy
command -v uv >/dev/null 2>&1 || { echo "Run sh scripts/bootstrap.sh --gpu first." >&2; exit 1; }
if [ ! -x .venv-task/bin/python ]; then
  uv venv --python 3.11.15 .venv-task
fi
.venv-task/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), "Expected Python 3.11.15; use a separate environment rather than overwrite this one"'
uv pip sync --python .venv-task/bin/python requirements/task.lock
