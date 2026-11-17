#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
export PATH="$RSI_ROOT/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv is required; no system packages will be changed." >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$RSI_ROOT/.venv"
export UV_CACHE_DIR="$RSI_ROOT/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RSI_ROOT/.local/python"
export UV_LINK_MODE=copy
uv sync --frozen --python 3.11.15
echo "Project environment ready: $RSI_ROOT/.venv"
