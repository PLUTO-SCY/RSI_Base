#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
command -v uv >/dev/null 2>&1 || { echo "uv is required; no system packages will be changed." >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$RSI_ROOT/.venv"
export UV_LINK_MODE=copy
uv sync --frozen --python 3.11.15
echo "Project environment ready: $RSI_ROOT/.venv"
