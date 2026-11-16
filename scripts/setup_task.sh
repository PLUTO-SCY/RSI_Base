#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
uv venv --python 3.11.15 .venv-task
uv pip sync --python .venv-task/bin/python requirements/task.lock
