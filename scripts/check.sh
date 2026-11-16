#!/bin/sh
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
.venv/bin/ruff check --config pyproject.toml src scripts tests
.venv/bin/ruff format --config pyproject.toml --check src scripts tests
.venv/bin/pytest
.venv/bin/python scripts/scan_secrets.py
