#!/bin/sh
# Current harness workflow, with local tools visible only to this command/children.
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
export PATH="$RSI_ROOT/.local/bin:$RSI_ROOT/.local/claude-cli/node_modules/.bin:$PATH"
export UV_CACHE_DIR="$RSI_ROOT/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RSI_ROOT/.local/python"
export HF_HOME="$RSI_ROOT/.local/hf"
export XDG_CACHE_HOME="$RSI_ROOT/.local/cache"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
RSI_COMMAND=${1:---help}
if [ "$#" -gt 0 ]; then shift; fi
case "$RSI_COMMAND" in
  --help|-h|help)
    echo "Usage: sh scripts/harness.sh COMMAND [arguments]"
    echo "  setup [--gpu] [--train] [--rl] [--claude] [--sandbox]  Install project-local environments/tools"
    echo "  data                      Rebuild/verify pinned public data (idempotent)"
    echo "  model [arguments]         Download/resume the pinned Qwen checkpoint"
    echo "  credentials               Save a hidden project-local API credential"
    echo "  doctor [--ready] [--local-gpu]  Check prerequisites; no model generation"
    echo "  serve [vLLM arguments]    Serve on RSI_GPU/RSI_TASK_PORT (foreground)"
    echo "  smoke [--meta]            Two validation queries; optional ONE Meta call"
    echo "  run PHASE [arguments]    calibrate/search/suite/finalize; explicit full run"
    echo "  sft PHASE [arguments]    collect/train/evaluate/test for LoRA model updates"
    echo "  rl PHASE [arguments]     train/evaluate/test for LoRA GRPO on a fixed Task prompt"
    echo "  serve-adapter [--run DIR]  Serve base + immutable LoRA on port 18001"
    echo "  runtime [arguments]      Save local model/runtime fingerprints"
    echo "  inspect [suite]          Print experiment progress"
    echo "  report [arguments]       Export aggregate JSON/PNG/SVG"
    echo "  check                     Run offline checks"
    exit 0 ;;
  setup) exec sh scripts/bootstrap.sh "$@" ;;
  serve) exec sh scripts/serve_task.sh "$@" ;;
esac
[ -x .venv/bin/python ] || { echo "Run sh scripts/harness.sh setup first." >&2; exit 1; }
case "$RSI_COMMAND" in
  data) exec .venv/bin/python scripts/prepare_data.py "$@" ;;
  model)
    [ -x .venv-task/bin/python ] || { echo "Run setup --gpu first (or use an existing remote Task endpoint)." >&2; exit 1; }
    exec .venv-task/bin/python scripts/download_task.py "$@" ;;
  credentials) exec .venv/bin/python scripts/configure_credentials.py "$@" ;;
  doctor) exec .venv/bin/python scripts/doctor.py "$@" ;;
  smoke) exec .venv/bin/python scripts/smoke_harness.py "$@" ;;
  run) exec .venv/bin/python scripts/run_memory.py "$@" ;;
  sft) exec .venv/bin/python scripts/run_sft.py "$@" ;;
  rl) exec .venv/bin/python scripts/run_rl.py "$@" ;;
  serve-adapter) exec .venv/bin/python scripts/serve_lora.py "$@" ;;
  runtime)
    [ -x .venv-task/bin/python ] || { echo "Run setup --gpu first." >&2; exit 1; }
    exec .venv-task/bin/python scripts/record_task_runtime.py --output .local/task-runtime-memory.json "$@" ;;
  inspect) exec .venv/bin/python scripts/inspect_memory.py "$@" ;;
  report) exec .venv/bin/python scripts/export_memory.py "$@" ;;
  check) exec sh scripts/check.sh "$@" ;;
  *) echo "Unknown command: $RSI_COMMAND. Run sh scripts/harness.sh --help." >&2; exit 2 ;;
esac
