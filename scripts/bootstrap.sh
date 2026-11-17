#!/bin/sh
# Install only project-local tools/environments; leave shell profiles untouched.
set -eu
RSI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RSI_ROOT"
RSI_WITH_GPU=0
RSI_WITH_CLAUDE=0
RSI_WITH_SANDBOX=0
for RSI_OPTION in "$@"; do
  case "$RSI_OPTION" in
    --gpu) RSI_WITH_GPU=1 ;;
    --claude) RSI_WITH_CLAUDE=1 ;;
    --sandbox) RSI_WITH_SANDBOX=1 ;;
    --help|-h)
      echo "Usage: sh scripts/bootstrap.sh [--gpu] [--claude] [--sandbox]"
      echo "Core Python environment is always installed. GPU and Claude tools are optional."
      exit 0 ;;
    *) echo "Unknown option: $RSI_OPTION" >&2; exit 2 ;;
  esac
done
export PATH="$RSI_ROOT/.local/bin:$RSI_ROOT/.local/claude-cli/node_modules/.bin:$PATH"
export UV_CACHE_DIR="$RSI_ROOT/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RSI_ROOT/.local/python"
export UV_LINK_MODE=copy
if ! command -v uv >/dev/null 2>&1 || [ "$(uv --version | cut -d ' ' -f 2)" != "0.11.17" ]; then
  command -v curl >/dev/null 2>&1 || { echo "Install curl first." >&2; exit 1; }
  mkdir -p .local/installers .local/bin
  curl --fail --location --silent --show-error https://astral.sh/uv/0.11.17/install.sh \
    --output .local/installers/uv-0.11.17.sh
  UV_UNMANAGED_INSTALL="$RSI_ROOT/.local/bin" UV_NO_MODIFY_PATH=1 sh .local/installers/uv-0.11.17.sh
fi
sh scripts/setup.sh
if [ "$RSI_WITH_SANDBOX" = 1 ]; then
  .venv/bin/python scripts/setup_sandbox.py
fi
if [ "$RSI_WITH_CLAUDE" = 1 ]; then
  command -v npm >/dev/null 2>&1 || { echo "Install Node.js >=18 and npm, or supply an existing Claude Code CLI." >&2; exit 1; }
  node -e 'if (Number(process.versions.node.split(".")[0]) < 18) process.exit(1)' || {
    echo "Pinned Claude Code 2.1.169 requires Node.js >=18 for npm installation." >&2; exit 1;
  }
  mkdir -p .local/claude-cli
  cp requirements/claude-code/package.json requirements/claude-code/package-lock.json .local/claude-cli/
  npm ci --prefix .local/claude-cli --cache .local/npm-cache --no-audit --no-fund
  CLAUDE_CONFIG_DIR="$RSI_ROOT/.local/claude-bootstrap" DISABLE_AUTOUPDATER=1 \
    .local/claude-cli/node_modules/.bin/claude --version
fi
if [ "$RSI_WITH_GPU" = 1 ]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "Local serving requires an installed NVIDIA driver (nvidia-smi)." >&2; exit 1; }
  sh scripts/setup_task.sh
fi
echo "Ready. Next: sh scripts/harness.sh data; sh scripts/harness.sh doctor"
