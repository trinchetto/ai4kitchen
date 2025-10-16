#!/usr/bin/env bash

# Convenience wrapper to run the CLIP mini-training smoke test locally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/mini_train_clip.py" "$@"
