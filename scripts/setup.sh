#!/usr/bin/env bash

# Bootstrap a development environment: ensure virtualenv, install deps, run tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: No Python interpreter found in PATH." >&2
    exit 1
  fi
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  VENV_PATH="${REPO_ROOT}/.venv"
  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "Creating virtual environment at ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
else
  echo "Using active virtual environment at ${VIRTUAL_ENV}"
fi

echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e .[dev]

echo "Running test suite..."
pytest
