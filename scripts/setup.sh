#!/usr/bin/env bash

# Bootstrap a development environment: ensure virtualenv, install deps, run tests.

SCRIPT_MODE="exec"
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  SCRIPT_MODE="source"
fi

set -euo pipefail

main() {
  if [[ "${SCRIPT_MODE}" == "exec" ]]; then
    echo "ℹ️ Running setup in standalone mode; activate the virtualenv later with 'source .venv/bin/activate' if needed."
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  cd "${REPO_ROOT}"

  PYTHON_BIN="${PYTHON_BIN:-}"
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "🐍 Searching for a Python interpreter..."
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    else
      echo "❌ Error: No Python interpreter found in PATH." >&2
      return 1
    fi
  fi

  VENV_PATH="${REPO_ROOT}/.venv"

  if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${VENV_PATH}" ]]; then
    echo "⚠️  Detected active virtual environment at ${VIRTUAL_ENV}; switching to project venv."
    if command -v deactivate >/dev/null 2>&1; then
      deactivate
    fi
  fi

  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "🛠️  Creating virtual environment at ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  fi

  echo "🧪 Activating virtual environment..."
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  echo "✅ Using virtual environment at ${VIRTUAL_ENV}"

  echo "📦 Installing project dependencies..."
  python -m pip install --upgrade pip
  python -m pip install -e .[dev]

  echo "🧵 Running test suite..."
  pytest

  echo "🎉 Setup complete!"
}

if main "$@"; then
  STATUS=0
else
  STATUS=$?
fi

if [[ "${SCRIPT_MODE}" == "source" ]]; then
  return "${STATUS}"
else
  exit "${STATUS}"
fi
