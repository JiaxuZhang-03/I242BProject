#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Create a local Python environment and install project dependencies.

Usage:
  ./setup_environment.sh

Optional environment variables:
  PYTHON_BIN      Python executable used to create the virtual environment. Default: python3
  ENV_DIR         Virtual environment directory. Default: .venv
  CREATE_VENV     Whether to create/use a local venv. Default: 1
                  Set CREATE_VENV=0 to install into the active environment.
  PIP_EXTRA_ARGS  Extra arguments passed to pip install. Default: empty

Examples:
  ./setup_environment.sh
  PYTHON_BIN=/usr/bin/python3 ENV_DIR=.venv ./setup_environment.sh
  CREATE_VENV=0 ./setup_environment.sh
USAGE
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"
CREATE_VENV="${CREATE_VENV:-1}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

if [[ ! -f "requirements.txt" ]]; then
  echo "requirements.txt not found in ${REPO_ROOT}" >&2
  exit 1
fi

if [[ "${CREATE_VENV}" == "1" ]]; then
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment at ${ENV_DIR}"
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
  else
    echo "Using existing virtual environment at ${ENV_DIR}"
  fi
  PYTHON_FOR_INSTALL="${ENV_DIR}/bin/python"
else
  echo "Installing into the active Python environment"
  PYTHON_FOR_INSTALL="${PYTHON_BIN}"
fi

echo "Python executable: ${PYTHON_FOR_INSTALL}"
"${PYTHON_FOR_INSTALL}" -m pip install --upgrade pip
"${PYTHON_FOR_INSTALL}" -m pip install -r requirements.txt ${PIP_EXTRA_ARGS}

echo
echo "Environment setup complete."
if [[ "${CREATE_VENV}" == "1" ]]; then
  echo "Run the full pipeline with:"
  echo "  ./run_full_pipeline.sh"
  echo
  echo "Or activate the environment manually with:"
  echo "  source ${ENV_DIR}/bin/activate"
else
  echo "Run the full pipeline with:"
  echo "  PYTHON_BIN=${PYTHON_FOR_INSTALL} ./run_full_pipeline.sh"
fi
