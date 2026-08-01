#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${DRIFTSQL_BASE_PYTHON:-python3.11}"
VENV_DIR="${PROJECT_ROOT}/.venv"

if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Python interpreter not found: ${BASE_PYTHON}" >&2
  echo "Set DRIFTSQL_BASE_PYTHON to a Python 3.10+ environment with CUDA PyTorch, vLLM, and Ray." >&2
  exit 1
fi

"${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"

"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements/training-overlay.txt"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-deps -e "${PROJECT_ROOT}"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-deps -e "${PROJECT_ROOT}/third_party/verl"

echo "Environment ready: ${VENV_DIR}"
echo "Run: ${VENV_DIR}/bin/python scripts/check_environment.py"
