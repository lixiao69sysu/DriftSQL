#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

export DRIFTSQL_TMPDIR="${DRIFTSQL_TMPDIR:-${PROJECT_DIR}/data/tmp/service}"
exec .venv/bin/uvicorn driftsql.service.app:app \
  --host "${DRIFTSQL_SERVICE_HOST:-127.0.0.1}" \
  --port "${DRIFTSQL_SERVICE_PORT:-8001}"
