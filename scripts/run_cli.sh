#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

exec .venv/bin/python -m driftsql.cli.app chat \
  --url "${DRIFTSQL_SERVICE_URL:-http://127.0.0.1:8001}" "$@"
