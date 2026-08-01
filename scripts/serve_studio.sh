#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${DRIFTSQL_SKIP_FRONTEND_BUILD:-0}" != "1" ]]; then
  bash "${PROJECT_DIR}/scripts/build_frontend.sh"
fi

exec bash "${PROJECT_DIR}/scripts/serve_service.sh"
