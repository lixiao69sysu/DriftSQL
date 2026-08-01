#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}/frontend"

if [[ ! -x node_modules/.bin/vite ]]; then
  npm ci --no-audit --no-fund
fi
npm run typecheck
npm test
npm run build
