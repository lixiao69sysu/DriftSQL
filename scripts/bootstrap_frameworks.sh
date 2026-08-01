#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${PROJECT_ROOT}/frameworks.lock"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "Missing lock file: ${LOCK_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${LOCK_FILE}"
mkdir -p "${THIRD_PARTY_DIR}"

checkout_pinned() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local target="${THIRD_PARTY_DIR}/${name}"

  if [[ ! -d "${target}/.git" ]]; then
    mkdir -p "${target}"
    git -C "${target}" init
    git -C "${target}" remote add origin "${url}"
  fi

  if [[ "$(git -C "${target}" rev-parse HEAD 2>/dev/null || true)" != "${commit}" ]]; then
    git -C "${target}" fetch --depth 1 origin "${commit}"
    git -C "${target}" checkout --detach FETCH_HEAD
  fi

  local actual
  actual="$(git -C "${target}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "${name}: expected ${commit}, got ${actual}" >&2
    exit 1
  fi
  echo "${name}: ${actual}"
}

checkout_pinned "BIRD-RL" "${BIRD_RL_URL}" "${BIRD_RL_COMMIT}"
checkout_pinned "BIRD-Interact" "${BIRD_INTERACT_URL}" "${BIRD_INTERACT_COMMIT}"
checkout_pinned "verl" "${VERL_URL}" "${VERL_COMMIT}"
checkout_pinned "EvoSchema" "${EVOSCHEMA_URL}" "${EVOSCHEMA_COMMIT}"
