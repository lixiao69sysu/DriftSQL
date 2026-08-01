#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/frameworks.lock"

verify_revision() {
  local name="$1"
  local expected="$2"
  local target="${PROJECT_ROOT}/third_party/${name}"

  if [[ ! -d "${target}/.git" ]]; then
    echo "${name}: missing (${target})" >&2
    return 1
  fi

  local actual
  actual="$(git -C "${target}" rev-parse HEAD)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${name}: expected ${expected}, got ${actual}" >&2
    return 1
  fi
  echo "${name}: ${actual}"
}

verify_revision "BIRD-RL" "${BIRD_RL_COMMIT}"
verify_revision "BIRD-Interact" "${BIRD_INTERACT_COMMIT}"
verify_revision "verl" "${VERL_COMMIT}"
verify_revision "EvoSchema" "${EVOSCHEMA_COMMIT}"
