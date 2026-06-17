#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="${1:-${REPO_ROOT}/runs}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

if [ ! -d "${OUTPUTS_DIR}" ]; then
  echo "Missing outputs directory: ${OUTPUTS_DIR}" >&2
  exit 1
fi

run_split() {
  local split="$1"

  python -m scripts.calculate_human_model_feedback \
    --outputs-dir "${OUTPUTS_DIR}" \
    --readability-human-feedback-model-path "${REPO_ROOT}/modernbert_readability_${split}" \
    --interaction-human-feedback-model-path "${REPO_ROOT}/modernbert_interaction_${split}" \
    --hmf-split "${split}"
}

run_split validation
run_split test
