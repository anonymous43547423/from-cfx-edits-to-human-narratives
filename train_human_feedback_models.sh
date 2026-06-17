#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

train_split() {
  local split="$1"

  python "${REPO_ROOT}/scripts/train_human_feedback_model.py" \
    --readability-human-dataset-path "${REPO_ROOT}/datasets/human-feedback/${split}/readability-human-labeled.csv" \
    --interaction-match-human-dataset-path "${REPO_ROOT}/datasets/human-feedback/${split}/interaction-match-human-labeled.csv" \
    --readability-output-dir "${REPO_ROOT}/modernbert_readability_${split}" \
    --interaction-output-dir "${REPO_ROOT}/modernbert_interaction_${split}"
}

train_split validation
train_split test
