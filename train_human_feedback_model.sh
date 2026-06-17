#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="${1:-validation}"
DATASET_DIR="${REPO_ROOT}/datasets/human-feedback/${SPLIT}"
READABILITY_OUTPUT_DIR="${REPO_ROOT}/modernbert_readability"
INTERACTION_OUTPUT_DIR="${REPO_ROOT}/modernbert_interaction"
READABILITY_TARGET_DIR="${REPO_ROOT}/modernbert_readability_${SPLIT}"
INTERACTION_TARGET_DIR="${REPO_ROOT}/modernbert_interaction_${SPLIT}"

case "${SPLIT}" in
  validation|test)
    ;;
  *)
    echo "Usage: $0 [validation|test]" >&2
    exit 1
    ;;
esac

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "Missing dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

if [ -e "${READABILITY_OUTPUT_DIR}" ] || [ -e "${INTERACTION_OUTPUT_DIR}" ]; then
  echo "Refusing to overwrite existing default output directories." >&2
  echo "Move or remove ${READABILITY_OUTPUT_DIR} and ${INTERACTION_OUTPUT_DIR} first." >&2
  exit 1
fi

if [ -e "${READABILITY_TARGET_DIR}" ] || [ -e "${INTERACTION_TARGET_DIR}" ]; then
  echo "Refusing to overwrite split-specific output directories." >&2
  echo "Move or remove ${READABILITY_TARGET_DIR} and ${INTERACTION_TARGET_DIR} first." >&2
  exit 1
fi

python "${REPO_ROOT}/scripts/train_human_feedback_model.py" \
  --readability-human-dataset-path "${DATASET_DIR}/readability-human-labeled.csv" \
  --interaction-match-human-dataset-path "${DATASET_DIR}/interaction-match-human-labeled.csv"

mv "${READABILITY_OUTPUT_DIR}" "${READABILITY_TARGET_DIR}"
mv "${INTERACTION_OUTPUT_DIR}" "${INTERACTION_TARGET_DIR}"

echo "Saved ${SPLIT} models to:"
echo "  ${READABILITY_TARGET_DIR}"
echo "  ${INTERACTION_TARGET_DIR}"
