#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/eval_datasets"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/scripts/gather_eval_datasets.py" \
  --data-source-dir "${REPO_ROOT}/runs" \
  --n-readability-samples 64 \
  --n-interaction-match-samples 64 \
  --output-dir "${OUTPUT_DIR}" \
  --best-runs-only true \
  --random-seed 42
