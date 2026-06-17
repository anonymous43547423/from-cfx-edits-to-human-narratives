#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUTS_DIR="${1:-${REPO_ROOT}/runs}"
OUTPUT_TEX_PATH="${2:-${REPO_ROOT}/runs/main_results_table.tex}"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

if [ ! -d "${OUTPUTS_DIR}" ]; then
  echo "Missing outputs directory: ${OUTPUTS_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_TEX_PATH}")"

python "${REPO_ROOT}/scripts/generate_latex_results_table.py" \
  --outputs-dir "${OUTPUTS_DIR}" > "${OUTPUT_TEX_PATH}"

echo "Wrote LaTeX table to ${OUTPUT_TEX_PATH}"
