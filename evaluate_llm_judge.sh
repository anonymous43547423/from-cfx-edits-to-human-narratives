#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

mkdir -p "${REPO_ROOT}/runs"

python "${REPO_ROOT}/scripts/evaluate_llm_judge.py" \
  --readability-human-dataset-path "${REPO_ROOT}/datasets/human-feedback/test/readability-human-labeled.csv" \
  --model-id-evaluation "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  --evaluation-llm-batch-size 32 \
  --output-json-path "${REPO_ROOT}/runs/llm_judge_readability_test.json"
