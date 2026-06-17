#!/usr/bin/env bash
# Orchestrate train_a, train_b, DPO, and eval stages.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Log all invocation arguments for reference
# ---------------------------------------------------------------------------
echo "=== Invocation ==="
echo "Script: ${BASH_SOURCE[0]}"
echo "Argument count: $#"
for i in "$@"; do
  printf '  %q\n' "$i"
done
echo "=================="

# ---------------------------------------------------------------------------
# Boolean flags recognised by run_pipeline.py (needed for passthrough shift)
# ---------------------------------------------------------------------------
BOOLEAN_FLAGS=(--show-prompts --disable-reasoning --enable-wandb)

is_boolean_flag() {
  local arg="$1"
  for f in "${BOOLEAN_FLAGS[@]}"; do
    [[ "$arg" == "$f" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# Defaults for explicitly-handled args
# ---------------------------------------------------------------------------

# Train/eval split args (shared across train_a and train_b)
TRAIN_EVALUATION_LLM_BATCH_SIZE=""
EVAL_EVALUATION_LLM_BATCH_SIZE=""
TRAIN_MODEL_ID_EVALUATION=""
EVAL_MODEL_ID_EVALUATION=""
TRAIN_EVALUATION=()
EVAL_EVALUATION=()
TRAIN_SAMPLE_USER_COUNT=""
EVAL_SAMPLE_USER_COUNT=""

# Per-stage split args
TRAIN_A_MODEL_ID_GENERATION=""
TRAIN_B_MODEL_ID_GENERATION=""
DPO_MODEL_ID=""
TRAIN_A_GENERATION_BATCH_SIZE=""
TRAIN_B_GENERATION_BATCH_SIZE=""
EVAL_GENERATION_BATCH_SIZE=""
RANDOM_SEED=""

# DPO-specific args
DPO_REWARD=""
DPO_LEARNING_RATE=""
DPO_BETA=""
DPO_N_EPOCHS=""
DPO_EVAL_DATASET_SPLIT=""
DPO_LORA_R=""

# W&B sweep (optional): train_a/train_b once, then random search over DPO lr, beta, LoRA r
SWEEP=0
SWEEP_TIME_LIMIT_MINUTES=""
SWEEP_DPO_LR_MIN=""
SWEEP_DPO_LR_MAX=""
SWEEP_DPO_BETA_MIN=""
SWEEP_DPO_BETA_MAX=""
SWEEP_LORA_R=()

# Output
OUTPUT_DATASETS_PATH=""

# Log level (forwarded to both pipeline and DPO)
LOG_LEVEL=""

# Holdout partition for the full workflow (validation | test)
TARGET_SET=""

# Passthrough args (forwarded as-is to every run_pipeline.py invocation)
PASSTHROUGH_ARGS=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    # Train/eval split args
    --train-evaluation-llm-batch-size)
      TRAIN_EVALUATION_LLM_BATCH_SIZE="$2"; shift 2 ;;
    --eval-evaluation-llm-batch-size)
      EVAL_EVALUATION_LLM_BATCH_SIZE="$2"; shift 2 ;;
    --train-model-id-evaluation)
      TRAIN_MODEL_ID_EVALUATION="$2"; shift 2 ;;
    --eval-model-id-evaluation)
      EVAL_MODEL_ID_EVALUATION="$2"; shift 2 ;;
    --train-evaluation)
      shift
      TRAIN_EVALUATION=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        TRAIN_EVALUATION+=("$1")
        shift
      done
      ;;
    --eval-evaluation)
      shift
      EVAL_EVALUATION=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EVAL_EVALUATION+=("$1")
        shift
      done
      ;;
    --train-sample-user-count)
      TRAIN_SAMPLE_USER_COUNT="$2"; shift 2 ;;
    --eval-sample-user-count)
      EVAL_SAMPLE_USER_COUNT="$2"; shift 2 ;;
    # Per-stage split args
    --train-a-model-id-generation)
      TRAIN_A_MODEL_ID_GENERATION="$2"; shift 2 ;;
    --train-b-model-id-generation)
      TRAIN_B_MODEL_ID_GENERATION="$2"; shift 2 ;;
    --model-id)
      DPO_MODEL_ID="$2"; shift 2 ;;
    --train-a-generation-batch-size)
      TRAIN_A_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --train-b-generation-batch-size)
      TRAIN_B_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --eval-generation-batch-size)
      EVAL_GENERATION_BATCH_SIZE="$2"; shift 2 ;;
    --random-seed)
      RANDOM_SEED="$2"; shift 2 ;;
    # DPO-specific args
    --dpo-reward)
      DPO_REWARD="$2"; shift 2 ;;
    --dpo-learning-rate)
      DPO_LEARNING_RATE="$2"; shift 2 ;;
    --dpo-beta)
      DPO_BETA="$2"; shift 2 ;;
    --dpo-n-epochs)
      DPO_N_EPOCHS="$2"; shift 2 ;;
    --dpo-eval-dataset-split)
      DPO_EVAL_DATASET_SPLIT="$2"; shift 2 ;;
    --dpo-lora-r)
      DPO_LORA_R="$2"; shift 2 ;;
    --sweep)
      SWEEP=1; shift ;;
    --sweep-time-limit-minutes)
      SWEEP_TIME_LIMIT_MINUTES="$2"; shift 2 ;;
    --sweep-dpo-learning-rate-min)
      SWEEP_DPO_LR_MIN="$2"; shift 2 ;;
    --sweep-dpo-learning-rate-max)
      SWEEP_DPO_LR_MAX="$2"; shift 2 ;;
    --sweep-dpo-beta-min)
      SWEEP_DPO_BETA_MIN="$2"; shift 2 ;;
    --sweep-dpo-beta-max)
      SWEEP_DPO_BETA_MAX="$2"; shift 2 ;;
    --sweep-lora-r)
      shift
      SWEEP_LORA_R=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SWEEP_LORA_R+=("$1")
        shift
      done
      ;;
    # Output
    --output-datasets-path)
      OUTPUT_DATASETS_PATH="$2"; shift 2 ;;
    --target-set)
      TARGET_SET="$2"; shift 2 ;;
    --user-pool)
      echo "error: --user-pool is set per stage by the orchestrator; do not pass it at the top level" >&2
      exit 1
      ;;
    # Log level (forwarded to both pipeline and DPO)
    --log-level)
      LOG_LEVEL="$2"
      PASSTHROUGH_ARGS+=(--log-level "$2")
      shift 2
      ;;
    # Everything else is forwarded to run_pipeline.py as-is
    *)
      PASSTHROUGH_ARGS+=("$1")
      if is_boolean_flag "$1"; then
        shift
      else
        PASSTHROUGH_ARGS+=("$2")
        shift 2
      fi
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Build train-shared args (shared by train_a and train_b)
# ---------------------------------------------------------------------------
TRAIN_SHARED_ARGS=()

[[ -n "${TRAIN_EVALUATION_LLM_BATCH_SIZE}" ]]  && TRAIN_SHARED_ARGS+=(--evaluation-llm-batch-size "${TRAIN_EVALUATION_LLM_BATCH_SIZE}")
[[ -n "${TRAIN_MODEL_ID_EVALUATION}" ]]     && TRAIN_SHARED_ARGS+=(--model-id-evaluation "${TRAIN_MODEL_ID_EVALUATION}")
[[ ${#TRAIN_EVALUATION[@]} -gt 0 ]]            && TRAIN_SHARED_ARGS+=(--evaluation "${TRAIN_EVALUATION[@]}")
[[ -n "${TRAIN_SAMPLE_USER_COUNT}" ]]           && TRAIN_SHARED_ARGS+=(--sample-user-count "${TRAIN_SAMPLE_USER_COUNT}")

# ---------------------------------------------------------------------------
# Build eval-specific args
# ---------------------------------------------------------------------------
EVAL_STAGE_ARGS=()

[[ -n "${EVAL_EVALUATION_LLM_BATCH_SIZE}" ]]  && EVAL_STAGE_ARGS+=(--evaluation-llm-batch-size "${EVAL_EVALUATION_LLM_BATCH_SIZE}")
[[ -n "${EVAL_MODEL_ID_EVALUATION}" ]]     && EVAL_STAGE_ARGS+=(--model-id-evaluation "${EVAL_MODEL_ID_EVALUATION}")
[[ ${#EVAL_EVALUATION[@]} -gt 0 ]]            && EVAL_STAGE_ARGS+=(--evaluation "${EVAL_EVALUATION[@]}")
[[ -n "${EVAL_GENERATION_BATCH_SIZE}" ]]       && EVAL_STAGE_ARGS+=(--generation-batch-size "${EVAL_GENERATION_BATCH_SIZE}")
[[ -n "${EVAL_SAMPLE_USER_COUNT}" ]]           && EVAL_STAGE_ARGS+=(--sample-user-count "${EVAL_SAMPLE_USER_COUNT}")

# ---------------------------------------------------------------------------
# Resolve output base path (create a timestamped subdirectory)
# ---------------------------------------------------------------------------
if [[ -z "${OUTPUT_DATASETS_PATH}" ]]; then
  echo "error: --output-datasets-path is required" >&2
  exit 1
fi
if [[ -z "${DPO_MODEL_ID}" ]] || [[ -z "${DPO_REWARD}" ]] || [[ -z "${DPO_N_EPOCHS}" ]]; then
  echo "error: --model-id, --dpo-reward, and --dpo-n-epochs are required" >&2
  exit 1
fi
if [[ -z "${RANDOM_SEED}" ]]; then
  echo "error: --random-seed is required" >&2
  exit 1
fi
if [[ -z "${TARGET_SET}" ]]; then
  echo "error: --target-set is required (validation or test)" >&2
  exit 1
fi
if [[ "${TARGET_SET}" != "validation" && "${TARGET_SET}" != "test" ]]; then
  echo "error: --target-set must be validation or test" >&2
  exit 1
fi
if [[ "${SWEEP}" -eq 1 ]]; then
  if [[ -z "${SWEEP_TIME_LIMIT_MINUTES}" ]] || [[ -z "${SWEEP_DPO_LR_MIN}" ]] || [[ -z "${SWEEP_DPO_LR_MAX}" ]] \
    || [[ -z "${SWEEP_DPO_BETA_MIN}" ]] || [[ -z "${SWEEP_DPO_BETA_MAX}" ]]; then
    echo "error: with --sweep, --sweep-time-limit-minutes and all --sweep-dpo-*-min/max are required" >&2
    exit 1
  fi
  if [[ ${#SWEEP_LORA_R[@]} -eq 0 ]]; then
    echo "error: with --sweep, --sweep-lora-r with at least one rank is required" >&2
    exit 1
  fi
else
  if [[ -z "${DPO_LEARNING_RATE}" ]] || [[ -z "${DPO_BETA}" ]] || [[ -z "${DPO_LORA_R}" ]]; then
    echo "error: without --sweep, --dpo-learning-rate, --dpo-beta, and --dpo-lora-r are required" >&2
    exit 1
  fi
fi

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
BASE="${OUTPUT_DATASETS_PATH}/${TIMESTAMP}"
mkdir -p "${BASE}"

# ---------------------------------------------------------------------------
# Stage 1: train_a
# ---------------------------------------------------------------------------
echo "=== Stage 1: train_a ==="

TRAIN_A_STAGE_ARGS=()
[[ -n "${TRAIN_A_MODEL_ID_GENERATION}" ]]  && TRAIN_A_STAGE_ARGS+=(--model-id-generation "${TRAIN_A_MODEL_ID_GENERATION}")
[[ -n "${TRAIN_A_GENERATION_BATCH_SIZE}" ]]    && TRAIN_A_STAGE_ARGS+=(--generation-batch-size "${TRAIN_A_GENERATION_BATCH_SIZE}")

python "${REPO_ROOT}/scripts/run_pipeline.py" \
  "${PASSTHROUGH_ARGS[@]}" \
  "${TRAIN_SHARED_ARGS[@]}" \
  "${TRAIN_A_STAGE_ARGS[@]}" \
  --target-set "${TARGET_SET}" \
  --user-pool train \
  --random-seed "${RANDOM_SEED}" \
  --output-datasets-path "${BASE}/train_a" \
  --no-create-output-datasets-subdirectory

# ---------------------------------------------------------------------------
# Stage 2: train_b
# ---------------------------------------------------------------------------
echo "=== Stage 2: train_b ==="

TRAIN_B_STAGE_ARGS=()
[[ -n "${TRAIN_B_MODEL_ID_GENERATION}" ]]  && TRAIN_B_STAGE_ARGS+=(--model-id-generation "${TRAIN_B_MODEL_ID_GENERATION}")
[[ -n "${TRAIN_B_GENERATION_BATCH_SIZE}" ]]    && TRAIN_B_STAGE_ARGS+=(--generation-batch-size "${TRAIN_B_GENERATION_BATCH_SIZE}")

python "${REPO_ROOT}/scripts/run_pipeline.py" \
  "${PASSTHROUGH_ARGS[@]}" \
  "${TRAIN_SHARED_ARGS[@]}" \
  "${TRAIN_B_STAGE_ARGS[@]}" \
  --target-set "${TARGET_SET}" \
  --user-pool train \
  --random-seed "${RANDOM_SEED}" \
  --output-datasets-path "${BASE}/train_b" \
  --no-create-output-datasets-subdirectory

# ---------------------------------------------------------------------------
# Stage 3–4: DPO + eval (single run) or W&B sweep
# ---------------------------------------------------------------------------
if [[ "${SWEEP}" -eq 1 ]]; then
  echo "=== Stages 3+: W&B sweep (DPO + eval per trial) ==="

  SWEEP_DRIVER_ARGS=(
    --base-dir "${BASE}"
    --model-id "${DPO_MODEL_ID}"
    --dpo-reward "${DPO_REWARD}"
    --n-epochs "${DPO_N_EPOCHS}"
    --sweep-time-limit-minutes "${SWEEP_TIME_LIMIT_MINUTES}"
    --sweep-dpo-learning-rate-min "${SWEEP_DPO_LR_MIN}"
    --sweep-dpo-learning-rate-max "${SWEEP_DPO_LR_MAX}"
    --sweep-dpo-beta-min "${SWEEP_DPO_BETA_MIN}"
    --sweep-dpo-beta-max "${SWEEP_DPO_BETA_MAX}"
    --sweep-lora-r "${SWEEP_LORA_R[@]}"
  )
  [[ -n "${DPO_EVAL_DATASET_SPLIT}" ]] && SWEEP_DRIVER_ARGS+=(--eval-dataset-split "${DPO_EVAL_DATASET_SPLIT}")
  [[ -n "${LOG_LEVEL}" ]]              && SWEEP_DRIVER_ARGS+=(--log-level "${LOG_LEVEL}")

  python "${REPO_ROOT}/scripts/run_dpo_eval_sweep.py" \
    "${SWEEP_DRIVER_ARGS[@]}" \
    -- \
    "${PASSTHROUGH_ARGS[@]}" \
    "${EVAL_STAGE_ARGS[@]}" \
    --target-set "${TARGET_SET}" \
    --user-pool eval \
    --random-seed "${RANDOM_SEED}" \
    --no-create-output-datasets-subdirectory
else
  echo "=== Stage 3: DPO training ==="

  DPO_ARGS=(
    --model-id "${DPO_MODEL_ID}"
    --datasets-dir-a "${BASE}/train_a"
    --datasets-dir-b "${BASE}/train_b"
    --output-dir "${BASE}/dpo"
    --reward "${DPO_REWARD}"
    --learning-rate "${DPO_LEARNING_RATE}"
    --beta "${DPO_BETA}"
    --n-epochs "${DPO_N_EPOCHS}"
  )
  [[ -n "${LOG_LEVEL}" ]]              && DPO_ARGS+=(--log-level "${LOG_LEVEL}")
  [[ -n "${DPO_EVAL_DATASET_SPLIT}" ]] && DPO_ARGS+=(--eval-dataset-split "${DPO_EVAL_DATASET_SPLIT}")
  DPO_ARGS+=(--lora-r "${DPO_LORA_R}")

  python "${REPO_ROOT}/scripts/run_dpo.py" "${DPO_ARGS[@]}"

  echo "=== Stage 4: eval ==="

  python "${REPO_ROOT}/scripts/run_pipeline.py" \
    "${PASSTHROUGH_ARGS[@]}" \
    "${EVAL_STAGE_ARGS[@]}" \
    --model-id-generation "${BASE}/dpo/best_model" \
    --target-set "${TARGET_SET}" \
    --user-pool eval \
    --random-seed "${RANDOM_SEED}" \
    --output-datasets-path "${BASE}" \
    --no-create-output-datasets-subdirectory
fi

echo "=== All stages complete ==="
