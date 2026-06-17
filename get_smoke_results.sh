#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
fi

mkdir -p "${REPO_ROOT}/runs"

# =====================
# Experiment parameters
# =====================
TOP_K=5
N_CFX_INTERACTIONS=2
N_NON_CFX_INTERACTIONS=2
MIN_CFX_INTERACTIONS=3
MAX_CFX_REMOVALS=50
TARGET_CFX_RANK=5
N_JUDGED_INTERACTIONS=4

EVALUATION_USER_BATCH_SIZE=16

ATTRIBUTION_METHODS=("jaccard")
GENERATION_MODEL_IDS=("mistralai/Ministral-8B-Instruct-2410")

N_SAMPLED_FAITHFULNESS_INTERACTIONS=60
FAITHFULNESS_MATCH_THRESHOLD=0.5
N_FAITHFULNESS_INTERACTIONS_MIN_LIMIT=5
N_FAITHFULNESS_TRIALS=20
N_FAITHFULNESS_SAMPLES=3

N_SAMPLED_DISTANCE_PAIRS=40

TRAIN_EVALUATION_LLM_BATCH_SIZE=64
EVAL_EVALUATION_LLM_BATCH_SIZE=32

TRAIN_MODEL_ID_EVALUATION="Qwen/Qwen3-14B"
EVAL_MODEL_ID_EVALUATION="mistralai/Mistral-Small-3.2-24B-Instruct-2506"

TRAIN_EVALUATION=("cfx_match" "non_cfx_match" "readability")
EVAL_EVALUATION=("cfx_match" "non_cfx_match" "readability")

TRAIN_SAMPLE_USER_COUNT=4
EVAL_SAMPLE_USER_COUNT=4

TRAIN_A_GENERATION_BATCH_SIZE=8
TRAIN_B_GENERATION_BATCH_SIZE=8
EVAL_GENERATION_BATCH_SIZE=8

RANDOM_SEED=42

DPO_REWARD="correctness_informativeness_readability"
DPO_LEARNING_RATE=2e-4
DPO_BETA=0.05
DPO_N_EPOCHS=2
DPO_EVAL_DATASET_SPLIT=0.1
DPO_LORA_R=8

USE_SWEEP="${USE_SWEEP:-0}"
SWEEP_TIME_LIMIT_MINUTES=480
SWEEP_DPO_LR_MIN=5e-7
SWEEP_DPO_LR_MAX=1e-4
SWEEP_DPO_BETA_MIN=0.005
SWEEP_DPO_BETA_MAX=0.2
SWEEP_LORA_R=(8 16)

ENABLE_WANDB="${ENABLE_WANDB:-0}"
COMMON_FLAGS=(--disable-reasoning)

if [[ "${ENABLE_WANDB}" == "1" ]]; then
  COMMON_FLAGS+=(--enable-wandb)
fi

for ATTRIBUTION_METHOD in "${ATTRIBUTION_METHODS[@]}"; do
  for MODEL_ID_GENERATION in "${GENERATION_MODEL_IDS[@]}"; do
    VANILLA_OUTPUT_PATH="${REPO_ROOT}/runs/run_pipeline_${ATTRIBUTION_METHOD}_${MODEL_ID_GENERATION##*/}"
    DPO_OUTPUT_PATH="${REPO_ROOT}/runs/run_eval_eval_dpo_eval_${ATTRIBUTION_METHOD}_${MODEL_ID_GENERATION##*/}_dpo"

    # Pairwise DPO data is generated from the target LLM plus Gemma 3 12B.
    # For Gemma 3 12B itself, use Ministral 8B as the second LLM.
    TRAIN_A_MODEL_ID_GENERATION="${MODEL_ID_GENERATION}"
    TRAIN_B_MODEL_ID_GENERATION="google/gemma-3-12b-it"
    if [[ "${MODEL_ID_GENERATION}" == "google/gemma-3-12b-it" ]]; then
      TRAIN_B_MODEL_ID_GENERATION="mistralai/Ministral-8B-Instruct-2410"
    fi

    # Run the vanilla baseline first, then the corresponding DPO workflow.
    python "${REPO_ROOT}/scripts/run_pipeline.py" \
      --top-k "${TOP_K}" \
      --n-cfx-interactions "${N_CFX_INTERACTIONS}" \
      --n-non-cfx-interactions "${N_NON_CFX_INTERACTIONS}" \
      --min-cfx-interactions "${MIN_CFX_INTERACTIONS}" \
      --max-cfx-removals "${MAX_CFX_REMOVALS}" \
      --target-cfx-rank "${TARGET_CFX_RANK}" \
      --n-judged-interactions "${N_JUDGED_INTERACTIONS}" \
      --generation-batch-size "${EVAL_GENERATION_BATCH_SIZE}" \
      --evaluation-user-batch-size "${EVALUATION_USER_BATCH_SIZE}" \
      --evaluation-llm-batch-size "${EVAL_EVALUATION_LLM_BATCH_SIZE}" \
      --attribution-method "${ATTRIBUTION_METHOD}" \
      --n-sampled-faithfulness-interactions "${N_SAMPLED_FAITHFULNESS_INTERACTIONS}" \
      --faithfulness-match-threshold "${FAITHFULNESS_MATCH_THRESHOLD}" \
      --n-faithfulness-interactions-min-limit "${N_FAITHFULNESS_INTERACTIONS_MIN_LIMIT}" \
      --n-faithfulness-trials "${N_FAITHFULNESS_TRIALS}" \
      --n-faithfulness-samples "${N_FAITHFULNESS_SAMPLES}" \
      --n-sampled-distance-pairs "${N_SAMPLED_DISTANCE_PAIRS}" \
      --model-id-evaluation "${EVAL_MODEL_ID_EVALUATION}" \
      --evaluation "${EVAL_EVALUATION[@]}" \
      --sample-user-count "${EVAL_SAMPLE_USER_COUNT}" \
      --model-id-generation "${MODEL_ID_GENERATION}" \
      --random-seed "${RANDOM_SEED}" \
      --target-set test \
      --user-pool eval \
      "${COMMON_FLAGS[@]}" \
      --output-datasets-path "${VANILLA_OUTPUT_PATH}"

    DPO_ARGS=(
      --top-k "${TOP_K}"
      --n-cfx-interactions "${N_CFX_INTERACTIONS}"
      --n-non-cfx-interactions "${N_NON_CFX_INTERACTIONS}"
      --min-cfx-interactions "${MIN_CFX_INTERACTIONS}"
      --max-cfx-removals "${MAX_CFX_REMOVALS}"
      --target-cfx-rank "${TARGET_CFX_RANK}"
      --n-judged-interactions "${N_JUDGED_INTERACTIONS}"
      --evaluation-user-batch-size "${EVALUATION_USER_BATCH_SIZE}"
      --attribution-method "${ATTRIBUTION_METHOD}"
      --n-sampled-faithfulness-interactions "${N_SAMPLED_FAITHFULNESS_INTERACTIONS}"
      --faithfulness-match-threshold "${FAITHFULNESS_MATCH_THRESHOLD}"
      --n-faithfulness-interactions-min-limit "${N_FAITHFULNESS_INTERACTIONS_MIN_LIMIT}"
      --n-faithfulness-trials "${N_FAITHFULNESS_TRIALS}"
      --n-faithfulness-samples "${N_FAITHFULNESS_SAMPLES}"
      --n-sampled-distance-pairs "${N_SAMPLED_DISTANCE_PAIRS}"
      --train-evaluation-llm-batch-size "${TRAIN_EVALUATION_LLM_BATCH_SIZE}"
      --eval-evaluation-llm-batch-size "${EVAL_EVALUATION_LLM_BATCH_SIZE}"
      --train-model-id-evaluation "${TRAIN_MODEL_ID_EVALUATION}"
      --eval-model-id-evaluation "${EVAL_MODEL_ID_EVALUATION}"
      --train-evaluation "${TRAIN_EVALUATION[@]}"
      --eval-evaluation "${EVAL_EVALUATION[@]}"
      --train-sample-user-count "${TRAIN_SAMPLE_USER_COUNT}"
      --eval-sample-user-count "${EVAL_SAMPLE_USER_COUNT}"
      --train-a-generation-batch-size "${TRAIN_A_GENERATION_BATCH_SIZE}"
      --train-b-generation-batch-size "${TRAIN_B_GENERATION_BATCH_SIZE}"
      --eval-generation-batch-size "${EVAL_GENERATION_BATCH_SIZE}"
      --random-seed "${RANDOM_SEED}"
      --train-a-model-id-generation "${TRAIN_A_MODEL_ID_GENERATION}"
      --train-b-model-id-generation "${TRAIN_B_MODEL_ID_GENERATION}"
      --model-id "${MODEL_ID_GENERATION}"
      --dpo-reward "${DPO_REWARD}"
      --dpo-n-epochs "${DPO_N_EPOCHS}"
      --dpo-eval-dataset-split "${DPO_EVAL_DATASET_SPLIT}"
      --target-set test
      --output-datasets-path "${DPO_OUTPUT_PATH}"
      "${COMMON_FLAGS[@]}"
    )

    if [[ "${USE_SWEEP}" == "1" ]]; then
      DPO_ARGS+=(
        --sweep
        --sweep-time-limit-minutes "${SWEEP_TIME_LIMIT_MINUTES}"
        --sweep-dpo-learning-rate-min "${SWEEP_DPO_LR_MIN}"
        --sweep-dpo-learning-rate-max "${SWEEP_DPO_LR_MAX}"
        --sweep-dpo-beta-min "${SWEEP_DPO_BETA_MIN}"
        --sweep-dpo-beta-max "${SWEEP_DPO_BETA_MAX}"
        --sweep-lora-r "${SWEEP_LORA_R[@]}"
      )
    else
      DPO_ARGS+=(
        --dpo-learning-rate "${DPO_LEARNING_RATE}"
        --dpo-beta "${DPO_BETA}"
        --dpo-lora-r "${DPO_LORA_R}"
      )
    fi

    "${REPO_ROOT}/scripts/run_eval_eval_dpo_eval.sh" "${DPO_ARGS[@]}"
  done
done
