"""Direct Preference Optimization (DPO) training script using TRL."""

# ruff: noqa: E402

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from recsys_nle.cuda_utils import enable_expandable_segments

enable_expandable_segments()

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.tokenization_mistral_common import MistralCommonBackend
from trl import DPOConfig, DPOTrainer  # type: ignore[attr-defined]

from datasets import Dataset  # type: ignore[attr-defined]
from recsys_nle.pipeline.reward import REWARD_TERMS, RewardType, compute_reward

if TYPE_CHECKING:
    from pandas import DataFrame
    from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


class _DPOMistralTokenizer(MistralCommonBackend):
    """MistralCommonBackend that auto-sets ``continue_final_message`` for DPO conversations."""

    def apply_chat_template(  # type: ignore[override]
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        *,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Pass ``continue_final_message=True`` when the conversation ends with an assistant turn."""
        if (
            not continue_final_message
            and conversation
            and isinstance(conversation[-1], dict)
            and conversation[-1].get("role") == "assistant"
        ):
            continue_final_message = True
        return super().apply_chat_template(
            conversation,
            continue_final_message=continue_final_message,
            **kwargs,
        )


class _DPOTrainerWithEvalMetrics(DPOTrainer):
    """DPOTrainer whose ``log()`` mutates the incoming dict so ``evaluate()`` returns DPO metrics."""

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Merge DPO metrics in-place so ``Trainer.evaluate`` exposes them for best-metric selection."""
        mode = "train" if self.model is not None and self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs.update(metrics)
        super(DPOTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()


def _calculate_score(row: pd.Series, reward_type: RewardType, suffix: str) -> float:
    """Compute the reward as a linear combination of suffixed evaluation columns from ``row``."""
    values = {column: float(row[f"{column}{suffix}"]) for column in REWARD_TERMS[reward_type]}
    return compute_reward(values, reward_type)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPO training."""
    parser = argparse.ArgumentParser(
        description="Train a DPO model using two dataset directories.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model identifier for the base LLM.",
    )
    parser.add_argument(
        "--datasets-dir-a",
        type=Path,
        required=True,
        help="Path to the first datasets directory.",
    )
    parser.add_argument(
        "--datasets-dir-b",
        type=Path,
        required=True,
        help="Path to the second datasets directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for training checkpoints and the final best model.",
    )
    parser.add_argument(
        "--reward",
        choices=[reward.value for reward in RewardType],
        required=True,
        help="Reward type to use for score calculation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Learning rate for DPO training.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="DPO beta parameter controlling the KL penalty strength.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--eval-dataset-split",
        type=float,
        default=None,
        help=("Optional fraction of training pairs to reserve for evaluation (e.g. 0.1 for 10%%)."),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        required=True,
        help="LoRA rank r; lora_alpha is set to r*2.",
    )
    return parser.parse_args()


def _load_dataset_dir(dataset_dir: Path) -> DataFrame:
    """Load generation and evaluation datasets from a directory."""
    generation_path = dataset_dir / "generation.feather"
    evaluation_path = dataset_dir / "evaluation.feather"

    if not generation_path.exists():
        msg = f"Missing generation dataset at {generation_path}"
        raise FileNotFoundError(msg)
    if not evaluation_path.exists():
        msg = f"Missing evaluation dataset at {evaluation_path}"
        raise FileNotFoundError(msg)

    generation = pd.read_feather(
        generation_path,
        columns=["user_id", "explanation_conversation"],
    ).rename(columns={"explanation_conversation": "conversation"})
    evaluation = pd.read_feather(
        evaluation_path,
        columns=[
            "user_id",
            "explanation_cfx_pattern_match_mean",
            "explanation_non_cfx_pattern_match_mean",
            "readability_overall_mean",
            "faithfulness_removal_pvalue_complement",
            "faithfulness_replacement_pvalue_complement",
        ],
    )

    merged = generation.merge(evaluation, on="user_id", how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        missing = merged[merged["_merge"] != "both"]
        msg = (
            f"Mismatch between generation and evaluation datasets: found {len(missing)} rows "
            f"that do not exist in both datasets. Example: {missing.iloc[0].to_dict()}"
        )
        raise ValueError(msg)

    return merged.drop(columns=["_merge"])


def _parse_conversation(conversation_json: str) -> list[dict[str, str]]:
    """Parse the conversation JSON string into a list of message dictionaries."""
    try:
        messages = json.loads(conversation_json)
    except (json.JSONDecodeError, ValueError) as e:
        msg = f"Failed to parse conversation JSON: {e}"
        raise ValueError(msg) from e

    if not isinstance(messages, list):
        msg = "Parsed JSON is not a list"
        raise TypeError(msg)

    return messages


def _validate_and_select_pairs(df_a: DataFrame, df_b: DataFrame, reward_type: RewardType) -> list[dict[str, Any]]:
    """Match rows, validate conversations, and select chosen/rejected pairs."""
    merged = df_a.merge(df_b, on=["user_id"], how="outer", suffixes=("_a", "_b"), indicator=True)

    if (merged["_merge"] != "both").any():
        missing = merged[merged["_merge"] != "both"]
        msg = (
            f"Mismatch in datasets: found {len(missing)} rows that do not exist in both datasets. "
            f"Example: {missing.iloc[0].to_dict()}"
        )
        raise ValueError(msg)

    training_data = []
    wins_a = 0
    wins_b = 0
    skipped_missing = 0

    required_columns = [f"{column}{suffix}" for column in REWARD_TERMS[reward_type] for suffix in ("_a", "_b")]

    for _, row in merged.iterrows():
        if row[required_columns].isna().any():
            LOGGER.warning(
                "Skipping user_id=%s: missing score columns %s",
                row["user_id"],
                [c for c in required_columns if pd.isna(row[c])],
            )
            skipped_missing += 1
            continue

        score_a = _calculate_score(row, reward_type, "_a")
        score_b = _calculate_score(row, reward_type, "_b")

        conv_a_json = row["conversation_a"]
        conv_b_json = row["conversation_b"]

        if pd.isna(conv_a_json) or pd.isna(conv_b_json):
            msg = f"Missing conversation data for user_id={row['user_id']}"
            raise ValueError(msg)

        msgs_a = _parse_conversation(str(conv_a_json))
        msgs_b = _parse_conversation(str(conv_b_json))

        inputs_a = msgs_a[:-1]
        inputs_b = msgs_b[:-1]

        if inputs_a != inputs_b:
            msg = (
                f"Input messages differ for user_id={row['user_id']}.\n"
                f"A inputs: {json.dumps(inputs_a)}\n"
                f"B inputs: {json.dumps(inputs_b)}"
            )
            raise ValueError(msg)

        if score_a == score_b:
            continue

        if score_a > score_b:
            chosen = msgs_a
            rejected = msgs_b
            score_chosen = score_a
            score_rejected = score_b
            wins_a += 1
        else:
            chosen = msgs_b
            rejected = msgs_a
            score_chosen = score_b
            score_rejected = score_a
            wins_b += 1

        training_data.append(
            {
                "chosen": chosen,
                "rejected": rejected,
                "score_chosen": score_chosen,
                "score_rejected": score_rejected,
            }
        )

    LOGGER.info(
        "Pairwise winners: %d from dataset A, %d from dataset B (out of %d pairs); "
        "skipped %d pairs with missing scores.",
        wins_a,
        wins_b,
        wins_a + wins_b,
        skipped_missing,
    )
    return training_data


def _load_and_prepare_model(model_id: str) -> Any:
    """Load the causal LM and configure it for PEFT + gradient checkpointing."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    # Required so gradients flow into LoRA adapters when the PEFT-frozen base
    # is combined with gradient checkpointing; without it the checkpointed
    # blocks are treated as non-differentiable and ``lora_B`` stays at its
    # zero init (especially visible for multimodal wrappers like Gemma 3).
    model.enable_input_require_grads()  # type: ignore[no-untyped-call]
    model.config.use_cache = False
    return model


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer for a model, handling Mistral, multimodal, and text-only models.

    Mistral models use ``_DPOMistralTokenizer`` (a ``MistralCommonBackend``
    subclass) so that conversations ending with an assistant turn are accepted
    during DPO tokenization.

    Other models try ``AutoProcessor`` first so that multimodal models (e.g.
    Gemma 3) whose tokenizer lives inside a processor are handled correctly.
    Falls back to ``AutoTokenizer`` when ``AutoProcessor`` is unavailable.
    """
    if model_id.startswith("mistralai/"):
        tokenizer: PreTrainedTokenizerBase = _DPOMistralTokenizer.from_pretrained(model_id)
        LOGGER.info("Loaded _DPOMistralTokenizer for model %s.", model_id)
        return tokenizer

    try:
        result = AutoProcessor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
    except (KeyError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        LOGGER.info("Loaded tokenizer (%s) for model %s.", type(tokenizer).__name__, model_id)
        return tokenizer
    else:
        if hasattr(result, "tokenizer"):
            LOGGER.info("Extracted tokenizer from processor (%s) for model %s.", type(result).__name__, model_id)
            return result.tokenizer  # type: ignore[no-any-return]
        LOGGER.info("AutoProcessor returned tokenizer (%s) for model %s.", type(result).__name__, model_id)
        return result  # type: ignore[no-any-return]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    LOGGER.info("Loading datasets...")
    df_a = _load_dataset_dir(args.datasets_dir_a)
    df_b = _load_dataset_dir(args.datasets_dir_b)
    reward_type = RewardType(args.reward)

    LOGGER.info("Processing and validating pairs...")
    try:
        data_records = _validate_and_select_pairs(df_a, df_b, reward_type)
    except Exception:
        LOGGER.exception("Failed to process datasets")
        raise

    if not data_records:
        LOGGER.warning("No valid training pairs found (all scores might be equal).")
        return 0

    LOGGER.info("Found %d training pairs.", len(data_records))
    train_dataset = Dataset.from_list(data_records)
    eval_dataset = None
    if args.eval_dataset_split is not None:
        split = train_dataset.train_test_split(test_size=args.eval_dataset_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        LOGGER.info("Split data into %d train and %d eval pairs.", len(train_dataset), len(eval_dataset))

    LOGGER.info("Loading model and tokenizer...")
    model = _load_and_prepare_model(args.model_id)
    tokenizer = load_tokenizer(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        beta=args.beta,
        warmup_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        logging_steps=10,
        bf16=True,
        report_to="none",
        load_best_model_at_end=False,
        metric_for_best_model="eval_rewards/margins",
        greater_is_better=True,
        precompute_ref_log_probs=True,
    )

    # Multimodal models (e.g. Gemma 3) advertise a vision model_type which
    # causes TRL to route through the vision processing path (expecting an
    # "images" column).  Switch to the text sub-model type so TRL uses the
    # text-only tokenization path instead.
    if hasattr(model.config, "text_config"):
        model.config.model_type = model.config.text_config.model_type

    trainer = _DPOTrainerWithEvalMetrics(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    LOGGER.info("Starting DPO training...")
    trainer.train()

    best_ckpt: str | None = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        msg = (
            "No best checkpoint was recorded by the trainer. "
            "Ensure an eval dataset is provided and eval_strategy is configured."
        )
        raise RuntimeError(msg)

    best_model_dir = args.output_dir / "best_model"
    LOGGER.info("Copying best checkpoint %s to %s ...", best_ckpt, best_model_dir)
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(best_ckpt, best_model_dir)

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
