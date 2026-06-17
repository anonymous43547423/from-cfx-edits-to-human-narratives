"""Train ModernBERT classifiers on human-labeled readability and interaction-match data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas import DataFrame
    from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "answerdotai/ModernBERT-large"
RANDOM_STATE = 42
READABILITY_OUTPUT_DIR = Path("modernbert_readability")
INTERACTION_OUTPUT_DIR = Path("modernbert_interaction")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for human-feedback classifier training."""
    parser = argparse.ArgumentParser(
        description="Train ModernBERT classifiers on human-labeled feedback datasets.",
    )
    parser.add_argument(
        "--readability-human-dataset-path",
        type=Path,
        required=True,
        help="Path to the human-labeled readability CSV.",
    )
    parser.add_argument(
        "--interaction-match-human-dataset-path",
        type=Path,
        required=True,
        help="Path to the human-labeled interaction-match CSV.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args()


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute classification accuracy from trainer evaluation predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def drop_unlabeled_rows(df: DataFrame, label_col: str, dataset_name: str) -> DataFrame:
    """Drop rows with missing labels and log how many rows were omitted."""
    labeled_mask = df[label_col].notna()
    dropped = int((~labeled_mask).sum())
    if dropped == 0:
        return df
    kept = int(labeled_mask.sum())
    LOGGER.info(
        "Dropping %s unlabeled rows from %s (%s labeled rows kept).",
        dropped,
        dataset_name,
        kept,
    )
    return df.loc[labeled_mask].copy()


def make_dataset(
    df: DataFrame,
    text_cols: Sequence[str],
    label_col: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Tokenize dataframe text columns and build a torch-formatted Hugging Face dataset."""
    texts1 = df[text_cols[0]].fillna("").astype(str).tolist()

    if len(text_cols) == 1:
        encodings = tokenizer(
            texts1,
            truncation=True,
        )
    else:
        texts2 = df[text_cols[1]].fillna("").astype(str).tolist()
        encodings = tokenizer(
            texts1,
            texts2,
            truncation=True,
        )

    dataset = Dataset.from_dict(
        {
            **encodings,
            "label": df[label_col].astype(int).tolist(),
        },
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    return dataset


def train_model(
    df: DataFrame,
    text_cols: Sequence[str],
    label_col: str,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    collator: DataCollatorWithPadding,
) -> Trainer:
    """Train a sequence classifier and save the best model to ``output_dir``."""
    train_df, test_df = train_test_split(
        df,
        test_size=100,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )

    train_ds = make_dataset(train_df, text_cols, label_col, tokenizer)
    test_ds = make_dataset(test_df, text_cols, label_col, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.2,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    train_metrics = trainer.predict(train_ds).metrics
    val_metrics = trainer.predict(test_ds).metrics

    LOGGER.info("%s train accuracy: %s", output_dir, train_metrics["test_accuracy"])
    LOGGER.info("%s validation accuracy: %s", output_dir, val_metrics["test_accuracy"])

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    return trainer


def main() -> int:
    """CLI entry point for training human-feedback classifiers."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    readability_df = pd.read_csv(args.readability_human_dataset_path)
    interaction_df = pd.read_csv(args.interaction_match_human_dataset_path)
    readability_df = drop_unlabeled_rows(readability_df, "overall", "readability dataset")
    interaction_df = drop_unlabeled_rows(interaction_df, "score", "interaction-match dataset")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)

    LOGGER.info("Training readability classifier -> %s", READABILITY_OUTPUT_DIR)
    train_model(
        df=readability_df,
        text_cols=["explanation_text"],
        label_col="overall",
        output_dir=READABILITY_OUTPUT_DIR,
        tokenizer=tokenizer,
        collator=collator,
    )

    LOGGER.info("Training interaction-match classifier -> %s", INTERACTION_OUTPUT_DIR)
    train_model(
        df=interaction_df,
        text_cols=["explanation_text", "interaction_description"],
        label_col="score",
        output_dir=INTERACTION_OUTPUT_DIR,
        tokenizer=tokenizer,
        collator=collator,
    )

    LOGGER.info("Saved models to %s and %s", READABILITY_OUTPUT_DIR, INTERACTION_OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
