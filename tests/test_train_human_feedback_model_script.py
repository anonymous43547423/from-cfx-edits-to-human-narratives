# ruff: noqa: S101, PLR2004
"""Tests for scripts/train_human_feedback_model.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts import train_human_feedback_model as script


def test_parse_args_requires_dataset_paths(monkeypatch: Any) -> None:
    """parse_args exits when required dataset paths are omitted."""
    monkeypatch.setattr("sys.argv", ["train_human_feedback_model.py"])
    with pytest.raises(SystemExit):
        script.parse_args()


def test_parse_args_maps_paths(monkeypatch: Any, tmp_path: Path) -> None:
    """parse_args maps human dataset path arguments to Path attributes."""
    readability_path = tmp_path / "readability.csv"
    interaction_path = tmp_path / "interaction.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_human_feedback_model.py",
            "--readability-human-dataset-path",
            str(readability_path),
            "--interaction-match-human-dataset-path",
            str(interaction_path),
        ],
    )
    args = script.parse_args()
    assert args.readability_human_dataset_path == readability_path
    assert args.interaction_match_human_dataset_path == interaction_path
    assert args.readability_output_dir == script.READABILITY_OUTPUT_DIR
    assert args.interaction_output_dir == script.INTERACTION_OUTPUT_DIR
    assert args.log_level == "INFO"


def test_parse_args_maps_custom_output_dirs(tmp_path: Path) -> None:
    """parse_args maps custom model output directories."""
    readability_path = tmp_path / "readability.csv"
    interaction_path = tmp_path / "interaction.csv"
    readability_output_dir = tmp_path / "modernbert_readability_validation"
    interaction_output_dir = tmp_path / "modernbert_interaction_validation"

    args = script.parse_args(
        [
            "--readability-human-dataset-path",
            str(readability_path),
            "--interaction-match-human-dataset-path",
            str(interaction_path),
            "--readability-output-dir",
            str(readability_output_dir),
            "--interaction-output-dir",
            str(interaction_output_dir),
        ],
    )

    assert args.readability_output_dir == readability_output_dir
    assert args.interaction_output_dir == interaction_output_dir


def test_make_dataset_single_and_pair() -> None:
    """make_dataset tokenizes one or two text columns and includes labels."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }

    single_df = pd.DataFrame(
        {
            "explanation_text": ["hello", "world"],
            "overall": [0, 1],
        },
    )
    single_ds = script.make_dataset(
        single_df,
        ["explanation_text"],
        "overall",
        tokenizer,
    )
    tokenizer.assert_called_with(["hello", "world"], truncation=True)
    assert "label" in single_ds.column_names
    assert single_ds[0]["label"] == 0

    tokenizer.reset_mock()
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }
    pair_df = pd.DataFrame(
        {
            "explanation_text": ["exp a", "exp b"],
            "interaction_description": ["int a", "int b"],
            "score": [1, 0],
        },
    )
    pair_ds = script.make_dataset(
        pair_df,
        ["explanation_text", "interaction_description"],
        "score",
        tokenizer,
    )
    tokenizer.assert_called_with(
        ["exp a", "exp b"],
        ["int a", "int b"],
        truncation=True,
    )
    assert "label" in pair_ds.column_names
    assert pair_ds[1]["label"] == 0


def test_main_drops_rows_with_missing_human_labels(monkeypatch: Any, tmp_path: Path) -> None:
    """Main omits rows where human labels are missing before training."""
    readability_path = tmp_path / "readability.csv"
    interaction_path = tmp_path / "interaction.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_human_feedback_model.py",
            "--readability-human-dataset-path",
            str(readability_path),
            "--interaction-match-human-dataset-path",
            str(interaction_path),
        ],
    )

    readability_df = pd.DataFrame(
        {
            "explanation_text": ["a", "b", "c"],
            "overall": [1, None, 0],
        },
    )
    interaction_df = pd.DataFrame(
        {
            "explanation_text": ["x", "y", "z"],
            "interaction_description": ["i1", "i2", "i3"],
            "score": [None, 1, 0],
        },
    )

    train_calls: list[dict[str, Any]] = []

    def _record_train_model(**kwargs: Any) -> MagicMock:
        train_calls.append(kwargs)
        return MagicMock()

    with (
        patch("scripts.train_human_feedback_model.pd.read_csv", side_effect=[readability_df, interaction_df]),
        patch.object(script, "AutoTokenizer") as mock_tokenizer_cls,
        patch.object(script, "DataCollatorWithPadding"),
        patch.object(script, "train_model", side_effect=_record_train_model),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        assert script.main() == 0

    assert len(train_calls) == 2
    assert train_calls[0]["df"]["overall"].tolist() == [1, 0]
    assert train_calls[1]["df"]["score"].tolist() == [1, 0]


def test_main_trains_both_models(monkeypatch: Any, tmp_path: Path) -> None:
    """Main loads both CSVs and trains readability and interaction classifiers."""
    readability_path = tmp_path / "readability.csv"
    interaction_path = tmp_path / "interaction.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_human_feedback_model.py",
            "--readability-human-dataset-path",
            str(readability_path),
            "--interaction-match-human-dataset-path",
            str(interaction_path),
        ],
    )

    readability_df = pd.DataFrame({"explanation_text": ["a"], "overall": [1]})
    interaction_df = pd.DataFrame(
        {
            "explanation_text": ["b"],
            "interaction_description": ["{year=1990}"],
            "score": [0],
        },
    )

    train_calls: list[dict[str, Any]] = []

    def _record_train_model(**kwargs: Any) -> MagicMock:
        train_calls.append(kwargs)
        return MagicMock()

    with (
        patch("scripts.train_human_feedback_model.pd.read_csv", side_effect=[readability_df, interaction_df]),
        patch.object(script, "AutoTokenizer") as mock_tokenizer_cls,
        patch.object(script, "DataCollatorWithPadding"),
        patch.object(script, "train_model", side_effect=_record_train_model),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        assert script.main() == 0

    assert len(train_calls) == 2

    readability_call = train_calls[0]
    assert readability_call["df"] is readability_df
    assert readability_call["text_cols"] == ["explanation_text"]
    assert readability_call["label_col"] == "overall"
    assert readability_call["output_dir"] == script.READABILITY_OUTPUT_DIR

    interaction_call = train_calls[1]
    assert interaction_call["df"] is interaction_df
    assert interaction_call["text_cols"] == ["explanation_text", "interaction_description"]
    assert interaction_call["label_col"] == "score"
    assert interaction_call["output_dir"] == script.INTERACTION_OUTPUT_DIR


def test_main_uses_custom_output_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    """Main forwards custom model output directories to training."""
    readability_path = tmp_path / "readability.csv"
    interaction_path = tmp_path / "interaction.csv"
    readability_output_dir = tmp_path / "modernbert_readability_test"
    interaction_output_dir = tmp_path / "modernbert_interaction_test"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_human_feedback_model.py",
            "--readability-human-dataset-path",
            str(readability_path),
            "--interaction-match-human-dataset-path",
            str(interaction_path),
            "--readability-output-dir",
            str(readability_output_dir),
            "--interaction-output-dir",
            str(interaction_output_dir),
        ],
    )

    readability_df = pd.DataFrame({"explanation_text": ["a"], "overall": [1]})
    interaction_df = pd.DataFrame(
        {
            "explanation_text": ["b"],
            "interaction_description": ["{year=1990}"],
            "score": [0],
        },
    )

    train_calls: list[dict[str, Any]] = []

    def _record_train_model(**kwargs: Any) -> MagicMock:
        train_calls.append(kwargs)
        return MagicMock()

    with (
        patch("scripts.train_human_feedback_model.pd.read_csv", side_effect=[readability_df, interaction_df]),
        patch.object(script, "AutoTokenizer") as mock_tokenizer_cls,
        patch.object(script, "DataCollatorWithPadding"),
        patch.object(script, "train_model", side_effect=_record_train_model),
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        assert script.main() == 0

    assert train_calls[0]["output_dir"] == readability_output_dir
    assert train_calls[1]["output_dir"] == interaction_output_dir
