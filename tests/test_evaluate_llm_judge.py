# ruff: noqa: S101, PLR2004
"""Tests for scripts/evaluate_llm_judge.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from recsys_nle.nl_explanations.evaluation.base import EvaluationResult
from scripts import evaluate_llm_judge as script

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_args_requires_mode_and_model(monkeypatch: Any) -> None:
    """parse_args exits when required arguments are omitted."""
    monkeypatch.setattr("sys.argv", ["evaluate_llm_judge.py"])
    with pytest.raises(SystemExit):
        script.parse_args()


def test_parse_args_rejects_both_modes(tmp_path: Path) -> None:
    """parse_args rejects specifying both dataset mode flags."""
    readability_path = tmp_path / "readability-human-labeled.csv"
    interaction_path = tmp_path / "interaction-match-human-labeled.csv"
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--readability-human-dataset-path",
                str(readability_path),
                "--interaction-match-human-dataset-path",
                str(interaction_path),
                "--model-id-evaluation",
                "mistralai/Ministral-8B-Instruct-2410",
            ],
        )


def test_parse_args_maps_readability_paths(tmp_path: Path) -> None:
    """parse_args maps readability dataset path and evaluation model id."""
    dataset_path = tmp_path / "readability-human-labeled.csv"
    args = script.parse_args(
        [
            "--readability-human-dataset-path",
            str(dataset_path),
            "--model-id-evaluation",
            "mistralai/Ministral-8B-Instruct-2410",
            "--evaluation-llm-batch-size",
            "8",
        ],
    )
    assert args.readability_human_dataset_path == dataset_path
    assert args.interaction_match_human_dataset_path is None
    assert args.model_id_evaluation == "mistralai/Ministral-8B-Instruct-2410"
    assert args.evaluation_llm_batch_size == 8
    assert args.output_json_path is None


def test_parse_args_maps_interaction_paths(tmp_path: Path) -> None:
    """parse_args maps interaction-match dataset path and evaluation model id."""
    dataset_path = tmp_path / "interaction-match-human-labeled.csv"
    args = script.parse_args(
        [
            "--interaction-match-human-dataset-path",
            str(dataset_path),
            "--model-id-evaluation",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )
    assert args.interaction_match_human_dataset_path == dataset_path
    assert args.readability_human_dataset_path is None


def test_resolve_judge_mode_readability(tmp_path: Path) -> None:
    """resolve_judge_mode returns readability mode when that flag is set."""
    dataset_path = tmp_path / "readability-human-labeled.csv"
    args = script.parse_args(
        [
            "--readability-human-dataset-path",
            str(dataset_path),
            "--model-id-evaluation",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )
    mode, path = script.resolve_judge_mode(args)
    assert mode is script.JudgeMode.READABILITY
    assert path == dataset_path


def test_resolve_judge_mode_interaction_match(tmp_path: Path) -> None:
    """resolve_judge_mode returns interaction-match mode when that flag is set."""
    dataset_path = tmp_path / "interaction-match-human-labeled.csv"
    args = script.parse_args(
        [
            "--interaction-match-human-dataset-path",
            str(dataset_path),
            "--model-id-evaluation",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )
    mode, path = script.resolve_judge_mode(args)
    assert mode is script.JudgeMode.INTERACTION_MATCH
    assert path == dataset_path


def test_extract_valid_labeled_rows_filters_invalid_overall() -> None:
    """Only rows with binary overall labels are retained in readability mode."""
    df = pd.DataFrame(
        {
            "explanation_text": ["a", "b", "c", "d"],
            "overall": [1, "", 0.5, 0],
        },
    )
    explanations, labels = script.extract_valid_labeled_rows(df, mode=script.JudgeMode.READABILITY)
    assert explanations == ["a", "d"]
    assert labels == [1, 0]


def test_extract_valid_labeled_rows_filters_invalid_score() -> None:
    """Only rows with binary score labels are retained in interaction mode."""
    df = pd.DataFrame(
        {
            "explanation_text": ["a", "b", "c", "d"],
            "interaction_description": ["{year=1996}", "{year=1997}", "{year=1998}", "{year=1999}"],
            "score": [1, "", 0.5, 0],
        },
    )
    records = script.extract_valid_labeled_row_records(df, mode=script.JudgeMode.INTERACTION_MATCH)
    assert len(records) == 2
    assert records[0].explanation_text == "a"
    assert records[0].label == 1
    assert records[0].interaction_description == "{year=1996}"
    assert records[1].label == 0


def test_best_threshold_accuracy_perfect_separation() -> None:
    """Threshold search finds perfect accuracy when scores separate classes."""
    y_true = [0, 0, 1, 1]
    y_score = [0.2, 0.3, 0.8, 0.9]
    result = script.best_threshold_accuracy(y_true, y_score)
    assert result.accuracy == 1.0
    assert result.n_scored == 4
    assert 0.3 < result.best_threshold <= 0.8


def test_best_threshold_accuracy_tiebreak_smallest_threshold() -> None:
    """When multiple thresholds tie, the smallest threshold is chosen."""
    y_true = [0, 0, 1, 1]
    y_score = [0.5, 0.5, 0.5, 0.5]
    result = script.best_threshold_accuracy(y_true, y_score)
    assert result.accuracy == 0.5
    assert result.best_threshold == pytest.approx(-0.5)


def test_best_threshold_accuracy_ignores_nan_scores() -> None:
    """NaN model scores are excluded from metric computation."""
    y_true = [0, 1, 0]
    y_score = [0.1, float("nan"), 0.9]
    result = script.best_threshold_accuracy(y_true, y_score)
    assert result.n_scored == 2
    assert result.accuracy == 1.0


def test_compute_min_overall_score() -> None:
    """Min overall uses the minimum of all six subscores."""
    evaluation = EvaluationResult(
        judgment="ok",
        score=0.75,
        details={
            "fluency": 1.0,
            "grammar": 0.66,
            "length": 1.0,
            "illustrativeness": 1.0,
            "naturalness": 0.33,
            "specificity": 1.0,
            "overall": 0.75,
        },
    )
    assert script.compute_min_overall_score(evaluation) == pytest.approx(0.33)


def test_compute_min_overall_score_nan_when_missing_subscore() -> None:
    """Min overall is NaN when any subscore is missing."""
    evaluation = EvaluationResult(
        judgment="partial",
        score=float("nan"),
        details={
            "fluency": 1.0,
            "grammar": 0.66,
            "length": 1.0,
            "illustrativeness": 1.0,
            "naturalness": 1.0,
            "overall": float("nan"),
        },
    )
    assert pd.isna(script.compute_min_overall_score(evaluation))


def test_compute_threshold_fp_fn() -> None:
    """FP/FN rows are derived from judge-score predictions at the chosen threshold."""
    records = [
        script.ValidLabeledRow(
            row_index=0,
            explanation_text="false positive",
            label=0,
            source={"user_id": 1},
        ),
        script.ValidLabeledRow(
            row_index=1,
            explanation_text="false negative",
            label=1,
            source={"user_id": 2},
        ),
        script.ValidLabeledRow(
            row_index=2,
            explanation_text="true positive",
            label=1,
            source={"user_id": 3},
        ),
    ]
    false_positives, false_negatives = script.compute_threshold_fp_fn(
        records,
        [1.0, 0.5, 1.0],
        1.0,
        label_column="overall",
    )
    assert len(false_positives) == 1
    assert false_positives[0]["explanation_text"] == "false positive"
    assert false_positives[0]["judge_score"] == 1.0
    assert false_positives[0]["prediction"] == 1
    assert false_positives[0]["user_id"] == 1
    assert len(false_negatives) == 1
    assert false_negatives[0]["explanation_text"] == "false negative"
    assert false_negatives[0]["judge_score"] == 0.5
    assert false_negatives[0]["prediction"] == 0


def test_format_classification_table_renders_ascii_table() -> None:
    """Classification rows render as a readable fixed-width table."""
    table = script.format_classification_table(
        [
            {
                "row_index": 0,
                "overall": 0,
                "prediction": 1,
                "judge_score": 1.0,
                "explanation_text": "example explanation",
            },
        ],
        columns=("row_index", "overall", "prediction", "judge_score", "explanation_text"),
    )
    assert "row_index" in table
    assert "explanation_text" in table
    assert "example explanation" in table
    assert "-+-" in table


def test_evaluate_readability_judge_with_mocked_llm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end readability mode computes both mean and min overall accuracies."""
    dataset_path = tmp_path / "readability-human-labeled.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "explanation_text": ["false positive exp", "false negative exp", "true positive exp"],
            "overall": [0, 1, 1],
        },
    ).to_csv(dataset_path, index=False)

    perfect_json = (
        '{"fluency": 1.0, "grammar": 1.0, "length": 1.0, '
        '"illustrativeness": 1.0, "naturalness": 1.0, "specificity": 1.0}'
    )
    mixed_json = (
        '{"fluency": 1.0, "grammar": 0.0, "length": 1.0, '
        '"illustrativeness": 0.0, "naturalness": 1.0, "specificity": 0.0}'
    )

    mock_llm = MagicMock()
    mock_llm.generate_batch.return_value = [perfect_json, mixed_json, perfect_json]

    def _fake_build_llm(_model_id: str) -> MagicMock:
        return mock_llm

    output_path = tmp_path / "report.json"
    with patch.object(script, "_build_evaluation_llm", side_effect=_fake_build_llm):
        code = script.main(
            [
                "--readability-human-dataset-path",
                str(dataset_path),
                "--model-id-evaluation",
                "mistralai/Ministral-8B-Instruct-2410",
                "--output-json-path",
                str(output_path),
            ],
        )

    assert code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["mode"] == "readability"
    assert report["n_rows"] == 3
    assert report["n_valid_rows"] == 3
    assert report["mean_overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["min_overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["mean_overall"]["best_threshold"] == pytest.approx(-0.5)
    assert len(report["mean_overall"]["false_positives"]) == 1
    assert report["mean_overall"]["false_positives"][0]["explanation_text"] == "false positive exp"
    assert report["mean_overall"]["false_negatives"] == []

    captured = capsys.readouterr()
    assert "False positives (mean_overall" in captured.out
    assert "false positive exp" in captured.out
    assert "False negatives (mean_overall" in captured.out
    assert "(none)" in captured.out
    assert "-+-" in captured.out

    mock_llm.generate_batch.assert_called_once()
    call_kwargs = mock_llm.generate_batch.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 256
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["batch_size"] == 4
    mock_llm.close.assert_called_once()


def test_evaluate_interaction_match_judge_with_mocked_llm(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end interaction mode computes score threshold accuracy."""
    dataset_path = tmp_path / "interaction-match-human-labeled.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "explanation_text": ["pattern A", "pattern B", "pattern C"],
            "interaction_description": ["{year=1996}", "{year=1997}", "{year=1998}"],
            "score": [0, 1, 1],
        },
    ).to_csv(dataset_path, index=False)

    mock_llm = MagicMock()
    mock_llm.generate_batch.return_value = [
        '{"judgment": "matches", "score": 1.0}',
        '{"judgment": "partial", "score": 0.33}',
        '{"judgment": "matches", "score": 1.0}',
    ]

    def _fake_build_llm(_model_id: str) -> MagicMock:
        return mock_llm

    output_path = tmp_path / "interaction-report.json"
    with patch.object(script, "_build_evaluation_llm", side_effect=_fake_build_llm):
        code = script.main(
            [
                "--interaction-match-human-dataset-path",
                str(dataset_path),
                "--model-id-evaluation",
                "mistralai/Ministral-8B-Instruct-2410",
                "--output-json-path",
                str(output_path),
            ],
        )

    assert code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["mode"] == "interaction_match"
    assert report["n_rows"] == 3
    assert report["n_valid_rows"] == 3
    assert report["score"]["accuracy"] == pytest.approx(2 / 3)
    assert len(report["score"]["false_positives"]) == 1
    assert report["score"]["false_positives"][0]["explanation_text"] == "pattern A"
    assert report["score"]["false_negatives"] == []

    captured = capsys.readouterr()
    assert "False positives (score" in captured.out
    assert "pattern A" in captured.out
    assert "False negatives (score" in captured.out
    assert "(none)" in captured.out

    mock_llm.generate_batch.assert_called_once()
    messages_batch = mock_llm.generate_batch.call_args.args[0]
    assert len(messages_batch) == 3
    assert "pattern A" in messages_batch[0][1]["content"]
    assert "{year=1996}" in messages_batch[0][1]["content"]
    mock_llm.close.assert_called_once()


def test_main_returns_error_for_missing_dataset(tmp_path: Path) -> None:
    """Main returns exit code 1 when the dataset path is invalid."""
    missing = tmp_path / "missing.csv"
    code = script.main(
        [
            "--readability-human-dataset-path",
            str(missing),
            "--model-id-evaluation",
            "mistralai/Ministral-8B-Instruct-2410",
        ],
    )
    assert code == 1
