# ruff: noqa: S101
"""Tests for the ACCENT Table 2 LaTeX generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

import pandas as pd
import pytest

from scripts.generate_latex_accent_table2 import (
    ExampleMetrics,
    ExampleRow,
    generate_latex,
    load_example_rows,
    render_latex,
    sample_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

_TEST_SAMPLE_SIZE = 3


class _ExampleInputRow(TypedDict, total=False):
    """Typed input row used by the test feather-writing helper."""

    user_id: int
    explanation_text: str
    cfx_mean: float
    non_cfx_mean: float
    contrast_mean: float
    readability_mean: float
    readability_hmf: float
    cfx_hmf_scores: list[float]
    non_cfx_hmf_scores: list[float]


def _create_trial_dir(
    outputs_dir: Path,
    run_id: str,
    *,
    validation_score: float,
) -> Path:
    """Create one sweep trial directory with the summary files needed for selection."""
    experiment_dir = outputs_dir / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
    timestamp_dir = experiment_dir / "2026-06-02T22:19:50+00:00"
    sweep_dir = timestamp_dir / "sweep"
    trial_dir = sweep_dir / run_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    trials_path = sweep_dir / "trials.json"
    payload = json.loads(trials_path.read_text(encoding="utf-8")) if trials_path.is_file() else {"trials": []}
    payload.setdefault("trials", []).append({"run_id": run_id})
    trials_path.write_text(json.dumps(payload), encoding="utf-8")
    (trial_dir / "run_summary.json").write_text(
        json.dumps({"config": {}, "results": {"reward_composite": 1.0}}),
        encoding="utf-8",
    )
    (trial_dir / "run_human_model_feedback_summary.json").write_text(
        json.dumps(
            {
                "config": {},
                "results": {
                    "validation": {"reward_composite_human_feedback_model": validation_score},
                    "test": {"reward_composite_human_feedback_model": validation_score - 0.1},
                },
            }
        ),
        encoding="utf-8",
    )
    return trial_dir


def _write_required_feathers(run_leaf: Path, rows: list[_ExampleInputRow]) -> None:
    """Write the minimal feather files needed by the ACCENT example-table generator."""
    generation = pd.DataFrame(
        {
            "user_id": [row["user_id"] for row in rows],
            "explanation_text": [row["explanation_text"] for row in rows],
        }
    )
    evaluation_data: dict[str, list[object]] = {
        "user_id": [row["user_id"] for row in rows],
        "explanation_cfx_pattern_match_mean": [row["cfx_mean"] for row in rows],
        "explanation_non_cfx_pattern_match_mean": [row["non_cfx_mean"] for row in rows],
        "readability_overall_mean": [row["readability_mean"] for row in rows],
    }
    if any("contrast_mean" in row for row in rows):
        evaluation_data["explanation_pattern_contrast_mean"] = [row.get("contrast_mean", float("nan")) for row in rows]
    evaluation = pd.DataFrame(evaluation_data)
    readability_hmf = pd.DataFrame(
        {
            "user_id": [row["user_id"] for row in rows],
            "readability_human_feedback_model_score": [row["readability_hmf"] for row in rows],
        }
    )

    cfx_rows: list[dict[str, object]] = []
    non_cfx_rows: list[dict[str, object]] = []
    interaction_id = 1
    for row in rows:
        user_id = row["user_id"]
        for score in row["cfx_hmf_scores"]:
            cfx_rows.append(
                {
                    "interaction_id": interaction_id,
                    "human_feedback_model_score": score,
                    "user_id": user_id,
                }
            )
            interaction_id += 1
        for score in row["non_cfx_hmf_scores"]:
            non_cfx_rows.append(
                {
                    "interaction_id": interaction_id,
                    "human_feedback_model_score": score,
                    "user_id": user_id,
                }
            )
            interaction_id += 1

    generation.to_feather(run_leaf / "generation.feather")
    evaluation.to_feather(run_leaf / "evaluation.feather")
    readability_hmf.to_feather(run_leaf / "evaluation_human_feedback_model.feather")
    pd.DataFrame(cfx_rows).to_feather(run_leaf / "cfx_match_details_human_feedback_model.feather")
    pd.DataFrame(non_cfx_rows).to_feather(run_leaf / "non_cfx_match_details_human_feedback_model.feather")


def test_load_example_rows_uses_informativeness_fallback_and_hmf_means(tmp_path: Path) -> None:
    """Load rows should compute per-example fallback contrast and HMF averages."""
    run_leaf = _create_trial_dir(tmp_path, "wv4br5l7", validation_score=1.5)
    _write_required_feathers(
        run_leaf,
        [
            {
                "user_id": 7,
                "explanation_text": "Because you watched quiet dramas",
                "cfx_mean": 0.9,
                "non_cfx_mean": 0.2,
                "readability_mean": 0.8,
                "readability_hmf": 1.0,
                "cfx_hmf_scores": [0.6, 0.8],
                "non_cfx_hmf_scores": [0.1, 0.3],
            }
        ],
    )

    rows = load_example_rows(run_leaf)

    assert len(rows) == 1
    metrics = rows[0].metrics
    assert metrics.correctness_llm == pytest.approx(0.9)
    assert metrics.correctness_human_calibrated == pytest.approx(0.7)
    assert metrics.informativeness_llm == pytest.approx(0.7)
    assert metrics.informativeness_human_calibrated == pytest.approx(0.5)
    assert metrics.linguistic_quality_llm == pytest.approx(0.8)
    assert metrics.linguistic_quality_human_calibrated == pytest.approx(1.0)


def test_sample_rows_is_reproducible_with_seed() -> None:
    """Sampling should be reproducible when the caller passes the same seed."""
    rows = [
        ExampleRow(
            user_id=user_id,
            explanation_text=f"example {user_id}",
            metrics=ExampleMetrics(
                correctness_llm=0.9,
                correctness_human_calibrated=0.8,
                informativeness_llm=0.7,
                informativeness_human_calibrated=0.6,
                linguistic_quality_llm=0.5,
                linguistic_quality_human_calibrated=0.4,
            ),
        )
        for user_id in range(1, 7)
    ]

    first = sample_rows(rows, sample_size=_TEST_SAMPLE_SIZE, seed=17)
    second = sample_rows(rows, sample_size=_TEST_SAMPLE_SIZE, seed=17)

    assert first == second
    assert len(first) == _TEST_SAMPLE_SIZE
    assert len({row.user_id for row in first}) == _TEST_SAMPLE_SIZE


def test_render_latex_emits_paired_headers() -> None:
    """Rendering should include the paired multi-level headers for the new metrics."""
    latex = render_latex(
        [
            ExampleRow(
                user_id=1,
                explanation_text="Because you watched mysteries",
                metrics=ExampleMetrics(
                    correctness_llm=0.8,
                    correctness_human_calibrated=0.7,
                    informativeness_llm=0.6,
                    informativeness_human_calibrated=0.5,
                    linguistic_quality_llm=0.9,
                    linguistic_quality_human_calibrated=1.0,
                ),
            )
        ]
    )

    assert r"\multicolumn{2}{c}{\textbf{Correctness $\uparrow$}}" in latex
    assert r"\multicolumn{2}{c}{\textbf{Informativeness $\uparrow$}}" in latex
    assert r"\multicolumn{2}{c}{\textbf{Linguistic Quality $\uparrow$}}" in latex
    assert r"\textbf{LLM}" in latex
    assert r"\textbf{H-cal.}" in latex
    assert r"\begin{tabular}{@{}p{6.2cm}cccccc@{}}" in latex
    assert "Each invocation samples five explanations at random." not in latex


def test_generate_latex_uses_validation_best_run_and_escapes_text(tmp_path: Path) -> None:
    """Generation should pick the validation-best trial and escape explanation text."""
    other_run = _create_trial_dir(tmp_path, "aaaa1111", validation_score=0.2)
    selected_run = _create_trial_dir(tmp_path, "wv4br5l7", validation_score=1.5)
    _write_required_feathers(
        other_run,
        [
            {
                "user_id": idx,
                "explanation_text": f"other explanation {idx}",
                "cfx_mean": 0.6,
                "non_cfx_mean": 0.2,
                "contrast_mean": 0.4,
                "readability_mean": 0.7,
                "readability_hmf": 0.8,
                "cfx_hmf_scores": [0.7],
                "non_cfx_hmf_scores": [0.3],
            }
            for idx in range(1, 6)
        ],
    )
    _write_required_feathers(
        selected_run,
        [
            {
                "user_id": idx,
                "explanation_text": ("selected & explanation_100% one" if idx == 1 else f"selected explanation {idx}"),
                "cfx_mean": 0.8,
                "non_cfx_mean": 0.3,
                "contrast_mean": 0.5,
                "readability_mean": 0.9,
                "readability_hmf": 1.0,
                "cfx_hmf_scores": [0.9],
                "non_cfx_hmf_scores": [0.2],
            }
            for idx in range(1, 6)
        ],
    )

    latex = generate_latex(tmp_path, seed=5)

    assert "selected explanation 2" in latex
    assert r"selected \& explanation\_100\% one" in latex
    assert "other explanation 1" not in latex
    assert r"\label{tab:example_readability}" in latex


def test_generate_latex_raises_clear_error_when_feathers_are_missing(tmp_path: Path) -> None:
    """Generation should fail clearly when the selected trial lacks required feathers."""
    _create_trial_dir(tmp_path, "wv4br5l7", validation_score=1.5)

    with pytest.raises(FileNotFoundError) as exc_info:
        generate_latex(tmp_path)

    message = str(exc_info.value)
    assert "Required feather files are missing under" in message
    assert "generation.feather" in message
    assert "evaluation_human_feedback_model.feather" in message
