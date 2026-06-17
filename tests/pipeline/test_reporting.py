# ruff: noqa: S101
"""Tests for pipeline reporting helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pytest

from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
from recsys_nle.pipeline.reporting import (
    PipelineReporter,
    _build_correctness_claims_frame,
    _build_correctness_extraction_frame,
    _build_evaluation_summary_frame,
    _build_nle_cf_faithfulness_frame,
    _build_text_frame,
    _collect_user_recommendations,
)


def test_collect_user_recommendations_orders_by_rank() -> None:
    """It sorts recommendations by rank when available."""
    recommendations = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 101, "score": 0.9, "rank": 2},
            {"user_id": 1, "movie_id": 102, "score": 0.95, "rank": 1},
            {"user_id": 2, "movie_id": 201, "score": 0.8, "rank": 1},
        ]
    )

    ordered = _collect_user_recommendations(recommendations, user_id=1, top_k=2)

    assert ordered["rank"].tolist() == [1, 2]
    assert ordered["movie_id"].tolist() == [102, 101]


def test_collect_user_recommendations_uses_score_when_rank_missing() -> None:
    """It falls back to score ordering when rank is missing."""
    recommendations = pd.DataFrame(
        [
            {"user_id": 3, "movie_id": 301, "score": 0.75},
            {"user_id": 3, "movie_id": 302, "score": 0.85},
            {"user_id": 3, "movie_id": 303, "score": 0.65},
        ]
    )

    ordered = _collect_user_recommendations(recommendations, user_id=3, top_k=2)

    assert ordered["movie_id"].tolist() == [302, 301]
    assert ordered["score"].tolist() == [0.85, 0.75]


def test_collect_user_recommendations_returns_empty_for_unknown_user() -> None:
    """It returns an empty frame for users without recommendations."""
    recommendations = pd.DataFrame(
        [
            {"user_id": 4, "movie_id": 401, "score": 0.7, "rank": 1},
        ]
    )

    ordered = _collect_user_recommendations(recommendations, user_id=999, top_k=1)

    assert ordered.empty
    assert list(ordered.columns) == ["user_id", "movie_id", "score", "rank"]


def _build_result_with_prompts() -> NaturalLanguageExplanationResult:
    """Construct a result object populated with prompts for helper tests."""
    plausibility_eval = EvaluationResult(judgment="plausible", score=0.9, prompt="plausibility prompt")

    # CFX match evaluation with both per_claim_scores and per_interaction_scores for testing
    cfx_match_details = {
        "per_claim_scores": [
            {"claim": "Enjoys classic sci-fi", "judgment": "supported", "score": 1.0},
        ],
        "claim_extraction": {
            "claims": ["Enjoys classic sci-fi"],
            "prompt": "extraction prompt text",
        },
        "per_interaction_scores": [
            {"interaction": "1. Sample Movie (id=1)", "judgment": "aligned", "score": 1.0},
        ],
    }
    explanation_cfx_match = EvaluationResult(
        judgment="cfx match",
        score=0.85,
        details=cfx_match_details,
        prompt="cfx match evaluation prompt",
    )

    return NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="Step-by-step reasoning",
        explanation="Because you seek out classic sci-fi adventures",
        reasoning_prompt="reasoning prompt text",
        explanation_prompt="explanation prompt text",
        explanation_plausibility=plausibility_eval,
        explanation_cfx_match=explanation_cfx_match,
        explanation_confidence=0.9,
    )


def test_build_evaluation_summary_frame_includes_prompt_column() -> None:
    """It includes prompt metadata when requested."""
    result = _build_result_with_prompts()

    frame, warnings = _build_evaluation_summary_frame(result, include_prompt=True)

    assert warnings == []
    assert list(frame.columns) == ["metric", "prompt", "score", "judgment"]
    assert frame.loc[frame["metric"] == "explanation_plausibility", "prompt"].iloc[0] == "plausibility prompt"


def test_build_correctness_tables_include_prompts() -> None:
    """It returns prompt-aware correctness and extraction tables."""
    result = _build_result_with_prompts()

    claims_frame, warnings = _build_correctness_claims_frame(result, include_prompt=True)
    extraction_frame = _build_correctness_extraction_frame(result)

    assert warnings == []
    assert list(claims_frame.columns) == ["prompt", "claim", "judgment", "score"]
    assert claims_frame["prompt"].tolist() == ["cfx match evaluation prompt"]
    assert list(extraction_frame.columns) == ["prompt", "claims"]
    assert len(extraction_frame) == 1
    row = extraction_frame.iloc[0]
    assert row["prompt"] == "extraction prompt text"
    assert row["claims"] == "Enjoys classic sci-fi"


def test_build_faithfulness_frame_includes_prompt_column() -> None:
    """It populates prompt values for CFX match rows."""
    result = _build_result_with_prompts()

    frame, warnings = _build_nle_cf_faithfulness_frame(result, include_prompt=True)

    assert warnings == []
    assert list(frame.columns) == ["prompt", "interaction", "judgment", "score"]
    assert frame["prompt"].iloc[0] == "cfx match evaluation prompt"


def test_text_frame_includes_confidence_column() -> None:
    """It builds a reasoning and explanation table that exposes confidence values."""
    result = _build_result_with_prompts()
    reporter = PipelineReporter()
    del reporter  # Reporter is unused; kept for future-proofing the test.

    frame, _columns = _build_text_frame(result, include_prompt=False)

    assert "confidence" in frame.columns

    explanation_row = frame[frame["field"] == "explanation"].iloc[0]
    expected_explanation_confidence = 0.9
    assert float(explanation_row["confidence"]) == expected_explanation_confidence


def test_log_recommendation_samples_limits_to_five_users(monkeypatch: pytest.MonkeyPatch) -> None:
    """It limits detailed output to the first five users."""
    reporter = PipelineReporter(logger=logging.getLogger("test"))
    recorded_users: list[int] = []

    def _record_snapshot(_self: PipelineReporter, *, user_id: int, **_: object) -> None:
        recorded_users.append(user_id)

    monkeypatch.setattr(PipelineReporter, "_log_user_snapshot", _record_snapshot)

    reporter.log_recommendation_samples(
        sampled_users=list(range(20)),
        recommendations=pd.DataFrame(columns=["user_id"]),
        top_k=1,
        user_attributions=None,
        user_nle_results=None,
        show_prompts=False,
    )

    assert recorded_users == list(range(5))


def test_log_recommendation_samples_prioritizes_detail_users(monkeypatch: pytest.MonkeyPatch) -> None:
    """It keeps users with detailed results within the first five."""
    reporter = PipelineReporter(logger=logging.getLogger("test"))
    recorded_users: list[int] = []

    def _record_snapshot(_self: PipelineReporter, *, user_id: int, **_: object) -> None:
        recorded_users.append(user_id)

    monkeypatch.setattr(PipelineReporter, "_log_user_snapshot", _record_snapshot)

    first_result = _build_result_with_prompts()
    first_result.user_id = 100
    second_result = _build_result_with_prompts()
    second_result.user_id = 101

    reporter.log_recommendation_samples(
        sampled_users=list(range(12)),
        recommendations=pd.DataFrame(columns=["user_id"]),
        top_k=1,
        user_attributions=None,
        user_nle_results={100: first_result, 101: second_result},
        show_prompts=False,
    )

    assert recorded_users == [100, 101, 0, 1, 2]
