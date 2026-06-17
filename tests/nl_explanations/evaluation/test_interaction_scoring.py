# ruff: noqa: S101, PLR2004
"""Unit tests for interaction-based explanation evaluation metrics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from recsys_nle.nl_explanations.evaluation.interaction_scoring import (
    InteractionScoringEvaluator,
    build_interaction_descriptions,
    build_single_interaction_scoring_messages,
)


class _TestEvaluator(InteractionScoringEvaluator):
    """Concrete test implementation of InteractionScoringEvaluator."""


def test_build_interaction_descriptions_creates_concise_lines() -> None:
    """It serialises interaction payloads into descriptions with key attributes."""
    items = [
        {"movie_title": "Inception", "movie_id": 1, "rating": 5},
        {"title": "The Matrix", "weight": 0.8},
        {"name": "Interstellar", "importance": 0.9},
        {},
    ]
    descriptions = build_interaction_descriptions(items)
    expected_count = 4
    assert len(descriptions) == expected_count
    assert descriptions[0] == "{}"
    assert "movie_id" not in descriptions[0]
    assert "rating=5" not in descriptions[0]

    assert descriptions[1] == "{}"
    assert "weight=0.8" not in descriptions[1]

    assert descriptions[2] == "{}"

    assert descriptions[3] == "{}"


def test_build_single_interaction_scoring_messages_format() -> None:
    """It creates properly formatted messages for single interaction scoring."""
    messages = build_single_interaction_scoring_messages(
        interaction_description="1. Inception (id=1, rating=5)",
        target_text="Science fiction films",
    )
    expected_message_count = 2
    assert len(messages) == expected_message_count
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Inception" in messages[1]["content"]
    assert "Science fiction films" in messages[1]["content"]
    assert '{"judgment":' in messages[1]["content"]
    assert "score" in messages[1]["content"]


def test_build_all_prompts_returns_prompts_per_interaction() -> None:
    """It builds one prompt per interaction."""
    evaluator = _TestEvaluator()
    interactions = pd.DataFrame(
        [
            {"movie_title": "Movie A", "movie_id": 1},
            {"movie_title": "Movie B", "movie_id": 2},
            {"movie_title": "Movie C", "movie_id": 3},
        ]
    )

    prompts = evaluator.build_all_prompts(
        explanation="Action films",
        interactions=interactions,
    )

    assert len(prompts) == 3
    for interaction_desc, messages in prompts:
        assert isinstance(interaction_desc, str)
        assert isinstance(messages, list)
        assert len(messages) == 2


def test_build_all_prompts_empty_interactions() -> None:
    """It returns empty list for empty interactions DataFrame."""
    evaluator = _TestEvaluator()
    interactions = pd.DataFrame()

    prompts = evaluator.build_all_prompts(
        explanation="Test pattern",
        interactions=interactions,
    )

    assert prompts == []


def test_build_all_prompts_empty_explanation() -> None:
    """It returns empty list for empty explanation text."""
    evaluator = _TestEvaluator()
    interactions = pd.DataFrame([{"movie_title": "Test Movie", "movie_id": 1}])

    prompts = evaluator.build_all_prompts(
        explanation="   ",
        interactions=interactions,
    )

    assert prompts == []


def test_parse_single_interaction_result_valid_json() -> None:
    """It parses valid JSON output correctly."""
    evaluator = _TestEvaluator()
    raw_output = '{"judgment": "Good match", "score": 1.0}'

    result = evaluator.parse_single_interaction_result(raw_output, "Test Movie")

    assert result["interaction"] == "Test Movie"
    assert result["score"] == 1.0
    assert result["judgment"] == "Good match"


def test_parse_single_interaction_result_markdown_fencing() -> None:
    """It parses JSON wrapped in markdown code fences."""
    evaluator = _TestEvaluator()
    raw_output = '```json\n{"judgment": "Partial", "score": 0.66}\n```'

    result = evaluator.parse_single_interaction_result(raw_output, "Test Movie")

    assert result["score"] == 0.66


def test_parse_single_interaction_result_invalid_json() -> None:
    """It returns NaN score for invalid JSON."""
    evaluator = _TestEvaluator()
    raw_output = "This is not valid JSON"

    result = evaluator.parse_single_interaction_result(raw_output, "Test Movie")

    score = result["score"]
    assert isinstance(score, float)
    assert math.isnan(score)


def test_parse_single_interaction_result_missing_judgment() -> None:
    """It handles missing judgment field gracefully."""
    evaluator = _TestEvaluator()
    raw_output = '{"score": 0.66}'

    result = evaluator.parse_single_interaction_result(raw_output, "Test Movie")

    assert result["score"] == 0.66
    assert "judgment" not in result  # No judgment key when empty


def test_aggregate_results_computes_mean() -> None:
    """It aggregates per-interaction scores into mean."""
    evaluator = _TestEvaluator()
    per_interaction_scores = [
        {"interaction": "Movie A", "score": 1.0},
        {"interaction": "Movie B", "score": 0.66},
        {"interaction": "Movie C", "score": 0.33},
        {"interaction": "Movie D", "score": 0.0},
    ]

    result = evaluator.aggregate_results(per_interaction_scores)

    expected_mean = (1.0 + 0.66 + 0.33 + 0.0) / 4
    assert result.score == pytest.approx(expected_mean)
    assert result.details is not None
    per_interaction_scores_list = result.details["per_interaction_scores"]
    assert isinstance(per_interaction_scores_list, list)
    assert len(per_interaction_scores_list) == 4


def test_aggregate_results_ignores_nan_scores() -> None:
    """It ignores NaN scores when computing the mean."""
    evaluator = _TestEvaluator()
    per_interaction_scores = [
        {"interaction": "Movie A", "score": 1.0},
        {"interaction": "Movie B", "score": float("nan")},
        {"interaction": "Movie C", "score": 0.5},
    ]

    result = evaluator.aggregate_results(per_interaction_scores)

    # Mean of valid scores only: (1.0 + 0.5) / 2 = 0.75
    assert result.score == 0.75


def test_aggregate_results_all_nan() -> None:
    """It returns NaN when all scores are NaN."""
    evaluator = _TestEvaluator()
    per_interaction_scores = [
        {"interaction": "Movie A", "score": float("nan")},
        {"interaction": "Movie B", "score": float("nan")},
    ]

    result = evaluator.aggregate_results(per_interaction_scores)

    assert math.isnan(result.score)
    assert result.details is not None
    assert "warnings" in result.details


def test_aggregate_results_empty() -> None:
    """It returns NaN score when no interactions were scored."""
    evaluator = _TestEvaluator()

    result = evaluator.aggregate_results([])

    assert math.isnan(result.score)
    assert result.details is not None
    assert result.details["per_interaction_scores"] == []


def test_build_empty_result() -> None:
    """It builds an empty result with the provided reason."""
    evaluator = _TestEvaluator()

    result = evaluator.build_empty_result(reason="No interactions available.")

    assert math.isnan(result.score)
    assert "No interactions" in result.judgment
    assert result.details is not None
    assert result.details["per_interaction_scores"] == []


def test_build_prompt_raises_not_implemented() -> None:
    """The base build_prompt method raises NotImplementedError."""
    evaluator = _TestEvaluator()
    interactions = pd.DataFrame([{"movie_title": "Test", "movie_id": 1}])

    with pytest.raises(NotImplementedError):
        evaluator.build_prompt(explanation="Test", interactions=interactions)
