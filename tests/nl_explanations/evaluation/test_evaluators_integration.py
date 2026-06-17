# ruff: noqa: S101, PLR2004
"""Unit tests for LLM-based evaluation metrics using mocks."""

from __future__ import annotations

import math

import pandas as pd

from recsys_nle.nl_explanations.evaluation import (
    CFXMatchEvaluator,
    EvaluationResult,
    NonCFXMatchEvaluator,
    PlausibilityEvaluator,
    ReadabilityEvaluator,
)

# Sample test data
_TEST_EXPLANATION = "Because you enjoyed classic 1980s science fiction action films"
_TEST_CFX_INTERACTION = pd.DataFrame([{"movie_title": "The Terminator", "movie_id": 1, "rating": 5.0, "weight": 0.8}])
_TEST_NON_CFX_INTERACTION = pd.DataFrame([{"movie_title": "Sleepless in Seattle", "movie_id": 10, "rating": 3.0}])


class TestPlausibilityEvaluator:
    """Tests for PlausibilityEvaluator prompt building and result parsing."""

    def test_build_prompt_returns_chat_messages(self) -> None:
        """It returns a list of chat messages for plausibility evaluation."""
        evaluator = PlausibilityEvaluator()
        messages = evaluator.build_prompt(explanation=_TEST_EXPLANATION)

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert _TEST_EXPLANATION in messages[1]["content"]

    def test_parse_result_valid_json(self) -> None:
        """It parses valid JSON responses into EvaluationResult."""
        evaluator = PlausibilityEvaluator()
        raw_output = '{"judgment": "The explanation is plausible.", "score": 0.85}'

        result = evaluator.parse_result(raw_output)

        assert isinstance(result, EvaluationResult)
        assert result.judgment == "The explanation is plausible."
        assert result.score == 0.85

    def test_parse_result_invalid_json(self) -> None:
        """It handles invalid JSON with NaN score and warning."""
        evaluator = PlausibilityEvaluator()
        raw_output = "This is not valid JSON"

        result = evaluator.parse_result(raw_output)

        assert isinstance(result, EvaluationResult)
        assert math.isnan(result.score)
        assert result.details is not None
        assert "warnings" in result.details

    def test_parse_result_score_clamping(self) -> None:
        """It clamps scores to [0, 1] range."""
        evaluator = PlausibilityEvaluator()

        result_high = evaluator.parse_result('{"judgment": "test", "score": 1.5}')
        assert result_high.score == 1.0

        result_low = evaluator.parse_result('{"judgment": "test", "score": -0.5}')
        assert result_low.score == 0.0


class TestReadabilityEvaluator:
    """Tests for ReadabilityEvaluator prompt building and result parsing."""

    def test_build_prompt_returns_chat_messages(self) -> None:
        """It returns a list of chat messages for readability evaluation."""
        evaluator = ReadabilityEvaluator()
        messages = evaluator.build_prompt(explanation=_TEST_EXPLANATION)

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert _TEST_EXPLANATION in messages[1]["content"]
        assert '"fluency"' in messages[1]["content"]
        assert '"grammar"' in messages[1]["content"]
        assert '"naturalness"' in messages[1]["content"]
        assert '"specificity"' in messages[1]["content"]

    def test_parse_result_valid_json(self) -> None:
        """It parses valid JSON responses into EvaluationResult."""
        evaluator = ReadabilityEvaluator()
        raw_output = (
            '{"fluency": 1.0, "grammar": 0.66, "length": 0.33, '
            '"illustrativeness": 0.33, "naturalness": 0.0, "specificity": 0.66}'
        )

        result = evaluator.parse_result(raw_output)

        assert isinstance(result, EvaluationResult)
        assert result.details is not None
        assert result.details["fluency"] == 1.0
        assert result.details["grammar"] == 0.66
        assert result.details["length"] == 0.33
        assert result.details["illustrativeness"] == 0.33
        assert result.details["naturalness"] == 0.0
        assert result.details["specificity"] == 0.66
        expected = (1.0 + 0.66 + 0.33 + 0.33 + 0.0 + 0.66) / 6
        assert result.score == expected

    def test_parse_result_missing_key(self) -> None:
        """It returns NaN overall when a subscore is missing."""
        evaluator = ReadabilityEvaluator()
        raw_output = '{"fluency": 1.0, "grammar": 0.66, "length": 0.33, "naturalness": 0.33, "specificity": 0.33}'

        result = evaluator.parse_result(raw_output)

        assert math.isnan(result.score)
        assert result.details is not None
        assert "warnings" in result.details


class TestCFXMatchEvaluator:
    """Tests for CFXMatchEvaluator prompt building and result aggregation."""

    def test_build_all_prompts_returns_prompts_per_interaction(self) -> None:
        """It returns one prompt per interaction for CFX match evaluation."""
        evaluator = CFXMatchEvaluator()
        prompts = evaluator.build_all_prompts(
            explanation=_TEST_EXPLANATION,
            interactions=_TEST_CFX_INTERACTION,
        )

        assert isinstance(prompts, list)
        assert len(prompts) == 1  # One interaction
        interaction_desc, messages = prompts[0]
        assert isinstance(interaction_desc, str)
        assert "movie_id" not in interaction_desc
        assert interaction_desc.startswith("{")
        assert interaction_desc.endswith("}")
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_build_all_prompts_empty_interactions(self) -> None:
        """It returns empty list when no interactions are provided."""
        evaluator = CFXMatchEvaluator()
        prompts = evaluator.build_all_prompts(
            explanation=_TEST_EXPLANATION,
            interactions=pd.DataFrame(),
        )

        assert prompts == []

    def test_parse_single_interaction_result(self) -> None:
        """It parses per-interaction scoring output correctly."""
        evaluator = CFXMatchEvaluator()
        raw_output = '{"judgment": "Matches well", "score": 1.0}'
        interaction_desc = "The Terminator (1984)"

        result = evaluator.parse_single_interaction_result(raw_output, interaction_desc)

        assert result["interaction"] == interaction_desc
        assert result["score"] == 1.0
        assert result["judgment"] == "Matches well"

    def test_aggregate_results(self) -> None:
        """It aggregates per-interaction scores into final result."""
        evaluator = CFXMatchEvaluator()
        per_interaction_scores = [
            {"interaction": "Movie A", "score": 1.0, "judgment": "Good match"},
            {"interaction": "Movie B", "score": 0.66, "judgment": "Partial match"},
        ]

        result = evaluator.aggregate_results(per_interaction_scores)

        assert isinstance(result, EvaluationResult)
        assert result.score == (1.0 + 0.66) / 2
        assert result.details is not None
        result_scores = result.details["per_interaction_scores"]
        assert isinstance(result_scores, list)
        assert len(result_scores) == 2

    def test_aggregate_results_empty(self) -> None:
        """It returns NaN score when no interactions were scored."""
        evaluator = CFXMatchEvaluator()

        result = evaluator.aggregate_results([])

        assert math.isnan(result.score)
        assert result.details is not None
        assert result.details["per_interaction_scores"] == []

    def test_build_empty_result(self) -> None:
        """It builds an empty result with the provided reason."""
        evaluator = CFXMatchEvaluator()
        result = evaluator.build_empty_result(reason="No CFX interactions available.")

        assert math.isnan(result.score)
        assert "No CFX interactions" in result.judgment


class TestNonCFXMatchEvaluator:
    """Tests for NonCFXMatchEvaluator prompt building and result aggregation."""

    def test_build_all_prompts_returns_prompts_per_interaction(self) -> None:
        """It returns one prompt per interaction for non-CFX match evaluation."""
        evaluator = NonCFXMatchEvaluator()
        prompts = evaluator.build_all_prompts(
            explanation=_TEST_EXPLANATION,
            interactions=_TEST_NON_CFX_INTERACTION,
        )

        assert isinstance(prompts, list)
        assert len(prompts) == 1
        interaction_desc, messages = prompts[0]
        assert "movie_id" not in interaction_desc
        assert interaction_desc.startswith("{")
        assert interaction_desc.endswith("}")
        assert len(messages) == 2

    def test_build_all_prompts_empty_interactions(self) -> None:
        """It returns empty list when no interactions are provided."""
        evaluator = NonCFXMatchEvaluator()
        prompts = evaluator.build_all_prompts(
            explanation=_TEST_EXPLANATION,
            interactions=pd.DataFrame(),
        )

        assert prompts == []

    def test_aggregate_results_single_score(self) -> None:
        """It handles single-interaction aggregation correctly."""
        evaluator = NonCFXMatchEvaluator()
        per_interaction_scores = [
            {"interaction": "Movie A", "score": 0.33, "judgment": "Weak match"},
        ]

        result = evaluator.aggregate_results(per_interaction_scores)

        assert result.score == 0.33
        assert "Non-CFX match" in result.judgment


class TestEdgeCases:
    """Tests for edge cases in evaluator behavior."""

    def test_cfx_match_multiple_interactions(self) -> None:
        """It builds prompts for multiple CFX interactions."""
        evaluator = CFXMatchEvaluator()
        interactions = pd.DataFrame(
            [
                {"movie_title": "The Terminator", "movie_id": 1, "rating": 5.0},
                {"movie_title": "RoboCop", "movie_id": 2, "rating": 4.5},
                {"movie_title": "Blade Runner", "movie_id": 3, "rating": 4.0},
            ]
        )

        prompts = evaluator.build_all_prompts(
            explanation=_TEST_EXPLANATION,
            interactions=interactions,
        )

        assert len(prompts) == 3

    def test_parse_result_with_markdown_fencing(self) -> None:
        """It parses JSON wrapped in markdown code fences."""
        evaluator = PlausibilityEvaluator()
        raw_output = '```json\n{"judgment": "Good", "score": 0.75}\n```'

        result = evaluator.parse_result(raw_output)

        assert result.score == 0.75
        assert result.judgment == "Good"

    def test_parse_single_interaction_missing_judgment(self) -> None:
        """It handles missing judgment field gracefully."""
        evaluator = CFXMatchEvaluator()
        raw_output = '{"score": 0.66}'

        result = evaluator.parse_single_interaction_result(raw_output, "Test Movie")

        assert result["score"] == 0.66
        assert "judgment" not in result  # No judgment key when empty

    def test_aggregate_nan_scores_ignored(self) -> None:
        """It ignores NaN scores when computing the mean."""
        evaluator = CFXMatchEvaluator()
        per_interaction_scores = [
            {"interaction": "Movie A", "score": 1.0},
            {"interaction": "Movie B", "score": float("nan")},
            {"interaction": "Movie C", "score": 0.5},
        ]

        result = evaluator.aggregate_results(per_interaction_scores)

        # Mean of valid scores only: (1.0 + 0.5) / 2 = 0.75
        assert result.score == 0.75
