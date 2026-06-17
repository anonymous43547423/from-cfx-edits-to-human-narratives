# ruff: noqa: S101, PLR2004, SLF001, RUF059
"""Unit tests for faithfulness evaluation metrics."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import torch

from recsys_nle.nl_explanations.evaluation.faithfulness import (
    FaithfulnessConfig,
    FaithfulnessRemovalEvaluator,
    FaithfulnessReplacementEvaluator,
    ScoredItem,
    _create_cfx_excluded_history,
)


def _create_mock_recommender(*, num_items: int = 100) -> MagicMock:
    """Create a mock recommender wrapper for testing."""
    mock = MagicMock()
    mock.num_items = num_items
    mock.device = torch.device("cpu")

    def mock_get_item_score(user_history: torch.Tensor, _target_item: int) -> float:
        return float(user_history.sum().item())

    mock.get_item_score = mock_get_item_score
    return mock


def _create_scored_items(n_items: int, high_score_count: int, low_score: float = 0.2) -> list[ScoredItem]:
    """Create scored items with specified distribution."""
    items: list[ScoredItem] = []
    for i in range(n_items):
        score = 0.8 if i < high_score_count else low_score
        items.append(
            ScoredItem(
                item_id=i,
                score=score,
                interaction_description=f"Item {i}",
            )
        )
    return items


class TestBaseFaithfulnessEvaluator:
    """Tests for BaseFaithfulnessEvaluator."""

    def test_split_by_similarity_separates_items_correctly(self) -> None:
        """It splits items into similar and dissimilar based on threshold."""
        evaluator = FaithfulnessRemovalEvaluator()

        scored_items = [
            ScoredItem(item_id=1, score=0.9, interaction_description="High 1"),
            ScoredItem(item_id=2, score=0.7, interaction_description="High 2"),
            ScoredItem(item_id=3, score=0.5, interaction_description="Border"),
            ScoredItem(item_id=4, score=0.3, interaction_description="Low 1"),
            ScoredItem(item_id=5, score=0.1, interaction_description="Low 2"),
        ]

        similar, dissimilar, nan_items = evaluator._split_by_similarity(scored_items, threshold=0.5)

        assert len(similar) == 3  # 0.9, 0.7, 0.5 (>= threshold)
        assert len(dissimilar) == 2  # 0.3, 0.1 (< threshold)
        assert len(nan_items) == 0

        similar_ids = {item.item_id for item in similar}
        assert similar_ids == {1, 2, 3}

        dissimilar_ids = {item.item_id for item in dissimilar}
        assert dissimilar_ids == {4, 5}

    def test_split_by_similarity_ignores_nan_scores(self) -> None:
        """It ignores items with NaN scores when splitting."""
        evaluator = FaithfulnessRemovalEvaluator()

        scored_items = [
            ScoredItem(item_id=1, score=0.9, interaction_description="Valid"),
            ScoredItem(item_id=2, score=float("nan"), interaction_description="NaN"),
            ScoredItem(item_id=3, score=0.3, interaction_description="Low"),
        ]

        similar, dissimilar, nan_items = evaluator._split_by_similarity(scored_items, threshold=0.5)

        assert len(similar) == 1
        assert len(dissimilar) == 1
        assert len(nan_items) == 1

    def test_build_nan_result(self) -> None:
        """It builds a NaN result with the given reason."""
        evaluator = FaithfulnessRemovalEvaluator()
        result = evaluator._build_nan_result("Test reason")

        assert math.isnan(result.score)
        assert "Test reason" in result.judgment

    def test_build_success_result(self) -> None:
        """It builds a result with correct score and details."""
        evaluator = FaithfulnessRemovalEvaluator()
        result = evaluator._build_success_result(
            median_score=7.0,
            n_candidates=20,
            n_trials=5,
            n_evaluated=12,
            n_samples_per_trial=3,
            metric_name="test_metric",
        )

        assert result.score == 7.0
        assert result.details is not None
        assert result.details["n_candidates"] == 20
        assert result.details["n_trials"] == 5
        assert result.details["n_evaluated"] == 12
        assert result.details["n_samples_per_trial"] == 3
        assert result.details["median_score"] == 7.0


class TestFaithfulnessRemovalEvaluator:
    """Tests for FaithfulnessRemovalEvaluator."""

    def test_compute_results_returns_nan_when_insufficient_similar_items(self) -> None:
        """It returns NaN for regular metric when insufficient similar items."""
        evaluator = FaithfulnessRemovalEvaluator()
        mock_recommender = _create_mock_recommender()

        # Only 2 similar items, but we need 5 to meet the min limit.
        scored_items = _create_scored_items(n_items=10, high_score_count=2)

        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=5,
            n_faithfulness_trials=2,
            n_faithfulness_samples=1,
        )

        user_history = torch.ones(100, dtype=torch.float32)
        regular, baseline = evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert math.isnan(regular.score)
        assert "Insufficient" in regular.judgment

    def test_compute_results_uses_dissimilar_candidates_for_baseline(self) -> None:
        """It uses only dissimilar items for baseline candidates."""
        evaluator = FaithfulnessRemovalEvaluator()
        mock_recommender = _create_mock_recommender()

        # 8 similar items (high score), only 2 dissimilar (baseline insufficient)
        scored_items = _create_scored_items(n_items=10, high_score_count=8)

        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=3,
            n_faithfulness_trials=2,
            n_faithfulness_samples=1,
        )

        user_history = torch.ones(100, dtype=torch.float32)
        regular, baseline = evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        # Regular should succeed, baseline should fail with only dissimilar candidates
        assert not math.isnan(regular.score)
        assert math.isnan(baseline.score)
        assert "Insufficient dissimilar" in baseline.judgment

    def test_compute_results_baseline_succeeds_with_dissimilar_items(self) -> None:
        """It returns success for baseline when dissimilar candidates are sufficient."""
        evaluator = FaithfulnessRemovalEvaluator()
        mock_recommender = _create_mock_recommender()

        # Only 2 similar items, but 8 dissimilar items (baseline sufficient)
        scored_items = _create_scored_items(n_items=10, high_score_count=2)

        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=3,
            n_faithfulness_trials=2,
            n_faithfulness_samples=1,
        )

        user_history = torch.ones(100, dtype=torch.float32)
        regular, baseline = evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert math.isnan(regular.score)
        assert "Insufficient similar" in regular.judgment
        assert not math.isnan(baseline.score)

    def test_compute_results_reports_median_score(self) -> None:
        """It reports the median score for removal."""
        evaluator = FaithfulnessRemovalEvaluator()
        mock_recommender = _create_mock_recommender()

        scored_items = _create_scored_items(n_items=3, high_score_count=3)
        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=3,
        )

        user_history = torch.ones(100, dtype=torch.float32)
        regular, _ = evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert regular.score == 97.0

    def test_removal_trials_remove_sampled_items(self) -> None:
        """It removes sampled candidates for removal."""
        evaluator = FaithfulnessRemovalEvaluator()
        captured_histories: list[torch.Tensor] = []

        def get_item_score(user_history: torch.Tensor, _target_item: int) -> float:
            captured_histories.append(user_history.clone())
            return float(user_history.sum().item())

        mock_recommender = MagicMock()
        mock_recommender.num_items = 30
        mock_recommender.device = torch.device("cpu")
        mock_recommender.get_item_score = get_item_score

        scored_items = [
            ScoredItem(item_id=10, score=0.9, interaction_description="High 10"),
            ScoredItem(item_id=20, score=0.1, interaction_description="Low 20"),
        ]
        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=1,
        )
        user_history = torch.ones(30, dtype=torch.float32)

        evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert len(captured_histories) == 2
        regular_history = captured_histories[0]
        assert regular_history[10] == 0.0
        baseline_history = captured_histories[1]
        assert baseline_history[20] == 0.0

    def test_cfx_items_retained_in_baseline_history(self) -> None:
        """It keeps CFX items in history while sampling candidates."""
        evaluator = FaithfulnessRemovalEvaluator()
        captured_histories: list[torch.Tensor] = []

        def get_item_score(user_history: torch.Tensor, _target_item: int) -> float:
            captured_histories.append(user_history.clone())
            return float(user_history.sum().item())

        mock_recommender = MagicMock()
        mock_recommender.num_items = 30
        mock_recommender.device = torch.device("cpu")
        mock_recommender.get_item_score = get_item_score

        scored_items = [ScoredItem(item_id=10, score=0.9, interaction_description="High 10")]
        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=1,
        )
        user_history = torch.ones(30, dtype=torch.float32)

        evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
            cfx_item_ids=[5, 15],
        )

        updated_history = captured_histories[0]
        # CFX items (5 and 15) should remain present in history
        assert updated_history[5].item() == 1.0
        assert updated_history[15].item() == 1.0
        # The candidate item should still be zeroed out
        assert updated_history[10].item() == 0.0
        # Other items should remain 1.0
        assert updated_history[0].item() == 1.0


class TestFaithfulnessReplacementEvaluator:
    """Tests for FaithfulnessReplacementEvaluator."""

    def test_compute_results_reports_median_score(self) -> None:
        """It reports the median score for replacement."""
        evaluator = FaithfulnessReplacementEvaluator()
        mock_recommender = _create_mock_recommender()

        scored_items = _create_scored_items(n_items=3, high_score_count=3)
        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=3,
        )

        user_history = torch.zeros(100, dtype=torch.float32)
        regular, _ = evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert regular.score == 3.0

    def test_replacement_trials_add_candidates(self) -> None:
        """It adds sampled candidates for replacement."""
        evaluator = FaithfulnessReplacementEvaluator()
        captured_histories: list[torch.Tensor] = []

        def get_item_score(user_history: torch.Tensor, _target_item: int) -> float:
            captured_histories.append(user_history.clone())
            return float(user_history.sum().item())

        mock_recommender = MagicMock()
        mock_recommender.num_items = 30
        mock_recommender.device = torch.device("cpu")
        mock_recommender.get_item_score = get_item_score

        scored_items = [
            ScoredItem(item_id=10, score=0.9, interaction_description="High 10"),
            ScoredItem(item_id=20, score=0.1, interaction_description="Low 20"),
        ]
        config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=20,
            match_threshold=0.5,
            n_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=1,
        )
        user_history = torch.zeros(30, dtype=torch.float32)

        evaluator.compute_results_from_scores(
            scored_items=scored_items,
            user_history=user_history,
            target_item=0,
            recommender=mock_recommender,
            config=config,
        )

        assert len(captured_histories) == 2
        regular_history = captured_histories[0]
        assert float(regular_history[10]) == 1.0
        baseline_history = captured_histories[1]
        assert float(baseline_history[20]) == 1.0


class TestBuildAllPromptsWithIds:
    """Tests for build_all_prompts_with_ids method."""

    def test_returns_movie_ids_with_prompts(self) -> None:
        """It returns movie IDs alongside prompts."""
        evaluator = FaithfulnessRemovalEvaluator()

        interactions = pd.DataFrame(
            [
                {"movie_id": 1, "movie_title": "Movie A"},
                {"movie_id": 2, "movie_title": "Movie B"},
                {"movie_id": 3, "movie_title": "Movie C"},
            ]
        )

        prompts = evaluator.build_all_prompts_with_ids(
            explanation="Test pattern",
            interactions=interactions,
        )

        assert len(prompts) == 3
        movie_ids = [p[0] for p in prompts]
        assert movie_ids == [1, 2, 3]

    def test_returns_empty_for_empty_interactions(self) -> None:
        """It returns empty list for empty interactions."""
        evaluator = FaithfulnessRemovalEvaluator()
        prompts = evaluator.build_all_prompts_with_ids(
            explanation="Test pattern",
            interactions=pd.DataFrame(),
        )
        assert prompts == []

    def test_returns_empty_for_empty_explanation(self) -> None:
        """It returns empty list for empty explanation."""
        evaluator = FaithfulnessRemovalEvaluator()
        interactions = pd.DataFrame([{"movie_id": 1, "movie_title": "Movie A"}])
        prompts = evaluator.build_all_prompts_with_ids(
            explanation="   ",
            interactions=interactions,
        )
        assert prompts == []


class TestCreateCfxExcludedHistory:
    """Tests for _create_cfx_excluded_history helper function."""

    def test_zeros_cfx_items(self) -> None:
        """It creates a history tensor with CFX item indices zeroed out."""
        original = torch.ones(10)
        cfx_item_ids = [1, 3, 7]

        result = _create_cfx_excluded_history(original, cfx_item_ids)

        # CFX items should be zeroed
        assert result[1].item() == 0.0
        assert result[3].item() == 0.0
        assert result[7].item() == 0.0

        # Non-CFX items should remain 1.0
        assert result[0].item() == 1.0
        assert result[2].item() == 1.0
        assert result[4].item() == 1.0

        # Original should be unchanged
        assert original[1].item() == 1.0
        assert original[3].item() == 1.0

    def test_handles_empty_cfx_ids(self) -> None:
        """It returns unchanged history when CFX item IDs are empty."""
        original = torch.ones(5)

        result = _create_cfx_excluded_history(original, [])

        assert torch.equal(result, original)
        # Ensure it's a copy, not the same object
        result[0] = 0.0
        assert original[0].item() == 1.0

    def test_ignores_out_of_range_ids(self) -> None:
        """It ignores CFX item IDs that are out of range."""
        original = torch.ones(5)
        cfx_item_ids = [1, 10, -1]  # 10 and -1 are out of range

        result = _create_cfx_excluded_history(original, cfx_item_ids)

        # Only index 1 should be zeroed (10 and -1 are out of range)
        assert result[1].item() == 0.0
        assert result[0].item() == 1.0
        assert result[2].item() == 1.0
        assert result[3].item() == 1.0
        assert result[4].item() == 1.0
