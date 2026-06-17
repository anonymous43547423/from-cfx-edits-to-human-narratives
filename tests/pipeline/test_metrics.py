"""Tests for the unified metric aggregation helpers."""

# ruff: noqa: S101

from __future__ import annotations

import math

import pytest

from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.pipeline.metrics import MetricSummary, flatten_summary, safe_score, summarise_metric


class TestSummariseMetric:
    """Tests for summarise_metric."""

    def test_basic_aggregation(self) -> None:
        """It computes success rate, mean, and std for finite values."""
        result = summarise_metric([1.0, 2.0, 3.0], total=3)
        assert result.success_rate == pytest.approx(1.0)
        assert result.mean == pytest.approx(2.0)
        assert math.isfinite(result.std)

    def test_nan_values_excluded_from_mean_and_std(self) -> None:
        """It excludes NaN values from mean/std and reduces success rate."""
        result = summarise_metric([1.0, float("nan"), 3.0], total=3)
        assert result.success_rate == pytest.approx(2 / 3)
        assert result.mean == pytest.approx(2.0)
        assert math.isfinite(result.std)

    def test_all_nan_returns_nan_mean(self) -> None:
        """It returns NaN mean/std when all values are NaN."""
        result = summarise_metric([float("nan"), float("nan")], total=2)
        assert result.success_rate == pytest.approx(0.0)
        assert math.isnan(result.mean)
        assert math.isnan(result.std)

    def test_empty_values_returns_nan(self) -> None:
        """It returns NaN for an empty value list with positive total."""
        result = summarise_metric([], total=5)
        assert result.success_rate == pytest.approx(0.0)
        assert math.isnan(result.mean)
        assert math.isnan(result.std)

    def test_zero_total_returns_nan_success_rate(self) -> None:
        """It returns NaN success rate when total is zero."""
        result = summarise_metric([], total=0)
        assert math.isnan(result.success_rate)

    def test_single_value_returns_nan_std(self) -> None:
        """It returns NaN std when fewer than 2 finite values exist."""
        result = summarise_metric([5.0], total=1)
        assert result.success_rate == pytest.approx(1.0)
        assert result.mean == pytest.approx(5.0)
        assert math.isnan(result.std)

    def test_inf_values_excluded(self) -> None:
        """It excludes infinity values like NaN."""
        result = summarise_metric([1.0, float("inf"), 3.0], total=3)
        assert result.success_rate == pytest.approx(2 / 3)
        assert result.mean == pytest.approx(2.0)


class TestFlattenSummary:
    """Tests for flatten_summary."""

    def test_produces_prefixed_keys(self) -> None:
        """It returns a dict with {prefix}_success_rate, _mean, _std keys."""
        summary = MetricSummary(success_rate=0.5, mean=1.0, std=0.1)
        result = flatten_summary(summary, "my_metric")
        assert result == {
            "my_metric_success_rate": 0.5,
            "my_metric_mean": 1.0,
            "my_metric_std": 0.1,
        }


class TestSafeScore:
    """Tests for safe_score."""

    def test_returns_score_from_evaluation(self) -> None:
        """It extracts the numeric score from an EvaluationResult."""
        evaluation = EvaluationResult(judgment="ok", score=0.75)
        assert safe_score(evaluation) == pytest.approx(0.75)

    def test_returns_nan_for_none(self) -> None:
        """It returns NaN when evaluation is None."""
        assert math.isnan(safe_score(None))

    def test_returns_nan_for_nan_score(self) -> None:
        """It returns NaN when the score is NaN."""
        evaluation = EvaluationResult(judgment="fail", score=float("nan"))
        assert math.isnan(safe_score(evaluation))

    def test_does_not_clamp(self) -> None:
        """It does not clamp scores outside [0, 1]."""
        evaluation = EvaluationResult(judgment="ok", score=1.5)
        assert safe_score(evaluation) == pytest.approx(1.5)

    def test_negative_score_not_clamped(self) -> None:
        """It does not clamp negative scores."""
        evaluation = EvaluationResult(judgment="ok", score=-0.5)
        assert safe_score(evaluation) == pytest.approx(-0.5)
