"""Tests for explanation run summary metrics."""

# ruff: noqa: S101

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import ttest_ind

from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
from recsys_nle.nl_explanations.workflow import ExplanationResult
from recsys_nle.pipeline.run_summary import compute_explanation_statistics
from recsys_nle.pipeline.workflow import CfxSearchOutcome, PipelineResult

P_VALUE_COMPLEMENT_LT_0_1_THRESHOLD = 0.9
P_VALUE_COMPLEMENT_LT_0_05_THRESHOLD = 0.95
P_VALUE_COMPLEMENT_LT_0_01_THRESHOLD = 0.99


def _make_eval(trial_scores: list[float]) -> EvaluationResult:
    """Create an evaluation result with trial scores."""
    return EvaluationResult(judgment="ok", score=0.0, details={"trial_scores": trial_scores})


def test_compute_explanation_statistics_adds_faithfulness_pvalues() -> None:
    """It adds mean faithfulness p-values for available trial scores."""
    user_one = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="",
        explanation="",
        explanation_plausibility=_make_eval([]),
        explanation_cfx_match=_make_eval([]),
        faithfulness_removal=_make_eval([0.1, 0.2, 0.15]),
        faithfulness_removal_baseline=_make_eval([0.9, 0.8, 0.85]),
        faithfulness_replacement=_make_eval([0.9, 0.8, 0.85]),
        faithfulness_replacement_baseline=_make_eval([0.1, 0.2, 0.15]),
    )
    user_two = NaturalLanguageExplanationResult(
        user_id=2,
        reasoning="",
        explanation="",
        explanation_plausibility=_make_eval([]),
        explanation_cfx_match=_make_eval([]),
        faithfulness_removal=_make_eval([0.2]),
        faithfulness_removal_baseline=_make_eval([0.3]),
        faithfulness_replacement=_make_eval([0.2]),
        faithfulness_replacement_baseline=_make_eval([0.3]),
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: user_one, 2: user_two})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[1, 2],
    )

    stats = compute_explanation_statistics(
        pipeline_result,
        enabled_evaluations={"faithfulness_removal", "faithfulness_replacement"},
    )

    expected_removal = (
        1.0
        - ttest_ind(
            [0.1, 0.2, 0.15],
            [0.9, 0.8, 0.85],
            equal_var=False,
            alternative="less",
        ).pvalue
    )
    expected_replacement = (
        1.0
        - ttest_ind(
            [0.9, 0.8, 0.85],
            [0.1, 0.2, 0.15],
            equal_var=False,
            alternative="greater",
        ).pvalue
    )
    expected_removal_lt_0_1 = 1.0 if expected_removal > P_VALUE_COMPLEMENT_LT_0_1_THRESHOLD else 0.0
    expected_removal_lt_0_05 = 1.0 if expected_removal > P_VALUE_COMPLEMENT_LT_0_05_THRESHOLD else 0.0
    expected_removal_lt_0_01 = 1.0 if expected_removal > P_VALUE_COMPLEMENT_LT_0_01_THRESHOLD else 0.0
    expected_replacement_lt_0_1 = 1.0 if expected_replacement > P_VALUE_COMPLEMENT_LT_0_1_THRESHOLD else 0.0
    expected_replacement_lt_0_05 = 1.0 if expected_replacement > P_VALUE_COMPLEMENT_LT_0_05_THRESHOLD else 0.0
    expected_replacement_lt_0_01 = 1.0 if expected_replacement > P_VALUE_COMPLEMENT_LT_0_01_THRESHOLD else 0.0

    assert math.isfinite(stats["faithfulness_removal_pvalue_complement"])
    assert math.isfinite(stats["faithfulness_replacement_pvalue_complement"])
    assert stats["faithfulness_removal_pvalue_complement"] == pytest.approx(expected_removal)
    assert stats["faithfulness_replacement_pvalue_complement"] == pytest.approx(expected_replacement)
    assert stats["faithfulness_removal_pvalue_success_rate"] == pytest.approx(0.5)
    assert stats["faithfulness_replacement_pvalue_success_rate"] == pytest.approx(0.5)
    assert stats["faithfulness_removal_pvalue_lt_0_1"] == pytest.approx(expected_removal_lt_0_1)
    assert stats["faithfulness_removal_pvalue_lt_0_05"] == pytest.approx(expected_removal_lt_0_05)
    assert stats["faithfulness_removal_pvalue_lt_0_01"] == pytest.approx(expected_removal_lt_0_01)
    assert stats["faithfulness_replacement_pvalue_lt_0_1"] == pytest.approx(expected_replacement_lt_0_1)
    assert stats["faithfulness_replacement_pvalue_lt_0_05"] == pytest.approx(expected_replacement_lt_0_05)
    assert stats["faithfulness_replacement_pvalue_lt_0_01"] == pytest.approx(expected_replacement_lt_0_01)


def test_compute_explanation_statistics_adds_distance_summary() -> None:
    """It adds summary stats for distance metrics when available."""
    evaluation = EvaluationResult(judgment="ok", score=0.5)
    user_one = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="",
        explanation="",
        explanation_plausibility=evaluation,
        explanation_cfx_match=evaluation,
        faithfulness_removal=evaluation,
        faithfulness_removal_baseline=evaluation,
        faithfulness_replacement=evaluation,
        faithfulness_replacement_baseline=evaluation,
    )
    user_two = NaturalLanguageExplanationResult(
        user_id=2,
        reasoning="",
        explanation="",
        explanation_plausibility=evaluation,
        explanation_cfx_match=evaluation,
        faithfulness_removal=evaluation,
        faithfulness_removal_baseline=evaluation,
        faithfulness_replacement=evaluation,
        faithfulness_replacement_baseline=evaluation,
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: user_one, 2: user_two})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[1, 2],
        distance_metrics_by_user={
            1: {"user_based_mean_cfx_distance": 0.2, "user_based_mean_separation": 0.5},
            2: {"user_based_mean_cfx_distance": float("nan"), "user_based_mean_separation": float("nan")},
        },
    )

    stats = compute_explanation_statistics(pipeline_result, enabled_evaluations=set())

    assert stats["user_based_mean_cfx_distance_success_rate"] == pytest.approx(0.5)
    assert stats["user_based_mean_cfx_distance_mean"] == pytest.approx(0.2)
    assert math.isnan(stats["user_based_mean_cfx_distance_std"])
    assert stats["user_based_mean_separation_success_rate"] == pytest.approx(0.5)
    assert stats["user_based_mean_separation_mean"] == pytest.approx(0.5)
    assert math.isnan(stats["user_based_mean_separation_std"])


def test_compute_explanation_statistics_adds_pattern_contrast() -> None:
    """It adds per-user pattern contrast (cfx - non_cfx) when both evaluations are enabled."""
    user_one = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="r",
        explanation="e",
        explanation_plausibility=EvaluationResult(judgment="ok", score=0.5),
        explanation_cfx_match=EvaluationResult(judgment="ok", score=0.8),
        explanation_non_cfx_match=EvaluationResult(judgment="ok", score=0.3),
    )
    user_two = NaturalLanguageExplanationResult(
        user_id=2,
        reasoning="r",
        explanation="e",
        explanation_plausibility=EvaluationResult(judgment="ok", score=0.5),
        explanation_cfx_match=EvaluationResult(judgment="ok", score=0.6),
        explanation_non_cfx_match=EvaluationResult(judgment="ok", score=0.5),
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: user_one, 2: user_two})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[1, 2],
    )

    stats = compute_explanation_statistics(
        pipeline_result,
        enabled_evaluations={"cfx_match", "non_cfx_match"},
    )

    # Per-user differences: user_one = 0.8 - 0.3 = 0.5, user_two = 0.6 - 0.5 = 0.1
    expected_mean = (0.5 + 0.1) / 2
    assert stats["explanation_pattern_contrast_mean"] == pytest.approx(expected_mean)
    assert stats["explanation_pattern_contrast_success_rate"] == pytest.approx(1.0)
    assert math.isfinite(stats["explanation_pattern_contrast_std"])


def test_compute_explanation_statistics_pattern_contrast_handles_nan() -> None:
    """It excludes users with NaN cfx or non_cfx scores from the contrast mean."""
    user_one = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="r",
        explanation="e",
        explanation_plausibility=EvaluationResult(judgment="ok", score=0.5),
        explanation_cfx_match=EvaluationResult(judgment="ok", score=0.9),
        explanation_non_cfx_match=EvaluationResult(judgment="ok", score=0.2),
    )
    # user_two has non_cfx_match = None -> NaN -> contrast is NaN for this user
    user_two = NaturalLanguageExplanationResult(
        user_id=2,
        reasoning="r",
        explanation="e",
        explanation_plausibility=EvaluationResult(judgment="ok", score=0.5),
        explanation_cfx_match=EvaluationResult(judgment="ok", score=0.6),
        explanation_non_cfx_match=None,
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: user_one, 2: user_two})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[1, 2],
    )

    stats = compute_explanation_statistics(
        pipeline_result,
        enabled_evaluations={"cfx_match", "non_cfx_match"},
    )

    # Only user_one contributes: 0.9 - 0.2 = 0.7
    assert stats["explanation_pattern_contrast_mean"] == pytest.approx(0.7)
    assert stats["explanation_pattern_contrast_success_rate"] == pytest.approx(0.5)
    # std is NaN with a single finite sample
    assert math.isnan(stats["explanation_pattern_contrast_std"])


def test_compute_explanation_statistics_omits_pattern_contrast_when_only_cfx_match() -> None:
    """It does not add pattern contrast keys when only one of cfx/non_cfx is enabled."""
    user = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="r",
        explanation="e",
        explanation_plausibility=EvaluationResult(judgment="ok", score=0.5),
        explanation_cfx_match=EvaluationResult(judgment="ok", score=0.8),
        explanation_non_cfx_match=EvaluationResult(judgment="ok", score=0.3),
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: user})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[1],
    )

    stats = compute_explanation_statistics(
        pipeline_result,
        enabled_evaluations={"cfx_match"},
    )

    assert "explanation_pattern_contrast_mean" not in stats
    assert "explanation_pattern_contrast_std" not in stats
    assert "explanation_pattern_contrast_success_rate" not in stats


def test_compute_explanation_statistics_adds_cfx_rates() -> None:
    """It reports summary rates for the CFX search outcomes."""
    explanations = ExplanationResult(dataset=None, results_by_user={})
    pipeline_result = PipelineResult(
        recommendations=pd.DataFrame(),
        user_attributions={},
        cfx_interactions=pd.DataFrame(),
        explanations=explanations,
        all_interactions=pd.DataFrame(),
        sampled_user_ids=[],
        cfx_search_outcome=CfxSearchOutcome(
            n_valid=3,
            n_no_cfx=1,
            n_below_min_interactions=1,
        ),
    )

    stats = compute_explanation_statistics(pipeline_result, enabled_evaluations=set())

    assert stats["cfx_success_rate"] == pytest.approx(0.8)
    assert stats["cfx_simple_rate"] == pytest.approx(0.25)
