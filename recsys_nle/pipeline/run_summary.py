"""Helpers for aggregating explanation evaluation statistics for the pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from scipy.stats import ttest_ind

from recsys_nle.core.attribution import AttributionMethod
from recsys_nle.nl_explanations.evaluation.readability import (
    READABILITY_SUBSCORE_KEYS,
)
from recsys_nle.nl_explanations.evaluation.readability import (
    extract_subscore as _readability_subscore,
)
from recsys_nle.pipeline.distance_metrics import _DISTANCE_METRIC_KEYS
from recsys_nle.pipeline.metrics import flatten_summary, safe_score, summarise_metric

MIN_TRIAL_SAMPLES = 2
P_VALUE_COMPLEMENT_LT_0_1_THRESHOLD = 0.9
P_VALUE_COMPLEMENT_LT_0_05_THRESHOLD = 0.95
P_VALUE_COMPLEMENT_LT_0_01_THRESHOLD = 0.99

if TYPE_CHECKING:
    from collections.abc import Collection

    from recsys_nle.nl_explanations.evaluation import EvaluationResult
    from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
    from recsys_nle.pipeline.config import PipelineConfig
    from recsys_nle.pipeline.workflow import PipelineResult


def _compute_generation_stats(results: list[NaturalLanguageExplanationResult]) -> dict[str, float | int]:
    """Compute success rates for reasoning and explanations."""
    total = len(results)
    if total == 0:
        return {"total_examples": 0}

    reasoning_success = sum(1 for item in results if item.reasoning.strip())
    explanation_success = sum(1 for item in results if item.explanation.strip())

    return {
        "total_examples": total,
        "reasoning_generation_success_rate": reasoning_success / total,
        "explanation_generation_success_rate": explanation_success / total,
    }


def _extract_trial_scores(evaluation: EvaluationResult | None) -> list[float]:
    """Extract finite trial scores from an evaluation result."""
    if evaluation is None or not hasattr(evaluation, "details"):
        return []
    details = evaluation.details or {}
    trial_scores = details.get("trial_scores")
    if not isinstance(trial_scores, Sequence) or isinstance(trial_scores, str):
        return []

    scores: list[float] = []
    for score_raw in trial_scores:
        try:
            score = float(score_raw) if score_raw is not None else float("nan")
        except (TypeError, ValueError):
            score = float("nan")
        if math.isfinite(score):
            scores.append(score)
    return scores


def _compute_faithfulness_pvalue_complement(
    regular_eval: EvaluationResult | None,
    baseline_eval: EvaluationResult | None,
    *,
    alternative: Literal["less", "greater"],
) -> float:
    """Compute 1 - p-value for Welch's t-test between trial score samples."""
    regular_scores = _extract_trial_scores(regular_eval)
    baseline_scores = _extract_trial_scores(baseline_eval)
    if len(regular_scores) < MIN_TRIAL_SAMPLES or len(baseline_scores) < MIN_TRIAL_SAMPLES:
        return float("nan")

    result = ttest_ind(
        regular_scores,
        baseline_scores,
        equal_var=False,
        alternative=alternative,
    )
    return float(1.0 - result.pvalue)


def _compute_evaluation_block(
    results: list[NaturalLanguageExplanationResult],
    *,
    extractor: Callable[[NaturalLanguageExplanationResult], EvaluationResult | None],
    prefix: str,
) -> dict[str, float]:
    """Compute success rate, mean, and std for a single evaluation dimension."""
    scores = [safe_score(extractor(item)) for item in results]
    return flatten_summary(summarise_metric(scores, len(results)), prefix)


def _compute_pvalue_stats(pvalues: list[float], prefix: str) -> dict[str, float]:
    """Compute significance frequency and success rate for p-value complements."""
    total = len(pvalues)
    if total == 0:
        return {
            f"{prefix}_pvalue_lt_0_1": float("nan"),
            f"{prefix}_pvalue_lt_0_05": float("nan"),
            f"{prefix}_pvalue_lt_0_01": float("nan"),
            f"{prefix}_pvalue_success_rate": float("nan"),
        }

    valid = [pvalue for pvalue in pvalues if math.isfinite(pvalue)]
    valid_count = len(valid)
    success_rate = valid_count / total
    if valid_count == 0:
        return {
            f"{prefix}_pvalue_lt_0_1": float("nan"),
            f"{prefix}_pvalue_lt_0_05": float("nan"),
            f"{prefix}_pvalue_lt_0_01": float("nan"),
            f"{prefix}_pvalue_success_rate": success_rate,
        }

    return {
        f"{prefix}_pvalue_lt_0_1": sum(1 for pvalue in valid if pvalue > P_VALUE_COMPLEMENT_LT_0_1_THRESHOLD)
        / valid_count,
        f"{prefix}_pvalue_lt_0_05": sum(1 for pvalue in valid if pvalue > P_VALUE_COMPLEMENT_LT_0_05_THRESHOLD)
        / valid_count,
        f"{prefix}_pvalue_lt_0_01": sum(1 for pvalue in valid if pvalue > P_VALUE_COMPLEMENT_LT_0_01_THRESHOLD)
        / valid_count,
        f"{prefix}_pvalue_success_rate": success_rate,
    }


def _normalise_config_value(value: Any) -> Any:
    """Normalise configuration values for JSON serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AttributionMethod):
        return value.value
    if isinstance(value, dict):
        return {key: _normalise_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_config_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalise_config_value(item) for item in value]
    return value


def _config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert PipelineConfig dataclass to a JSON-serializable dict."""
    return cast("dict[str, Any]", _normalise_config_value(asdict(config)))


def _compute_pattern_match_stats(
    results: list[NaturalLanguageExplanationResult],
    evaluations: set[str],
) -> dict[str, float]:
    """Compute cfx match, non-cfx match, and per-user pattern contrast stats."""
    stats: dict[str, float] = {}
    if "cfx_match" in evaluations:
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.explanation_cfx_match,
                prefix="explanation_cfx_pattern_match",
            )
        )
    if "non_cfx_match" in evaluations:
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.explanation_non_cfx_match,
                prefix="explanation_non_cfx_pattern_match",
            )
        )
    if "cfx_match" in evaluations and "non_cfx_match" in evaluations:
        contrast_scores = [
            safe_score(item.explanation_cfx_match) - safe_score(item.explanation_non_cfx_match) for item in results
        ]
        stats.update(
            flatten_summary(
                summarise_metric(contrast_scores, len(results)),
                "explanation_pattern_contrast",
            )
        )
    return stats


def _compute_distance_summary(
    distance_metrics_by_user: dict[int, dict[str, float]] | Any,
) -> dict[str, float]:
    """Compute success rates, means, and stds for all distance metrics."""
    total_users = len(distance_metrics_by_user)
    summary: dict[str, float] = {}
    for key in _DISTANCE_METRIC_KEYS:
        values = [
            float(metrics.get(key, float("nan"))) for metrics in distance_metrics_by_user.values() if key in metrics
        ]
        summary.update(flatten_summary(summarise_metric(values, total_users), key))
    return summary


def compute_explanation_statistics(
    pipeline_result: PipelineResult,
    *,
    enabled_evaluations: Collection[str] | None = None,
) -> dict[str, Any]:
    """Summarise generation and evaluation statistics for the evaluated explanations.

    The returned mapping contains ratios of successful generations and evaluations,
    as well as mean scores for each evaluation dimension, using only parsable
    (non-NaN) scores in the aggregates. When no evaluated explanations are
    available, an empty dictionary is returned.
    """
    explanations = pipeline_result.explanations
    outcome = pipeline_result.cfx_search_outcome
    n_with_cfx = outcome.n_valid + outcome.n_below_min_interactions
    total_users = n_with_cfx + outcome.n_no_cfx
    stats: dict[str, Any] = {
        "cfx_success_rate": n_with_cfx / total_users if total_users > 0 else float("nan"),
        "cfx_simple_rate": outcome.n_below_min_interactions / n_with_cfx if n_with_cfx > 0 else float("nan"),
    }
    if explanations is None or not explanations.results_by_user:
        return stats

    results = list(explanations.results_by_user.values())
    stats.update(_compute_generation_stats(results))

    evaluations = {
        name.strip().lower()
        for name in (enabled_evaluations or ("plausibility", "readability", "cfx_match", "non_cfx_match"))
    }

    if "plausibility" in evaluations:
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.explanation_plausibility,
                prefix="explanation_plausibility",
            )
        )

    stats.update(_compute_pattern_match_stats(results, evaluations))

    if "readability" in evaluations:
        total = len(results)
        for key in READABILITY_SUBSCORE_KEYS:
            scores = [_readability_subscore(item.explanation_readability, key) for item in results]
            stats.update(flatten_summary(summarise_metric(scores, total), f"readability_{key}"))
        overall_scores = [safe_score(item.explanation_readability) for item in results]
        stats.update(flatten_summary(summarise_metric(overall_scores, total), "readability_overall"))

    if "faithfulness_removal" in evaluations:
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.faithfulness_removal,
                prefix="faithfulness_removal",
            )
        )
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.faithfulness_removal_baseline,
                prefix="faithfulness_removal_baseline",
            )
        )
        removal_pvalues = [
            _compute_faithfulness_pvalue_complement(
                item.faithfulness_removal,
                item.faithfulness_removal_baseline,
                alternative="less",
            )
            for item in results
        ]
        stats["faithfulness_removal_pvalue_complement"] = summarise_metric(removal_pvalues, len(results)).mean
        stats.update(_compute_pvalue_stats(removal_pvalues, "faithfulness_removal"))

    if "faithfulness_replacement" in evaluations:
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.faithfulness_replacement,
                prefix="faithfulness_replacement",
            )
        )
        stats.update(
            _compute_evaluation_block(
                results,
                extractor=lambda item: item.faithfulness_replacement_baseline,
                prefix="faithfulness_replacement_baseline",
            )
        )
        replacement_pvalues = [
            _compute_faithfulness_pvalue_complement(
                item.faithfulness_replacement,
                item.faithfulness_replacement_baseline,
                alternative="greater",
            )
            for item in results
        ]
        stats["faithfulness_replacement_pvalue_complement"] = summarise_metric(replacement_pvalues, len(results)).mean
        stats.update(_compute_pvalue_stats(replacement_pvalues, "faithfulness_replacement"))

    if pipeline_result.distance_metrics_by_user:
        stats.update(_compute_distance_summary(pipeline_result.distance_metrics_by_user))

    return stats


def build_run_summary(
    pipeline_result: PipelineResult,
    config: PipelineConfig,
    *,
    enabled_evaluations: Collection[str] | None = None,
) -> dict[str, Any]:
    """Build the run summary containing config and evaluation results."""
    return {
        "config": _config_to_dict(config),
        "results": compute_explanation_statistics(
            pipeline_result,
            enabled_evaluations=enabled_evaluations,
        ),
    }
