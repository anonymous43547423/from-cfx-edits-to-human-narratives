"""Unified metric aggregation helpers for pipeline evaluation statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from recsys_nle.nl_explanations.evaluation.base import EvaluationResult

_MIN_STD_SAMPLES = 2


@dataclass(slots=True, frozen=True)
class MetricSummary:
    """Aggregated statistics for a single metric across users."""

    success_rate: float
    mean: float
    std: float


def summarise_metric(values: Sequence[float], total: int) -> MetricSummary:
    """Compute success rate, mean, and std over finite values."""
    finite = [v for v in values if math.isfinite(v)]
    n = len(finite)
    return MetricSummary(
        success_rate=n / total if total > 0 else float("nan"),
        mean=float(sum(finite) / n) if n else float("nan"),
        std=float(statistics.stdev(finite)) if n >= _MIN_STD_SAMPLES else float("nan"),
    )


def flatten_summary(summary: MetricSummary, prefix: str) -> dict[str, float]:
    """Expand a MetricSummary into a flat dict with prefixed keys."""
    return {
        f"{prefix}_success_rate": summary.success_rate,
        f"{prefix}_mean": summary.mean,
        f"{prefix}_std": summary.std,
    }


def safe_score(evaluation: EvaluationResult | None) -> float:
    """Return a numeric score or NaN; no clamping (evaluators handle bounds)."""
    if evaluation is None:
        return float("nan")
    score = float(evaluation.score)
    if not math.isfinite(score):
        return float("nan")
    return score
