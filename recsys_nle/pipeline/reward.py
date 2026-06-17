"""DPO reward definitions as linear combinations of pipeline evaluation aggregates."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping


class RewardType(StrEnum):
    """Supported reward types for DPO training and sweep objectives."""

    INFORMATIVENESS = "informativeness"
    CORRECTNESS = "correctness"
    CORRECTNESS_INFORMATIVENESS = "correctness_informativeness"
    CORRECTNESS_INFORMATIVENESS_READABILITY = "correctness_informativeness_readability"
    FAITHFULNESS = "faithfulness"


# Each reward is a linear combination of evaluation columns: score = sum(coeff * values[col]).
REWARD_TERMS: dict[RewardType, dict[str, float]] = {
    RewardType.INFORMATIVENESS: {
        "explanation_cfx_pattern_match_mean": 1.0,
        "explanation_non_cfx_pattern_match_mean": -1.0,
    },
    RewardType.CORRECTNESS: {
        "explanation_cfx_pattern_match_mean": 1.0,
    },
    RewardType.CORRECTNESS_INFORMATIVENESS: {
        "explanation_cfx_pattern_match_mean": 1.5,
        "explanation_non_cfx_pattern_match_mean": -1.0,
    },
    RewardType.CORRECTNESS_INFORMATIVENESS_READABILITY: {
        "explanation_cfx_pattern_match_mean": 1.5,
        "explanation_non_cfx_pattern_match_mean": -1.0,
        "readability_overall_mean": 1.5,
    },
    RewardType.FAITHFULNESS: {
        "faithfulness_removal_pvalue_complement": 1.0,
        "faithfulness_replacement_pvalue_complement": 1.0,
    },
}


def compute_reward(values: Mapping[str, float], reward_type: RewardType) -> float:
    """Return the reward as a linear combination of aggregate metric columns in ``values``."""
    return sum(coeff * float(values[column]) for column, coeff in REWARD_TERMS[reward_type].items())


def reward_composite_for_results(
    results: Mapping[str, object],
    reward_type: RewardType,
) -> tuple[float, frozenset[str]]:
    """Return ``(reward_composite, missing_columns)`` for pipeline ``results`` aggregates.

    The score is NaN if any required key is missing, non-numeric, nested, or non-finite.
    When the score is finite, ``missing_columns`` is empty; otherwise it lists unusable keys.
    """
    missing: set[str] = set()
    metric_values: dict[str, float] = {}
    for column in REWARD_TERMS[reward_type]:
        raw = results.get(column)
        if raw is None or not isinstance(raw, (int, float, str)):
            missing.add(column)
            continue
        try:
            value = float(raw)
        except ValueError:
            missing.add(column)
            continue
        if not math.isfinite(value):
            missing.add(column)
            continue
        metric_values[column] = value
    if missing:
        return float("nan"), frozenset(missing)
    return compute_reward(metric_values, reward_type), frozenset()


def apply_reward_composite_to_summary_results(
    results: MutableMapping[str, object],
    reward_type: RewardType,
) -> frozenset[str]:
    """Write ``reward_composite`` and ``reward_metric_name`` into ``results``; return missing columns."""
    score, missing = reward_composite_for_results(results, reward_type)
    results["reward_composite"] = score
    results["reward_metric_name"] = reward_type.value
    return missing
