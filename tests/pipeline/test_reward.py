# ruff: noqa: S101, SLF001, PLR2004
"""Tests for DPO reward definitions shared with the pipeline."""

from __future__ import annotations

import math

import pandas as pd

from recsys_nle.pipeline.reward import (
    REWARD_TERMS,
    RewardType,
    apply_reward_composite_to_summary_results,
    compute_reward,
    reward_composite_for_results,
)
from scripts import run_dpo


def test_compute_reward_matches_run_dpo_row_score() -> None:
    """``compute_reward`` matches ``run_dpo._calculate_score`` for suffixed rows."""
    row = pd.Series(
        {
            "explanation_cfx_pattern_match_mean_a": 0.8,
            "explanation_non_cfx_pattern_match_mean_a": 0.2,
            "readability_overall_mean_a": 0.5,
            "faithfulness_removal_pvalue_complement_a": 0.9,
            "faithfulness_replacement_pvalue_complement_a": 0.85,
        }
    )
    for reward_type in RewardType:
        assert compute_reward(
            {c: float(row[f"{c}_a"]) for c in REWARD_TERMS[reward_type]},
            reward_type,
        ) == run_dpo._calculate_score(row, reward_type, "_a")


def test_reward_composite_for_results_missing_key() -> None:
    """Missing aggregates yield NaN and report missing columns."""
    results: dict[str, object] = {"explanation_cfx_pattern_match_mean": 0.5}
    score, missing = reward_composite_for_results(results, RewardType.INFORMATIVENESS)
    assert math.isnan(score)
    assert "explanation_non_cfx_pattern_match_mean" in missing


def test_apply_reward_composite_to_summary_results() -> None:
    """In-place summary update adds ``reward_composite`` and metric name."""
    results: dict[str, object] = {
        "explanation_cfx_pattern_match_mean": 1.0,
        "explanation_non_cfx_pattern_match_mean": 0.0,
    }
    missing = apply_reward_composite_to_summary_results(results, RewardType.CORRECTNESS_INFORMATIVENESS)
    assert not missing
    assert results["reward_metric_name"] == RewardType.CORRECTNESS_INFORMATIVENESS.value
    assert results["reward_composite"] == 1.5
