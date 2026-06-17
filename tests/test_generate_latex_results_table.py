# ruff: noqa: S101, PLR2004, FBT003
"""Tests for the LaTeX results table generator script."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from scripts.generate_latex_results_table import (
    MetricValue,
    RowMetrics,
    RunIdentity,
    RunVariant,
    _build_table_rows,
    _is_counterfactual_cell,
    build_row_metrics,
    cfx_interaction_count_metric_for_run_dir,
    cfx_interaction_count_metric_from_dataframe,
    discover_runs,
    format_mean_std,
    format_single,
    generate_latex,
    parse_run_directory_name,
    rank_column,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# parse_run_directory_name
# ---------------------------------------------------------------------------


class TestParseRunDirectoryName:
    """Tests for directory name parsing."""

    def test_vanilla_cosine_gemma(self) -> None:
        result = parse_run_directory_name("run_pipeline_cosine_gemma-3-12b-it")
        assert result == RunIdentity(method="cosine", model="gemma-3-12b-it", is_dpo=False)

    def test_vanilla_jaccard_ministral(self) -> None:
        result = parse_run_directory_name("run_pipeline_jaccard_Ministral-8B-Instruct-2410")
        assert result == RunIdentity(method="jaccard", model="Ministral-8B-Instruct-2410", is_dpo=False)

    def test_dpo_after_method(self) -> None:
        result = parse_run_directory_name("run_pipeline_cosine_dpo_gemma-3-12b-it")
        assert result is not None
        assert result.method == "cosine"
        assert result.model == "gemma-3-12b-it"
        assert result.is_dpo is True

    def test_dpo_before_method(self) -> None:
        result = parse_run_directory_name("run_pipeline_dpo_lxr_gemma-3-12b-it")
        assert result is not None
        assert result.method == "lxr"
        assert result.model == "gemma-3-12b-it"
        assert result.is_dpo is True

    def test_spinrec(self) -> None:
        result = parse_run_directory_name("run_pipeline_spinrec_gemma-3-12b-it")
        assert result == RunIdentity(method="spinrec", model="gemma-3-12b-it", is_dpo=False)

    def test_vanilla_cosine_qwen3_8b(self) -> None:
        result = parse_run_directory_name("run_pipeline_cosine_Qwen3-8B")
        assert result == RunIdentity(method="cosine", model="Qwen3-8B", is_dpo=False)

    def test_lime(self) -> None:
        result = parse_run_directory_name("run_pipeline_lime_Ministral-8B-Instruct-2410")
        assert result == RunIdentity(method="lime", model="Ministral-8B-Instruct-2410", is_dpo=False)

    def test_no_prefix_returns_none(self) -> None:
        assert parse_run_directory_name("some_other_dir") is None

    def test_unknown_method_returns_none(self) -> None:
        assert parse_run_directory_name("run_pipeline_unknown_gemma-3-12b-it") is None

    def test_no_model_returns_none(self) -> None:
        assert parse_run_directory_name("run_pipeline_cosine") is None

    def test_empty_suffix_returns_none(self) -> None:
        assert parse_run_directory_name("run_pipeline_") is None

    def test_dpo_prefix_ministral(self) -> None:
        result = parse_run_directory_name("run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo")
        assert result == RunIdentity(method="accent", model="Ministral-8B-Instruct-2410", is_dpo=True)

    def test_dpo_prefix_gemma(self) -> None:
        result = parse_run_directory_name("run_eval_eval_dpo_eval_cosine_gemma-3-12b-it_dpo")
        assert result == RunIdentity(method="cosine", model="gemma-3-12b-it", is_dpo=True)

    def test_dpo_prefix_all_methods(self) -> None:
        for method in ("jaccard", "cosine", "lime", "shap", "accent", "lxr", "spinrec"):
            result = parse_run_directory_name(f"run_eval_eval_dpo_eval_{method}_gemma-3-12b-it_dpo")
            assert result is not None, f"Failed for method {method}"
            assert result.method == method
            assert result.model == "gemma-3-12b-it"
            assert result.is_dpo is True

    def test_dpo_prefix_empty_suffix_returns_none(self) -> None:
        assert parse_run_directory_name("run_eval_eval_dpo_eval_") is None


# ---------------------------------------------------------------------------
# format_mean_std / format_single
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for metric value formatting."""

    def test_format_mean_std_both_finite(self) -> None:
        metric = MetricValue(mean=0.41, std=0.12)
        assert format_mean_std(metric) == "0.41 $\\pm$ 0.12"

    def test_format_mean_std_nan_mean_returns_blank(self) -> None:
        metric = MetricValue(mean=float("nan"), std=0.05)
        assert format_mean_std(metric) == ""

    def test_format_mean_std_nan_std_omits_pm(self) -> None:
        metric = MetricValue(mean=0.50, std=float("nan"))
        assert format_mean_std(metric) == "0.50"

    def test_format_single_finite(self) -> None:
        assert format_single(0.92) == "0.92"

    def test_format_single_nan_returns_blank(self) -> None:
        assert format_single(float("nan")) == ""

    def test_rounding(self) -> None:
        metric = MetricValue(mean=0.784, std=0.127)
        text = format_mean_std(metric)
        assert text == "0.78 $\\pm$ 0.13"


# ---------------------------------------------------------------------------
# rank_column
# ---------------------------------------------------------------------------


class TestRankColumn:
    """Tests for column ranking logic."""

    def test_basic_ranking(self) -> None:
        values = [0.3, 0.5, 0.4]
        ranks = rank_column(values)
        assert ranks == [None, 1, 2]

    def test_nan_values_get_none(self) -> None:
        values = [float("nan"), 0.5, float("nan")]
        ranks = rank_column(values)
        assert ranks == [None, 1, None]

    def test_tie_for_best(self) -> None:
        values = [0.5, 0.5, 0.3]
        ranks = rank_column(values)
        assert ranks == [1, 1, 2]

    def test_tie_for_second(self) -> None:
        values = [0.5, 0.4, 0.4]
        ranks = rank_column(values)
        assert ranks == [1, 2, 2]

    def test_all_equal(self) -> None:
        values = [0.5, 0.5, 0.5]
        ranks = rank_column(values)
        assert ranks == [1, 1, 1]

    def test_all_nan(self) -> None:
        values = [float("nan"), float("nan")]
        ranks = rank_column(values)
        assert ranks == [None, None]

    def test_single_value(self) -> None:
        ranks = rank_column([0.5])
        assert ranks == [1]

    def test_empty(self) -> None:
        assert rank_column([]) == []

    def test_lower_is_better_ranking(self) -> None:
        values = [0.5, 0.3, 0.4]
        ranks = rank_column(values, higher_is_better=False)
        assert ranks == [None, 1, 2]

    def test_lower_is_better_tie_for_best(self) -> None:
        values = [0.5, 0.3, 0.3]
        ranks = rank_column(values, higher_is_better=False)
        assert ranks == [2, 1, 1]


# ---------------------------------------------------------------------------
# build_row_metrics
# ---------------------------------------------------------------------------


class TestCfxInteractionCountMetric:
    """Tests for CFX size computation from interactions.feather rows."""

    def test_counterfactual_cell_predicate(self) -> None:
        assert _is_counterfactual_cell(1) is True
        assert _is_counterfactual_cell(True) is True
        assert _is_counterfactual_cell(0) is False
        assert _is_counterfactual_cell(False) is False
        assert _is_counterfactual_cell("true") is True

    def test_multiple_users_with_cfx_rows(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "is_counterfactual": [1, 1, 0, 1, 0],
            },
        )
        metric = cfx_interaction_count_metric_from_dataframe(interactions)
        assert metric is not None
        assert metric.mean == pytest.approx(1.5)
        assert metric.std == pytest.approx(math.sqrt(0.5))

    def test_single_user_has_nan_std(self) -> None:
        interactions = pd.DataFrame(
            {
                "user_id": [1, 1],
                "is_counterfactual": [1, 1],
            },
        )
        metric = cfx_interaction_count_metric_from_dataframe(interactions)
        assert metric is not None
        assert metric.mean == pytest.approx(2.0)
        assert math.isnan(metric.std)

    def test_no_cfx_rows_returns_none(self) -> None:
        interactions = pd.DataFrame({"user_id": [1], "is_counterfactual": [0]})
        assert cfx_interaction_count_metric_from_dataframe(interactions) is None

    def test_missing_feather_returns_nan_metric(self, tmp_path: Path) -> None:
        metric = cfx_interaction_count_metric_for_run_dir(tmp_path)
        assert math.isnan(metric.mean)
        assert math.isnan(metric.std)


class TestBuildRowMetrics:
    """Tests for assembling row metrics from result dicts."""

    def test_full_vanilla_and_dpo(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "vanilla"
        run_dir.mkdir()
        pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "is_counterfactual": [1, 1, 0, 1, 0],
            },
        ).to_feather(run_dir / "interactions.feather")

        dpo = RunVariant(
            results={
                "explanation_cfx_pattern_match_mean": 0.60,
                "explanation_cfx_pattern_match_std": 0.11,
                "explanation_pattern_contrast_mean": 0.80,
                "explanation_pattern_contrast_std": 0.08,
                "readability_overall_mean": 0.85,
                "readability_overall_std": 0.05,
            },
            run_dir=tmp_path / "dpo",
            hmf_results={
                "explanation_cfx_pattern_human_feedback_model_match_mean": 0.42,
                "explanation_cfx_pattern_human_feedback_model_match_std": 0.06,
                "explanation_pattern_human_feedback_model_contrast_mean": 0.12,
                "explanation_pattern_human_feedback_model_contrast_std": 0.04,
                "readability_human_feedback_model_score_mean": 0.88,
                "readability_human_feedback_model_score_std": 0.03,
            },
        )
        vanilla = RunVariant(
            results={
                "cfx_success_rate": 0.92,
                "cfx_simple_rate": 0.15,
                "explanation_cfx_pattern_match_mean": 0.55,
                "explanation_cfx_pattern_match_std": 0.10,
                "explanation_pattern_contrast_mean": 0.78,
                "explanation_pattern_contrast_std": 0.09,
                "readability_overall_mean": 0.81,
                "readability_overall_std": 0.07,
            },
            run_dir=run_dir,
            hmf_results={
                "explanation_cfx_pattern_human_feedback_model_match_mean": 0.40,
                "explanation_cfx_pattern_human_feedback_model_match_std": 0.05,
                "explanation_pattern_human_feedback_model_contrast_mean": 0.10,
                "explanation_pattern_human_feedback_model_contrast_std": 0.03,
                "readability_human_feedback_model_score_mean": 0.91,
                "readability_human_feedback_model_score_std": 0.02,
            },
        )
        row = build_row_metrics(vanilla, dpo)
        assert row.cfx_size.mean == pytest.approx(1.5)
        assert row.cfx_size.std == pytest.approx(math.sqrt(0.5))
        assert row.cfx_success_rate == pytest.approx(0.92)
        assert row.cfx_simple_rate == pytest.approx(0.15)
        assert row.correctness_vanilla.mean == pytest.approx(0.55)
        assert row.correctness_dpo.mean == pytest.approx(0.60)
        assert row.informativeness_vanilla.mean == pytest.approx(0.78)
        assert row.informativeness_dpo.mean == pytest.approx(0.80)
        assert row.readability_vanilla.mean == pytest.approx(0.81)
        assert row.readability_dpo.std == pytest.approx(0.05)
        assert row.correctness_cal_vanilla.mean == pytest.approx(0.40)
        assert row.correctness_cal_dpo.mean == pytest.approx(0.42)
        assert row.informativeness_cal_vanilla.mean == pytest.approx(0.10)
        assert row.informativeness_cal_dpo.mean == pytest.approx(0.12)
        assert row.readability_cal_vanilla.mean == pytest.approx(0.91)
        assert row.readability_cal_dpo.mean == pytest.approx(0.88)

    def test_missing_dpo_gives_nan(self, tmp_path: Path) -> None:
        vanilla = RunVariant(results={"cfx_success_rate": 0.9}, run_dir=tmp_path)
        row = build_row_metrics(vanilla, None)
        assert math.isnan(row.correctness_dpo.mean)
        assert math.isnan(row.informativeness_dpo.mean)
        assert math.isnan(row.readability_dpo.mean)
        assert math.isnan(row.correctness_cal_dpo.mean)
        assert math.isnan(row.informativeness_cal_dpo.mean)
        assert math.isnan(row.readability_cal_dpo.mean)

    def test_missing_hmf_gives_nan_calibrated_metrics(self, tmp_path: Path) -> None:
        vanilla = RunVariant(
            results={"explanation_cfx_pattern_match_mean": 0.55},
            run_dir=tmp_path,
        )
        row = build_row_metrics(vanilla, None)
        assert row.correctness_vanilla.mean == pytest.approx(0.55)
        assert math.isnan(row.correctness_cal_vanilla.mean)
        assert math.isnan(row.informativeness_cal_vanilla.mean)
        assert math.isnan(row.readability_cal_vanilla.mean)

    def test_both_none_gives_all_nan(self) -> None:
        row = build_row_metrics(None, None)
        assert math.isnan(row.cfx_size.mean)
        assert math.isnan(row.cfx_success_rate)
        assert math.isnan(row.cfx_simple_rate)
        assert math.isnan(row.readability_vanilla.mean)
        assert math.isnan(row.readability_cal_vanilla.mean)


# ---------------------------------------------------------------------------
# discover_runs (integration with filesystem fixtures)
# ---------------------------------------------------------------------------


def _write_run_summary(run_dir: Path, results: dict[str, object], *, target_set: str = "test") -> None:
    """Write a minimal run_summary.json into a timestamped subdirectory."""
    ts_dir = run_dir / "2026-01-01T00:00:00+00:00"
    ts_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": {"target_set": target_set}, "results": results}
    (ts_dir / "run_summary.json").write_text(json.dumps(summary))


def _write_hmf_summary(run_dir: Path, results: dict[str, object], *, split: str = "test") -> None:
    """Write a minimal run_human_model_feedback_summary.json into a timestamped subdirectory."""
    ts_dir = run_dir / "2026-01-01T00:00:00+00:00"
    ts_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": {}, "results": {split: results}}
    (ts_dir / "run_human_model_feedback_summary.json").write_text(json.dumps(summary))


def _write_interactions_feather(run_dir: Path, interactions: pd.DataFrame) -> None:
    """Write interactions.feather next to run_summary.json for CFX-size computation."""
    ts_dir = run_dir / "2026-01-01T00:00:00+00:00"
    ts_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_feather(ts_dir / "interactions.feather")


class TestDiscoverRuns:
    """Tests for filesystem discovery."""

    def test_discovers_vanilla_and_dpo(self, tmp_path: Path) -> None:
        vanilla_dir = tmp_path / "run_pipeline_cosine_gemma-3-12b-it"
        dpo_dir = tmp_path / "run_pipeline_cosine_dpo_gemma-3-12b-it"
        _write_run_summary(vanilla_dir, {"cfx_success_rate": 0.9})
        _write_run_summary(dpo_dir, {"cfx_success_rate": 0.91})

        runs = discover_runs(tmp_path)
        key = ("cosine", "gemma-3-12b-it")
        assert key in runs
        assert "vanilla" in runs[key]
        assert "dpo" in runs[key]
        assert runs[key]["vanilla"].results["cfx_success_rate"] == 0.9

    def test_discovers_hmf_summaries(self, tmp_path: Path) -> None:
        vanilla_dir = tmp_path / "run_pipeline_cosine_gemma-3-12b-it"
        _write_run_summary(vanilla_dir, {"cfx_success_rate": 0.9})
        _write_hmf_summary(
            vanilla_dir,
            {
                "explanation_cfx_pattern_human_feedback_model_match_mean": 0.33,
                "explanation_cfx_pattern_human_feedback_model_match_std": 0.05,
            },
        )

        runs = discover_runs(tmp_path)
        key = ("cosine", "gemma-3-12b-it")
        vanilla_hmf = runs[key]["vanilla"].hmf_results
        assert vanilla_hmf is not None
        assert vanilla_hmf["explanation_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(0.33)

    def test_ignores_non_matching_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "some_other_dir").mkdir()
        runs = discover_runs(tmp_path)
        assert runs == {}

    def test_picks_latest_timestamp(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_pipeline_shap_gemma-3-12b-it"
        old_ts = run_dir / "2025-01-01T00:00:00+00:00"
        new_ts = run_dir / "2026-06-01T00:00:00+00:00"
        old_ts.mkdir(parents=True)
        new_ts.mkdir(parents=True)
        (old_ts / "run_summary.json").write_text(
            json.dumps({"config": {"target_set": "test"}, "results": {"cfx_success_rate": 0.5}}),
        )
        (new_ts / "run_summary.json").write_text(
            json.dumps({"config": {"target_set": "test"}, "results": {"cfx_success_rate": 0.9}}),
        )

        runs = discover_runs(tmp_path)
        assert runs[("shap", "gemma-3-12b-it")]["vanilla"].results["cfx_success_rate"] == 0.9

    def test_vanilla_uses_split_aware_latest_hmf_timestamp(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410"
        test_ts = run_dir / "2026-01-01T00:00:00+00:00"
        validation_ts = run_dir / "2026-02-01T00:00:00+00:00"
        test_ts.mkdir(parents=True)
        validation_ts.mkdir(parents=True)
        test_ts.joinpath("run_summary.json").write_text(
            json.dumps({"config": {"target_set": "test"}, "results": {"cfx_success_rate": 0.91}}),
        )
        validation_ts.joinpath("run_summary.json").write_text(
            json.dumps({"config": {"target_set": "validation"}, "results": {"cfx_success_rate": 0.22}}),
        )
        test_ts.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "test": {"explanation_cfx_pattern_human_feedback_model_match_mean": 0.66},
                    },
                },
            ),
        )
        validation_ts.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "validation": {
                            "explanation_cfx_pattern_human_feedback_model_match_mean": 0.11,
                        },
                    },
                },
            ),
        )

        runs = discover_runs(tmp_path)
        vanilla = runs[("accent", "Ministral-8B-Instruct-2410")]["vanilla"]
        assert vanilla.run_dir == validation_ts
        assert vanilla.results["cfx_success_rate"] == pytest.approx(0.22)
        assert vanilla.hmf_results is not None
        assert vanilla.hmf_results["explanation_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(
            0.66,
        )

    def test_vanilla_uses_latest_timestamp_when_split_exists_there(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410"
        test_ts = run_dir / "2026-01-01T00:00:00+00:00"
        validation_ts = run_dir / "2026-02-01T00:00:00+00:00"
        test_ts.mkdir(parents=True)
        validation_ts.mkdir(parents=True)
        test_ts.joinpath("run_summary.json").write_text(
            json.dumps({"config": {"target_set": "test"}, "results": {"cfx_success_rate": 0.91}}),
        )
        validation_ts.joinpath("run_summary.json").write_text(
            json.dumps({"config": {"target_set": "validation"}, "results": {"cfx_success_rate": 0.22}}),
        )
        test_ts.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "test": {"explanation_cfx_pattern_human_feedback_model_match_mean": 0.66},
                    },
                },
            ),
        )
        validation_ts.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "test": {
                            "explanation_cfx_pattern_human_feedback_model_match_mean": 0.11,
                        },
                    },
                },
            ),
        )

        runs = discover_runs(tmp_path)
        vanilla = runs[("accent", "Ministral-8B-Instruct-2410")]["vanilla"]
        assert vanilla.run_dir == validation_ts
        assert vanilla.results["cfx_success_rate"] == pytest.approx(0.22)
        assert vanilla.hmf_results is not None
        assert vanilla.hmf_results["explanation_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(
            0.11,
        )

    def test_discovers_dpo_prefix_runs(self, tmp_path: Path) -> None:
        vanilla_dir = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410"
        dpo_dir = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        _write_run_summary(vanilla_dir, {"cfx_success_rate": 0.8})
        _write_run_summary(dpo_dir, {"explanation_pattern_contrast_mean": 0.75})

        runs = discover_runs(tmp_path)
        key = ("accent", "Ministral-8B-Instruct-2410")
        assert key in runs
        assert "vanilla" in runs[key]
        assert "dpo" in runs[key]
        assert runs[key]["vanilla"].results["cfx_success_rate"] == 0.8
        assert runs[key]["dpo"].results["explanation_pattern_contrast_mean"] == 0.75

    def test_discovers_dpo_eval_sweep_best_trial(self, tmp_path: Path) -> None:
        """Select by validation calibrated composite, display test split metrics."""
        dpo_root = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_ts = dpo_root / "2026-01-01T00:00:00+00:00"
        test_ts = dpo_root / "2026-01-02T00:00:00+00:00"
        val_sweep = val_ts / "sweep"
        test_sweep = test_ts / "sweep"
        val_low = val_sweep / "low"
        val_high = val_sweep / "high"
        test_low = test_sweep / "low"
        test_high = test_sweep / "high"
        for trial_dir in (val_low, val_high, test_low, test_high):
            trial_dir.mkdir(parents=True)
        val_low.joinpath("run_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {"explanation_cfx_pattern_match_mean": 0.11},
                },
            ),
        )
        val_high.joinpath("run_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {"explanation_cfx_pattern_match_mean": 0.99},
                },
            ),
        )
        test_low.joinpath("run_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {"explanation_cfx_pattern_match_mean": 0.33},
                },
            ),
        )
        test_high.joinpath("run_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {"explanation_cfx_pattern_match_mean": 0.44},
                },
            ),
        )
        val_low.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {"validation": {"reward_composite_human_feedback_model": 1.0}},
                },
            ),
        )
        val_high.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "validation": {"reward_composite_human_feedback_model": 2.0},
                    },
                },
            ),
        )
        test_high.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps(
                {
                    "config": {},
                    "results": {
                        "test": {"explanation_cfx_pattern_human_feedback_model_match_mean": 0.77},
                    },
                },
            ),
        )

        trials_payload = {
            "top_k": 2,
            "trials": [
                {
                    "run_id": "low",
                    "score": 1.0,
                    "trial_dir": "/other-machine/lowtrial",
                    "retained": True,
                },
                {
                    "run_id": "high",
                    "score": 2.0,
                    "trial_dir": "/other-machine/hightrial",
                    "retained": True,
                },
            ],
        }
        val_sweep.joinpath("trials.json").write_text(json.dumps(trials_payload))
        test_sweep.joinpath("trials.json").write_text(json.dumps(trials_payload))

        runs = discover_runs(tmp_path)
        key = ("accent", "Ministral-8B-Instruct-2410")
        assert runs[key]["dpo"].results["explanation_cfx_pattern_match_mean"] == pytest.approx(0.44)
        assert runs[key]["dpo"].run_dir == test_high.resolve()
        dpo_hmf = runs[key]["dpo"].hmf_results
        assert dpo_hmf is not None
        assert dpo_hmf["explanation_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(0.77)

    def test_dpo_best_trial_tie_uses_deterministic_run_id_order(self, tmp_path: Path) -> None:
        dpo_root = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        test_ts = dpo_root / "2026-01-02T00:00:00+00:00"
        test_sweep = test_ts / "sweep"
        high = test_sweep / "high"
        low = test_sweep / "low"
        for trial_dir in (high, low):
            trial_dir.mkdir(parents=True)
        high.joinpath("run_summary.json").write_text(
            json.dumps({"config": {}, "results": {"explanation_cfx_pattern_match_mean": 0.44}}),
        )
        low.joinpath("run_summary.json").write_text(
            json.dumps({"config": {}, "results": {"explanation_cfx_pattern_match_mean": 0.33}}),
        )
        high.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps({"config": {}, "results": {"validation": {"reward_composite_human_feedback_model": 1.0}}}),
        )
        low.joinpath("run_human_model_feedback_summary.json").write_text(
            json.dumps({"config": {}, "results": {"validation": {"reward_composite_human_feedback_model": 1.0}}}),
        )
        trials_payload = {
            "top_k": 2,
            "trials": [{"run_id": "low", "score": 1.0}, {"run_id": "high", "score": 1.0}],
        }
        test_sweep.joinpath("trials.json").write_text(json.dumps(trials_payload))

        runs = discover_runs(tmp_path)
        key = ("accent", "Ministral-8B-Instruct-2410")
        assert runs[key]["dpo"].results["explanation_cfx_pattern_match_mean"] == pytest.approx(0.44)

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_runs(tmp_path / "nonexistent") == {}


# ---------------------------------------------------------------------------
# End-to-end table generation
# ---------------------------------------------------------------------------


class TestGenerateLatex:
    """Tests for end-to-end LaTeX generation."""

    def test_generates_valid_table_structure(self, tmp_path: Path) -> None:
        # Use the first model in _MODEL_ORDER so method-level columns are populated.
        run_root = tmp_path / "run_pipeline_cosine_Ministral-8B-Instruct-2410"
        _write_run_summary(
            run_root,
            {
                "cfx_success_rate": 0.93,
                "cfx_simple_rate": 0.12,
                "explanation_cfx_pattern_match_mean": 0.55,
                "explanation_cfx_pattern_match_std": 0.07,
                "explanation_pattern_contrast_mean": 0.80,
                "explanation_pattern_contrast_std": 0.08,
                "readability_overall_mean": 0.83,
                "readability_overall_std": 0.06,
            },
        )
        _write_hmf_summary(
            run_root,
            {
                "explanation_cfx_pattern_human_feedback_model_match_mean": 0.40,
                "explanation_cfx_pattern_human_feedback_model_match_std": 0.05,
                "explanation_pattern_human_feedback_model_contrast_mean": 0.10,
                "explanation_pattern_human_feedback_model_contrast_std": 0.03,
                "readability_human_feedback_model_score_mean": 0.91,
                "readability_human_feedback_model_score_std": 0.02,
            },
        )
        _write_interactions_feather(
            run_root,
            pd.DataFrame(
                {
                    "user_id": [1, 1, 2, 2, 3],
                    "is_counterfactual": [1, 1, 0, 1, 0],
                },
            ),
        )

        latex = generate_latex(tmp_path)
        assert r"\begin{table*}[t]" in latex
        assert r"\end{table*}" in latex
        assert r"\begin{tabular}" in latex
        assert "Cosine" in latex
        assert "Ministral" in latex
        assert "1.50 $\\pm$ 0.71" in latex
        assert "0.93" in latex
        assert "0.12" in latex
        assert r"CFX Success\\Rate" in latex
        assert latex.index(r"CFX\\Size $\downarrow$") < latex.index(r"Simple CFX\\Rate")
        assert "Correctness" in latex
        assert "Informativeness" in latex
        assert "Linguistic Quality" in latex
        assert "(cal.)" not in latex
        assert "human-calibrated scores" in latex
        assert "Vanilla LLM & DPO" in latex
        assert "V & D" not in latex
        assert "0.40 $\\pm$ 0.05" in latex
        assert "0.91 $\\pm$ 0.02" in latex
        assert "Fluency" not in latex
        assert "Overall" not in latex
        assert "Specif." not in latex
        assert "Sep." not in latex

    def test_cfx_size_highlighting_prefers_lower_values(self, tmp_path: Path) -> None:
        small_root = tmp_path / "run_pipeline_cosine_Ministral-8B-Instruct-2410"
        large_root = tmp_path / "run_pipeline_lxr_Ministral-8B-Instruct-2410"
        for run_root, success_rate in ((small_root, 0.9), (large_root, 0.95)):
            _write_run_summary(run_root, {"cfx_success_rate": success_rate})
        _write_interactions_feather(
            small_root,
            pd.DataFrame({"user_id": [1, 2], "is_counterfactual": [1, 1]}),
        )
        _write_interactions_feather(
            large_root,
            pd.DataFrame(
                {
                    "user_id": [1, 1, 2, 2, 3, 3],
                    "is_counterfactual": [1, 1, 1, 1, 1, 1],
                },
            ),
        )

        latex = generate_latex(tmp_path)
        assert r"\textbf{1.00 $\pm$ 0.00}" in latex
        assert r"\underline{2.00 $\pm$ 0.00}" in latex

    def test_highlighting_applied(self, tmp_path: Path) -> None:
        # Use the same model for both methods so both have LLM-level metrics.
        cosine_root = tmp_path / "run_pipeline_cosine_Ministral-8B-Instruct-2410"
        lxr_root = tmp_path / "run_pipeline_lxr_Ministral-8B-Instruct-2410"
        _write_run_summary(
            cosine_root,
            {
                "cfx_success_rate": 0.90,
                "cfx_simple_rate": 0.10,
                "readability_overall_mean": 0.80,
                "readability_overall_std": 0.05,
            },
        )
        _write_run_summary(
            lxr_root,
            {
                "cfx_success_rate": 0.95,
                "cfx_simple_rate": 0.20,
                "readability_overall_mean": 0.90,
                "readability_overall_std": 0.03,
            },
        )
        _write_hmf_summary(
            cosine_root,
            {
                "readability_human_feedback_model_score_mean": 0.80,
                "readability_human_feedback_model_score_std": 0.05,
            },
        )
        _write_hmf_summary(
            lxr_root,
            {
                "readability_human_feedback_model_score_mean": 0.90,
                "readability_human_feedback_model_score_std": 0.03,
            },
        )

        latex = generate_latex(tmp_path)
        # LXR has the best readability so it should be bold.
        assert r"\textbf{0.90 $\pm$ 0.03}" in latex
        # Cosine has second-best readability so it should be underlined.
        assert r"\underline{0.80 $\pm$ 0.05}" in latex

    def test_empty_outputs_dir(self, tmp_path: Path) -> None:
        latex = generate_latex(tmp_path)
        assert r"\begin{table*}[t]" in latex
        assert r"\end{table*}" in latex

    def test_method_order(self, tmp_path: Path) -> None:
        for method in ("lxr", "jaccard", "cosine"):
            _write_run_summary(
                tmp_path / f"run_pipeline_{method}_gemma-3-12b-it",
                {"cfx_success_rate": 0.9},
            )
        latex = generate_latex(tmp_path)
        jaccard_pos = latex.index("Jaccard")
        cosine_pos = latex.index("Cosine")
        lxr_pos = latex.index("LXR")
        assert jaccard_pos < cosine_pos < lxr_pos


class TestBuildTableRows:
    """Tests for the internal _build_table_rows helper."""

    def test_blank_cells_not_highlighted(self) -> None:
        nan_mv = MetricValue(float("nan"), float("nan"))
        row = RowMetrics(
            cfx_size=nan_mv,
            cfx_success_rate=float("nan"),
            cfx_simple_rate=float("nan"),
            correctness_vanilla=nan_mv,
            correctness_dpo=nan_mv,
            informativeness_vanilla=nan_mv,
            informativeness_dpo=nan_mv,
            readability_vanilla=nan_mv,
            readability_dpo=nan_mv,
            correctness_cal_vanilla=nan_mv,
            correctness_cal_dpo=nan_mv,
            informativeness_cal_vanilla=nan_mv,
            informativeness_cal_dpo=nan_mv,
            readability_cal_vanilla=nan_mv,
            readability_cal_dpo=nan_mv,
        )
        rows = _build_table_rows([("cosine", "gemma-3-12b-it", row)])
        assert len(rows[0]) == 9
        for cell in rows[0]:
            assert "textbf" not in cell
            assert "underline" not in cell
