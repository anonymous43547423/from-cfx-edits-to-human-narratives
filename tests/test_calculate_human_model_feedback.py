# ruff: noqa: S101, PLR2004, SLF001
"""Tests for scripts/calculate_human_model_feedback.py."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts import calculate_human_model_feedback as script
from scripts.gather_eval_datasets import interaction_description_from_exported_interaction_row


def _experiment_with_timestamp(root: Path, name: str, ts_name: str = "2026-01-01T00:00:00+00:00") -> Path:
    """Create ``root/name/ts_name`` and return the timestamp path."""
    ts = root / name / ts_name
    ts.mkdir(parents=True, exist_ok=True)
    return ts


def _write_generation(run_dir: Path, user_ids: list[int], text_prefix: str) -> None:
    """Write ``generation.feather`` for the given users."""
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = pd.DataFrame(
        {
            "user_id": user_ids,
            "explanation_text": [f"{text_prefix}-{uid}" for uid in user_ids],
        },
    )
    gen.to_feather(run_dir / "generation.feather")


def _write_run_summary(
    run_dir: Path,
    *,
    target_set: str,
    user_pool: str = "eval",
    reward_metric_name: str = "correctness",
) -> None:
    """Write a minimal sibling ``run_summary.json`` used by HMF split logic."""
    payload = {
        "config": {"target_set": target_set, "user_pool": user_pool},
        "results": {"reward_metric_name": reward_metric_name},
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_match_feathers(
    run_dir: Path,
    *,
    user_ids: list[int],
    explanation_text_by_user: dict[int, str],
    interactions: pd.DataFrame,
    cfx_rows: pd.DataFrame | None = None,
    non_cfx_rows: pd.DataFrame | None = None,
) -> None:
    """Write generation, interactions, and optional match-detail feathers."""
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = pd.DataFrame(
        [{"user_id": uid, "explanation_text": explanation_text_by_user.get(uid, "")} for uid in user_ids],
    )
    gen.to_feather(run_dir / "generation.feather")
    interactions.to_feather(run_dir / "interactions.feather")
    if cfx_rows is not None and not cfx_rows.empty:
        cfx_rows.to_feather(run_dir / "cfx_match_details.feather")
    if non_cfx_rows is not None and not non_cfx_rows.empty:
        non_cfx_rows.to_feather(run_dir / "non_cfx_match_details.feather")


def _write_hmf_summary(run_dir: Path, *, validation: dict[str, object] | None = None) -> None:
    """Write a nested-split HMF summary fixture."""
    results: dict[str, object] = {}
    if validation is not None:
        results["validation"] = validation
    payload = {
        "config": {
            "readability_human_feedback_model_path": "m",
            "interaction_human_feedback_model_path": "m",
            "batch_size": 8,
            "hmf_split": "validation",
        },
        "results": results,
    }
    (run_dir / "run_human_model_feedback_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _assert_e2e_summary(summary: dict[str, object], model_path: Path) -> None:
    """Assert expected config and results in the e2e summary fixture."""
    assert summary["config"] == {
        "readability_human_feedback_model_path": str(model_path),
        "interaction_human_feedback_model_path": str(model_path),
        "batch_size": 8,
        "hmf_split": "validation",
        "target_set": "validation",
        "user_pool": "eval",
    }
    run_results = summary["results"]
    assert isinstance(run_results, dict)
    validation = run_results["validation"]
    assert isinstance(validation, dict)
    assert validation["readability_human_feedback_model_score_mean"] == pytest.approx(0.5)
    assert validation["readability_human_feedback_model_score_success_rate"] == pytest.approx(1.0)
    assert validation["readability_human_feedback_model_score_std"] == pytest.approx(math.sqrt(0.5))
    assert validation["explanation_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(0.0)
    assert validation["explanation_cfx_pattern_human_feedback_model_match_success_rate"] == pytest.approx(0.5)
    assert validation["explanation_cfx_pattern_human_feedback_model_match_std"] is None
    assert validation["explanation_non_cfx_pattern_human_feedback_model_match_mean"] == pytest.approx(1.0)
    assert validation["explanation_non_cfx_pattern_human_feedback_model_match_success_rate"] == pytest.approx(0.5)
    assert validation["explanation_non_cfx_pattern_human_feedback_model_match_std"] is None
    assert validation["explanation_pattern_human_feedback_model_contrast_mean"] is None
    assert validation["explanation_pattern_human_feedback_model_contrast_success_rate"] == pytest.approx(0.0)
    assert validation["explanation_pattern_human_feedback_model_contrast_std"] is None
    assert validation["reward_metric_name_human_feedback_model"] == "correctness"
    assert validation["reward_composite_human_feedback_model"] == pytest.approx(0.0)


class TestDiscoverRunLeaves:
    """Run-leaf discovery across independent and sweep outputs."""

    def test_non_sweep_timestamp_dir(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_pipeline_a")
        _write_generation(ts, [1, 2], "a")

        leaves = script.discover_run_leaves(tmp_path)
        assert leaves == [ts]

    def test_sweep_pools_all_trials(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_eval_sweep")
        sweep = ts / "sweep"
        t1 = sweep / "t1"
        t2 = sweep / "t2"
        _write_generation(t1, [1], "t1")
        _write_generation(t2, [2], "t2")

        leaves = script.discover_run_leaves(tmp_path)
        assert leaves == [t1, t2]

    def test_skips_dirs_without_generation(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_pipeline_a")
        (ts / "sweep" / "empty_trial").mkdir(parents=True)

        leaves = script.discover_run_leaves(tmp_path)
        assert leaves == []

    def test_multiple_timestamps(self, tmp_path: Path) -> None:
        top = tmp_path / "run_pipeline_a"
        ts1 = top / "2026-01-01T00:00:00+00:00"
        ts2 = top / "2026-01-02T00:00:00+00:00"
        _write_generation(ts1, [1], "a")
        _write_generation(ts2, [2], "b")

        leaves = script.discover_run_leaves(tmp_path)
        assert leaves == [ts1, ts2]


class TestLoadMatchDetailsInputs:
    """Interaction input building for inference."""

    def test_interaction_description_matches_gather_helper(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        interactions = pd.DataFrame(
            {
                "interaction_id": [0],
                "user_id": [1],
                "item_id": [42],
                "rating": [5.0],
                "attribution_score": [0.4],
                "is_counterfactual": [True],
            },
        )
        _write_match_feathers(
            run_dir,
            user_ids=[1],
            explanation_text_by_user={1: "exp text"},
            interactions=interactions,
            cfx_rows=pd.DataFrame(
                {"user_id": [1], "interaction_id": [0], "score": [1.0], "judgment": ["yes"]},
            ),
        )

        inputs = script.load_match_details_inputs(run_dir, "cfx_match_details.feather")
        assert inputs is not None
        row = interactions.iloc[0].to_dict()
        row_for_desc = {str(k): v for k, v in row.items()}
        expected = interaction_description_from_exported_interaction_row(row_for_desc)
        assert inputs.iloc[0]["interaction_description"] == expected
        assert inputs.iloc[0]["explanation_text"] == "exp text"


class TestPredictLabels:
    """Batched argmax inference."""

    def test_single_sequence_argmax(self) -> None:
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        model = MagicMock()
        model.return_value.logits = torch.tensor(
            [
                [2.0, 0.5],
                [0.1, 3.0],
            ],
        )

        labels = script.predict_labels(
            model,
            tokenizer,
            ["a", "b"],
            batch_size=2,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        assert labels == [0, 1]
        tokenizer.assert_called_once_with(
            ["a", "b"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    def test_pair_sequence_argmax(self) -> None:
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        model = MagicMock()
        model.return_value.logits = torch.tensor([[0.2, 1.5]])

        labels = script.predict_labels(
            model,
            tokenizer,
            ["exp"],
            ["desc"],
            batch_size=8,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        assert labels == [1]
        tokenizer.assert_called_once_with(
            ["exp"],
            ["desc"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )


class TestHumanModelFeedbackRunConfig:
    """Run configuration serialization for summary JSON."""

    def test_from_args_and_to_dict(self, tmp_path: Path) -> None:
        readability_path = tmp_path / "readability-model"
        interaction_path = tmp_path / "interaction-model"
        args = script.parse_args(
            [
                "--outputs-dir",
                str(tmp_path),
                "--readability-human-feedback-model-path",
                str(readability_path),
                "--interaction-human-feedback-model-path",
                str(interaction_path),
                "--hmf-split",
                "validation",
                "--batch-size",
                "16",
            ],
        )

        config = script.HumanModelFeedbackRunConfig.from_args(args)

        assert config.readability_human_feedback_model_path == readability_path
        assert config.interaction_human_feedback_model_path == interaction_path
        assert config.batch_size == 16
        assert config.to_dict() == {
            "readability_human_feedback_model_path": str(readability_path),
            "interaction_human_feedback_model_path": str(interaction_path),
            "hmf_split": "validation",
            "batch_size": 16,
        }


class TestComputeHumanModelFeedbackResults:
    """Run-level aggregation of human-feedback scores."""

    def test_contrast_when_user_has_both_match_scores(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _write_generation(run_dir, [1], "a")

        results = script.compute_human_model_feedback_results(
            run_dir,
            readability_df=None,
            cfx_match_df=pd.DataFrame(
                {"user_id": [1, 1], "human_feedback_model_score": [1, 1]},
            ),
            non_cfx_match_df=pd.DataFrame(
                {"user_id": [1], "human_feedback_model_score": [0]},
            ),
            scored_cfx_match=True,
            scored_non_cfx_match=True,
        )

        assert results["explanation_pattern_human_feedback_model_contrast_mean"] == pytest.approx(1.0)
        assert results["explanation_pattern_human_feedback_model_contrast_success_rate"] == pytest.approx(1.0)
        assert math.isnan(results["explanation_pattern_human_feedback_model_contrast_std"])


class TestSplitSelection:
    """Run-leaf selection for validation and test HMF passes."""

    def test_validation_split_selects_all_discovered_runs(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_ts = exp / "2026-01-01T00:00:00+00:00" / "sweep"
        test_ts = exp / "2026-01-02T00:00:00+00:00" / "sweep"
        val_trial = val_ts / "low"
        test_trial = test_ts / "high"
        for run in (val_trial, test_trial):
            _write_generation(run, [1], "x")
        _write_run_summary(val_trial, target_set="validation")
        _write_run_summary(test_trial, target_set="test")

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        selected = script._select_run_leaves(metadata, "validation")
        assert {selection.run_leaf for selection in selected} == {val_trial, test_trial}

    def test_test_split_selects_only_best_validation_trial(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        sweep_ts = exp / "2026-01-02T00:00:00+00:00" / "sweep"
        trial_low = sweep_ts / "low"
        trial_high = sweep_ts / "high"
        for run in (trial_low, trial_high):
            _write_generation(run, [1], "x")
        _write_run_summary(trial_low, target_set="test")
        _write_run_summary(trial_high, target_set="test")
        _write_hmf_summary(
            trial_low,
            validation={script._HMF_REWARD_COMPOSITE_KEY: 0.1},
        )
        _write_hmf_summary(
            trial_high,
            validation={script._HMF_REWARD_COMPOSITE_KEY: 0.9},
        )

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        selected = script._select_run_leaves(metadata, "test")
        assert [s.run_leaf for s in selected] == [trial_high]
        assert selected[0].validation_results_for_merge is not None
        assert selected[0].validation_results_for_merge[script._HMF_REWARD_COMPOSITE_KEY] == pytest.approx(0.9)

    def test_test_split_uses_deterministic_tiebreak_on_validation_score(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        sweep_ts = exp / "2026-01-02T00:00:00+00:00" / "sweep"
        trial_low = sweep_ts / "low"
        trial_high = sweep_ts / "high"
        for run in (trial_low, trial_high):
            _write_generation(run, [1], "x")
            _write_run_summary(run, target_set="test")
            _write_hmf_summary(run, validation={script._HMF_REWARD_COMPOSITE_KEY: 0.5})

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        selected = script._select_run_leaves(metadata, "test")
        # Lexicographic tie-break picks "high" over "low".
        assert [s.run_leaf for s in selected] == [trial_high]

    def test_test_split_fails_when_validation_hmf_is_unavailable(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_trial = exp / "2026-01-01T00:00:00+00:00" / "sweep" / "low"
        test_trial = exp / "2026-01-02T00:00:00+00:00" / "sweep" / "low"
        _write_generation(val_trial, [1], "v")
        _write_generation(test_trial, [1], "t")
        _write_run_summary(val_trial, target_set="validation")
        _write_run_summary(test_trial, target_set="test")

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        with pytest.raises(ValueError, match="Validation HMF calibrated composite not available"):
            script._select_run_leaves(metadata, "test")

    def test_test_split_fails_when_best_run_id_is_duplicated(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        first = exp / "2026-01-01T00:00:00+00:00" / "sweep" / "high"
        second = exp / "2026-01-02T00:00:00+00:00" / "sweep" / "high"
        for run in (first, second):
            _write_generation(run, [1], "x")
            _write_run_summary(run, target_set="test")
        _write_hmf_summary(first, validation={script._HMF_REWARD_COMPOSITE_KEY: 0.9})

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        with pytest.raises(ValueError, match="Duplicate test sweep run id"):
            script._select_run_leaves(metadata, "test")

    def test_test_split_includes_nonsweep_vanilla_run(self, tmp_path: Path) -> None:
        vanilla_test = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410" / "2026-01-02T00:00:00+00:00"
        _write_generation(vanilla_test, [1], "v")
        _write_run_summary(vanilla_test, target_set="test")

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        selected = script._select_run_leaves(metadata, "test")
        assert [selection.run_leaf for selection in selected] == [vanilla_test]
        assert selected[0].validation_results_for_merge is None

    def test_test_split_includes_vanilla_and_best_dpo_trial(self, tmp_path: Path) -> None:
        dpo_exp = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        sweep_ts = dpo_exp / "2026-01-02T00:00:00+00:00" / "sweep"
        trial_low = sweep_ts / "low"
        trial_high = sweep_ts / "high"
        for run in (trial_low, trial_high):
            _write_generation(run, [1], "x")
        _write_run_summary(trial_low, target_set="test")
        _write_run_summary(trial_high, target_set="test")
        _write_hmf_summary(trial_low, validation={script._HMF_REWARD_COMPOSITE_KEY: 0.1})
        _write_hmf_summary(trial_high, validation={script._HMF_REWARD_COMPOSITE_KEY: 0.9})

        vanilla_test = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410" / "2026-01-02T00:00:00+00:00"
        _write_generation(vanilla_test, [2], "v")
        _write_run_summary(vanilla_test, target_set="test")

        metadata = [script._build_run_leaf_metadata(path) for path in script.discover_run_leaves(tmp_path)]
        selected = script._select_run_leaves(metadata, "test")
        selected_paths = {selection.run_leaf for selection in selected}
        assert selected_paths == {trial_high, vanilla_test}
        by_path = {selection.run_leaf: selection for selection in selected}
        assert by_path[vanilla_test].validation_results_for_merge is None
        assert by_path[trial_high].validation_results_for_merge is not None


class TestWriteHumanModelFeedbackSummary:
    """Tests for summary JSON serialization."""

    def test_nan_written_as_null(self, tmp_path: Path) -> None:
        summary = {
            "config": {},
            "results": {
                "validation": {
                    "score_mean": 0.5,
                    "reward_composite_human_feedback_model": float("nan"),
                },
            },
        }
        script.write_human_model_feedback_summary(tmp_path, summary)
        raw = (tmp_path / "run_human_model_feedback_summary.json").read_text(encoding="utf-8")
        # Must be valid JSON (no bare NaN token).
        parsed = json.loads(raw)
        assert parsed["results"]["validation"]["reward_composite_human_feedback_model"] is None
        assert parsed["results"]["validation"]["score_mean"] == pytest.approx(0.5)

    def test_finite_values_preserved(self, tmp_path: Path) -> None:
        summary = {"config": {}, "results": {"test": {"score_mean": 0.75, "score_std": 0.1}}}
        script.write_human_model_feedback_summary(tmp_path, summary)
        parsed = json.loads((tmp_path / "run_human_model_feedback_summary.json").read_text(encoding="utf-8"))
        assert parsed["results"]["test"]["score_mean"] == pytest.approx(0.75)
        assert parsed["results"]["test"]["score_std"] == pytest.approx(0.1)


class TestMainEndToEnd:
    """End-to-end scoring with mocked transformers models."""

    def test_main_writes_all_output_feathers(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_pipeline_a")
        interactions = pd.DataFrame(
            {
                "interaction_id": [0, 1],
                "user_id": [1, 2],
                "item_id": [10, 11],
                "rating": [5.0, 4.0],
                "attribution_score": [0.5, 0.4],
                "is_counterfactual": [True, False],
            },
        )
        _write_match_feathers(
            ts,
            user_ids=[1, 2],
            explanation_text_by_user={1: "exp-1", 2: "exp-2"},
            interactions=interactions,
            cfx_rows=pd.DataFrame(
                {
                    "user_id": [1],
                    "interaction_id": [0],
                    "score": [1.0],
                    "judgment": ["yes"],
                },
            ),
            non_cfx_rows=pd.DataFrame(
                {
                    "user_id": [2],
                    "interaction_id": [1],
                    "score": [0.0],
                    "judgment": ["no"],
                },
            ),
        )
        _write_run_summary(ts, target_set="validation", reward_metric_name="correctness")

        readability_model = MagicMock()
        interaction_model = MagicMock()
        readability_logits = torch.tensor([[3.0, 0.1], [0.2, 2.0]])

        def _readability_forward(**_kwargs: Any) -> MagicMock:
            out = MagicMock()
            out.logits = readability_logits
            return out

        interaction_call = {"n": 0}

        def _interaction_forward(**_kwargs: Any) -> MagicMock:
            out = MagicMock()
            if interaction_call["n"] == 0:
                out.logits = torch.tensor([[2.0, 0.1]])
            else:
                out.logits = torch.tensor([[0.1, 2.5]])
            interaction_call["n"] += 1
            return out

        readability_model.side_effect = _readability_forward
        interaction_model.side_effect = _interaction_forward
        readability_model.to.return_value = readability_model
        interaction_model.to.return_value = interaction_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        model_path = tmp_path / "model"
        model_path.mkdir()

        with (
            patch.object(script, "AutoTokenizer") as mock_tok_cls,
            patch.object(script, "AutoModelForSequenceClassification") as mock_model_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.side_effect = [readability_model, interaction_model]
            code = script.main(
                [
                    "--outputs-dir",
                    str(tmp_path),
                    "--readability-human-feedback-model-path",
                    str(model_path),
                    "--interaction-human-feedback-model-path",
                    str(model_path),
                    "--hmf-split",
                    "validation",
                    "--batch-size",
                    "8",
                ],
            )

        assert code == 0

        readability_out = pd.read_feather(ts / "evaluation_human_feedback_model.feather")
        assert list(readability_out.columns) == ["user_id", "readability_human_feedback_model_score"]
        assert len(readability_out) == 2
        assert readability_out["readability_human_feedback_model_score"].tolist() == [0, 1]

        cfx_out = pd.read_feather(ts / "cfx_match_details_human_feedback_model.feather")
        assert list(cfx_out.columns) == ["interaction_id", "human_feedback_model_score", "user_id"]
        assert len(cfx_out) == 1
        assert cfx_out["human_feedback_model_score"].tolist() == [0]

        non_cfx_out = pd.read_feather(ts / "non_cfx_match_details_human_feedback_model.feather")
        assert list(non_cfx_out.columns) == ["interaction_id", "human_feedback_model_score", "user_id"]
        assert len(non_cfx_out) == 1
        assert non_cfx_out["human_feedback_model_score"].tolist() == [1]

        summary_path = ts / "run_human_model_feedback_summary.json"
        assert summary_path.is_file()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _assert_e2e_summary(summary, model_path)

    def test_parse_args_requires_model_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """parse_args exits when required paths are omitted."""
        monkeypatch.setattr("sys.argv", ["calculate_human_model_feedback.py"])
        with pytest.raises(SystemExit):
            script.parse_args()
