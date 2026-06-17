# ruff: noqa: S101, PLR2004
"""Tests for scripts/gather_eval_datasets.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from recsys_nle.nl_explanations.evaluation.cfx_match import CFXMatchEvaluator
from recsys_nle.nl_explanations.evaluation.readability import READABILITY_SUBSCORE_KEYS
from scripts.gather_eval_datasets import (
    _row_dedup_key,
    _rows_with_sequential_ids,
    gather_interaction_match_rows,
    gather_readability_rows,
    interaction_description_from_exported_interaction_row,
    main,
    parse_args,
    raise_if_eval_outputs_exist,
    write_interaction_match_csvs,
    write_readability_csvs,
)


def _eval_frame(user_ids: list[int]) -> pd.DataFrame:
    """Build a minimal evaluation.feather-compatible frame."""
    rows: dict[str, object] = {"user_id": user_ids}
    for i, key in enumerate(READABILITY_SUBSCORE_KEYS):
        rows[f"readability_{key}_mean"] = [0.5 + j * 0.1 + i * 0.01 for j in range(len(user_ids))]
    rows["readability_overall_mean"] = [0.75 + j * 0.05 for j in range(len(user_ids))]
    return pd.DataFrame(rows)


def _write_run_leaf(run_dir: Path, user_ids: list[int], text_prefix: str) -> None:
    """Write generation.feather and evaluation.feather under ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    gen = pd.DataFrame(
        {
            "user_id": user_ids,
            "explanation_text": [f"{text_prefix}-{uid}" for uid in user_ids],
        },
    )
    gen.to_feather(run_dir / "generation.feather")
    _eval_frame(user_ids).to_feather(run_dir / "evaluation.feather")


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


def _experiment_with_timestamp(root: Path, name: str, ts_name: str = "2026-01-01T00:00:00+00:00") -> Path:
    """Create ``root/name/ts_name`` and return the timestamp path."""
    ts = root / name / ts_name
    ts.mkdir(parents=True, exist_ok=True)
    return ts


def _write_run_summary(run_dir: Path, *, target_set: str) -> None:
    """Write minimal run_summary.json with target_set."""
    (run_dir / "run_summary.json").write_text(
        json.dumps({"config": {"target_set": target_set}, "results": {}}),
        encoding="utf-8",
    )


def _write_hmf_summary(run_dir: Path, *, validation_score: float) -> None:
    """Write validation HMF composite summary."""
    (run_dir / "run_human_model_feedback_summary.json").write_text(
        json.dumps(
            {
                "config": {},
                "results": {"validation": {"reward_composite_human_feedback_model": validation_score}},
            },
        ),
        encoding="utf-8",
    )


def _write_trials_json(sweep_dir: Path, run_ids: list[str]) -> None:
    """Write sweep trials.json with provided run ids."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    payload = {"top_k": len(run_ids), "trials": [{"run_id": run_id, "score": 0.0} for run_id in run_ids]}
    (sweep_dir / "trials.json").write_text(json.dumps(payload), encoding="utf-8")


class TestParseArgs:
    """Argument parsing behavior for gather_eval_datasets CLI."""

    def test_requires_best_runs_only(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--data-source-dir",
                    str(tmp_path),
                    "--n-readability-samples",
                    "1",
                    "--n-interaction-match-samples",
                    "1",
                ],
            )

    def test_parses_best_runs_only_true_false(self, tmp_path: Path) -> None:
        true_args = parse_args(
            [
                "--data-source-dir",
                str(tmp_path),
                "--n-readability-samples",
                "1",
                "--n-interaction-match-samples",
                "1",
                "--best-runs-only",
                "true",
            ],
        )
        false_args = parse_args(
            [
                "--data-source-dir",
                str(tmp_path),
                "--n-readability-samples",
                "1",
                "--n-interaction-match-samples",
                "1",
                "--best-runs-only",
                "false",
            ],
        )
        assert true_args.best_runs_only is True
        assert false_args.best_runs_only is False

    def test_rejects_invalid_best_runs_only_value(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--data-source-dir",
                    str(tmp_path),
                    "--n-readability-samples",
                    "1",
                    "--n-interaction-match-samples",
                    "1",
                    "--best-runs-only",
                    "yes",
                ],
            )


class TestGatherReadabilityRows:
    """Filesystem integration tests."""

    def test_samples_from_multiple_tops(self, tmp_path: Path) -> None:
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "run_pipeline_a"), list(range(10)), "a")
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "run_pipeline_b"), list(range(10, 20)), "b")

        ai, human = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=20,
            random_seed=0,
        )
        assert len(ai) == len(human) == 20
        paths: list[str] = [str(r["generation_feather_path"]) for r in ai]
        assert any("run_pipeline_a" in p for p in paths)
        assert any("run_pipeline_b" in p for p in paths)

        dedup_keys = {_row_dedup_key(row) for row in ai}
        assert len(dedup_keys) == 20

        for row in human:
            for key in READABILITY_SUBSCORE_KEYS:
                assert row[key] == ""
            assert row["overall"] == ""

        for a, h in zip(ai, human, strict=True):
            assert a["generation_feather_path"] == h["generation_feather_path"]
            assert a["user_id"] == h["user_id"]
            assert a["explanation_text"] == h["explanation_text"]
            assert a["overall"] != ""

    def test_sweep_samples_from_trials_and_other_tops(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_eval_sweep")
        sweep = ts / "sweep"
        _write_run_leaf(sweep / "t1", [1, 2], "t1")
        _write_run_leaf(sweep / "t2", [3, 4, 5], "t2")

        _write_run_leaf(_experiment_with_timestamp(tmp_path, "other_run"), [10, 11, 12, 13], "o")

        ai, _ = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=4,
            random_seed=99,
        )
        assert len(ai) == 4
        norm_paths = [str(r["generation_feather_path"]).replace("\\", "/") for r in ai]
        assert any("/sweep/" in p for p in norm_paths)
        assert any("other_run" in p for p in norm_paths)

    def test_skips_empty_top_level_dir(self, tmp_path: Path) -> None:
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "run_with_data"), [1, 2, 3], "ok")
        _experiment_with_timestamp(tmp_path, "run_failed_empty")

        ai, human = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=2,
            random_seed=0,
        )
        assert len(ai) == len(human) == 2
        assert all("run_with_data" in str(r["generation_feather_path"]) for r in ai)

    def test_skips_duplicate_rows(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "dup_run")
        ts.mkdir(parents=True, exist_ok=True)
        gen = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "explanation_text": ["dup", "dup", "unique"],
            },
        )
        gen.to_feather(ts / "generation.feather")
        dup_eval = _eval_frame([1])
        eval_df = pd.concat([dup_eval, dup_eval, _eval_frame([2])], ignore_index=True)
        eval_df.to_feather(ts / "evaluation.feather")

        ai, _ = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=2,
            random_seed=0,
        )
        assert len(ai) == 2
        assert len({_row_dedup_key(row) for row in ai}) == 2

    def test_warns_on_consecutive_duplicates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "warn_run"), [1, 2], "w")

        row_pick_count = 0
        original_choice = random.Random.choice

        def patched_choice(self: random.Random, seq: list[object]) -> object:
            if seq and isinstance(seq[0], dict):
                nonlocal row_pick_count
                row_pick_count += 1
                if row_pick_count <= 101:
                    return seq[0]
                return seq[1]
            return original_choice(self, seq)

        monkeypatch.setattr(random.Random, "choice", patched_choice)

        with caplog.at_level("WARNING"):
            ai, _ = gather_readability_rows(
                data_source_dir=tmp_path,
                n_samples=2,
                random_seed=0,
            )

        assert len(ai) == 2
        assert any("Skipped 100 consecutive duplicate rows while sampling" in r.message for r in caplog.records)

    def test_raises_if_not_exactly_one_timestamp_subdir(self, tmp_path: Path) -> None:
        top = tmp_path / "run_pipeline_x"
        top.mkdir()
        (top / "a").mkdir()
        (top / "b").mkdir()

        with pytest.raises(ValueError, match="exactly one timestamp"):
            gather_readability_rows(data_source_dir=tmp_path, n_samples=1, random_seed=0)

    def test_raises_when_unique_pool_exhausted(self, tmp_path: Path) -> None:
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "run_a"), [1], "a")
        _write_run_leaf(_experiment_with_timestamp(tmp_path, "run_b"), [2, 3, 4], "b")

        with pytest.raises(ValueError, match="Not enough unique explanations"):
            gather_readability_rows(data_source_dir=tmp_path, n_samples=5, random_seed=0)

    def test_best_runs_only_true_selects_best_sweep_trial(self, tmp_path: Path) -> None:
        top = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_sweep = top / "2026-01-01T00:00:00+00:00" / "sweep"
        test_sweep = top / "2026-01-02T00:00:00+00:00" / "sweep"
        val_low = val_sweep / "low"
        val_high = val_sweep / "high"
        test_low = test_sweep / "low"
        test_high = test_sweep / "high"
        _write_run_leaf(val_low, [1, 2], "val-low")
        _write_run_leaf(val_high, [3, 4], "val-high")
        _write_run_leaf(test_low, [5, 6], "test-low")
        _write_run_leaf(test_high, [7, 8], "test-high")
        _write_hmf_summary(val_low, validation_score=0.1)
        _write_hmf_summary(val_high, validation_score=0.9)
        _write_trials_json(val_sweep, ["low", "high"])
        _write_trials_json(test_sweep, ["low", "high"])

        ai, _ = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=2,
            random_seed=7,
            best_runs_only=True,
        )
        paths = {str(row["generation_feather_path"]) for row in ai}
        assert all("/high/" in path.replace("\\", "/") for path in paths)

    def test_best_runs_only_true_includes_vanilla_test_and_best_dpo(self, tmp_path: Path) -> None:
        dpo_top = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_sweep = dpo_top / "2026-01-01T00:00:00+00:00" / "sweep"
        test_sweep = dpo_top / "2026-01-02T00:00:00+00:00" / "sweep"
        val_low = val_sweep / "low"
        val_high = val_sweep / "high"
        test_low = test_sweep / "low"
        test_high = test_sweep / "high"
        _write_run_leaf(val_low, [1], "v-low")
        _write_run_leaf(val_high, [2], "v-high")
        _write_run_leaf(test_low, [3], "t-low")
        _write_run_leaf(test_high, [4], "t-high")
        _write_hmf_summary(val_low, validation_score=0.2)
        _write_hmf_summary(val_high, validation_score=0.8)
        _write_trials_json(val_sweep, ["low", "high"])
        _write_trials_json(test_sweep, ["low", "high"])

        vanilla_top = tmp_path / "run_pipeline_accent_Ministral-8B-Instruct-2410"
        vanilla_test = vanilla_top / "2026-01-02T00:00:00+00:00"
        _write_run_leaf(vanilla_test, [10], "vanilla")
        _write_run_summary(vanilla_test, target_set="test")

        ai, _ = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=2,
            random_seed=3,
            best_runs_only=True,
        )
        sources = [str(row["generation_feather_path"]).replace("\\", "/") for row in ai]
        assert any("run_pipeline_accent_Ministral-8B-Instruct-2410" in source for source in sources)
        assert any("/high/" in source for source in sources)

    def test_best_runs_only_true_fails_when_non_sweep_summary_missing(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_pipeline_x")
        _write_run_leaf(ts, [1], "x")
        with pytest.raises(ValueError, match=r"Missing latest non-sweep run leaf with run_summary\.json"):
            gather_readability_rows(data_source_dir=tmp_path, n_samples=1, random_seed=0, best_runs_only=True)

    def test_best_runs_only_true_fails_when_validation_hmf_missing(self, tmp_path: Path) -> None:
        top = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_sweep = top / "2026-01-01T00:00:00+00:00" / "sweep"
        test_sweep = top / "2026-01-02T00:00:00+00:00" / "sweep"
        val_trial = val_sweep / "low"
        test_trial = test_sweep / "low"
        _write_run_leaf(val_trial, [1], "v")
        _write_run_leaf(test_trial, [2], "t")
        _write_run_summary(val_trial, target_set="validation")
        _write_run_summary(test_trial, target_set="test")
        _write_trials_json(val_sweep, ["low"])
        _write_trials_json(test_sweep, ["low"])

        with pytest.raises(ValueError, match="Validation HMF calibrated composite not available"):
            gather_readability_rows(data_source_dir=tmp_path, n_samples=1, random_seed=0, best_runs_only=True)

    def test_best_runs_only_true_uses_deterministic_tiebreak_on_validation(self, tmp_path: Path) -> None:
        top = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_sweep = top / "2026-01-01T00:00:00+00:00" / "sweep"
        test_sweep = top / "2026-01-02T00:00:00+00:00" / "sweep"
        for run_id in ("low", "high"):
            val_trial = val_sweep / run_id
            test_trial = test_sweep / run_id
            _write_run_leaf(val_trial, [1], f"v-{run_id}")
            _write_run_leaf(test_trial, [2], f"t-{run_id}")
            _write_hmf_summary(val_trial, validation_score=0.5)
        _write_trials_json(val_sweep, ["low", "high"])
        _write_trials_json(test_sweep, ["low", "high"])

        ai, _ = gather_readability_rows(
            data_source_dir=tmp_path,
            n_samples=1,
            random_seed=0,
            best_runs_only=True,
        )
        source = str(ai[0]["generation_feather_path"]).replace("\\", "/")
        assert "/high/" in source

    def test_best_runs_only_true_fails_when_best_trial_missing_in_test(self, tmp_path: Path) -> None:
        top = tmp_path / "run_eval_eval_dpo_eval_accent_Ministral-8B-Instruct-2410_dpo"
        val_sweep = top / "2026-01-01T00:00:00+00:00" / "sweep"
        test_sweep = top / "2026-01-02T00:00:00+00:00" / "sweep"
        val_trial = val_sweep / "high"
        _write_run_leaf(val_trial, [1], "v")
        _write_run_summary(val_trial, target_set="validation")
        _write_hmf_summary(val_trial, validation_score=0.9)
        _write_trials_json(val_sweep, ["high"])
        _write_trials_json(test_sweep, ["low"])
        test_low = test_sweep / "low"
        _write_run_leaf(test_low, [2], "t")
        _write_run_summary(test_low, target_set="test")

        with pytest.raises(ValueError, match="not present in latest sweep timestamp"):
            gather_readability_rows(data_source_dir=tmp_path, n_samples=1, random_seed=0, best_runs_only=True)


class TestGatherInteractionMatchRows:
    """Interaction-match sampling."""

    def _minimal_interactions(self, user_id: int, item_id: int) -> pd.DataFrame:
        """Single interaction row with a new ``interaction_id`` index."""
        return pd.DataFrame(
            {
                "interaction_id": [0],
                "user_id": [user_id],
                "item_id": [item_id],
                "rating": [5.0],
                "attribution_score": [0.4],
                "is_counterfactual": [True],
            },
        )

    def test_samples_from_multiple_tops(self, tmp_path: Path) -> None:
        for name, uid in (("run_a", 1), ("run_b", 2)):
            ts = _experiment_with_timestamp(tmp_path, name)
            inter = self._minimal_interactions(uid, 0)
            _write_match_feathers(
                ts,
                user_ids=[uid],
                explanation_text_by_user={uid: f"e-{uid}"},
                interactions=inter,
                cfx_rows=pd.DataFrame(
                    {"user_id": [uid], "interaction_id": [0], "score": [0.5], "judgment": ["a"]},
                ),
                non_cfx_rows=pd.DataFrame(
                    {"user_id": [uid], "interaction_id": [0], "score": [0.2], "judgment": ["b"]},
                ),
            )

        ai, human = gather_interaction_match_rows(
            data_source_dir=tmp_path,
            n_samples=4,
            random_seed=0,
        )
        assert len(ai) == len(human) == 4
        assert any("run_a" in str(r["match_details_feather_path"]) for r in ai)
        assert any("run_b" in str(r["match_details_feather_path"]) for r in ai)
        output_keys = {(r["match_details_feather_path"], r["user_id"], r["interaction_id"], r["score"]) for r in ai}
        assert len(output_keys) == 4

        for a, h in zip(ai, human, strict=True):
            for key in (
                "match_details_feather_path",
                "user_id",
                "interaction_id",
                "explanation_text",
                "interaction_description",
            ):
                assert a[key] == h[key]
            assert h["score"] == ""
            assert "judgment" not in h

    def test_pools_cfx_and_non_cfx_uniformly_in_one_top(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "one_top")
        inter = self._minimal_interactions(1, 0)
        _write_match_feathers(
            ts,
            user_ids=[1],
            explanation_text_by_user={1: "expl"},
            interactions=inter,
            cfx_rows=pd.DataFrame(
                {"user_id": [1], "interaction_id": [0], "score": [0.9], "judgment": ["cfx"]},
            ),
            non_cfx_rows=pd.DataFrame(
                {"user_id": [1], "interaction_id": [0], "score": [0.1], "judgment": ["non"]},
            ),
        )
        ai, _ = gather_interaction_match_rows(data_source_dir=tmp_path, n_samples=2, random_seed=42)
        paths = [Path(p).name for p in (str(r["match_details_feather_path"]) for r in ai)]
        assert "cfx_match_details.feather" in paths
        assert "non_cfx_match_details.feather" in paths

    def test_sweep_samples_from_trials_and_other_tops(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_sweep_im")
        sweep = ts / "sweep"
        for tname, uid in (("t1", 1), ("t2", 2)):
            inter = self._minimal_interactions(uid, 0)
            _write_match_feathers(
                sweep / tname,
                user_ids=[uid],
                explanation_text_by_user={uid: f"x-{uid}"},
                interactions=inter,
                cfx_rows=pd.DataFrame(
                    {"user_id": [uid], "interaction_id": [0], "score": [0.5], "judgment": ["j"]},
                ),
            )
        other = _experiment_with_timestamp(tmp_path, "other_im")
        inter2 = self._minimal_interactions(3, 0)
        _write_match_feathers(
            other,
            user_ids=[3],
            explanation_text_by_user={3: "x-3"},
            interactions=inter2,
            cfx_rows=pd.DataFrame(
                {"user_id": [3], "interaction_id": [0], "score": [0.5], "judgment": ["j"]},
            ),
        )

        ai, _ = gather_interaction_match_rows(data_source_dir=tmp_path, n_samples=2, random_seed=7)
        assert len(ai) == 2
        norm_paths = [str(r["match_details_feather_path"]).replace("\\", "/") for r in ai]
        assert any("/sweep/" in p for p in norm_paths)
        assert any("other_im" in p for p in norm_paths)

    def test_raises_when_unique_pool_exhausted(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "small")
        inter = self._minimal_interactions(1, 0)
        _write_match_feathers(
            ts,
            user_ids=[1],
            explanation_text_by_user={1: "e"},
            interactions=inter,
            cfx_rows=pd.DataFrame(
                {"user_id": [1], "interaction_id": [0], "score": [0.5], "judgment": ["j"]},
            ),
        )
        with pytest.raises(ValueError, match="Not enough unique interaction-match rows"):
            gather_interaction_match_rows(data_source_dir=tmp_path, n_samples=2, random_seed=0)

    def test_interaction_description_computed_only_for_sampled_rows(self, tmp_path: Path) -> None:
        """Many candidate match rows share one interaction; descriptions match eager computation."""
        ts = _experiment_with_timestamp(tmp_path, "defer_desc")
        n_candidates = 40
        inter = pd.DataFrame(
            {
                "interaction_id": [0],
                "user_id": [1],
                "item_id": [1],
                "rating": [5.0],
                "attribution_score": [0.5],
                "is_counterfactual": [True],
            },
        )
        row_for_desc: dict[str, object] = {
            "interaction_id": 0,
            "user_id": 1,
            "item_id": 1,
            "rating": 5.0,
            "attribution_score": 0.5,
            "is_counterfactual": True,
        }
        expected_desc = interaction_description_from_exported_interaction_row(row_for_desc)
        _write_match_feathers(
            ts,
            user_ids=[1],
            explanation_text_by_user={1: "one-explanation"},
            interactions=inter,
            cfx_rows=pd.DataFrame(
                {
                    "user_id": [1] * n_candidates,
                    "interaction_id": [0] * n_candidates,
                    "score": [float(i) / 100 for i in range(n_candidates)],
                    "judgment": ["j"] * n_candidates,
                },
            ),
        )
        ai, human = gather_interaction_match_rows(
            data_source_dir=tmp_path,
            n_samples=3,
            random_seed=99,
        )
        assert len(ai) == len(human) == 3
        for r in ai:
            assert r["interaction_description"] == expected_desc
            assert "_interaction_desc_source" not in r


class TestInteractionDescriptionConsistent:
    """``interaction_description`` matches interaction-scoring preparation."""

    def test_matches_evaluator_single_row(self) -> None:
        """Exported ``item_id`` row yields the same line as evaluator ``movie_id`` frame."""
        item_id = 1
        rating = 4.5
        exported: dict[str, object] = {"item_id": item_id, "rating": rating, "attribution_score": 0.3}
        evaluator = CFXMatchEvaluator()
        direct = evaluator.build_all_prompts(
            explanation="some pattern text",
            interactions=pd.DataFrame([{"movie_id": item_id, "rating": rating, "weight": 0.3, "importance": 0.3}]),
        )[0][0]
        assert interaction_description_from_exported_interaction_row(exported) == direct


class TestRowsWithSequentialIds:
    """Sequential id assignment for exported CSV rows."""

    def test_assigns_ids_from_one(self) -> None:
        rows: list[dict[str, object]] = [{"explanation_text": "a"}, {"explanation_text": "b"}]
        numbered = _rows_with_sequential_ids(rows, ["id", "explanation_text"])
        assert numbered == [{"id": 1, "explanation_text": "a"}, {"id": 2, "explanation_text": "b"}]


class TestWriteReadabilityCsvs:
    """CSV output tests."""

    def test_roundtrip_columns_match(self, tmp_path: Path) -> None:
        stub_path = str(tmp_path / "stub-generation.feather")
        scores = dict.fromkeys(READABILITY_SUBSCORE_KEYS, 0.5)
        ai_rows = [
            {
                "generation_feather_path": stub_path,
                "user_id": 1,
                "explanation_text": "hello",
                **scores,
                "overall": 0.75,
            },
        ]
        human_rows = [{**dict(ai_rows[0]), **dict.fromkeys(READABILITY_SUBSCORE_KEYS, ""), "overall": ""}]

        ai_path, human_path = write_readability_csvs(
            output_dir=tmp_path,
            ai_rows=ai_rows,
            human_rows=human_rows,
        )
        ai_df = pd.read_csv(ai_path)
        human_df = pd.read_csv(human_path)
        assert list(ai_df.columns) == list(human_df.columns)
        assert next(iter(ai_df.columns)) == "id"
        assert ai_df.loc[0, "id"] == 1
        assert pd.isna(human_df.loc[0, "fluency"]) or human_df.loc[0, "fluency"] == ""
        assert pd.isna(human_df.loc[0, "specificity"]) or human_df.loc[0, "specificity"] == ""


class TestWriteInteractionMatchCsvs:
    """Interaction-match CSV columns."""

    def test_human_has_no_judgment_column(self, tmp_path: Path) -> None:
        ai_rows = [
            {
                "match_details_feather_path": "/x/cfx_match_details.feather",
                "user_id": 1,
                "interaction_id": 0,
                "explanation_text": "e",
                "interaction_description": "{year=2000}",
                "score": 0.5,
                "judgment": "gj",
            },
        ]
        human_rows = [
            {
                "match_details_feather_path": ai_rows[0]["match_details_feather_path"],
                "user_id": 1,
                "interaction_id": 0,
                "explanation_text": "e",
                "interaction_description": "{year=2000}",
                "score": "",
            },
        ]
        ai_p, hum_p = write_interaction_match_csvs(
            output_dir=tmp_path,
            ai_rows=ai_rows,
            human_rows=human_rows,
        )
        ai_df = pd.read_csv(ai_p)
        hum_df = pd.read_csv(hum_p)
        assert "judgment" in ai_df.columns
        assert "judgment" not in hum_df.columns
        assert next(iter(ai_df.columns)) == "id"
        assert ai_df.loc[0, "id"] == 1
        assert hum_df.loc[0, "id"] == 1


class TestMainCli:
    """Smoke test for CLI."""

    def test_main_zero_exit(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_one")
        _write_run_leaf(ts, [1, 2], "x")
        inter = pd.DataFrame(
            {
                "interaction_id": [0, 1],
                "user_id": [1, 2],
                "item_id": [0, 1],
                "rating": [5.0, 4.0],
                "attribution_score": [0.5, 0.4],
                "is_counterfactual": [True, False],
            },
        )
        _write_match_feathers(
            ts,
            user_ids=[1, 2],
            explanation_text_by_user={1: "a", 2: "b"},
            interactions=inter,
            cfx_rows=pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "interaction_id": [0, 1],
                    "score": [0.8, 0.7],
                    "judgment": ["ok", "ok"],
                },
            ),
            non_cfx_rows=pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "interaction_id": [0, 1],
                    "score": [0.2, 0.3],
                    "judgment": ["n", "n"],
                },
            ),
        )
        out = tmp_path / "out"
        code = main(
            [
                "--data-source-dir",
                str(tmp_path),
                "--n-readability-samples",
                "2",
                "--n-interaction-match-samples",
                "2",
                "--output-dir",
                str(out),
                "--random-seed",
                "1",
                "--best-runs-only",
                "false",
            ],
        )
        assert code == 0
        assert (out / "readability-ai-labeled.csv").is_file()
        assert (out / "readability-human-labeled.csv").is_file()
        assert (out / "interaction-match-ai-labeled.csv").is_file()
        assert (out / "interaction-match-human-labeled.csv").is_file()
        readability_df = pd.read_csv(out / "readability-human-labeled.csv")
        interaction_df = pd.read_csv(out / "interaction-match-human-labeled.csv")
        assert list(readability_df["id"]) == list(range(1, len(readability_df) + 1))
        assert list(interaction_df["id"]) == list(range(1, len(interaction_df) + 1))


class TestNoOverwriteOutputs:
    """CLI refuses to run when output CSVs already exist."""

    def test_raise_if_eval_outputs_exist_empty_dir(self, tmp_path: Path) -> None:
        """Absent files do not raise."""
        raise_if_eval_outputs_exist(tmp_path / "missing_parent" / "out")

    def test_raise_if_eval_outputs_exist_when_present(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "readability-ai-labeled.csv").write_text("stub\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Refusing to overwrite"):
            raise_if_eval_outputs_exist(out)

    def test_main_does_not_write_when_any_output_exists(self, tmp_path: Path) -> None:
        ts = _experiment_with_timestamp(tmp_path, "run_one")
        _write_run_leaf(ts, [1, 2], "x")
        inter = pd.DataFrame(
            {
                "interaction_id": [0, 1],
                "user_id": [1, 2],
                "item_id": [0, 1],
                "rating": [5.0, 4.0],
                "attribution_score": [0.5, 0.4],
                "is_counterfactual": [True, False],
            },
        )
        _write_match_feathers(
            ts,
            user_ids=[1, 2],
            explanation_text_by_user={1: "a", 2: "b"},
            interactions=inter,
            cfx_rows=pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "interaction_id": [0, 1],
                    "score": [0.8, 0.7],
                    "judgment": ["ok", "ok"],
                },
            ),
            non_cfx_rows=pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "interaction_id": [0, 1],
                    "score": [0.2, 0.3],
                    "judgment": ["n", "n"],
                },
            ),
        )
        out = tmp_path / "out"
        out.mkdir()
        (out / "interaction-match-human-labeled.csv").write_text("keep\n", encoding="utf-8")
        code = main(
            [
                "--data-source-dir",
                str(tmp_path),
                "--n-readability-samples",
                "2",
                "--n-interaction-match-samples",
                "2",
                "--output-dir",
                str(out),
                "--random-seed",
                "1",
                "--best-runs-only",
                "false",
            ],
        )
        assert code == 1
        assert (out / "interaction-match-human-labeled.csv").read_text(encoding="utf-8") == "keep\n"
        assert not (out / "readability-ai-labeled.csv").exists()
        assert not (out / "readability-human-labeled.csv").exists()
        assert not (out / "interaction-match-ai-labeled.csv").exists()
