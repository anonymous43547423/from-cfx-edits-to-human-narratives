"""Tests for the datasets exporter helper."""

# ruff: noqa: S101, TC003

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from recsys_nle.core.attribution import UserAttribution
from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
from recsys_nle.nl_explanations.workflow import ExplanationResult
from recsys_nle.pipeline.config import OutputConfig
from recsys_nle.pipeline.datasets_exporter import DatasetsExporter
from recsys_nle.pipeline.workflow import PipelineResult


def _mock_movie_metadata_index() -> dict[int, dict[str, object]]:
    """Return mock metadata for test item IDs."""
    return {
        10: {"movie_title": "Test Movie 10", "title": "Test Movie 10", "genres": ["Action", "Comedy"], "year": 2020},
        20: {"movie_title": "Test Movie 20", "title": "Test Movie 20", "genres": ["Drama"], "year": 2021},
        100: {"movie_title": "Test Movie 100", "title": "Test Movie 100", "genres": ["Sci-Fi"], "year": 2022},
    }


def _make_pipeline_result() -> PipelineResult:
    """Construct a minimal pipeline result suitable for dataset export tests."""
    recommendations = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 10, "score": 0.9, "rank": 1},
            {"user_id": 2, "movie_id": 20, "score": 0.8, "rank": 1},
        ]
    )

    cfx_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [10],
            "rating": [4.0],
            "weight": [0.5],
            "importance": [0.5],
        }
    )
    non_cfx_interactions = pd.DataFrame(
        {
            "movie_id": [100],
            "rating": [3.0],
        }
    )
    all_interactions = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 4.0, "attribution_score": 0.5, "is_counterfactual": True},
            {
                "user_id": 1,
                "item_id": 100,
                "rating": 3.0,
                "attribution_score": float("nan"),
                "is_counterfactual": False,
            },
        ]
    )
    user_attributions: dict[int, UserAttribution] = {
        1: UserAttribution(
            user_id=1,
            cfx_interactions=cfx_interactions[["movie_id", "rating", "weight", "importance"]],
            non_cfx_interactions=non_cfx_interactions,
        ),
    }

    # Build a simple explanation result with per-interaction faithfulness details.
    base_eval = EvaluationResult(judgment="ok", score=0.8)
    faithfulness_details = {
        "per_interaction_scores": [
            {"interaction": '{year=2020, genres="Action, Comedy"}', "item_id": 10, "judgment": "aligned", "score": 1.0},
        ]
    }
    faithfulness_eval = EvaluationResult(judgment="faithful", score=0.8, details=faithfulness_details)
    removal_details = {
        "per_interaction_scores": [
            {"interaction": "1. Sample Movie (id=10)", "item_id": 10, "match_score": 0.9},
            {"interaction": "2. Other Movie (id=100)", "item_id": 100, "match_score": float("nan")},
        ],
        "trial_scores": [0.4, 0.5],
    }
    removal_eval = EvaluationResult(judgment="removal", score=0.4, details=removal_details)
    removal_baseline_eval = EvaluationResult(
        judgment="removal baseline",
        score=0.2,
        details={
            "per_interaction_scores": [
                {
                    "interaction": "2. Other Movie (id=100)",
                    "item_id": 100,
                    "match_score": float("nan"),
                }
            ],
            "trial_scores": [0.2],
        },
    )
    replacement_eval = EvaluationResult(
        judgment="replacement",
        score=0.3,
        details={
            "per_interaction_scores": [{"interaction": "1. Sample Movie (id=10)", "item_id": 10, "match_score": 0.2}],
            "trial_scores": [0.3, 0.25],
        },
    )
    replacement_baseline_eval = EvaluationResult(
        judgment="replacement baseline",
        score=0.1,
        details={
            "per_interaction_scores": [{"interaction": "2. Other Movie (id=100)", "item_id": 100, "match_score": 0.3}],
            "trial_scores": [0.1],
        },
    )

    nle_result = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="Reasoning text",
        explanation="Because you liked similar titles",
        explanation_plausibility=base_eval,
        explanation_cfx_match=faithfulness_eval,
        faithfulness_removal=removal_eval,
        faithfulness_removal_baseline=removal_baseline_eval,
        faithfulness_replacement=replacement_eval,
        faithfulness_replacement_baseline=replacement_baseline_eval,
    )
    explanations = ExplanationResult(dataset=None, results_by_user={1: nle_result})

    return PipelineResult(
        recommendations=recommendations,
        user_attributions=user_attributions,
        cfx_interactions=cfx_interactions,
        explanations=explanations,
        all_interactions=all_interactions,
        sampled_user_ids=[1],
    )


def test_exporter_writes_filtered_datasets(tmp_path: Path) -> None:  # noqa: PLR0915
    """It writes per-run datasets limited to sampled users and referenced items."""
    output_root = tmp_path / "outputs"
    config = OutputConfig(output_datasets_path=output_root)
    exporter = DatasetsExporter(config)

    pipeline_result = _make_pipeline_result()
    with patch(
        "recsys_nle.pipeline.datasets_exporter._movie_metadata_index",
        _mock_movie_metadata_index,
    ):
        exporter.export(pipeline_result)

    assert output_root.is_dir()
    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    users_path = run_dir / "users.feather"
    items_path = run_dir / "items.feather"
    item_genres_path = run_dir / "item_genres.feather"
    interactions_path = run_dir / "interactions.feather"
    recommendations_path = run_dir / "recommendations.feather"
    generation_path = run_dir / "generation.feather"
    evaluation_path = run_dir / "evaluation.feather"
    cfx_match_details_path = run_dir / "cfx_match_details.feather"
    non_cfx_match_details_path = run_dir / "non_cfx_match_details.feather"
    faithfulness_removal_path = run_dir / "faithfulness_removal.feather"
    faithfulness_removal_baseline_path = run_dir / "faithfulness_removal_baseline.feather"
    faithfulness_replacement_path = run_dir / "faithfulness_replacement.feather"
    faithfulness_replacement_baseline_path = run_dir / "faithfulness_replacement_baseline.feather"
    faithfulness_removal_trials_path = run_dir / "faithfulness_removal_trials.feather"
    faithfulness_removal_baseline_trials_path = run_dir / "faithfulness_removal_baseline_trials.feather"
    faithfulness_replacement_trials_path = run_dir / "faithfulness_replacement_trials.feather"
    faithfulness_replacement_baseline_trials_path = run_dir / "faithfulness_replacement_baseline_trials.feather"

    for path in (
        users_path,
        items_path,
        item_genres_path,
        interactions_path,
        recommendations_path,
        generation_path,
        evaluation_path,
        cfx_match_details_path,
        non_cfx_match_details_path,
        faithfulness_removal_path,
        faithfulness_removal_baseline_path,
        faithfulness_replacement_path,
        faithfulness_replacement_baseline_path,
        faithfulness_removal_trials_path,
        faithfulness_removal_baseline_trials_path,
        faithfulness_replacement_trials_path,
        faithfulness_replacement_baseline_trials_path,
    ):
        assert path.is_file()

    users = pd.read_feather(users_path)
    assert list(users.columns) == ["user_id"]
    assert users["user_id"].tolist() == [1]

    recommendations = pd.read_feather(recommendations_path)
    assert set(recommendations["user_id"].unique().tolist()) == {1}
    assert set(recommendations.columns) == {"user_id", "item_id", "rank", "score"}

    interactions = pd.read_feather(interactions_path)
    assert set(interactions["user_id"].unique().tolist()) == {1}
    assert set(interactions.columns) == {
        "interaction_id",
        "user_id",
        "item_id",
        "rating",
        "attribution_score",
        "is_counterfactual",
    }
    assert sorted(interactions["item_id"].tolist()) == [10, 100]

    items = pd.read_feather(items_path)
    assert set(items.columns) == {"item_id", "title", "year"}
    assert set(items["item_id"].tolist()) == {10, 100}

    item_genres = pd.read_feather(item_genres_path)
    assert "item_id" in item_genres.columns
    assert "genre" in item_genres.columns
    assert set(item_genres["item_id"].unique().tolist()) == {10, 100}

    generation = pd.read_feather(generation_path)
    assert set(generation.columns) == {
        "user_id",
        "reasoning_enabled",
        "reasoning_text",
        "explanation_text",
        "explanation_confidence",
        "explanation_conversation",
    }
    assert set(generation["user_id"].unique().tolist()) == {1}
    assert generation["reasoning_enabled"].all()  # All have reasoning in our test data
    assert generation["reasoning_text"].notna().all()

    evaluation = pd.read_feather(evaluation_path)
    assert set(evaluation.columns) == {
        "user_id",
        "explanation_plausibility",
        "readability_fluency_mean",
        "readability_grammar_mean",
        "readability_length_mean",
        "readability_illustrativeness_mean",
        "readability_naturalness_mean",
        "readability_specificity_mean",
        "readability_overall_mean",
        "explanation_cfx_pattern_match_mean",
        "explanation_cfx_pattern_match_success_rate",
        "explanation_non_cfx_pattern_match_mean",
        "explanation_non_cfx_pattern_match_success_rate",
        "overall_faithfulness_removal_score",
        "overall_faithfulness_removal_baseline_score",
        "overall_faithfulness_replacement_score",
        "overall_faithfulness_replacement_baseline_score",
        "faithfulness_removal_pvalue_complement",
        "faithfulness_replacement_pvalue_complement",
        "user_based_mean_cfx_distance",
        "user_based_median_cfx_distance",
        "user_based_mean_non_cfx_distance",
        "user_based_median_non_cfx_distance",
        "user_based_mean_cfx_non_cfx_distance",
        "user_based_median_cfx_non_cfx_distance",
        "user_based_mean_separation",
        "user_based_median_separation",
        "item_based_mean_cfx_distance",
        "item_based_median_cfx_distance",
        "item_based_mean_non_cfx_distance",
        "item_based_median_non_cfx_distance",
        "item_based_mean_cfx_non_cfx_distance",
        "item_based_median_cfx_non_cfx_distance",
        "item_based_mean_separation",
        "item_based_median_separation",
    }
    assert set(evaluation["user_id"].unique().tolist()) == {1}
    assert evaluation["explanation_cfx_pattern_match_mean"].tolist() == [0.8]
    assert evaluation["explanation_cfx_pattern_match_success_rate"].tolist() == [1.0]

    cfx_match_details = pd.read_feather(cfx_match_details_path)
    assert set(cfx_match_details["user_id"].unique().tolist()) == {1}
    assert "interaction_id" in cfx_match_details.columns
    # All per-interaction entries should map back to a known interaction id.
    assert cfx_match_details["interaction_id"].isna().sum() == 0
    interaction_id_by_item = {int(row["item_id"]): int(row["interaction_id"]) for _, row in interactions.iterrows()}
    assert cfx_match_details["interaction_id"].tolist() == [interaction_id_by_item[10]]

    # Each interaction should have a single explanation score row.
    counts_by_attr = cfx_match_details.groupby("interaction_id")["interaction_id"].count()
    assert set(counts_by_attr.tolist()) == {1}

    faithfulness_removal = pd.read_feather(faithfulness_removal_path)
    assert set(faithfulness_removal.columns) == {
        "user_id",
        "interaction_id",
        "item_id",
        "match_score",
    }
    assert set(faithfulness_removal["user_id"].unique().tolist()) == {1}
    assert faithfulness_removal["match_score"].isna().sum() == 1

    faithfulness_removal_baseline = pd.read_feather(faithfulness_removal_baseline_path)
    assert set(faithfulness_removal_baseline.columns) == {
        "user_id",
        "interaction_id",
        "item_id",
        "match_score",
    }
    assert set(faithfulness_removal_baseline["user_id"].unique().tolist()) == {1}
    assert faithfulness_removal_baseline["match_score"].isna().sum() == 1

    faithfulness_replacement = pd.read_feather(faithfulness_replacement_path)
    assert set(faithfulness_replacement.columns) == {
        "user_id",
        "interaction_id",
        "item_id",
        "match_score",
    }
    assert set(faithfulness_replacement["user_id"].unique().tolist()) == {1}

    faithfulness_replacement_baseline = pd.read_feather(faithfulness_replacement_baseline_path)
    assert set(faithfulness_replacement_baseline.columns) == {
        "user_id",
        "interaction_id",
        "item_id",
        "match_score",
    }
    assert set(faithfulness_replacement_baseline["user_id"].unique().tolist()) == {1}

    faithfulness_removal_trials = pd.read_feather(faithfulness_removal_trials_path)
    assert set(faithfulness_removal_trials.columns) == {"user_id", "trial_no", "score"}
    assert set(faithfulness_removal_trials["user_id"].unique().tolist()) == {1}
    assert set(faithfulness_removal_trials["trial_no"].tolist()) == {0, 1}

    faithfulness_removal_baseline_trials = pd.read_feather(faithfulness_removal_baseline_trials_path)
    assert set(faithfulness_removal_baseline_trials.columns) == {"user_id", "trial_no", "score"}
    assert set(faithfulness_removal_baseline_trials["user_id"].unique().tolist()) == {1}
    assert set(faithfulness_removal_baseline_trials["trial_no"].tolist()) == {0}

    faithfulness_replacement_trials = pd.read_feather(faithfulness_replacement_trials_path)
    assert set(faithfulness_replacement_trials.columns) == {"user_id", "trial_no", "score"}
    assert set(faithfulness_replacement_trials["user_id"].unique().tolist()) == {1}
    assert set(faithfulness_replacement_trials["trial_no"].tolist()) == {0, 1}

    faithfulness_replacement_baseline_trials = pd.read_feather(faithfulness_replacement_baseline_trials_path)
    assert set(faithfulness_replacement_baseline_trials.columns) == {"user_id", "trial_no", "score"}
    assert set(faithfulness_replacement_baseline_trials["user_id"].unique().tolist()) == {1}
    assert set(faithfulness_replacement_baseline_trials["trial_no"].tolist()) == {0}


def test_exporter_writes_directly_when_subdirectory_disabled(tmp_path: Path) -> None:
    """It writes datasets directly to the output path when subdirectory creation is disabled."""
    output_root = tmp_path / "outputs"
    config = OutputConfig(
        output_datasets_path=output_root,
        create_output_datasets_subdirectory=False,
    )
    exporter = DatasetsExporter(config)

    pipeline_result = _make_pipeline_result()
    with patch(
        "recsys_nle.pipeline.datasets_exporter._movie_metadata_index",
        _mock_movie_metadata_index,
    ):
        run_dir = exporter.export(pipeline_result)

    assert run_dir == output_root
    # No timestamped subdirectory should have been created.
    subdirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(subdirs) == 0

    # Feather files should exist directly in output_root.
    assert (output_root / "users.feather").is_file()
    assert (output_root / "generation.feather").is_file()
    assert (output_root / "evaluation.feather").is_file()


def test_exporter_handles_string_user_ids_in_results_by_user(tmp_path: Path) -> None:
    """It handles string user_id keys in results_by_user with integer sampled_user_ids."""
    # This tests the scenario where _assemble_results stores keys as strings
    # but sampled_user_ids contains integers.
    recommendations = pd.DataFrame([{"user_id": 1, "movie_id": 10, "score": 0.9, "rank": 1}])
    cfx_interactions = pd.DataFrame(
        {"user_id": [1], "movie_id": [10], "rating": [4.0], "weight": [0.5], "importance": [0.5]}
    )
    non_cfx_interactions = pd.DataFrame({"movie_id": [100], "rating": [3.0]})
    user_attributions: dict[int, UserAttribution] = {
        1: UserAttribution(
            user_id=1,
            cfx_interactions=cfx_interactions[["movie_id", "rating", "weight", "importance"]],
            non_cfx_interactions=non_cfx_interactions,
        ),
    }
    all_interactions = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 4.0, "attribution_score": 0.5, "is_counterfactual": True},
        ]
    )

    base_eval = EvaluationResult(judgment="ok", score=0.8)
    faithfulness_details_data = {
        "per_interaction_scores": [
            {"interaction": '{year=2020, genres="Action, Comedy"}', "item_id": 10, "judgment": "aligned", "score": 1.0},
        ]
    }
    faithfulness_eval = EvaluationResult(judgment="faithful", score=0.8, details=faithfulness_details_data)

    nle_result = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="Reasoning text",
        explanation="Because you liked similar titles",
        explanation_plausibility=base_eval,
        explanation_cfx_match=faithfulness_eval,
    )
    # Key is int
    explanations = ExplanationResult(dataset=None, results_by_user={1: nle_result})

    pipeline_result = PipelineResult(
        recommendations=recommendations,
        user_attributions=user_attributions,
        cfx_interactions=cfx_interactions,
        explanations=explanations,
        all_interactions=all_interactions,
        sampled_user_ids=[1],  # Integer user ID
    )

    output_root = tmp_path / "outputs"
    config = OutputConfig(output_datasets_path=output_root)
    exporter = DatasetsExporter(config)
    exporter.export(pipeline_result)

    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    generation = pd.read_feather(run_dir / "generation.feather")
    expected_row_count = 1
    assert len(generation) == expected_row_count
    assert set(generation["user_id"].unique().tolist()) == {1}
    assert set(generation.columns) == {
        "user_id",
        "reasoning_enabled",
        "reasoning_text",
        "explanation_text",
        "explanation_confidence",
        "explanation_conversation",
    }

    evaluation = pd.read_feather(run_dir / "evaluation.feather")
    assert len(evaluation) == expected_row_count
    assert set(evaluation["user_id"].unique().tolist()) == {1}
    assert set(evaluation.columns) == {
        "user_id",
        "explanation_plausibility",
        "readability_fluency_mean",
        "readability_grammar_mean",
        "readability_length_mean",
        "readability_illustrativeness_mean",
        "readability_naturalness_mean",
        "readability_specificity_mean",
        "readability_overall_mean",
        "explanation_cfx_pattern_match_mean",
        "explanation_cfx_pattern_match_success_rate",
        "explanation_non_cfx_pattern_match_mean",
        "explanation_non_cfx_pattern_match_success_rate",
        "overall_faithfulness_removal_score",
        "overall_faithfulness_removal_baseline_score",
        "overall_faithfulness_replacement_score",
        "overall_faithfulness_replacement_baseline_score",
        "faithfulness_removal_pvalue_complement",
        "faithfulness_replacement_pvalue_complement",
        "user_based_mean_cfx_distance",
        "user_based_median_cfx_distance",
        "user_based_mean_non_cfx_distance",
        "user_based_median_non_cfx_distance",
        "user_based_mean_cfx_non_cfx_distance",
        "user_based_median_cfx_non_cfx_distance",
        "user_based_mean_separation",
        "user_based_median_separation",
        "item_based_mean_cfx_distance",
        "item_based_median_cfx_distance",
        "item_based_mean_non_cfx_distance",
        "item_based_median_non_cfx_distance",
        "item_based_mean_cfx_non_cfx_distance",
        "item_based_median_cfx_non_cfx_distance",
        "item_based_mean_separation",
        "item_based_median_separation",
    }

    cfx_match_details = pd.read_feather(run_dir / "cfx_match_details.feather")
    assert len(cfx_match_details) == expected_row_count
    assert set(cfx_match_details["user_id"].unique().tolist()) == {1}
