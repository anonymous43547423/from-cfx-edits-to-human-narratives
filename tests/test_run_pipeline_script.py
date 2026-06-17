# ruff: noqa: S101, SLF001, TC003
"""Tests for the run_pipeline command-line script."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import Dataset  # type: ignore[attr-defined]
from recsys_nle.core.attribution import AttributionMethod, UserAttribution
from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
from recsys_nle.nl_explanations.workflow import ExplanationResult
from recsys_nle.pipeline.config import PipelineConfig
from recsys_nle.pipeline.workflow import PipelineResult
from scripts import run_pipeline


def make_recommendations_frame() -> pd.DataFrame:
    """Construct a minimal recommendations DataFrame."""
    return pd.DataFrame({"user_id": [1], "movie_id": [101], "score": [0.9]})


def make_cfx_interactions() -> tuple[dict[int, UserAttribution], pd.DataFrame]:
    """Construct a minimal attribution mapping and summary."""
    cfx_interactions = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [101],
            "rating": [4.0],
            "weight": [0.4],
            "importance": [0.4],
        }
    )
    non_cfx_interactions = pd.DataFrame(
        {
            "movie_id": [200],
            "rating": [3.0],
        }
    )
    user_attr: dict[int, UserAttribution] = {
        1: UserAttribution(
            user_id=1,
            cfx_interactions=cfx_interactions[["movie_id", "rating", "weight", "importance"]],
            non_cfx_interactions=non_cfx_interactions,
        ),
    }
    return user_attr, cfx_interactions


def make_explanation_result() -> ExplanationResult:
    """Create an explanation result optionally containing a dataset."""
    dataset = Dataset.from_list(
        [
            {
                "user_id": "1",
                "recommendations": [{"movie_id": 101}],
                "cfx_interactions": [{"movie_id": 101}],
                "non_cfx_interactions": [{"movie_id": 200}],
                "reasoning": "Reasoning text",
                "explanation": "Because you liked similar titles",
                "explanation_plausibility": {"judgment": "ok", "score": 0.6},
                "explanation_cfx_match": {"judgment": "ok", "score": 0.6},
                "explanation_non_cfx_match": {"judgment": "ok", "score": 0.6},
            }
        ]
    )
    evaluation = EvaluationResult(judgment="ok", score=0.6)
    result = NaturalLanguageExplanationResult(
        user_id=1,
        reasoning="Reasoning text",
        explanation="Because you liked similar titles",
        explanation_plausibility=evaluation,
        explanation_cfx_match=evaluation,
        explanation_non_cfx_match=evaluation,
    )
    return ExplanationResult(dataset=dataset, results_by_user={1: result})


def make_pipeline_result() -> PipelineResult:
    """Combine pipeline stage results into a unified object."""
    recommendations = make_recommendations_frame()
    user_attributions, cfx_interactions = make_cfx_interactions()
    explanations = make_explanation_result()
    return PipelineResult(
        recommendations=recommendations,
        user_attributions=user_attributions,
        cfx_interactions=cfx_interactions,
        explanations=explanations,
    )


def build_args(tmp_path: Path, **overrides: Any) -> Namespace:
    """Build CLI argument namespaces with sensible defaults."""
    defaults: dict[str, Any] = {
        "data_home": tmp_path / "data",
        "top_k": 5,
        "sample_user_count": 3,
        "generation_batch_size": 4,
        "evaluation_user_batch_size": 4,
        "evaluation_llm_batch_size": 4,
        "model_id_generation": "test/model-generation",
        "model_id_evaluation": "test/model-evaluation",
        "evaluation": ["plausibility", "correctness", "faithfulness"],
        "num_factors": None,
        "reg": None,
        "num_iterations": None,
        "alpha": None,
        "random_state": None,
        "random_seed": 42,
        "attribution_method": "lxr",
        "n_cfx_interactions": 3,
        "n_non_cfx_interactions": 3,
        "min_cfx_interactions": 1,
        "n_judged_interactions": None,
        "max_cfx_removals": 5,
        "target_cfx_rank": 3,
        "num_samples": 10,
        "output_datasets_path": tmp_path / "outputs",
        "disable_reasoning": False,
        "log_level": "INFO",
        "n_sampled_faithfulness_interactions": 20,
        "faithfulness_match_threshold": 0.5,
        "n_faithfulness_interactions_min_limit": 1,
        "n_faithfulness_trials": 2,
        "n_faithfulness_samples": 1,
        "n_sampled_distance_pairs": 5,
        "create_output_datasets_subdirectory": True,
        "target_set": "test",
        "user_pool": "eval",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_build_pipeline_config_respects_disable_reasoning(tmp_path: Path) -> None:
    """Propagate the disable_reasoning flag into the explanation config."""
    args = build_args(tmp_path, disable_reasoning=True)

    # Use the real _build_pipeline_config but avoid running the full pipeline.
    config = run_pipeline._build_pipeline_config(args)

    assert isinstance(config, PipelineConfig)
    assert config.explanation.disable_reasoning is True


def test_build_pipeline_config_sets_attribution_method(tmp_path: Path) -> None:
    """It stores the CLI attribution method in the config."""
    args = build_args(tmp_path, attribution_method="cosine")

    config = run_pipeline._build_pipeline_config(args)

    assert config.attribution.method == AttributionMethod.COSINE


def test_build_pipeline_config_sets_min_cfx_interactions(tmp_path: Path) -> None:
    """It stores the minimum CFX interaction threshold in the config."""
    expected_min = 4
    args = build_args(tmp_path, min_cfx_interactions=expected_min)

    config = run_pipeline._build_pipeline_config(args)

    assert config.attribution.min_cfx_interactions == expected_min


def test_build_pipeline_config_propagates_create_output_datasets_subdirectory(tmp_path: Path) -> None:
    """It propagates the create_output_datasets_subdirectory flag into the output config."""
    args_true = build_args(tmp_path, create_output_datasets_subdirectory=True)
    config_true = run_pipeline._build_pipeline_config(args_true)
    assert config_true.outputs.create_output_datasets_subdirectory is True

    args_false = build_args(tmp_path, create_output_datasets_subdirectory=False)
    config_false = run_pipeline._build_pipeline_config(args_false)
    assert config_false.outputs.create_output_datasets_subdirectory is False


def test_build_pipeline_config_propagates_target_set_and_user_pool(tmp_path: Path) -> None:
    """It propagates --target-set and --user-pool into the pipeline config."""
    args = build_args(tmp_path, target_set="validation", user_pool="train")
    config = run_pipeline._build_pipeline_config(args)
    assert config.target_set == "validation"
    assert config.user_pool == "train"

    args_test = build_args(tmp_path, target_set="test", user_pool="eval")
    config_test = run_pipeline._build_pipeline_config(args_test)
    assert config_test.target_set == "test"
    assert config_test.user_pool == "eval"
