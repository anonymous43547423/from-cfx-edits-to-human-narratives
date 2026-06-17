# ruff: noqa: S101

"""Tests for pipeline workflow orchestration."""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from recsys_nle.core.attribution import AttributionConfig, AttributionMethod, UserAttribution
from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.llm import OpenAIChatLLMClient
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
from recsys_nle.nl_explanations.workflow import ExplanationConfig, ExplanationResult
from recsys_nle.pipeline.config import PipelineConfig, TargetSet, UserPool
from recsys_nle.pipeline.workflow import (
    AttributionResult,
    CfxSearchOutcome,
    PipelineWorkflow,
    _build_evaluation_llm,
    _raise_if_einfra_generation,
    _resolve_user_pool,
    run_attribution_and_cfx_algorithms,
)


def make_lxr_result(
    with_attributions: bool,
) -> tuple[pd.DataFrame, Mapping[int, UserAttribution], pd.DataFrame]:
    """Construct minimal LXR outputs for testing."""
    recommendations = pd.DataFrame(
        [
            {"user_id": 1, "movie_id": 101, "score": 0.9, "rank": 1},
        ]
    )
    if with_attributions:
        cfx_interactions = pd.DataFrame(
            [
                {"movie_id": 101, "rating": 4.0, "weight": 0.4, "importance": 0.4},
            ]
        )
        non_cfx_interactions = pd.DataFrame(
            [
                {"movie_id": 200, "rating": 3.0},
            ]
        )
        user_attributions: Mapping[int, UserAttribution] = {
            1: UserAttribution(
                user_id=1,
                cfx_interactions=cfx_interactions,
                non_cfx_interactions=non_cfx_interactions,
            ),
        }
        cfx_summary = cfx_interactions.assign(user_id=1)[["user_id", "movie_id", "rating", "weight", "importance"]]
    else:
        user_attributions = {}
        cfx_summary = pd.DataFrame(
            columns=["user_id", "movie_id", "rating", "weight", "importance"],
        )
    return recommendations, dict(user_attributions), cfx_summary


def _minimal_pipeline_config(**overrides: Any) -> PipelineConfig:
    """Build a minimal pipeline config for workflow tests."""
    defaults: dict[str, Any] = {
        "explanation": ExplanationConfig(
            model_id_generation="test/model",
            model_id_evaluation="test/model",
            n_faithfulness_interactions_min_limit=1,
            n_faithfulness_trials=1,
            n_faithfulness_samples=1,
            n_cfx_interactions=3,
        ),
        "attribution": AttributionConfig(
            method=AttributionMethod.LXR,
            max_cfx_removals=5,
            target_cfx_rank=3,
            min_cfx_interactions=1,
        ),
        "target_set": "test",
        "user_pool": "eval",
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_pipeline_run_invokes_explanations_with_all_user_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """It forwards recommendations and attributions into the explanation workflow."""
    config = _minimal_pipeline_config()

    lxr_result = make_lxr_result(with_attributions=True)

    def _load_components() -> tuple[object, int, pd.DataFrame, object, object, object]:
        """Return lightweight dummy LXR components."""
        device = object()
        num_items = 0
        test_data = pd.DataFrame()
        items_array: object = object()
        recommender: object = object()
        explainer: object = object()
        return device, num_items, test_data, items_array, recommender, explainer

    seen_args: dict[str, Any] = {}

    def _run_attribution(**kwargs: Any) -> AttributionResult:
        """Return fixed LXR-style outputs."""
        seen_args["attribution_kwargs"] = kwargs
        return AttributionResult(
            recommendations=lxr_result[0],
            user_attributions=lxr_result[1],
            cfx_interactions=lxr_result[2],
            all_interactions=pd.DataFrame(),
            user_histories={},
            user_targets={},
            cfx_search_outcome=CfxSearchOutcome(),
        )

    def _run_explanations(
        _self: PipelineWorkflow,
        **kwargs: Any,
    ) -> tuple[ExplanationResult, list[int]]:
        """Capture explanation stage arguments and return a minimal result."""
        seen_args["explanation_kwargs"] = kwargs
        evaluation = EvaluationResult(judgment="ok", score=0.5)
        nle = NaturalLanguageExplanationResult(
            user_id=1,
            reasoning="Because you liked similar titles",
            explanation="Because you liked Movie 101",
            explanation_plausibility=evaluation,
            explanation_cfx_match=evaluation,
        )
        return ExplanationResult(dataset=None, results_by_user={1: nle}), [1]

    def _build_distance_context(*_args: Any, **_kwargs: Any) -> object:
        """Return a dummy distance context."""
        return object()

    def _compute_distance_metrics(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        """Return fixed distance metrics for assertions."""
        return {"user_based_mean_cfx_distance": 0.5}

    monkeypatch.setattr("recsys_nle.pipeline.workflow._load_lxr_components", _load_components)
    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow.run_attribution_and_cfx_algorithms",
        _run_attribution,
    )
    monkeypatch.setattr(PipelineWorkflow, "_run_explanations_stage", _run_explanations)
    monkeypatch.setattr("recsys_nle.pipeline.workflow.build_distance_context", _build_distance_context)
    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow.compute_all_distance_metrics_for_user",
        _compute_distance_metrics,
    )

    workflow = PipelineWorkflow()
    result = workflow.run(config)

    assert seen_args["explanation_kwargs"]["user_attributions"] == lxr_result[1]
    assert seen_args["attribution_kwargs"]["method"] == AttributionMethod.LXR
    assert seen_args["attribution_kwargs"]["min_cfx_interactions"] == 1
    assert result.explanations is not None
    assert 1 in result.explanations.results_by_user
    assert result.distance_metrics_by_user == {1: {"user_based_mean_cfx_distance": 0.5}}


def test_pipeline_run_resolves_user_pool_before_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """It slices test_data via target_set and user_pool before attribution."""
    config = _minimal_pipeline_config(target_set="validation", user_pool="train")

    lxr_result = make_lxr_result(with_attributions=True)
    full_test_data = pd.DataFrame({"item_0": [1.0] * 6, "user_id": list(range(10, 16))})
    resolved = pd.DataFrame({"item_0": [1.0], "user_id": [15]})

    def _load_components() -> tuple[object, int, pd.DataFrame, object, object, object]:
        """Return dummy components with six-user test matrix."""
        return object(), 3, full_test_data, object(), object(), object()

    seen: dict[str, Any] = {}

    def _resolve(test_data: pd.DataFrame, *, user_pool: str, target_set: str) -> pd.DataFrame:
        """Capture resolution inputs and return a stub slice."""
        seen["resolve"] = (test_data is full_test_data, user_pool, target_set)
        return resolved

    def _run_attribution(**kwargs: Any) -> AttributionResult:
        """Capture user_data passed to attribution."""
        seen["user_data"] = kwargs["user_data"]
        return AttributionResult(
            recommendations=lxr_result[0],
            user_attributions=lxr_result[1],
            cfx_interactions=lxr_result[2],
            all_interactions=pd.DataFrame(),
            user_histories={},
            user_targets={},
            cfx_search_outcome=CfxSearchOutcome(),
        )

    monkeypatch.setattr("recsys_nle.pipeline.workflow._load_lxr_components", _load_components)
    monkeypatch.setattr("recsys_nle.pipeline.workflow._resolve_user_pool", _resolve)
    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow.run_attribution_and_cfx_algorithms",
        _run_attribution,
    )
    monkeypatch.setattr(
        PipelineWorkflow,
        "_run_explanations_stage",
        lambda _self, **_kw: (ExplanationResult(dataset=None, results_by_user={}), []),
    )
    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow.build_distance_context",
        lambda *_a, **_kw: object(),
    )

    PipelineWorkflow().run(config)

    assert seen["resolve"] == (True, "train", "validation")
    assert seen["user_data"] is resolved


def test_resolve_user_pool_all_combinations() -> None:
    """Each user_pool and target_set pair selects the expected third(s)."""
    user_ids = list(range(10, 16))
    test_data = pd.DataFrame(
        {"item_0": [1.0] * len(user_ids)},
        index=user_ids,
    )

    cases: list[tuple[UserPool, TargetSet, list[int]]] = [
        ("eval", "validation", [12, 13]),
        ("eval", "test", [10, 11]),
        ("train", "validation", [14, 15]),
        ("train", "test", [12, 13, 14, 15]),
    ]
    for user_pool, target_set, expected_ids in cases:
        resolved = _resolve_user_pool(
            test_data,
            user_pool=user_pool,
            target_set=target_set,
        )
        assert list(resolved["user_id"]) == expected_ids, f"{user_pool}/{target_set}"


def test_user_pool_slices_are_disjoint_where_expected() -> None:
    """Eval and train pools do not overlap within the same target_set mode."""
    user_ids = list(range(10, 16))
    test_data = pd.DataFrame(
        {"item_0": [1.0] * len(user_ids)},
        index=user_ids,
    )

    for ts in ("validation", "test"):
        eval_ids = set(_resolve_user_pool(test_data, user_pool="eval", target_set=ts)["user_id"])
        train_ids = set(_resolve_user_pool(test_data, user_pool="train", target_set=ts)["user_id"])
        assert eval_ids.isdisjoint(train_ids)


def test_attribution_uses_pre_sliced_user_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_attribution_and_cfx_algorithms processes all users in the provided slice."""
    user_ids = list(range(10, 16))
    test_data = pd.DataFrame(
        {
            "item_0": [1.0] * len(user_ids),
            "item_1": [0.0] * len(user_ids),
            "item_2": [0.0] * len(user_ids),
        },
        index=user_ids,
    )
    sliced = _resolve_user_pool(test_data, user_pool="eval", target_set="test")

    device = torch.device("cpu")
    num_items = 3
    items_array = np.eye(num_items, dtype=np.float32)

    class _StubRecommender(torch.nn.Module):
        """Minimal recommender stub returning constant scores."""

        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            """Return small positive scores for all items."""
            return torch.ones(1, num_items) * 0.5

    class _StubExplainer(torch.nn.Module):
        """Minimal explainer stub."""

        def forward(self, _x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
            """Return dummy attribution scores."""
            return torch.zeros(num_items)

    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow.get_counterfactual_explanation",
        lambda *_a, **_kw: [],
    )
    monkeypatch.setattr(
        "recsys_nle.pipeline.workflow._load_attribution_assets",
        lambda **_kw: type(
            "FakeAssets",
            (),
            {
                "all_items_tensor": torch.tensor(items_array, dtype=torch.float32),
                "kw_dict": {},
                "jaccard_dict": None,
                "cosine_dict": None,
                "item_to_cluster": None,
                "shap_values": None,
                "lime": None,
                "train_array": None,
                "pop_array": None,
            },
        )(),
    )

    result = run_attribution_and_cfx_algorithms(
        device=device,
        num_items=num_items,
        user_data=sliced,
        items_array=items_array,
        recommender=_StubRecommender(),  # type: ignore[arg-type]
        explainer=_StubExplainer(),  # type: ignore[arg-type]
        top_k=1,
        n_non_cfx_interactions=0,
        max_cfx_removals=1,
        target_cfx_rank=1,
        min_cfx_interactions=1,
        max_users=10,
        random_seed=42,
        method=AttributionMethod.LXR,
    )

    assert set(result.recommendations["user_id"].unique()) == {10, 11}


def test_einfra_generation_is_rejected() -> None:
    """EINFRA-prefixed models must not be used for explanation generation."""
    with pytest.raises(ValueError, match="LLM judge evaluation"):
        _raise_if_einfra_generation("EINFRA/qwen2.5-coder:32b-instruct-q8_0")


def test_build_evaluation_llm_einfra(monkeypatch: pytest.MonkeyPatch) -> None:
    """EINFRA/ evaluation uses OpenAI-compatible client with stripped model id."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.OpenAI",
        lambda **_kwargs: MagicMock(),
    )

    client = _build_evaluation_llm("EINFRA/remote-id")
    assert isinstance(client, OpenAIChatLLMClient)
    assert client.model == "remote-id"
