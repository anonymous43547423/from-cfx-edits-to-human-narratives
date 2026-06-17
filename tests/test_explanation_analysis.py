"""Unit tests for individual attribution (explanation) functions."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from recsys_nle.cf_explanations.LXR.explanation_analysis import (
    find_accent_mask,
    find_accent_mask_refactored,
    find_cosine_mask,
    find_jaccard_mask,
    find_lxr_mask,
    find_shapley_mask,
    find_spinrec_mask,
    get_counterfactual_explanation,
)
from recsys_nle.cf_explanations.LXR.recommenders_architecture import VAE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ITEMS = 20


def _make_user_vector() -> np.ndarray[Any, Any]:
    """Create a binary user-interaction vector with a few items interacted."""
    vec = np.zeros(NUM_ITEMS, dtype=np.int64)
    vec[1] = 1
    vec[3] = 1
    vec[7] = 1
    vec[12] = 1
    return vec


def _make_vae(device: torch.device) -> VAE:
    """Create a tiny VAE recommender for testing."""
    config = {
        "enc_dims": [16, 8],
        "dropout": 0.0,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200_000,
    }
    kw = {"device": device, "num_items": NUM_ITEMS}
    model: VAE = VAE(config, **kw)  # type: ignore[no-untyped-call]
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class _TinyExplainer(nn.Module):
    """Minimal explainer that returns sigmoid scores for testing LXR."""

    def __init__(self, num_items: int) -> None:
        """Initialise with a single linear layer."""
        super().__init__()
        self.fc = nn.Linear(num_items * 2, num_items)

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        """Return per-item importance scores in (0, 1)."""
        combined = torch.cat((user_tensor, item_tensor), dim=-1)
        return torch.sigmoid(self.fc(combined))


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------


def test_find_jaccard_mask_returns_scores_for_interacted_items() -> None:
    """It returns similarity scores only for items in the user history."""
    user_vector = _make_user_vector()
    target_item = 5
    jaccard_dict = {(1, 5): 0.8, (3, 5): 0.6, (7, 5): 0.3, (12, 5): 0.1}

    result = find_jaccard_mask(user_vector, target_item, jaccard_dict)  # type: ignore[no-untyped-call]

    assert set(result.keys()) == {1, 3, 7, 12}
    assert result[1] == 0.8
    assert result[3] == 0.6


def test_find_jaccard_mask_returns_zero_for_missing_pairs() -> None:
    """It assigns 0 when a (item, target) pair is not in the dict."""
    user_vector = _make_user_vector()
    target_item = 5
    jaccard_dict: dict[tuple[int, int], float] = {(1, 5): 0.5}

    result = find_jaccard_mask(user_vector, target_item, jaccard_dict)  # type: ignore[no-untyped-call]

    assert result[3] == 0
    assert result[7] == 0


def test_find_jaccard_mask_excludes_target_from_history() -> None:
    """It removes the target item from the user history before scoring."""
    user_vector = _make_user_vector()
    target_item = 3  # item 3 is in the user history
    jaccard_dict: dict[tuple[int, int], float] = {(3, 3): 0.9, (1, 3): 0.4, (7, 3): 0.2, (12, 3): 0.1}

    result = find_jaccard_mask(user_vector, target_item, jaccard_dict)  # type: ignore[no-untyped-call]

    # Item 3 should be excluded because user_hist[target_item] is set to 0
    assert 3 not in result
    assert set(result.keys()) == {1, 7, 12}


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


def test_find_cosine_mask_returns_scores_for_interacted_items() -> None:
    """It returns cosine similarity scores only for items in the user history."""
    user_vector = _make_user_vector()
    target_item = 5
    cosine_dict = {(1, 5): 0.9, (3, 5): 0.7, (7, 5): 0.4, (12, 5): 0.2}

    result = find_cosine_mask(user_vector, target_item, cosine_dict)  # type: ignore[no-untyped-call]

    assert set(result.keys()) == {1, 3, 7, 12}
    assert result[1] == 0.9
    assert result[12] == 0.2


def test_find_cosine_mask_excludes_target_from_history() -> None:
    """It removes the target item from the user history before scoring."""
    user_vector = _make_user_vector()
    target_item = 7
    cosine_dict: dict[tuple[int, int], float] = {(1, 7): 0.5, (3, 7): 0.3, (7, 7): 1.0, (12, 7): 0.1}

    result = find_cosine_mask(user_vector, target_item, cosine_dict)  # type: ignore[no-untyped-call]

    assert 7 not in result


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def test_find_shapley_mask_returns_cluster_values_for_interacted_items() -> None:
    """It returns SHAP cluster values for items the user interacted with."""
    user_tensor = torch.tensor(_make_user_vector(), dtype=torch.float32)
    user_id = 42

    # shap_values: rows with [user_id, cluster_0_val, cluster_1_val, ...]
    shap_values = np.array(
        [
            [42, 0.5, -0.3, 0.1],
            [99, 0.9, 0.8, 0.7],
        ]
    )
    # Map each interacted item to a cluster index
    item_to_cluster = {1: 0, 3: 1, 7: 2, 12: 0}

    result = find_shapley_mask(user_tensor, user_id, shap_values, item_to_cluster)  # type: ignore[no-untyped-call]

    assert set(result.keys()) == {1, 3, 7, 12}
    assert result[1] == 0.5  # cluster 0
    assert result[3] == -0.3  # cluster 1
    assert result[7] == 0.1  # cluster 2
    assert result[12] == 0.5  # cluster 0


# ---------------------------------------------------------------------------
# LXR
# ---------------------------------------------------------------------------


def test_find_lxr_mask_returns_scores_for_positively_masked_items() -> None:
    """It returns importance scores only for items with positive masked values."""
    device = torch.device("cpu")
    explainer = _TinyExplainer(NUM_ITEMS).to(device)
    explainer.eval()

    user_tensor = torch.tensor(_make_user_vector(), dtype=torch.float32, device=device)
    item_tensor = torch.zeros(NUM_ITEMS, dtype=torch.float32, device=device)
    item_tensor[5] = 1.0

    result = find_lxr_mask(user_tensor, item_tensor, explainer)  # type: ignore[no-untyped-call]

    assert isinstance(result, dict)
    # Only items where user_tensor > 0 AND explainer score > 0 can appear
    for item_id in result:
        assert user_tensor[item_id].item() > 0
        assert result[item_id] > 0


# ---------------------------------------------------------------------------
# Accent
# ---------------------------------------------------------------------------


def _make_accent_fixtures(
    device: torch.device,
) -> tuple[torch.Tensor, VAE, np.ndarray[Any, Any], dict[str, object], int]:
    """Build all inputs needed to call find_accent_mask."""
    recommender = _make_vae(device)
    items_array = np.eye(NUM_ITEMS)
    all_items_tensor = torch.tensor(items_array, dtype=torch.float32, device=device)
    kw_dict: dict[str, object] = {
        "device": device,
        "num_items": NUM_ITEMS,
        "all_items_tensor": all_items_tensor,
        "items_array": items_array,
        "output_type": "multiple",
        "recommender_name": "VAE",
    }
    user_tensor = torch.tensor(_make_user_vector(), dtype=torch.float32, device=device)
    target_item = 5
    return user_tensor, recommender, items_array, kw_dict, target_item


def test_find_accent_mask_runs_on_cpu() -> None:
    """It completes without device errors when everything is on CPU."""
    device = torch.device("cpu")
    user_tensor, recommender, items_array, kw_dict, target_item = _make_accent_fixtures(device)

    result = find_accent_mask(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        NUM_ITEMS,
        items_array,
        kw_dict,
        device,
        top_k=2,
    )

    assert isinstance(result, dict)
    # Should only contain items that were in the user's history
    for item_id in result:
        assert _make_user_vector()[item_id] == 1


def test_find_accent_mask_returns_float_scores() -> None:
    """It returns float attribution scores for each interacted item."""
    device = torch.device("cpu")
    user_tensor, recommender, items_array, kw_dict, target_item = _make_accent_fixtures(device)

    result = find_accent_mask(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        NUM_ITEMS,
        items_array,
        kw_dict,
        device,
        top_k=2,
    )

    for score in result.values():
        assert isinstance(score, float)


def test_find_accent_mask_refactored_returns_gap_influences() -> None:
    """The refactored ACCENT returns gap influences between rec and candidates."""
    device = torch.device("cpu")
    user_tensor, recommender, items_array, kw_dict, target_item = _make_accent_fixtures(device)

    result = find_accent_mask_refactored(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        NUM_ITEMS,
        items_array,
        kw_dict,
        device,
        top_k=3,
    )

    # Should return scores for historical items
    assert isinstance(result, dict)
    assert len(result) > 0

    # All scores should be float
    for score in result.values():
        assert isinstance(score, float)

    # Should only contain items from user's history
    user_vector = _make_user_vector()
    for item_id in result:
        assert user_vector[item_id] == 1


def test_get_counterfactual_explanation_negates_accent_scores() -> None:
    """Accent scores are negated in get_counterfactual_explanation so high = supports rec."""
    device = torch.device("cpu")
    user_tensor, recommender, items_array, kw_dict, target_item = _make_accent_fixtures(device)
    user_vector = _make_user_vector()

    # Get raw accent scores (high = supports alternatives).
    # Use default top_k=5 to match what get_counterfactual_explanation passes.
    raw_accent = find_accent_mask(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        NUM_ITEMS,
        items_array,
        kw_dict,
        device,
    )

    # Get scores via get_counterfactual_explanation (should be negated)
    sorted_items = get_counterfactual_explanation(  # type: ignore[no-untyped-call]
        user_tensor,
        user_vector,
        0,  # user_id (unused by accent)
        target_item,
        None,  # explainer (unused by accent)
        recommender,
        items_array,
        device,
        method="accent",
        kw_dict=kw_dict,
        num_items=NUM_ITEMS,
    )

    pipeline_scores = dict(sorted_items)

    # Each pipeline score should be the negation of the raw accent score
    for item_id, raw_score in raw_accent.items():
        assert item_id in pipeline_scores
        assert pipeline_scores[item_id] == -raw_score, (
            f"item {item_id}: expected {-raw_score}, got {pipeline_scores[item_id]}"
        )


# ---------------------------------------------------------------------------
# SPINRec
# ---------------------------------------------------------------------------


def _make_spinrec_fixtures(
    device: torch.device,
) -> tuple[torch.Tensor, VAE, torch.Tensor, np.ndarray[Any, Any], int]:
    """Build all inputs needed to call find_spinrec_mask."""
    recommender = _make_vae(device)
    all_items_tensor = torch.eye(NUM_ITEMS, dtype=torch.float32, device=device)
    pop_array = np.full(NUM_ITEMS, 0.1, dtype=np.float32)
    user_tensor = torch.tensor(_make_user_vector(), dtype=torch.float32, device=device)
    target_item = 5
    return user_tensor, recommender, all_items_tensor, pop_array, target_item


def test_find_spinrec_mask_returns_float_scores_for_interacted_items() -> None:
    """It returns float attribution scores only for items in the user history."""
    device = torch.device("cpu")
    user_tensor, recommender, all_items_tensor, pop_array, target_item = _make_spinrec_fixtures(device)

    result = find_spinrec_mask(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        all_items_tensor,
        device,
        train_array=None,
        pop_array=pop_array,
        method="sample_items_by_pop",
    )

    assert isinstance(result, dict)
    assert len(result) > 0
    for score in result.values():
        assert isinstance(score, float)


def test_find_spinrec_mask_runs_with_base_method() -> None:
    """It completes successfully using the zero-baseline method."""
    device = torch.device("cpu")
    user_tensor, recommender, all_items_tensor, _, target_item = _make_spinrec_fixtures(device)

    result = find_spinrec_mask(  # type: ignore[no-untyped-call]
        user_tensor,
        target_item,
        recommender,
        all_items_tensor,
        device,
        train_array=None,
        pop_array=None,
        method="base",
    )

    assert isinstance(result, dict)
    for score in result.values():
        assert isinstance(score, float)
