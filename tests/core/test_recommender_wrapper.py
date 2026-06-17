# ruff: noqa: S101, PLR2004
"""Unit tests for RecommenderWrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from recsys_nle.core.recommender_wrapper import RecommenderWrapper


def _create_mock_recommender(num_items: int = 10) -> MagicMock:
    """Create a mock recommender that returns predictable scores."""

    def mock_forward(_user_tensor: torch.Tensor) -> torch.Tensor:
        # Return scores where item i has score (num_items - i)
        return torch.tensor(
            [[float(num_items - i) for i in range(num_items)]],
            dtype=torch.float32,
        )

    return MagicMock(side_effect=mock_forward)


def test_get_item_rank_returns_correct_rank_for_top_item() -> None:
    """It returns rank 1 for the highest scoring uninteracted item."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    # User has not interacted with any items
    user_history = torch.zeros(num_items, dtype=torch.float32)
    # Item 0 should have the highest score (10.0), so rank 1
    rank = wrapper.get_item_rank(user_history, target_item=0)
    assert rank == 1


def test_get_item_rank_excludes_interacted_items() -> None:
    """It excludes items the user has already interacted with from ranking."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    # User has interacted with item 0 (the highest scoring one)
    user_history = torch.zeros(num_items, dtype=torch.float32)
    user_history[0] = 1.0

    # Item 1 should now be rank 1 since item 0 is excluded
    rank = wrapper.get_item_rank(user_history, target_item=1)
    assert rank == 1

    # Item 0 should still be at some position (masked out)
    # Actually, when the item is interacted, it gets score 0 from catalog masking
    # So it would be at the end of the ranking
    rank_0 = wrapper.get_item_rank(user_history, target_item=0)
    assert rank_0 > 1


def test_get_item_rank_returns_correct_rank_for_lower_items() -> None:
    """It returns correct ranks for items with lower scores."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    user_history = torch.zeros(num_items, dtype=torch.float32)

    # Item 0 has score 10, item 1 has score 9, etc.
    # So item 5 should be at rank 6
    rank = wrapper.get_item_rank(user_history, target_item=5)
    assert rank == 6


def test_get_top_k_items_returns_correct_items() -> None:
    """It returns the top k items in descending score order."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    user_history = torch.zeros(num_items, dtype=torch.float32)

    top_3 = wrapper.get_top_k_items(user_history, k=3)
    assert len(top_3) == 3
    # Items 0, 1, 2 have scores 10, 9, 8 respectively
    assert top_3 == [0, 1, 2]


def test_get_top_k_items_excludes_interacted_items() -> None:
    """It excludes interacted items from top-k results."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    # User has interacted with items 0 and 1
    user_history = torch.zeros(num_items, dtype=torch.float32)
    user_history[0] = 1.0
    user_history[1] = 1.0

    top_3 = wrapper.get_top_k_items(user_history, k=3)
    assert len(top_3) == 3
    # Items 2, 3, 4 should be top 3 since 0, 1 are excluded
    assert top_3 == [2, 3, 4]


def test_get_item_score_returns_score_for_target_item() -> None:
    """It returns the score for a specific target item."""
    num_items = 10
    mock_recommender = _create_mock_recommender(num_items)
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=mock_recommender,
        num_items=num_items,
        device=device,
    )

    user_history = torch.zeros(num_items, dtype=torch.float32)
    score = wrapper.get_item_score(user_history, target_item=3)
    assert score == 7.0


def test_num_items_property() -> None:
    """It exposes the number of items through a property."""
    num_items = 42
    wrapper = RecommenderWrapper(
        recommender=MagicMock(),
        num_items=num_items,
        device=torch.device("cpu"),
    )
    assert wrapper.num_items == num_items


def test_device_property() -> None:
    """It exposes the device through a property."""
    device = torch.device("cpu")
    wrapper = RecommenderWrapper(
        recommender=MagicMock(),
        num_items=10,
        device=device,
    )
    assert wrapper.device == device
