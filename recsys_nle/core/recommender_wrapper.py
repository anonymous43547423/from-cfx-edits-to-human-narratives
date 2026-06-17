"""Wrapper for VAE recommender providing counterfactual testing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from recsys_nle.cf_explanations.LXR.recommenders_architecture import VAE


class RecommenderWrapper:
    """Wrapper for VAE recommender providing counterfactual testing operations."""

    def __init__(
        self,
        recommender: VAE,
        num_items: int,
        device: torch.device,
    ) -> None:
        """Initialise with recommender model and configuration."""
        self._recommender = recommender
        self._num_items = num_items
        self._device = device

    @property
    def num_items(self) -> int:
        """Return the number of items in the catalog."""
        return self._num_items

    @property
    def device(self) -> torch.device:
        """Return the device used for computation."""
        return self._device

    def get_item_rank(
        self,
        user_history: torch.Tensor,
        target_item: int,
    ) -> int:
        """Return the rank (1-indexed) of target_item in recommendations."""
        user_tensor = user_history.to(self._device)
        user_res = self._recommender(user_tensor)[:, : self._num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_scores = torch.mul(user_res.squeeze(), user_catalog)

        sorted_indices = torch.argsort(user_scores, descending=True)
        rank_position = (sorted_indices == target_item).nonzero(as_tuple=True)[0]

        if len(rank_position) == 0:
            return self._num_items + 1

        return int(rank_position[0].item()) + 1

    def get_item_score(self, user_history: torch.Tensor, target_item: int) -> float:
        """Return the recommendation score for target_item."""
        if target_item < 0 or target_item >= self._num_items:
            return float("nan")

        user_tensor = user_history.to(self._device)
        user_res = self._recommender(user_tensor)[:, : self._num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_scores = torch.mul(user_res.squeeze(), user_catalog)
        return float(user_scores[target_item].item())

    def get_top_k_items(self, user_history: torch.Tensor, k: int) -> list[int]:
        """Return top-k recommended item IDs for the given user history."""
        user_tensor = user_history.to(self._device)
        user_res = self._recommender(user_tensor)[:, : self._num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_scores = torch.mul(user_res.squeeze(), user_catalog)

        _, top_k_indices = torch.topk(user_scores, min(k, self._num_items))
        return top_k_indices.cpu().tolist()
