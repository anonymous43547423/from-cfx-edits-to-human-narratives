"""Attribution configuration and helpers decoupled from implementation details."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import pandas as pd


@dataclass(slots=True)
class UserAttribution:
    """Container for a single user's CFX and non-CFX interaction explanations."""

    user_id: int
    cfx_interactions: pd.DataFrame
    non_cfx_interactions: pd.DataFrame


class AttributionMethod(Enum):
    """Supported feature attribution algorithms for counterfactual explanations."""

    LXR = "lxr"
    JACCARD = "jaccard"
    COSINE = "cosine"
    LIME = "lime"
    ACCENT = "accent"
    SHAP = "shap"
    SPINREC = "spinrec"


@dataclass(slots=True)
class AttributionConfig:
    """Configuration for user-level explanation attribution generation."""

    method: AttributionMethod
    max_cfx_removals: int
    target_cfx_rank: int
    min_cfx_interactions: int
    recommendation_count: int = 10
    n_non_cfx_interactions: int = 5
    num_samples: int = 500


@dataclass(slots=True)
class AttributionResult:
    """Outcome of computing user recommendation attributions."""

    user_attributions: dict[int, UserAttribution]
    explained_user_ids: list[int]


def summarise_interactions(interactions_by_user: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Flatten per-user interaction dataframes into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for user_id, interactions in interactions_by_user.items():
        frame = interactions.copy()
        if frame.empty:
            continue
        frame.insert(0, "user_id", user_id)
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    columns = ["user_id", "movie_id", "rating", "weight", "importance"]
    return pd.DataFrame(columns=columns)
