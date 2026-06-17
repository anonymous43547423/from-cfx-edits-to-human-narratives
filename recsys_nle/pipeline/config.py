"""Configuration structures for the end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from recsys_nle.core.attribution import AttributionConfig
    from recsys_nle.nl_explanations.workflow import ExplanationConfig

TargetSet = Literal["validation", "test"]
UserPool = Literal["train", "eval"]


@dataclass(slots=True)
class RecommendationConfig:
    """Configuration required to generate recommendation outputs."""

    top_k: int = 10


@dataclass(slots=True)
class OutputConfig:
    """Configuration for optional pipeline artifacts written to disk."""

    output_datasets_path: Path | None = None
    n_sampled_distance_pairs: int = 100
    create_output_datasets_subdirectory: bool = True


@dataclass(slots=True)
class PipelineConfig:
    """Complete configuration for the RecSys NLE pipeline."""

    explanation: ExplanationConfig
    attribution: AttributionConfig
    target_set: TargetSet
    user_pool: UserPool
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 42
    max_users_for_attribution: int | None = None
    sample_user_count: int = 3
    show_prompts: bool = False
