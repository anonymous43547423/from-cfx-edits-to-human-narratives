"""Helpers for constructing NL explanation datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from datasets import Dataset  # type: ignore[attr-defined]
from recsys_nle.nl_explanations.payloads import (
    prepare_interaction_payload,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from recsys_nle.core.attribution import UserAttribution


def build_explanation_record(
    *,
    user_id: int,
    cfx_interactions: DataFrame,
    non_cfx_interactions: DataFrame,
) -> dict[str, object] | None:
    """Construct a serialisable explanation record for a single user."""
    # For dataset, store all CFX interactions (evaluators use all)
    cfx_payload = prepare_interaction_payload(cfx_interactions, max_items=None)
    if not cfx_payload:
        return None

    # Non-CFX is already limited to N during computation
    non_cfx_payload = prepare_interaction_payload(non_cfx_interactions, max_items=None)

    return {
        "user_id": user_id,
        "cfx_interactions": cfx_payload,
        "non_cfx_interactions": non_cfx_payload,
    }


def build_explanation_dataset(
    *,
    attributions: Mapping[int, UserAttribution],
    user_ids: Sequence[int],
) -> Dataset:
    """Create a Hugging Face dataset aligning recommendations with attribution context."""
    records: list[dict[str, object]] = []
    for user_id in user_ids:
        attribution = attributions.get(user_id)
        if attribution is None:
            continue

        record = build_explanation_record(
            user_id=user_id,
            cfx_interactions=attribution.cfx_interactions,
            non_cfx_interactions=attribution.non_cfx_interactions,
        )
        if record is not None:
            records.append(record)

    return Dataset.from_list(records)
