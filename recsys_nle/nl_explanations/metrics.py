"""Lightweight counters for tracking LLM usage during explanation runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LLMUsageMetrics:
    """Cumulative usage statistics for Hugging Face LLM calls."""

    hf_calls: int = 0
    hf_total_seconds: float = 0.0


_METRICS = LLMUsageMetrics()


def record_hf_call(duration_seconds: float) -> None:
    """Record the duration of a single Hugging Face LLM request."""
    if duration_seconds < 0.0:
        return
    _METRICS.hf_calls += 1
    _METRICS.hf_total_seconds += float(duration_seconds)


def get_llm_usage_metrics() -> LLMUsageMetrics:
    """Return a snapshot of the current LLM usage metrics."""
    return LLMUsageMetrics(
        hf_calls=_METRICS.hf_calls,
        hf_total_seconds=_METRICS.hf_total_seconds,
    )
