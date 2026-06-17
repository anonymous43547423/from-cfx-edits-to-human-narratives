# ruff: noqa: S101, PLR2004, SLF001

"""Tests for the natural-language explanation workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from datasets import Dataset  # type: ignore[attr-defined]
from recsys_nle.nl_explanations import workflow as workflow_module
from recsys_nle.nl_explanations.evaluation import EvaluationResult
from recsys_nle.nl_explanations.evaluation.faithfulness import FaithfulnessConfig, ScoredItem
from recsys_nle.nl_explanations.generator import GeneratedExplanation, UserExplanationInput
from recsys_nle.nl_explanations.workflow import ExplanationWorkflow

_RECORD_COUNT = 2
_PLAUSIBILITY_SCORE = 0.9


def _build_records() -> Sequence[dict[str, object]]:
    """Return in-memory records matching the dataset builder schema."""
    return [
        {
            "user_id": 1,
            "recommendations": [{"movie_id": 101, "score": 0.9, "rank": 1}],
            "cfx_interactions": [{"movie_id": 5, "movie_title": "Movie A", "rating": 4.0}],
            "non_cfx_interactions": [{"movie_id": 50, "movie_title": "Movie B", "rating": 3.0}],
        },
        {
            "user_id": 2,
            "recommendations": [{"movie_id": 202, "score": 0.8, "rank": 1}],
            "cfx_interactions": [{"movie_id": 6, "movie_title": "Movie C", "rating": 3.5}],
            "non_cfx_interactions": [{"movie_id": 60, "movie_title": "Movie D", "rating": 2.5}],
        },
    ]


@dataclass(slots=True)
class _StubGenerator:
    """Deterministic generator stub for testing batches."""

    calls: list[str]

    def generate(self, user_input: UserExplanationInput) -> GeneratedExplanation:
        """Record the call order and return canned content."""
        self.calls.append(str(user_input.user_id))
        user_str = str(user_input.user_id)
        return GeneratedExplanation(
            user_id=user_input.user_id,
            reasoning=f"reasoning-{user_str}",
            explanation=f"Because you enjoy {user_str}",
        )

    def generate_batch(
        self,
        inputs: Sequence[UserExplanationInput],
        *,
        batch_size: int | None = None,
    ) -> dict[int, GeneratedExplanation]:
        """Generate explanations for a batch of users."""
        _ = batch_size
        results: dict[int, GeneratedExplanation] = {}
        for user_input in inputs:
            generated = self.generate(user_input)
            results[user_input.user_id] = generated
        return results


@dataclass(slots=True)
class _BatchGenerator:
    """Generator stub exposing batch execution for testing."""

    batch_calls: list[list[str]]
    batch_sizes: list[int | None] = field(default_factory=list)

    def generate_batch(
        self,
        inputs: Sequence[UserExplanationInput],
        *,
        batch_size: int | None = None,
    ) -> dict[int, GeneratedExplanation]:
        """Record batch invocations and return canned content."""
        user_ids = [str(user_input.user_id) for user_input in inputs]
        self.batch_calls.append(user_ids)
        self.batch_sizes.append(batch_size)
        results: dict[int, GeneratedExplanation] = {}
        for user_input in inputs:
            user_str = str(user_input.user_id)
            results[user_input.user_id] = GeneratedExplanation(
                user_id=user_input.user_id,
                reasoning=f"reasoning-{user_str}",
                explanation=f"Because you enjoy {user_str}",
            )
        return results

    def generate(self, user_input: UserExplanationInput) -> GeneratedExplanation:  # pragma: no cover - guard path
        """Prevent fallback to single-item generation during tests."""
        message = f"generate() should not be called for user_id={user_input.user_id}"
        raise AssertionError(message)


@dataclass(slots=True)
class _ClosableGenerator:
    """Generator stub that records resource cleanup."""

    calls: list[str]
    events: list[str]

    def generate_batch(
        self,
        inputs: Sequence[UserExplanationInput],
        *,
        batch_size: int | None = None,
    ) -> dict[int, GeneratedExplanation]:
        """Generate explanations and capture call order."""
        _ = batch_size
        results: dict[int, GeneratedExplanation] = {}
        for user_input in inputs:
            self.calls.append(str(user_input.user_id))
            user_str = str(user_input.user_id)
            results[user_input.user_id] = GeneratedExplanation(
                user_id=user_input.user_id,
                reasoning=f"reasoning-{user_str}",
                explanation=f"Because you enjoy {user_str}",
            )
        return results

    def generate(self, user_input: UserExplanationInput) -> GeneratedExplanation:  # pragma: no cover - guard path
        """Prevent fallback to single-item generation during tests."""
        message = f"generate() should not be called for user_id={user_input.user_id}"
        raise AssertionError(message)

    def close(self) -> None:
        """Record the cleanup event."""
        self.events.append("generation_closed")


@dataclass(slots=True)
class _StubPlausibilityEvaluator:
    """Stub plausibility evaluator returning deterministic results."""

    calls: list[str]
    prefix: str = "plausibility"

    def build_prompt(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame | None = None,
    ) -> list[dict[str, str]]:
        """Build a plausibility evaluation prompt."""
        _ = interactions
        identifier = explanation.rsplit(None, 1)[-1]
        self.calls.append(f"{self.prefix}-prompt-{identifier}")
        return [
            {"role": "system", "content": "You are a plausibility judge."},
            {"role": "user", "content": f"Evaluate: {explanation}"},
        ]

    def parse_result(self, raw_output: str, *, prompt: str | None = None) -> EvaluationResult:
        """Parse raw output into evaluation result."""
        _ = (raw_output, prompt)
        return EvaluationResult(judgment=f"{self.prefix}-plausible", score=_PLAUSIBILITY_SCORE)


@dataclass(slots=True)
class _StubInteractionEvaluator:
    """Stub interaction-based evaluator returning deterministic results."""

    calls: list[str]
    prefix: str

    def build_all_prompts(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame,
    ) -> list[tuple[str, list[dict[str, str]]]]:
        """Build prompts for all interactions."""
        identifier = explanation.rsplit(None, 1)[-1]
        self.calls.append(f"{self.prefix}-prompts-{identifier}")
        prompts: list[tuple[str, list[dict[str, str]]]] = []
        for _, row in interactions.iterrows():
            movie_title = str(row.get("movie_title", row.get("movie_id", "")))
            interaction_desc = f"{movie_title}"
            messages = [
                {"role": "system", "content": "You are a match judge."},
                {"role": "user", "content": f"Check {movie_title} against {explanation}"},
            ]
            prompts.append((interaction_desc, messages))
        return prompts

    def build_all_prompts_with_ids(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame,
    ) -> list[tuple[int, str, list[dict[str, str]]]]:
        """Build prompts for all interactions with movie IDs."""
        prompts_with_ids: list[tuple[int, str, list[dict[str, str]]]] = []
        for (_, row), (interaction_desc, messages) in zip(
            interactions.iterrows(),
            self.build_all_prompts(explanation=explanation, interactions=interactions),
            strict=True,
        ):
            movie_id = int(row.get("movie_id", -1))
            prompts_with_ids.append((movie_id, interaction_desc, messages))
        return prompts_with_ids

    def parse_single_interaction_result(
        self,
        raw_output: str,
        interaction_description: str,
    ) -> Mapping[str, object]:
        """Parse a single interaction result."""
        _ = raw_output
        return {
            "interaction": interaction_description,
            "score": _PLAUSIBILITY_SCORE,
            "judgment": f"{self.prefix}-match",
        }

    def aggregate_results(
        self,
        per_interaction_scores: Sequence[Mapping[str, object]],
    ) -> EvaluationResult:
        """Aggregate per-interaction results."""
        return EvaluationResult(
            judgment=f"{self.prefix}-aggregated",
            score=_PLAUSIBILITY_SCORE,
            details={"per_interaction_scores": list(per_interaction_scores)},
        )

    def build_empty_result(self, *, reason: str) -> EvaluationResult:
        """Build an empty result."""
        return EvaluationResult(
            judgment=reason,
            score=float("nan"),
            details={"per_interaction_scores": []},
        )


@dataclass(slots=True)
class _StubEvaluationLLM:
    """Stub LLM client for evaluation that records calls."""

    calls: list[list[list[dict[str, str]]]]

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Return canned JSON responses for each prompt."""
        _ = (max_new_tokens, temperature, top_p, stop_sequences, batch_size)
        self.calls.append([[dict(msg) for msg in msgs] for msgs in messages_batch])
        return ['{"judgment": "stub", "score": 0.9}' for _ in messages_batch]

    def close(self) -> None:
        """No-op close for protocol compatibility."""


@dataclass(slots=True)
class _RecordingEvaluationLLM:
    """Evaluation LLM stub that records when called."""

    events: list[str]

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Record evaluation invocation and return canned scores."""
        _ = (messages_batch, max_new_tokens, temperature, top_p, stop_sequences, batch_size)
        self.events.append("evaluation_called")
        return ['{"judgment": "stub", "score": 0.9}' for _ in messages_batch]

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Avoid single-message generation in this test stub."""
        _ = (messages, max_new_tokens, temperature, top_p, stop_sequences)
        message = "generate() should not be called for _RecordingEvaluationLLM"
        raise AssertionError(message)

    def close(self) -> None:
        """Record the cleanup event."""
        self.events.append("evaluation_closed")


@dataclass(slots=True)
class _RecordingFaithfulnessEvaluator:
    """Faithfulness evaluator stub that records min-limit values and CFX item IDs."""

    calls: list[int]
    cfx_item_ids_received: list[list[int]] = field(default_factory=list)

    def compute_results_from_scores(
        self,
        *,
        scored_items: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: object,
        config: FaithfulnessConfig,
        cfx_item_ids: Sequence[int] = (),
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Record the min-limit and cfx_item_ids, return placeholder results."""
        _ = (scored_items, user_history, target_item, recommender)
        self.calls.append(config.n_interactions_min_limit)
        self.cfx_item_ids_received.append(list(cfx_item_ids))
        return (
            EvaluationResult(judgment="faithfulness", score=0.5),
            EvaluationResult(judgment="faithfulness-baseline", score=0.5),
        )


def test_enrich_dataset_uses_dataset_map_batches() -> None:
    """It generates and evaluates explanations via dataset.map batches."""
    generator = _StubGenerator(calls=[])
    evaluation_llm = _StubEvaluationLLM(calls=[])
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=evaluation_llm,  # type: ignore[arg-type]
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        generation_batch_size=2,
        evaluation_user_batch_size=2,
        evaluation_llm_batch_size=2,
    )

    dataset = Dataset.from_list(list(_build_records()))
    result = workflow.enrich_dataset(dataset)

    assert len(result.dataset) == _RECORD_COUNT
    assert generator.calls == ["1", "2"]
    assert set(result.results_by_user.keys()) == {1, 2}

    first_user = result.results_by_user[1]
    assert first_user.explanation == "Because you enjoy 1"
    assert pytest.approx(first_user.explanation_plausibility.score) == _PLAUSIBILITY_SCORE
    cfx_details = first_user.explanation_cfx_match.details or {}
    cfx_scores = cfx_details.get("per_interaction_scores", [])
    assert isinstance(cfx_scores, list)
    assert cfx_scores
    first_cfx_score = cfx_scores[0]
    assert isinstance(first_cfx_score, Mapping)
    assert first_cfx_score.get("item_id") == 5

    record = result.dataset[0]
    assert record["reasoning"] == "reasoning-1"
    assert record["explanation_plausibility"] == {
        "judgment": "plausibility-plausible",
        "score": _PLAUSIBILITY_SCORE,
    }


def test_generate_batch_prefers_batch_method() -> None:
    """It prefers generator.generate_batch when available."""
    generator = _BatchGenerator(batch_calls=[])
    evaluation_llm = _StubEvaluationLLM(calls=[])
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=evaluation_llm,  # type: ignore[arg-type]
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        generation_batch_size=2,
        evaluation_user_batch_size=2,
        evaluation_llm_batch_size=2,
    )

    dataset = Dataset.from_list(list(_build_records()))
    workflow.enrich_dataset(dataset)

    assert generator.batch_calls == [["1", "2"]]
    assert generator.batch_sizes == [2]


def test_evaluation_llm_called_with_batch() -> None:
    """It calls the evaluation LLM with batched prompts."""
    generator = _StubGenerator(calls=[])
    evaluation_llm = _StubEvaluationLLM(calls=[])
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=evaluation_llm,  # type: ignore[arg-type]
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        generation_batch_size=2,
        evaluation_user_batch_size=2,
        evaluation_llm_batch_size=2,
    )

    dataset = Dataset.from_list(list(_build_records()))
    workflow.enrich_dataset(dataset)

    # Should have called the evaluation LLM at least once with batched prompts
    assert len(evaluation_llm.calls) >= 1
    # Each user has: 1 plausibility + 1 CFX + 1 non-CFX = 3 prompts per user
    # With 2 users: 6 total prompts
    total_prompts = sum(len(batch) for batch in evaluation_llm.calls)
    assert total_prompts == 6


def test_enabled_evaluations_limit_prompts() -> None:
    """It builds prompts only for enabled evaluators."""
    generator = _StubGenerator(calls=[])
    evaluation_llm = _StubEvaluationLLM(calls=[])
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=evaluation_llm,  # type: ignore[arg-type]
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        generation_batch_size=2,
        evaluation_user_batch_size=2,
        evaluation_llm_batch_size=2,
        enabled_evaluations=("plausibility",),
    )

    dataset = Dataset.from_list(list(_build_records()))
    workflow.enrich_dataset(dataset)

    # Only plausibility evaluator should have been called
    assert len(plausibility.calls) == _RECORD_COUNT
    assert cfx_match.calls == []
    assert non_cfx_match.calls == []

    # LLM should only receive plausibility prompts (1 per user = 2 total)
    total_prompts = sum(len(batch) for batch in evaluation_llm.calls)
    assert total_prompts == 2


def test_generation_cleanup_happens_before_evaluation() -> None:
    """It releases generation resources before evaluation begins."""
    events: list[str] = []
    generator = _ClosableGenerator(calls=[], events=events)
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")
    evaluation_llm = _RecordingEvaluationLLM(events=events)

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=None,
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        generation_batch_size=2,
        evaluation_user_batch_size=2,
        evaluation_llm_batch_size=2,
    )

    dataset = Dataset.from_list(list(_build_records()))
    generated_dataset, generated_store = workflow.generate_dataset(dataset)

    generator.close()
    workflow.set_evaluation_llm_client(evaluation_llm)
    result = workflow.evaluate_dataset(dataset=generated_dataset, generated_store=generated_store)

    assert "generation_closed" in events
    assert "evaluation_called" in events
    assert events.index("generation_closed") < events.index("evaluation_called")
    assert len(result.dataset) == _RECORD_COUNT


def test_faithfulness_passes_min_limit() -> None:
    """It forwards the min-limit setting to faithfulness evaluators."""
    removal_evaluator = _RecordingFaithfulnessEvaluator(calls=[])
    replacement_evaluator = _RecordingFaithfulnessEvaluator(calls=[])

    workflow = ExplanationWorkflow(
        generator=_StubGenerator(calls=[]),  # type: ignore[arg-type]
        evaluation_llm_client=None,
        plausibility_evaluator=_StubPlausibilityEvaluator(calls=[]),  # type: ignore[arg-type]
        cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="cfx_match"),  # type: ignore[arg-type]
        non_cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="non_cfx_match"),  # type: ignore[arg-type]
        faithfulness_removal_evaluator=removal_evaluator,  # type: ignore[arg-type]
        faithfulness_replacement_evaluator=replacement_evaluator,  # type: ignore[arg-type]
        recommender=MagicMock(),
        enabled_evaluations=("faithfulness_removal", "faithfulness_replacement"),
    )

    context = workflow_module._EvaluationContext(
        user_id=1,
        cfx_interactions=pd.DataFrame(),
        cfx_interactions_full=pd.DataFrame(),
        non_cfx_interactions=pd.DataFrame(),
        explanation="stub",
        user_history=torch.ones(5),
        target_item=1,
    )
    scores = workflow_module._FaithfulnessScores(
        removal_scores=[ScoredItem(item_id=1, score=0.9, interaction_description="A")],
        replacement_scores=[ScoredItem(item_id=2, score=0.8, interaction_description="B")],
    )
    config = FaithfulnessConfig(
        n_sampled_faithfulness_interactions=5,
        match_threshold=0.5,
        n_interactions_min_limit=3,
        n_faithfulness_trials=1,
        n_faithfulness_samples=1,
    )

    workflow._compute_user_faithfulness(context, scores, config)

    assert removal_evaluator.calls == [3]
    assert replacement_evaluator.calls == [3]


def test_faithfulness_passes_cfx_item_ids_to_evaluators() -> None:
    """It passes CFX item IDs to faithfulness evaluators."""
    removal_evaluator = _RecordingFaithfulnessEvaluator(calls=[])
    replacement_evaluator = _RecordingFaithfulnessEvaluator(calls=[])

    workflow = ExplanationWorkflow(
        generator=_StubGenerator(calls=[]),  # type: ignore[arg-type]
        evaluation_llm_client=None,
        plausibility_evaluator=_StubPlausibilityEvaluator(calls=[]),  # type: ignore[arg-type]
        cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="cfx_match"),  # type: ignore[arg-type]
        non_cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="non_cfx_match"),  # type: ignore[arg-type]
        faithfulness_removal_evaluator=removal_evaluator,  # type: ignore[arg-type]
        faithfulness_replacement_evaluator=replacement_evaluator,  # type: ignore[arg-type]
        recommender=MagicMock(),
        enabled_evaluations=("faithfulness_removal", "faithfulness_replacement"),
    )

    # CFX interactions contain movie_ids 2 and 5
    cfx_interactions = pd.DataFrame({"movie_id": [2, 5], "movie_title": ["A", "B"]})

    context = workflow_module._EvaluationContext(
        user_id=1,
        cfx_interactions=cfx_interactions,
        cfx_interactions_full=cfx_interactions,
        non_cfx_interactions=pd.DataFrame(),
        explanation="stub",
        user_history=torch.ones(10),
        target_item=1,
    )
    scores = workflow_module._FaithfulnessScores(
        removal_scores=[ScoredItem(item_id=1, score=0.9, interaction_description="A")],
        replacement_scores=[ScoredItem(item_id=3, score=0.8, interaction_description="B")],
    )
    config = FaithfulnessConfig(
        n_sampled_faithfulness_interactions=5,
        match_threshold=0.5,
        n_interactions_min_limit=1,
        n_faithfulness_trials=1,
        n_faithfulness_samples=1,
    )

    workflow._compute_user_faithfulness(context, scores, config)

    # Both evaluators should receive the same CFX item IDs
    assert len(removal_evaluator.cfx_item_ids_received) == 1
    assert len(replacement_evaluator.cfx_item_ids_received) == 1

    assert removal_evaluator.cfx_item_ids_received[0] == [2, 5]
    assert replacement_evaluator.cfx_item_ids_received[0] == [2, 5]


def test_judged_interactions_do_not_limit_faithfulness_candidates() -> None:
    """It uses full interaction sets for faithfulness candidates."""
    generator = _StubGenerator(calls=[])
    plausibility = _StubPlausibilityEvaluator(calls=[])
    cfx_match = _StubInteractionEvaluator(calls=[], prefix="cfx_match")
    non_cfx_match = _StubInteractionEvaluator(calls=[], prefix="non_cfx_match")
    recommender = MagicMock()
    recommender.num_items = 10

    workflow = ExplanationWorkflow(
        generator=generator,  # type: ignore[arg-type]
        evaluation_llm_client=None,
        plausibility_evaluator=plausibility,  # type: ignore[arg-type]
        cfx_match_evaluator=cfx_match,  # type: ignore[arg-type]
        non_cfx_match_evaluator=non_cfx_match,  # type: ignore[arg-type]
        recommender=recommender,
        enabled_evaluations=("faithfulness_removal", "cfx_match", "non_cfx_match"),
    )

    batch: Mapping[str, Sequence[object]] = {
        "user_id": [1],
        "cfx_interactions": [[{"movie_id": 1}, {"movie_id": 2}, {"movie_id": 3}]],
        "non_cfx_interactions": [[{"movie_id": 100}, {"movie_id": 101}]],
        "explanation": ["stub"],
    }
    generated_store = {1: GeneratedExplanation(user_id=1, reasoning="reason", explanation="text")}
    user_history = torch.ones(10, dtype=torch.float32)
    faithfulness_config = FaithfulnessConfig(
        n_sampled_faithfulness_interactions=5,
        match_threshold=0.5,
        n_interactions_min_limit=1,
        n_faithfulness_trials=1,
        n_faithfulness_samples=1,
    )

    contexts = workflow._build_evaluation_contexts(
        batch=batch,
        generated_store=generated_store,
        n_judged_interactions=1,
        faithfulness_config=faithfulness_config,
        user_histories={1: user_history},
        user_targets={},
    )

    assert len(contexts) == 1
    context = contexts[0]
    assert len(context.cfx_interactions) == 1
    assert len(context.cfx_interactions_full) == 3
    assert len(context.non_cfx_interactions) == 1
    assert context.faithfulness_removal_candidates is not None
    assert len(context.faithfulness_removal_candidates) == 5
    removal_ids = set(context.faithfulness_removal_candidates["movie_id"].tolist())
    assert removal_ids.issubset({0, 4, 5, 6, 7, 8, 9})


def test_faithfulness_uses_full_cfx_items_when_sampled() -> None:
    """It uses full CFX items when evaluated interactions are sampled."""
    removal_evaluator = _RecordingFaithfulnessEvaluator(calls=[])
    replacement_evaluator = _RecordingFaithfulnessEvaluator(calls=[])

    workflow = ExplanationWorkflow(
        generator=_StubGenerator(calls=[]),  # type: ignore[arg-type]
        evaluation_llm_client=None,
        plausibility_evaluator=_StubPlausibilityEvaluator(calls=[]),  # type: ignore[arg-type]
        cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="cfx_match"),  # type: ignore[arg-type]
        non_cfx_match_evaluator=_StubInteractionEvaluator(calls=[], prefix="non_cfx_match"),  # type: ignore[arg-type]
        faithfulness_removal_evaluator=removal_evaluator,  # type: ignore[arg-type]
        faithfulness_replacement_evaluator=replacement_evaluator,  # type: ignore[arg-type]
        recommender=MagicMock(),
        enabled_evaluations=("faithfulness_removal", "faithfulness_replacement"),
    )

    cfx_full = pd.DataFrame({"movie_id": [2, 5, 7]})
    context = workflow_module._EvaluationContext(
        user_id=1,
        cfx_interactions=cfx_full.head(1),
        cfx_interactions_full=cfx_full,
        non_cfx_interactions=pd.DataFrame(),
        explanation="stub",
        user_history=torch.ones(10),
        target_item=1,
    )
    scores = workflow_module._FaithfulnessScores(
        removal_scores=[ScoredItem(item_id=1, score=0.9, interaction_description="A")],
        replacement_scores=[ScoredItem(item_id=3, score=0.8, interaction_description="B")],
    )
    config = FaithfulnessConfig(
        n_sampled_faithfulness_interactions=5,
        match_threshold=0.5,
        n_interactions_min_limit=1,
        n_faithfulness_trials=1,
        n_faithfulness_samples=1,
    )

    workflow._compute_user_faithfulness(context, scores, config)

    assert removal_evaluator.cfx_item_ids_received[0] == [2, 5, 7]
    assert replacement_evaluator.cfx_item_ids_received[0] == [2, 5, 7]


def test_extract_cfx_item_ids_returns_movie_ids() -> None:
    """It extracts movie_id values from CFX interactions DataFrame."""
    cfx_interactions = pd.DataFrame({"movie_id": [1, 3, 7], "movie_title": ["A", "B", "C"]})

    result = ExplanationWorkflow._extract_cfx_item_ids(cfx_interactions)

    assert result == [1, 3, 7]


def test_extract_cfx_item_ids_handles_empty_dataframe() -> None:
    """It returns empty list when CFX interactions are empty."""
    cfx_interactions = pd.DataFrame()

    result = ExplanationWorkflow._extract_cfx_item_ids(cfx_interactions)

    assert result == []


def test_extract_cfx_item_ids_handles_missing_column() -> None:
    """It returns empty list when movie_id column is missing."""
    cfx_interactions = pd.DataFrame({"movie_title": ["A", "B", "C"]})

    result = ExplanationWorkflow._extract_cfx_item_ids(cfx_interactions)

    assert result == []
