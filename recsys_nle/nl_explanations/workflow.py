"""Workflow for generating and evaluating natural-language explanations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd
import torch  # noqa: TC002

from datasets import Dataset  # type: ignore[attr-defined]
from recsys_nle.nl_explanations.dataset_builder import build_explanation_dataset
from recsys_nle.nl_explanations.evaluation.base import EvaluationResult
from recsys_nle.nl_explanations.evaluation.faithfulness import (
    FaithfulnessConfig,
    FaithfulnessRemovalEvaluator,
    FaithfulnessReplacementEvaluator,
    ScoredItem,
)
from recsys_nle.nl_explanations.generator import (
    ExplanationGenerator,
    GeneratedExplanation,
    UserExplanationInput,
)
from recsys_nle.nl_explanations.prompts import serialise_chat_messages
from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult

if TYPE_CHECKING:
    from recsys_nle.core.attribution import UserAttribution
    from recsys_nle.core.recommender_wrapper import RecommenderWrapper
    from recsys_nle.nl_explanations.evaluation import (
        CFXMatchEvaluator,
        NonCFXMatchEvaluator,
        PlausibilityEvaluator,
        ReadabilityEvaluator,
    )
    from recsys_nle.nl_explanations.evaluation.interaction_scoring import InteractionScoringEvaluator
    from recsys_nle.nl_explanations.llm import LLMClient


_SUPPORTED_EVALUATION_METRICS: tuple[str, ...] = (
    "plausibility",
    "readability",
    "cfx_match",
    "non_cfx_match",
    "faithfulness_removal",
    "faithfulness_replacement",
)


@dataclass(slots=True)
class ExplanationConfig:
    """Controls generation of natural-language explanations."""

    model_id_generation: str
    model_id_evaluation: str
    n_faithfulness_interactions_min_limit: int
    n_faithfulness_trials: int
    n_faithfulness_samples: int
    n_cfx_interactions: int = 5
    n_judged_interactions: int | None = None
    generation_batch_size: int | None = 4
    evaluation_user_batch_size: int | None = 4
    evaluation_llm_batch_size: int | None = 4
    disable_reasoning: bool = False
    enabled_evaluations: tuple[str, ...] = _SUPPORTED_EVALUATION_METRICS
    n_sampled_faithfulness_interactions: int = 20
    faithfulness_match_threshold: float = 0.5


@dataclass(slots=True)
class ExplanationResult:
    """Outputs obtained after running the explanation workflow."""

    dataset: Dataset
    results_by_user: dict[int, NaturalLanguageExplanationResult]


class ExplanationWorkflow:
    """Generate natural-language explanations for user recommendations."""

    def __init__(
        self,
        *,
        generator: ExplanationGenerator,
        evaluation_llm_client: LLMClient | None,
        plausibility_evaluator: PlausibilityEvaluator,
        readability_evaluator: ReadabilityEvaluator | None = None,
        cfx_match_evaluator: CFXMatchEvaluator,
        non_cfx_match_evaluator: NonCFXMatchEvaluator | None = None,
        faithfulness_removal_evaluator: FaithfulnessRemovalEvaluator | None = None,
        faithfulness_replacement_evaluator: FaithfulnessReplacementEvaluator | None = None,
        recommender: RecommenderWrapper | None = None,
        generation_batch_size: int | None = 4,
        evaluation_user_batch_size: int | None = 4,
        evaluation_llm_batch_size: int | None = 4,
        enabled_evaluations: Sequence[str] | None = None,
    ) -> None:
        """Initialise the workflow with generation and evaluation components."""
        self._generator = generator
        self._evaluation_llm = evaluation_llm_client
        self._plausibility_evaluator = plausibility_evaluator
        self._readability_evaluator = readability_evaluator
        self._cfx_match_evaluator = cfx_match_evaluator
        self._non_cfx_match_evaluator = non_cfx_match_evaluator
        self._faithfulness_removal_evaluator = faithfulness_removal_evaluator
        self._faithfulness_replacement_evaluator = faithfulness_replacement_evaluator
        self._recommender = recommender
        self._generation_batch_size = generation_batch_size
        self._evaluation_user_batch_size = evaluation_user_batch_size
        self._evaluation_llm_batch_size = evaluation_llm_batch_size
        self._enabled_evaluations = self._normalise_enabled_evaluations(enabled_evaluations)

    def set_evaluation_llm_client(self, evaluation_llm_client: LLMClient) -> None:
        """Attach or replace the evaluation LLM client."""
        self._evaluation_llm = evaluation_llm_client

    @staticmethod
    def _normalise_enabled_evaluations(enabled: Sequence[str] | None) -> set[str]:
        """Normalise enabled evaluation metric names, defaulting to all when unset."""
        if enabled is None:
            return set(_SUPPORTED_EVALUATION_METRICS)
        selected: set[str] = set()
        for name in enabled:
            lowered = str(name).strip().lower()
            if lowered in _SUPPORTED_EVALUATION_METRICS:
                selected.add(lowered)
        return selected or set(_SUPPORTED_EVALUATION_METRICS)

    def run(
        self,
        *,
        attributions: Mapping[int, UserAttribution],
        user_ids: Sequence[int],
        config: ExplanationConfig,
        user_histories: Mapping[int, torch.Tensor] | None = None,
        user_targets: Mapping[int, int] | None = None,
    ) -> ExplanationResult:
        """Build inputs from raw data, then generate explanations."""
        dataset = build_explanation_dataset(
            attributions=attributions,
            user_ids=user_ids,
        )

        faithfulness_config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=config.n_sampled_faithfulness_interactions,
            match_threshold=config.faithfulness_match_threshold,
            n_interactions_min_limit=config.n_faithfulness_interactions_min_limit,
            n_faithfulness_trials=config.n_faithfulness_trials,
            n_faithfulness_samples=config.n_faithfulness_samples,
        )

        return self.enrich_dataset(
            dataset,
            n_judged_interactions=config.n_judged_interactions,
            faithfulness_config=faithfulness_config,
            user_histories=user_histories,
            user_targets=user_targets,
        )

    def enrich_dataset(
        self,
        dataset: Dataset,
        n_judged_interactions: int | None = None,
        faithfulness_config: FaithfulnessConfig | None = None,
        user_histories: Mapping[int, torch.Tensor] | None = None,
        user_targets: Mapping[int, int] | None = None,
    ) -> ExplanationResult:
        """Generate explanations and evaluations for each dataset record."""
        generated_dataset, generated_store = self.generate_dataset(dataset)
        return self.evaluate_dataset(
            generated_dataset,
            generated_store=generated_store,
            n_judged_interactions=n_judged_interactions,
            faithfulness_config=faithfulness_config,
            user_histories=user_histories,
            user_targets=user_targets,
        )

    def enrich_records(self, records: Sequence[dict[str, object]]) -> ExplanationResult:
        """Generate explanations for in-memory records without building inputs."""
        dataset = Dataset.from_list(list(records))
        return self.enrich_dataset(dataset)

    def generate_dataset(
        self,
        dataset: Dataset,
    ) -> tuple[Dataset, dict[int, GeneratedExplanation]]:
        """Generate explanations for a dataset without evaluation."""
        generated_store: dict[int, GeneratedExplanation] = {}
        generated_dataset = dataset.map(
            self._generate_batch,
            batched=True,
            batch_size=self._generation_batch_size,
            load_from_cache_file=False,
            fn_kwargs={"generated_store": generated_store},
            new_fingerprint="explanation-generate-v1-dummy-fingerprint",
            desc="Generating explanations",
        )
        return generated_dataset, generated_store

    def evaluate_dataset(
        self,
        dataset: Dataset,
        *,
        generated_store: dict[int, GeneratedExplanation],
        n_judged_interactions: int | None = None,
        faithfulness_config: FaithfulnessConfig | None = None,
        user_histories: Mapping[int, torch.Tensor] | None = None,
        user_targets: Mapping[int, int] | None = None,
    ) -> ExplanationResult:
        """Evaluate explanations for a dataset and return assembled results."""
        evaluation_store: dict[int, dict[str, EvaluationResult | None]] = {}
        evaluated_dataset = dataset.map(
            self._evaluate_batch,
            batched=True,
            batch_size=self._evaluation_user_batch_size,
            load_from_cache_file=False,
            fn_kwargs={
                "generated_store": generated_store,
                "evaluation_store": evaluation_store,
                "n_judged_interactions": n_judged_interactions,
                "faithfulness_config": faithfulness_config,
                "user_histories": user_histories,
                "user_targets": user_targets,
            },
            new_fingerprint="explanation-evaluate-v1-dummy-fingerprint",
            desc="Evaluating explanations",
        )
        results_by_user = self._assemble_results(
            dataset=evaluated_dataset,
            generated_store=generated_store,
            evaluation_store=evaluation_store,
        )
        return ExplanationResult(dataset=evaluated_dataset, results_by_user=results_by_user)

    def _generate_batch(
        self,
        batch: Mapping[str, Sequence[object]],
        *,
        generated_store: dict[int, GeneratedExplanation],
    ) -> dict[str, list[Any]]:
        """Generate reasoning and explanations for a dataset batch."""
        user_inputs = self._build_user_inputs(batch)
        generated_map = self._generate_explanations(user_inputs)
        return self._store_and_collect_outputs(user_inputs, generated_map, generated_store)

    def _build_user_inputs(self, batch: Mapping[str, Sequence[object]]) -> list[UserExplanationInput]:
        """Parse batch data into user explanation inputs."""
        user_inputs: list[UserExplanationInput] = []
        for index, raw_user_id in enumerate(batch["user_id"]):
            user_id = int(raw_user_id)  # type: ignore[call-overload]
            user_inputs.append(
                UserExplanationInput(
                    user_id=user_id,
                    cfx_interactions=self._frame_from_payload(batch["cfx_interactions"][index]),
                    non_cfx_interactions=self._frame_from_payload(batch["non_cfx_interactions"][index]),
                )
            )
        return user_inputs

    def _store_and_collect_outputs(
        self,
        user_inputs: list[UserExplanationInput],
        generated_map: dict[int, GeneratedExplanation],
        generated_store: dict[int, GeneratedExplanation],
    ) -> dict[str, list[Any]]:
        """Store generated explanations and collect output columns."""
        reasoning: list[Any] = []
        explanations: list[Any] = []

        for user_input in user_inputs:
            user_id = user_input.user_id
            generated = generated_map.get(user_id)
            if generated is None:
                msg = f"Generated explanation missing for user_id={user_id}"
                raise KeyError(msg)
            reasoning.append(generated.reasoning)
            explanations.append(generated.explanation)
            generated_store[user_id] = generated

        return {"reasoning": reasoning, "explanation": explanations}

    def _evaluate_batch(
        self,
        batch: Mapping[str, Sequence[object]],
        *,
        generated_store: Mapping[int, GeneratedExplanation],
        evaluation_store: dict[int, dict[str, EvaluationResult | None]],
        n_judged_interactions: int | None = None,
        faithfulness_config: FaithfulnessConfig | None = None,
        user_histories: Mapping[int, torch.Tensor] | None = None,
        user_targets: Mapping[int, int] | None = None,
    ) -> dict[str, list[Any]]:
        """Evaluate generated explanations using batch-first LLM inference."""
        user_histories = user_histories or {}
        user_targets = user_targets or {}

        contexts = self._build_evaluation_contexts(
            batch, generated_store, n_judged_interactions, faithfulness_config, user_histories, user_targets
        )

        evaluations_by_user = self._run_llm_batch_evaluation(contexts, faithfulness_config)

        return self._store_and_build_output_columns(contexts, evaluations_by_user, evaluation_store)

    def _build_evaluation_contexts(
        self,
        batch: Mapping[str, Sequence[object]],
        generated_store: Mapping[int, GeneratedExplanation],
        n_judged_interactions: int | None,
        faithfulness_config: FaithfulnessConfig | None,
        user_histories: Mapping[int, torch.Tensor],
        user_targets: Mapping[int, int],
    ) -> list[_EvaluationContext]:
        """Build evaluation contexts for all users in the batch."""
        contexts: list[_EvaluationContext] = []

        for index, raw_user_id in enumerate(batch["user_id"]):
            user_id = int(raw_user_id)  # type: ignore[call-overload]
            if user_id not in generated_store:
                msg = f"Generated explanation missing for user_id={user_id}"
                raise KeyError(msg)

            cfx_frame_full = self._frame_from_payload(batch["cfx_interactions"][index])
            non_cfx_frame_full = self._frame_from_payload(batch["non_cfx_interactions"][index])
            explanation_text = str(batch["explanation"][index])

            cfx_frame = cfx_frame_full
            non_cfx_frame = non_cfx_frame_full
            if n_judged_interactions is not None:
                cfx_frame, non_cfx_frame = self._sample_judged_interactions(
                    user_id=user_id,
                    cfx_interactions=cfx_frame_full,
                    non_cfx_interactions=non_cfx_frame_full,
                    n_judged_interactions=n_judged_interactions,
                )

            cfx_item_ids = self._extract_cfx_item_ids(cfx_frame_full)
            user_history = user_histories.get(user_id)
            removal_candidates, replacement_candidates = self._prepare_faithfulness_candidates(
                user_id,
                faithfulness_config,
                user_history,
                cfx_item_ids,
            )

            contexts.append(
                _EvaluationContext(
                    user_id=user_id,
                    cfx_interactions=cfx_frame,
                    cfx_interactions_full=cfx_frame_full,
                    non_cfx_interactions=non_cfx_frame,
                    explanation=explanation_text,
                    user_history=user_history,
                    target_item=user_targets.get(user_id),
                    faithfulness_removal_candidates=removal_candidates,
                    faithfulness_replacement_candidates=replacement_candidates,
                )
            )

        return contexts

    def _run_llm_batch_evaluation(
        self,
        contexts: list[_EvaluationContext],
        faithfulness_config: FaithfulnessConfig | None,
    ) -> dict[int, dict[str, EvaluationResult | None]]:
        """Collect prompts, run LLM batch, and distribute results."""
        all_prompts = self._collect_all_evaluation_prompts(contexts)
        raw_outputs = self._execute_llm_batch(all_prompts)

        evaluations_by_user, faithfulness_scores_by_user = self._distribute_evaluation_results(
            contexts, all_prompts, list(raw_outputs)
        )

        if faithfulness_config is not None and self._recommender is not None:
            self._merge_faithfulness_results(
                evaluations_by_user, contexts, faithfulness_scores_by_user, faithfulness_config
            )

        return evaluations_by_user

    def _execute_llm_batch(self, all_prompts: list[_PromptInfo]) -> list[str]:
        """Execute LLM batch generation for all prompts."""
        if self._evaluation_llm is None:
            msg = "Evaluation LLM client is required to run evaluations."
            raise RuntimeError(msg)
        all_messages = [prompt_info.messages for prompt_info in all_prompts]
        if not all_messages:
            return []
        return list(
            self._evaluation_llm.generate_batch(
                all_messages,
                max_new_tokens=256,
                temperature=0.0,
                batch_size=self._evaluation_llm_batch_size,
            )
        )

    def _merge_faithfulness_results(
        self,
        evaluations_by_user: dict[int, dict[str, EvaluationResult | None]],
        contexts: Sequence[_EvaluationContext],
        faithfulness_scores_by_user: Mapping[int, _FaithfulnessScores],
        config: FaithfulnessConfig,
    ) -> None:
        """Compute and merge faithfulness results into evaluation results."""
        faithfulness_results = self._compute_faithfulness_results(
            contexts=contexts,
            faithfulness_scores_by_user=faithfulness_scores_by_user,
            config=config,
        )
        for user_id, faith_results in faithfulness_results.items():
            if user_id in evaluations_by_user:
                evaluations_by_user[user_id].update(faith_results)

    def _store_and_build_output_columns(
        self,
        contexts: list[_EvaluationContext],
        evaluations_by_user: dict[int, dict[str, EvaluationResult | None]],
        evaluation_store: dict[int, dict[str, EvaluationResult | None]],
    ) -> dict[str, list[Any]]:
        """Store evaluation results and build dataset output columns."""
        plausibility_payloads: list[Any] = []
        readability_payloads: list[Any] = []
        cfx_match_payloads: list[Any] = []
        non_cfx_match_payloads: list[Any] = []

        for context in contexts:
            evaluations = evaluations_by_user[context.user_id]
            evaluation_store[context.user_id] = evaluations
            plausibility_payloads.append(self._serialise_evaluation(evaluations["explanation_plausibility"]))
            readability_payloads.append(self._serialise_evaluation(evaluations["explanation_readability"]))
            cfx_match_payloads.append(self._serialise_evaluation(evaluations["explanation_cfx_match"]))
            non_cfx_match_payloads.append(self._serialise_evaluation(evaluations.get("explanation_non_cfx_match")))

        return {
            "explanation_plausibility": plausibility_payloads,
            "explanation_readability": readability_payloads,
            "explanation_cfx_match": cfx_match_payloads,
            "explanation_non_cfx_match": non_cfx_match_payloads,
        }

    def _prepare_faithfulness_candidates(
        self,
        user_id: int,
        faithfulness_config: FaithfulnessConfig | None,
        user_history: torch.Tensor | None,
        cfx_item_ids: Sequence[int],
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Prepare candidate interactions for faithfulness evaluation."""
        if faithfulness_config is None or user_history is None:
            return None, None

        n_sample = faithfulness_config.n_sampled_faithfulness_interactions
        removal_candidates: pd.DataFrame | None = None
        replacement_candidates: pd.DataFrame | None = None

        if "faithfulness_removal" in self._enabled_evaluations:
            removal_candidates = self._build_removal_candidates(
                user_history=user_history,
                cfx_item_ids=cfx_item_ids,
                n_sample=n_sample,
                seed=user_id,
            )

        if "faithfulness_replacement" in self._enabled_evaluations and self._recommender is not None:
            replacement_candidates = self._build_replacement_candidates(
                user_history=user_history,
                n_sample=n_sample,
                seed=user_id,
            )

        return removal_candidates, replacement_candidates

    def _compute_faithfulness_results(
        self,
        contexts: Sequence[_EvaluationContext],
        faithfulness_scores_by_user: Mapping[int, _FaithfulnessScores],
        config: FaithfulnessConfig,
    ) -> dict[int, dict[str, EvaluationResult | None]]:
        """Run counterfactual trials using pre-computed LLM scores."""
        return {
            context.user_id: self._compute_user_faithfulness(
                context,
                faithfulness_scores_by_user.get(context.user_id, _FaithfulnessScores()),
                config,
            )
            for context in contexts
        }

    def _compute_user_faithfulness(
        self,
        context: _EvaluationContext,
        scores: _FaithfulnessScores,
        config: FaithfulnessConfig,
    ) -> dict[str, EvaluationResult | None]:
        """Compute faithfulness results for a single user."""
        if context.user_history is None or context.target_item is None or self._recommender is None:
            return self._build_empty_faithfulness_results()

        cfx_item_ids = self._extract_cfx_item_ids(context.cfx_interactions_full)

        results: dict[str, EvaluationResult | None] = {}

        removal_results = self._compute_removal_faithfulness(context, scores, config, cfx_item_ids)
        replacement_results = self._compute_replacement_faithfulness(context, scores, config, cfx_item_ids)

        results.update(removal_results)
        results.update(replacement_results)
        return results

    @staticmethod
    def _build_empty_faithfulness_results() -> dict[str, EvaluationResult | None]:
        """Build empty results when faithfulness cannot be computed."""
        return {
            "faithfulness_removal": None,
            "faithfulness_removal_baseline": None,
            "faithfulness_replacement": None,
            "faithfulness_replacement_baseline": None,
        }

    @staticmethod
    def _extract_cfx_item_ids(cfx_interactions: pd.DataFrame) -> list[int]:
        """Extract CFX item IDs from interactions DataFrame."""
        if "movie_id" not in cfx_interactions.columns:
            return []
        return [int(movie_id) for movie_id in cfx_interactions["movie_id"]]

    def _compute_removal_faithfulness(
        self,
        context: _EvaluationContext,
        scores: _FaithfulnessScores,
        config: FaithfulnessConfig,
        cfx_item_ids: list[int],
    ) -> dict[str, EvaluationResult | None]:
        """Compute faithfulness removal results."""
        if not self._can_compute_removal(scores):
            return {"faithfulness_removal": None, "faithfulness_removal_baseline": None}

        assert context.user_history is not None  # noqa: S101
        assert context.target_item is not None  # noqa: S101
        assert self._recommender is not None  # noqa: S101
        assert self._faithfulness_removal_evaluator is not None  # noqa: S101

        regular, baseline = self._faithfulness_removal_evaluator.compute_results_from_scores(
            scored_items=scores.removal_scores,
            user_history=context.user_history,
            target_item=context.target_item,
            recommender=self._recommender,
            config=config,
            cfx_item_ids=cfx_item_ids,
        )
        return {"faithfulness_removal": regular, "faithfulness_removal_baseline": baseline}

    def _can_compute_removal(self, scores: _FaithfulnessScores) -> bool:
        """Check if removal faithfulness can be computed."""
        return (
            "faithfulness_removal" in self._enabled_evaluations
            and self._faithfulness_removal_evaluator is not None
            and bool(scores.removal_scores)
        )

    def _compute_replacement_faithfulness(
        self,
        context: _EvaluationContext,
        scores: _FaithfulnessScores,
        config: FaithfulnessConfig,
        cfx_item_ids: list[int],
    ) -> dict[str, EvaluationResult | None]:
        """Compute faithfulness replacement results."""
        if not self._can_compute_replacement(scores):
            return {"faithfulness_replacement": None, "faithfulness_replacement_baseline": None}

        assert context.user_history is not None  # noqa: S101
        assert context.target_item is not None  # noqa: S101
        assert self._recommender is not None  # noqa: S101
        assert self._faithfulness_replacement_evaluator is not None  # noqa: S101

        regular, baseline = self._faithfulness_replacement_evaluator.compute_results_from_scores(
            scored_items=scores.replacement_scores,
            user_history=context.user_history,
            target_item=context.target_item,
            recommender=self._recommender,
            config=config,
            cfx_item_ids=cfx_item_ids,
        )
        return {"faithfulness_replacement": regular, "faithfulness_replacement_baseline": baseline}

    def _can_compute_replacement(self, scores: _FaithfulnessScores) -> bool:
        """Check if replacement faithfulness can be computed."""
        return (
            "faithfulness_replacement" in self._enabled_evaluations
            and self._faithfulness_replacement_evaluator is not None
            and bool(scores.replacement_scores)
        )

    def _build_removal_candidates(
        self,
        user_history: torch.Tensor,
        cfx_item_ids: Sequence[int],
        n_sample: int,
        seed: int,
    ) -> pd.DataFrame:
        """Build candidates from interacted items excluding CFX."""
        if self._recommender is None:
            return pd.DataFrame(columns=["movie_id"])

        num_items = self._recommender.num_items
        user_history_np = user_history.cpu().numpy()
        cfx_id_set = {int(item_id) for item_id in cfx_item_ids}

        interacted = [i for i in range(num_items) if user_history_np[i] > 0 and i not in cfx_id_set]

        rng = random.Random(seed)  # noqa: S311
        sampled = rng.sample(interacted, min(n_sample, len(interacted)))

        return pd.DataFrame({"movie_id": sampled})

    def _build_replacement_candidates(
        self,
        user_history: torch.Tensor,
        n_sample: int,
        seed: int,
    ) -> pd.DataFrame:
        """Build candidates from items user hasn't interacted with."""
        if self._recommender is None:
            return pd.DataFrame(columns=["movie_id"])

        num_items = self._recommender.num_items
        user_history_np = user_history.cpu().numpy()

        non_interacted = [i for i in range(num_items) if user_history_np[i] == 0]

        rng = random.Random(seed)  # noqa: S311
        sampled = rng.sample(non_interacted, min(n_sample, len(non_interacted)))

        return pd.DataFrame({"movie_id": sampled})

    def _collect_all_evaluation_prompts(self, contexts: Sequence[_EvaluationContext]) -> list[_PromptInfo]:
        """Collect all evaluation prompts for all users in the batch."""
        all_prompts: list[_PromptInfo] = []
        for context in contexts:
            all_prompts.extend(self._collect_prompts_for_user(context))
        return all_prompts

    def _collect_prompts_for_user(self, context: _EvaluationContext) -> list[_PromptInfo]:
        """Collect evaluation prompts using the sampled interaction frames."""
        prompts: list[_PromptInfo] = []
        user_id = context.user_id

        if "plausibility" in self._enabled_evaluations:
            prompts.append(self._build_plausibility_prompt(user_id, context.explanation))

        if "readability" in self._enabled_evaluations and self._readability_evaluator is not None:
            prompts.append(self._build_readability_prompt(user_id, context.explanation))

        if "cfx_match" in self._enabled_evaluations:
            prompts.extend(self._build_interaction_match_prompts(user_id, "cfx_match", context))

        if "non_cfx_match" in self._enabled_evaluations and self._non_cfx_match_evaluator is not None:
            prompts.extend(self._build_interaction_match_prompts(user_id, "non_cfx_match", context))

        if "faithfulness_removal" in self._enabled_evaluations and self._faithfulness_removal_evaluator is not None:
            prompts.extend(self._build_faithfulness_prompts(user_id, "faithfulness_removal", context))

        if (
            "faithfulness_replacement" in self._enabled_evaluations
            and self._faithfulness_replacement_evaluator is not None
        ):
            prompts.extend(self._build_faithfulness_prompts(user_id, "faithfulness_replacement", context))

        return prompts

    def _build_plausibility_prompt(self, user_id: int, explanation: str) -> _PromptInfo:
        """Build a plausibility evaluation prompt."""
        messages = self._plausibility_evaluator.build_prompt(explanation=explanation)
        return _PromptInfo(
            user_id=user_id,
            evaluator_name="plausibility",
            interaction_key=None,
            messages=messages,
            prompt_text=serialise_chat_messages(messages),
        )

    def _build_readability_prompt(self, user_id: int, explanation: str) -> _PromptInfo:
        """Build a readability evaluation prompt."""
        assert self._readability_evaluator is not None  # noqa: S101
        messages = self._readability_evaluator.build_prompt(explanation=explanation)
        return _PromptInfo(
            user_id=user_id,
            evaluator_name="readability",
            interaction_key=None,
            messages=messages,
            prompt_text=serialise_chat_messages(messages),
        )

    def _build_interaction_match_prompts(
        self, user_id: int, evaluator_name: str, context: _EvaluationContext
    ) -> list[_PromptInfo]:
        """Build prompts for CFX or non-CFX interaction matching."""
        if evaluator_name == "cfx_match":
            return self._build_prompts_from_evaluator(
                user_id, evaluator_name, self._cfx_match_evaluator, context.cfx_interactions, context.explanation
            )
        return self._build_prompts_from_evaluator(
            user_id, evaluator_name, self._non_cfx_match_evaluator, context.non_cfx_interactions, context.explanation
        )

    def _build_prompts_from_evaluator(
        self,
        user_id: int,
        evaluator_name: str,
        evaluator: InteractionScoringEvaluator | None,
        interactions: pd.DataFrame,
        explanation: str,
    ) -> list[_PromptInfo]:
        """Build prompts from a given interaction scoring evaluator."""
        if evaluator is None or interactions.empty:
            return []

        prompts: list[_PromptInfo] = []
        for movie_id, interaction_desc, messages in evaluator.build_all_prompts_with_ids(
            explanation=explanation, interactions=interactions
        ):
            prompts.append(
                _PromptInfo(
                    user_id=user_id,
                    evaluator_name=evaluator_name,
                    interaction_key=interaction_desc,
                    messages=messages,
                    prompt_text=serialise_chat_messages(messages),
                    item_id=movie_id,
                )
            )
        return prompts

    def _build_faithfulness_prompts(
        self, user_id: int, evaluator_name: str, context: _EvaluationContext
    ) -> list[_PromptInfo]:
        """Build prompts for faithfulness scoring."""
        if evaluator_name == "faithfulness_removal":
            return self._build_faithfulness_prompts_from_evaluator(
                user_id,
                evaluator_name,
                self._faithfulness_removal_evaluator,
                context.faithfulness_removal_candidates,
                context.explanation,
            )
        return self._build_faithfulness_prompts_from_evaluator(
            user_id,
            evaluator_name,
            self._faithfulness_replacement_evaluator,
            context.faithfulness_replacement_candidates,
            context.explanation,
        )

    def _build_faithfulness_prompts_from_evaluator(
        self,
        user_id: int,
        evaluator_name: str,
        evaluator: FaithfulnessRemovalEvaluator | FaithfulnessReplacementEvaluator | None,
        candidates: pd.DataFrame | None,
        explanation: str,
    ) -> list[_PromptInfo]:
        """Build prompts from a faithfulness evaluator."""
        if evaluator is None or candidates is None or candidates.empty:
            return []

        prompts: list[_PromptInfo] = []
        scoring_name = f"{evaluator_name}_scoring"
        for movie_id, interaction_desc, messages in evaluator.build_all_prompts_with_ids(
            explanation=explanation, interactions=candidates
        ):
            prompts.append(
                _PromptInfo(
                    user_id=user_id,
                    evaluator_name=scoring_name,
                    interaction_key=interaction_desc,
                    messages=messages,
                    prompt_text=serialise_chat_messages(messages),
                    item_id=movie_id,
                )
            )
        return prompts

    # Type alias for grouped raw outputs
    _RawResultsMap = dict[str, list[tuple[str | None, str, int | None]]]

    def _distribute_evaluation_results(
        self,
        contexts: Sequence[_EvaluationContext],
        all_prompts: Sequence[_PromptInfo],
        raw_outputs: list[str],
    ) -> tuple[dict[int, dict[str, EvaluationResult | None]], dict[int, _FaithfulnessScores]]:
        """Parse raw outputs and aggregate results per user."""
        grouped = self._group_raw_outputs_by_user(contexts, all_prompts, raw_outputs)

        evaluations_by_user: dict[int, dict[str, EvaluationResult | None]] = {}
        faithfulness_scores_by_user: dict[int, _FaithfulnessScores] = {}

        for context in contexts:
            user_id = context.user_id
            results = grouped[user_id]

            evaluations_by_user[user_id] = {
                "explanation_plausibility": self._parse_plausibility_result(results, all_prompts),
                "explanation_readability": self._parse_readability_result(results, all_prompts, user_id),
                "explanation_cfx_match": self._parse_cfx_match_result(results),
                "explanation_non_cfx_match": self._parse_non_cfx_match_result(results),
            }
            faithfulness_scores_by_user[user_id] = self._parse_user_faithfulness_scores(results)

        return evaluations_by_user, faithfulness_scores_by_user

    def _group_raw_outputs_by_user(
        self,
        contexts: Sequence[_EvaluationContext],
        all_prompts: Sequence[_PromptInfo],
        raw_outputs: list[str],
    ) -> dict[int, _RawResultsMap]:
        """Group raw LLM outputs by user and evaluator name."""
        grouped: dict[int, ExplanationWorkflow._RawResultsMap] = {}
        for context in contexts:
            grouped[context.user_id] = {
                "plausibility": [],
                "readability": [],
                "cfx_match": [],
                "non_cfx_match": [],
                "faithfulness_removal_scoring": [],
                "faithfulness_replacement_scoring": [],
            }

        for prompt_info, raw_output in zip(all_prompts, raw_outputs, strict=False):
            grouped[prompt_info.user_id][prompt_info.evaluator_name].append(
                (prompt_info.interaction_key, raw_output, prompt_info.item_id)
            )

        return grouped

    def _parse_plausibility_result(
        self, results: _RawResultsMap, all_prompts: Sequence[_PromptInfo]
    ) -> EvaluationResult:
        """Parse plausibility evaluation result."""
        if "plausibility" not in self._enabled_evaluations:
            return EvaluationResult(judgment="Plausibility evaluation disabled by configuration.", score=float("nan"))

        if not results["plausibility"]:
            return EvaluationResult(judgment="Plausibility evaluation disabled by configuration.", score=float("nan"))

        _, raw_output, _ = results["plausibility"][0]
        prompt_text = all_prompts[0].prompt_text if all_prompts else None
        return self._plausibility_evaluator.parse_result(raw_output, prompt=prompt_text)

    def _parse_readability_result(
        self, results: _RawResultsMap, all_prompts: Sequence[_PromptInfo], user_id: int
    ) -> EvaluationResult:
        """Parse readability evaluation result."""
        if "readability" not in self._enabled_evaluations or self._readability_evaluator is None:
            return EvaluationResult(judgment="Readability evaluation disabled by configuration.", score=float("nan"))

        if not results["readability"]:
            return EvaluationResult(judgment="Readability evaluation disabled by configuration.", score=float("nan"))

        _, raw_output, _ = results["readability"][0]
        prompt_text = next(
            (
                prompt.prompt_text
                for prompt in all_prompts
                if prompt.evaluator_name == "readability" and prompt.user_id == user_id
            ),
            None,
        )
        return self._readability_evaluator.parse_result(raw_output, prompt=prompt_text)

    def _parse_cfx_match_result(self, results: _RawResultsMap) -> EvaluationResult:
        """Parse CFX match evaluation result."""
        if "cfx_match" not in self._enabled_evaluations:
            return EvaluationResult(judgment="CFX match evaluation disabled by configuration.", score=float("nan"))

        if results["cfx_match"]:
            scores = self._parse_interaction_results(self._cfx_match_evaluator, results["cfx_match"])
            return self._cfx_match_evaluator.aggregate_results(scores)

        return self._cfx_match_evaluator.build_empty_result(
            reason="No CFX interactions available for CFX match evaluation."
        )

    def _parse_non_cfx_match_result(self, results: _RawResultsMap) -> EvaluationResult | None:
        """Parse non-CFX match evaluation result."""
        if "non_cfx_match" not in self._enabled_evaluations or self._non_cfx_match_evaluator is None:
            return None

        if results["non_cfx_match"]:
            scores = self._parse_interaction_results(self._non_cfx_match_evaluator, results["non_cfx_match"])
            return self._non_cfx_match_evaluator.aggregate_results(scores)

        return self._non_cfx_match_evaluator.build_empty_result(
            reason="No non-CFX interactions available for Non-CFX match evaluation."
        )

    def _parse_user_faithfulness_scores(self, results: _RawResultsMap) -> _FaithfulnessScores:
        """Parse faithfulness scores for a user."""
        scores = _FaithfulnessScores()

        if results["faithfulness_removal_scoring"] and self._faithfulness_removal_evaluator is not None:
            scores.removal_scores = self._parse_faithfulness_item_scores(
                self._faithfulness_removal_evaluator, results["faithfulness_removal_scoring"]
            )

        if results["faithfulness_replacement_scoring"] and self._faithfulness_replacement_evaluator is not None:
            scores.replacement_scores = self._parse_faithfulness_item_scores(
                self._faithfulness_replacement_evaluator, results["faithfulness_replacement_scoring"]
            )

        return scores

    @staticmethod
    def _parse_interaction_results(
        evaluator: InteractionScoringEvaluator,
        results: list[tuple[str | None, str, int | None]],
    ) -> list[Mapping[str, object]]:
        """Parse raw interaction scoring outputs into structured results."""
        parsed_scores: list[Mapping[str, object]] = []
        for interaction_key, raw_output, item_id in results:
            parsed = dict(evaluator.parse_single_interaction_result(raw_output, interaction_key or ""))
            if item_id is not None:
                parsed["item_id"] = item_id
            parsed_scores.append(parsed)
        return parsed_scores

    @staticmethod
    def _parse_faithfulness_item_scores(
        evaluator: InteractionScoringEvaluator,
        results: list[tuple[str | None, str, int | None]],
    ) -> list[ScoredItem]:
        """Parse raw faithfulness scoring outputs into ScoredItem list."""
        scored_items: list[ScoredItem] = []
        for interaction_key, raw_output, item_id in results:
            if item_id is None:
                continue
            interaction_desc = interaction_key or ""
            parsed = evaluator.parse_single_interaction_result(raw_output, interaction_desc)
            raw_score = parsed.get("score", float("nan"))
            score = float(raw_score) if isinstance(raw_score, (int, float)) else float("nan")
            scored_items.append(ScoredItem(item_id=item_id, score=score, interaction_description=interaction_desc))
        return scored_items

    @staticmethod
    def _sample_judged_interactions(
        *,
        user_id: int,
        cfx_interactions: pd.DataFrame,
        non_cfx_interactions: pd.DataFrame,
        n_judged_interactions: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Deterministically sample CFX and non-CFX interactions for evaluation."""
        seed = user_id
        if len(cfx_interactions) > n_judged_interactions:
            cfx_interactions = cfx_interactions.sample(n=n_judged_interactions, random_state=seed)
        if len(non_cfx_interactions) > n_judged_interactions:
            non_cfx_interactions = non_cfx_interactions.sample(n=n_judged_interactions, random_state=seed)
        return cfx_interactions, non_cfx_interactions

    def _generate_explanations(
        self,
        user_inputs: Sequence[UserExplanationInput],
    ) -> dict[int, GeneratedExplanation]:
        """Generate explanations using batch support when available."""
        return self._generator.generate_batch(user_inputs, batch_size=self._generation_batch_size)

    @staticmethod
    def _serialise_evaluation(result: EvaluationResult | None) -> dict[str, object] | None:
        """Convert an evaluation result into a serialisable mapping."""
        if result is None:
            return None
        return {"judgment": result.judgment, "score": result.score}

    def _assemble_results(
        self,
        *,
        dataset: Dataset,
        generated_store: Mapping[int, GeneratedExplanation],
        evaluation_store: Mapping[int, Mapping[str, EvaluationResult | None]],
    ) -> dict[int, NaturalLanguageExplanationResult]:
        """Combine generated and evaluated data into result objects keyed by user."""
        results: dict[int, NaturalLanguageExplanationResult] = {}
        for record in dataset:
            user_id = int(record["user_id"])
            generated = generated_store.get(user_id)
            evaluations = evaluation_store.get(user_id)
            if generated is None or evaluations is None:
                continue
            explanation_plausibility = evaluations.get("explanation_plausibility")
            explanation_cfx_match = evaluations.get("explanation_cfx_match")
            if explanation_plausibility is None or explanation_cfx_match is None:
                continue
            results[user_id] = NaturalLanguageExplanationResult.from_components(
                generated,
                explanation_plausibility=explanation_plausibility,
                explanation_cfx_match=explanation_cfx_match,
                explanation_readability=evaluations.get("explanation_readability"),
                explanation_non_cfx_match=evaluations.get("explanation_non_cfx_match"),
                faithfulness_removal=evaluations.get("faithfulness_removal"),
                faithfulness_removal_baseline=evaluations.get("faithfulness_removal_baseline"),
                faithfulness_replacement=evaluations.get("faithfulness_replacement"),
                faithfulness_replacement_baseline=evaluations.get("faithfulness_replacement_baseline"),
            )
        return results

    @staticmethod
    def _frame_from_payload(payload: object) -> pd.DataFrame:
        """Convert a dataset payload into a pandas DataFrame."""
        if not isinstance(payload, Sequence):
            msg = f"Expected a sequence, got {type(payload)}"
            raise TypeError(msg)
        return pd.DataFrame(list(payload))


@dataclass(slots=True)
class _EvaluationContext:
    """Stores per-user data required for batch evaluation."""

    user_id: int
    cfx_interactions: pd.DataFrame
    cfx_interactions_full: pd.DataFrame
    non_cfx_interactions: pd.DataFrame
    explanation: str
    # Faithfulness-specific context (optional, set when faithfulness is enabled)
    user_history: torch.Tensor | None = None
    target_item: int | None = None
    faithfulness_removal_candidates: pd.DataFrame | None = None
    faithfulness_replacement_candidates: pd.DataFrame | None = None


@dataclass(slots=True)
class _PromptInfo:
    """Tracks metadata for a single evaluation prompt in the batch."""

    user_id: int
    evaluator_name: str
    interaction_key: str | None
    messages: list[dict[str, str]]
    prompt_text: str
    item_id: int | None = None


@dataclass(slots=True)
class _FaithfulnessScores:
    """Stores parsed faithfulness scores for a user."""

    removal_scores: list[ScoredItem] = field(default_factory=list)
    replacement_scores: list[ScoredItem] = field(default_factory=list)
