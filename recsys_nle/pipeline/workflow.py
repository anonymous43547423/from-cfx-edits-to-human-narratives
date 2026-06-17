"""End-to-end orchestration for recommendations, attributions, and explanations."""

from __future__ import annotations

import copy
import importlib
import pickle
import random
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
import torch

from recsys_nle.cf_explanations.LXR import help_functions as _hf_module
from recsys_nle.cf_explanations.LXR import recommenders_architecture as _ra_module
from recsys_nle.cf_explanations.LXR.explanation_analysis import (
    get_counterfactual_explanation,
    limit_cf_item_interactions,
)
from recsys_nle.cf_explanations.LXR.lime import LimeBase, distance_to_proximity
from recsys_nle.cf_explanations.LXR.recommenders_architecture import VAE
from recsys_nle.core.attribution import AttributionMethod, UserAttribution, summarise_interactions
from recsys_nle.core.movielens import MovielensArtifacts
from recsys_nle.core.recommender_wrapper import RecommenderWrapper
from recsys_nle.nl_explanations.dataset_builder import build_explanation_dataset
from recsys_nle.nl_explanations.evaluation import (
    CFXMatchEvaluator,
    FaithfulnessRemovalEvaluator,
    FaithfulnessReplacementEvaluator,
    NonCFXMatchEvaluator,
    PlausibilityEvaluator,
    ReadabilityEvaluator,
)
from recsys_nle.nl_explanations.evaluation.faithfulness import FaithfulnessConfig
from recsys_nle.nl_explanations.generator import ExplanationGenerator
from recsys_nle.nl_explanations.llm import (
    EINFRA_MODEL_PREFIX,
    HuggingFaceLLMClient,
    OpenAIChatLLMClient,
    einfra_api_model_id,
)
from recsys_nle.nl_explanations.payloads import _movie_metadata_index
from recsys_nle.nl_explanations.workflow import ExplanationResult, ExplanationWorkflow
from recsys_nle.pipeline.distance_metrics import (
    build_distance_context,
    compute_all_distance_metrics_for_user,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from recsys_nle.cf_explanations.LXR.LXR_training import Explainer as ExplainerType
    from recsys_nle.nl_explanations.llm import LLMClient
    from recsys_nle.pipeline.config import PipelineConfig, TargetSet, UserPool


def _raise_if_einfra_generation(model_id: str) -> None:
    """Reject EINFRA-prefixed models for local HF generation (evaluation only)."""
    if model_id.startswith(EINFRA_MODEL_PREFIX):
        msg = (
            "EINFRA OpenAI-compatible models are supported only for LLM judge evaluation. "
            "Use a Hugging Face model id or local path for --model-id-generation."
        )
        raise ValueError(msg)


def _build_evaluation_llm(model_id: str) -> LLMClient:
    """Construct an evaluation LLM client (Hugging Face or EINFRA OpenAI-compatible)."""
    if model_id.startswith(EINFRA_MODEL_PREFIX):
        return OpenAIChatLLMClient(model=einfra_api_model_id(model_id))
    return HuggingFaceLLMClient(model_id=model_id)


@dataclass(slots=True)
class CfxSearchOutcome:
    """Outcome counts from the CFX search across sampled users."""

    n_valid: int = 0
    n_no_cfx: int = 0
    n_below_min_interactions: int = 0


@dataclass(slots=True)
class PipelineResult:
    """Aggregate output emitted by the RecSys pipeline orchestration."""

    recommendations: DataFrame
    user_attributions: Mapping[int, UserAttribution]
    cfx_interactions: DataFrame
    explanations: ExplanationResult
    all_interactions: DataFrame = field(default_factory=pd.DataFrame)
    sampled_user_ids: list[int] = field(default_factory=list)
    distance_metrics_by_user: Mapping[int, dict[str, float]] = field(default_factory=dict)
    cfx_search_outcome: CfxSearchOutcome = field(default_factory=CfxSearchOutcome)


@dataclass(slots=True)
class AttributionAssets:
    """Runtime assets required to compute attribution methods."""

    all_items_tensor: torch.Tensor
    kw_dict: dict[str, object]
    jaccard_dict: dict[tuple[int, int], float] | None
    cosine_dict: dict[tuple[int, int], float] | None
    item_to_cluster: Any | None
    shap_values: Any | None
    lime: LimeBase | None
    train_array: np.ndarray[Any, Any] | None = None
    pop_array: np.ndarray[Any, Any] | None = None


@dataclass(slots=True)
class AttributionResult:
    """Result from the attribution and CFX computation."""

    recommendations: DataFrame
    user_attributions: Mapping[int, UserAttribution]
    cfx_interactions: DataFrame
    all_interactions: DataFrame
    user_histories: dict[int, torch.Tensor]
    user_targets: dict[int, int]
    cfx_search_outcome: CfxSearchOutcome


class PipelineWorkflow:
    """Coordinate LXR-based recommendation, attribution, and explanation workflows."""

    def run(self, config: PipelineConfig) -> PipelineResult:
        """Execute the full pipeline and return aggregated outputs."""
        device, num_items, test_data, items_array, recommender, explainer = _load_lxr_components()
        user_data = _resolve_user_pool(
            test_data,
            user_pool=config.user_pool,
            target_set=config.target_set,
        )

        result = run_attribution_and_cfx_algorithms(
            device=device,
            num_items=num_items,
            user_data=user_data,
            items_array=items_array,
            recommender=recommender,
            explainer=explainer,
            top_k=config.recommendation.top_k,
            n_non_cfx_interactions=config.attribution.n_non_cfx_interactions,
            max_cfx_removals=config.attribution.max_cfx_removals,
            target_cfx_rank=config.attribution.target_cfx_rank,
            min_cfx_interactions=config.attribution.min_cfx_interactions,
            max_users=config.max_users_for_attribution,
            random_seed=config.random_seed,
            method=config.attribution.method,
        )
        recommendations = result.recommendations
        user_attributions = result.user_attributions
        cfx_interactions = result.cfx_interactions
        all_interactions = result.all_interactions
        user_histories = result.user_histories
        user_targets = result.user_targets

        recommender_wrapper = RecommenderWrapper(
            recommender=recommender,
            num_items=num_items,
            device=device,
        )

        explanations, sampled_user_ids = self._run_explanations_stage(
            config=config,
            user_attributions=user_attributions,
            recommender_wrapper=recommender_wrapper,
            user_histories=user_histories,
            user_targets=user_targets,
        )

        distance_metrics_by_user: dict[int, dict[str, float]] = {}
        if sampled_user_ids:
            distance_context = build_distance_context(
                all_interactions,
                _movie_metadata_index(),
                sampled_user_ids,
            )
            n_pairs = config.outputs.n_sampled_distance_pairs
            for user_id in sampled_user_ids:
                distance_metrics_by_user[user_id] = compute_all_distance_metrics_for_user(
                    user_id,
                    all_interactions,
                    distance_context,
                    n_pairs,
                    random_seed=user_id,
                )

        return PipelineResult(
            recommendations=recommendations,
            user_attributions=user_attributions,
            cfx_interactions=cfx_interactions,
            explanations=explanations,
            all_interactions=all_interactions,
            sampled_user_ids=sampled_user_ids,
            distance_metrics_by_user=distance_metrics_by_user,
            cfx_search_outcome=result.cfx_search_outcome,
        )

    def _build_explanation_workflow(
        self,
        config: PipelineConfig,
        generator: ExplanationGenerator,
        recommender_wrapper: RecommenderWrapper | None = None,
    ) -> ExplanationWorkflow:
        """Initialise the explanation workflow with the provided generator."""
        plausibility_evaluator = PlausibilityEvaluator()
        readability_evaluator: ReadabilityEvaluator | None = None
        cfx_match_evaluator = CFXMatchEvaluator()
        non_cfx_match_evaluator = NonCFXMatchEvaluator()

        faithfulness_removal_evaluator: FaithfulnessRemovalEvaluator | None = None
        faithfulness_replacement_evaluator: FaithfulnessReplacementEvaluator | None = None

        enabled_evals = set(config.explanation.enabled_evaluations)
        if "readability" in enabled_evals:
            readability_evaluator = ReadabilityEvaluator()
        if "faithfulness_removal" in enabled_evals:
            faithfulness_removal_evaluator = FaithfulnessRemovalEvaluator()
        if "faithfulness_replacement" in enabled_evals:
            faithfulness_replacement_evaluator = FaithfulnessReplacementEvaluator()

        return ExplanationWorkflow(
            generator=generator,
            evaluation_llm_client=None,
            plausibility_evaluator=plausibility_evaluator,
            readability_evaluator=readability_evaluator,
            cfx_match_evaluator=cfx_match_evaluator,
            non_cfx_match_evaluator=non_cfx_match_evaluator,
            faithfulness_removal_evaluator=faithfulness_removal_evaluator,
            faithfulness_replacement_evaluator=faithfulness_replacement_evaluator,
            recommender=recommender_wrapper,
            generation_batch_size=config.explanation.generation_batch_size,
            evaluation_user_batch_size=config.explanation.evaluation_user_batch_size,
            evaluation_llm_batch_size=config.explanation.evaluation_llm_batch_size,
            enabled_evaluations=config.explanation.enabled_evaluations,
        )

    def _run_explanations_stage(
        self,
        *,
        config: PipelineConfig,
        user_attributions: Mapping[int, UserAttribution],
        recommender_wrapper: RecommenderWrapper | None = None,
        user_histories: Mapping[int, torch.Tensor] | None = None,
        user_targets: Mapping[int, int] | None = None,
    ) -> tuple[ExplanationResult, list[int]]:
        """Generate natural-language explanations for a sampled subset of users."""
        if not user_attributions:
            msg = "No user attributions provided"
            raise ValueError(msg)

        all_user_ids: list[int] = list(user_attributions.keys())
        max_users = config.sample_user_count
        selected_user_ids = all_user_ids[:max_users] if max_users > 0 else all_user_ids

        # Phase 1: Generation
        _raise_if_einfra_generation(config.explanation.model_id_generation)
        generator_llm = HuggingFaceLLMClient(model_id=config.explanation.model_id_generation)
        generator = ExplanationGenerator(
            llm_client=generator_llm,
            n_cfx_interactions=config.explanation.n_cfx_interactions,
            use_reasoning=not config.explanation.disable_reasoning,
        )
        workflow = self._build_explanation_workflow(config, generator, recommender_wrapper)
        dataset = build_explanation_dataset(attributions=user_attributions, user_ids=selected_user_ids)
        generated_dataset, generated_store = workflow.generate_dataset(dataset)
        generator.close()

        # Phase 2: Evaluation
        evaluation_llm = _build_evaluation_llm(config.explanation.model_id_evaluation)
        workflow.set_evaluation_llm_client(evaluation_llm)
        faithfulness_config = FaithfulnessConfig(
            n_sampled_faithfulness_interactions=config.explanation.n_sampled_faithfulness_interactions,
            match_threshold=config.explanation.faithfulness_match_threshold,
            n_interactions_min_limit=config.explanation.n_faithfulness_interactions_min_limit,
            n_faithfulness_trials=config.explanation.n_faithfulness_trials,
            n_faithfulness_samples=config.explanation.n_faithfulness_samples,
        )
        result = workflow.evaluate_dataset(
            generated_dataset,
            generated_store=generated_store,
            n_judged_interactions=config.explanation.n_judged_interactions,
            faithfulness_config=faithfulness_config,
            user_histories=user_histories,
            user_targets=user_targets,
        )
        evaluation_llm.close()

        return result, selected_user_ids


def _resolve_user_pool(
    test_data: pd.DataFrame,
    *,
    user_pool: UserPool,
    target_set: TargetSet,
) -> pd.DataFrame:
    """Return the row-order slice of ``test_data`` for the given pool and target set."""
    n_users = len(test_data)
    third = n_users // 3
    if third == 0 and n_users > 1:
        third = 1

    if user_pool == "eval":
        selected = test_data.iloc[third : 2 * third] if target_set == "validation" else test_data.iloc[:third]
    elif target_set == "validation":
        selected = test_data.iloc[2 * third :]
    else:
        selected = test_data.iloc[third:]

    result = selected.copy()
    result["user_id"] = result.index.astype(int)
    return result


def _get_data_paths() -> tuple[Path, Path]:
    """Return dataset and checkpoint directories used by LXR."""
    base_path = Path(__file__).resolve().parents[2]
    data_path = base_path / "datasets" / "ML1M"
    checkpoint_path = base_path / "checkpoints" / "recommenders" / "VAE"
    return data_path, checkpoint_path


def _load_lxr_components() -> tuple[torch.device, int, pd.DataFrame, np.ndarray[Any, Any], VAE, ExplainerType]:
    """Load LXR test data, item identities, and instantiate VAE and explainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path, checkpoint_path = _get_data_paths()

    train_data = pd.read_csv(data_path / "train_data_ML1M.csv", index_col=0)
    test_data = pd.read_csv(data_path / "test_data_ML1M.csv", index_col=0)
    num_items = train_data.shape[1]

    items_array = np.eye(num_items, dtype=np.float32)
    all_items_tensor = torch.tensor(items_array, dtype=torch.float32, device=device)

    with (data_path / "pop_dict_ML1M.pkl").open("rb") as handle:
        pop_dict = pickle.load(handle)  # noqa: S301
    pop_array = np.array([pop_dict.get(i, 0.0) for i in range(len(pop_dict))], dtype=np.float32)

    vae_config: dict[str, float | list[int]] = {
        "enc_dims": [512, 128],
        "dropout": 0.5,
        "anneal_cap": 0.2,
        "total_anneal_steps": 200_000,
    }
    vae_kwargs = {
        "device": device,
        "num_items": num_items,
        "pop_array": pop_array,
        "all_items_tensor": all_items_tensor,
        "items_array": items_array,
    }
    recommender = VAE(vae_config, **vae_kwargs)  # type: ignore[no-untyped-call]
    vae_state = torch.load(checkpoint_path / "VAE_ML1M_0.0007_128_10.pt", map_location=device)
    recommender.load_state_dict(vae_state)
    recommender.to(device)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad = False

    for name, source in (
        ("recommenders_architecture", _ra_module),
        ("help_functions", _hf_module),
    ):
        full_name = f"ipynb.fs.defs.{name}"
        shim = types.ModuleType(full_name)
        shim.__dict__.update(source.__dict__)
        sys.modules[full_name] = shim

    argv_backup = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        lxr_training = importlib.import_module("recsys_nle.cf_explanations.LXR.LXR_training")
        ExplainerClass = lxr_training.Explainer
    finally:
        sys.argv = argv_backup

    explainer = ExplainerClass(num_items, num_items, 128)
    expl_state = torch.load(
        checkpoint_path / "LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426LXRMAIN.pt",
        map_location=device,
    )
    explainer.load_state_dict(expl_state)
    explainer.to(device)
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad = False

    test_data = test_data.copy()
    test_data["user_id"] = test_data.index.astype(int)
    return device, num_items, test_data, items_array, recommender, explainer


def load_movies_metadata() -> MovielensArtifacts:
    """Load movie metadata (movie_id, title, genres)."""
    data_path, _ = _get_data_paths()
    movies_path = data_path / "data files" / "movies.dat"
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "MovieName", "Genre"],
        encoding="latin-1",
    )
    movies["movie_id"] = movies["MovieID"].astype(int) - 1
    movies["title"] = movies["MovieName"].astype(str)
    movies["genres"] = movies["Genre"].fillna("").apply(lambda value: value.split("|") if value else [])
    items = movies[["movie_id", "title", "genres"]].copy()

    metadata = items.copy()
    metadata["metadata_title"] = metadata["title"]
    metadata["metadata_genres"] = metadata["genres"]

    ratings = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    return MovielensArtifacts(ratings=ratings, items=items, metadata=metadata)


def _load_similarity_dict(path: Path, num_items: int) -> dict[tuple[int, int], float]:
    """Load a similarity dictionary and ensure symmetry."""
    with path.open("rb") as handle:
        raw_dict = pickle.load(handle)  # noqa: S301
    similarity_dict: dict[tuple[int, int], float] = {}
    for pair, score in raw_dict.items():
        similarity_dict[(int(pair[0]), int(pair[1]))] = float(score)
    for i in range(num_items):
        for j in range(i, num_items):
            similarity_dict[(j, i)] = similarity_dict[(i, j)]
    return similarity_dict


def _load_attribution_assets(
    *,
    method: AttributionMethod,
    data_path: Path,
    num_items: int,
    device: torch.device,
    items_array: np.ndarray[Any, Any],
) -> AttributionAssets:
    """Load attribution-specific data and helpers for the selected method."""
    all_items_tensor = torch.tensor(items_array, dtype=torch.float32, device=device)
    kw_dict = {
        "device": device,
        "num_items": num_items,
        "all_items_tensor": all_items_tensor,
        "items_array": items_array,
        "output_type": "multiple",
        "recommender_name": "VAE",
    }

    jaccard_dict: dict[tuple[int, int], float] | None = None
    cosine_dict: dict[tuple[int, int], float] | None = None
    item_to_cluster: Any | None = None
    shap_values: Any | None = None
    lime: LimeBase | None = None

    if method in {AttributionMethod.JACCARD, AttributionMethod.COSINE}:
        if method == AttributionMethod.JACCARD:
            jaccard_dict = _load_similarity_dict(data_path / "jaccard_based_sim_ML1M.pkl", num_items)
        else:
            cosine_dict = _load_similarity_dict(data_path / "cosine_based_sim_ML1M.pkl", num_items)

    if method == AttributionMethod.SHAP:
        with (data_path / "item_to_cluster_VAE_ML1M.pkl").open("rb") as handle:
            item_to_cluster = pickle.load(handle)  # noqa: S301
        with (data_path / "shap_values_VAE_ML1M.pkl").open("rb") as handle:
            shap_values = pickle.load(handle)  # noqa: S301

    if method == AttributionMethod.LIME:
        lime = LimeBase(distance_to_proximity)  # type: ignore[no-untyped-call]

    pop_array: np.ndarray[Any, Any] | None = None
    if method == AttributionMethod.SPINREC:
        with (data_path / "pop_dict_ML1M.pkl").open("rb") as handle:
            pop_dict = pickle.load(handle)  # noqa: S301
        pop_array = np.array([pop_dict.get(i, 0.0) for i in range(len(pop_dict))], dtype=np.float32)

    return AttributionAssets(
        all_items_tensor=all_items_tensor,
        kw_dict=kw_dict,
        jaccard_dict=jaccard_dict,
        cosine_dict=cosine_dict,
        item_to_cluster=item_to_cluster,
        shap_values=shap_values,
        lime=lime,
        pop_array=pop_array,
    )


def run_attribution_and_cfx_algorithms(  # noqa: C901, PLR0912, PLR0915
    *,
    device: torch.device,
    num_items: int,
    user_data: pd.DataFrame,
    items_array: np.ndarray[Any, Any],
    recommender: VAE,
    explainer: ExplainerType,
    top_k: int,
    n_non_cfx_interactions: int,
    max_cfx_removals: int,
    target_cfx_rank: int,
    min_cfx_interactions: int,
    max_users: int | None,
    random_seed: int,
    method: AttributionMethod,
) -> AttributionResult:
    """Run recommendations and counterfactual attributions."""
    data_path, _ = _get_data_paths()
    assets = _load_attribution_assets(
        method=method,
        data_path=data_path,
        num_items=num_items,
        device=device,
        items_array=items_array,
    )
    feature_columns = [column for column in user_data.columns if column != "user_id"]
    user_matrix = user_data[feature_columns].to_numpy(dtype=np.float32, copy=False)
    user_ids_int = user_data["user_id"].astype(int).tolist()

    all_user_vectors = list(user_matrix)
    all_user_ids: list[int] = list(user_ids_int)

    # Decide how many users to include in attribution computations.
    if max_users is None or max_users <= 0 or max_users >= len(all_user_ids):
        target_user_count = len(all_user_ids)
    else:
        target_user_count = max_users

    rng = random.Random(random_seed)  # noqa: S311
    shuffled_indices = list(range(len(all_user_ids)))
    rng.shuffle(shuffled_indices)

    if not all_user_vectors:
        empty_recs = pd.DataFrame(columns=["user_id", "movie_id", "score", "rank"])
        empty_interactions = pd.DataFrame(
            columns=["user_id", "movie_id", "rating", "weight", "importance"],
        )
        empty_all_interactions = pd.DataFrame(
            columns=["user_id", "item_id", "rating", "attribution_score", "is_counterfactual"]
        )
        return AttributionResult(
            recommendations=empty_recs,
            user_attributions={},
            cfx_interactions=empty_interactions,
            all_interactions=empty_all_interactions,
            user_histories={},
            user_targets={},
            cfx_search_outcome=CfxSearchOutcome(),
        )

    user_matrix = np.vstack(all_user_vectors).astype(np.float32, copy=False)

    user_histories: dict[int, torch.Tensor] = {}
    user_targets: dict[int, int] = {}

    records: list[dict[str, object]] = []
    for user_id, vector in zip(all_user_ids, user_matrix, strict=False):
        user_tensor = torch.tensor(vector, dtype=torch.float32, device=device)
        user_res = recommender(user_tensor)[:, :num_items]
        user_catalog = torch.ones_like(user_tensor) - user_tensor
        user_scores = torch.mul(user_res.squeeze(), user_catalog)
        scores_np = user_scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(scores_np).any():
            continue
        k = min(top_k, num_items)
        indices = np.argpartition(-scores_np, k - 1)[:k]
        ordered = indices[np.argsort(scores_np[indices])[::-1]]
        for rank, item_idx in enumerate(ordered, start=1):
            score = float(scores_np[item_idx])
            if not np.isfinite(score):
                continue
            records.append(
                {
                    "user_id": user_id,
                    "movie_id": int(item_idx),
                    "score": score,
                    "rank": rank,
                }
            )

    recommendations = (
        pd.DataFrame(records, columns=["user_id", "movie_id", "score", "rank"])
        if records
        else pd.DataFrame(columns=["user_id", "movie_id", "score", "rank"])
    )

    user_attributions: dict[int, UserAttribution] = {}
    all_interactions_rows: list[dict[str, object]] = []

    valid_user_count = 0
    cfx_search_outcome = CfxSearchOutcome()
    for row_index in shuffled_indices:
        if valid_user_count >= target_user_count:
            break
        user_id = all_user_ids[row_index]
        base_vector = user_matrix[row_index]
        if not np.any(base_vector > 0.0):
            continue

        user_recs = recommendations[recommendations["user_id"] == user_id]
        if user_recs.empty:
            continue
        target_movie_id = int(user_recs.iloc[0]["movie_id"])
        if target_movie_id < 0 or target_movie_id >= num_items:
            continue

        user_tensor = torch.tensor(base_vector, dtype=torch.float32, device=device)

        cfx_vector = (base_vector > 0.0).astype(np.int64)
        # lime.py hardcodes device="cpu" internally, so the recommender
        # and items tensor must live on CPU for that method.
        recommender_copy = copy.deepcopy(recommender)
        cfx_all_items_tensor = assets.all_items_tensor
        if method == AttributionMethod.LIME:
            recommender_copy = recommender_copy.cpu()
            cfx_all_items_tensor = cfx_all_items_tensor.cpu()
        all_scored_items = get_counterfactual_explanation(
            user_tensor,
            cfx_vector,
            user_id,
            target_movie_id,
            explainer,
            recommender_copy,
            items_array,
            device,
            method=method.value,
            jaccard_dict=assets.jaccard_dict,
            cosine_dict=assets.cosine_dict,
            shap_values=assets.shap_values,
            item_to_cluster=assets.item_to_cluster,
            lime=assets.lime,
            all_items_tensor=cfx_all_items_tensor,
            kw_dict=assets.kw_dict,
            num_items=num_items,
            train_array=assets.train_array,
            pop_array=assets.pop_array,
        )  # type: ignore[no-untyped-call]
        if not all_scored_items:
            cfx_search_outcome.n_no_cfx += 1
            continue

        minimal_cf_items, is_valid_cfx = limit_cf_item_interactions(
            all_scored_items,
            user_tensor,
            target_movie_id,
            recommender,
            num_items,
            device,
            max_cfx_removals,
            target_cfx_rank,
        )
        if not is_valid_cfx:
            cfx_search_outcome.n_no_cfx += 1
            continue

        # Store user history and target for faithfulness evaluation
        user_histories[user_id] = user_tensor.clone()
        user_targets[user_id] = target_movie_id

        all_interactions_rows.extend(
            _collect_all_interactions(
                user_id=user_id,
                all_scored_items=all_scored_items,
                minimal_cf_items=minimal_cf_items,
                base_vector=base_vector,
                num_items=num_items,
            )
        )

        # Build user_attributions using only the minimal CF set
        cfx_movie_ids: set[int] = set()
        rows_cf: list[dict[str, object]] = []
        for item_id, weight in minimal_cf_items:
            if item_id < 0 or item_id >= num_items:
                continue
            rating = float(base_vector[item_id])
            if rating <= 0.0:
                continue
            cfx_movie_ids.add(int(item_id))
            rows_cf.append(
                {
                    "movie_id": int(item_id),
                    "rating": rating,
                    "weight": float(weight),
                    "importance": float(abs(weight)),
                }
            )
        if len(rows_cf) < min_cfx_interactions:
            cfx_search_outcome.n_below_min_interactions += 1
            continue

        # Sample N non-CFX interactions from user's positive ratings
        non_cfx_candidates: list[dict[str, object]] = []
        for item_id in range(num_items):
            if item_id in cfx_movie_ids:
                continue
            rating = float(base_vector[item_id])
            if rating > 0.0:
                non_cfx_candidates.append({"movie_id": item_id, "rating": rating})

        # Sample up to n_non_cfx_interactions non-CFX interactions (not for cryptographic purposes)
        rng = random.Random(user_id)  # noqa: S311
        sample_size = min(n_non_cfx_interactions, len(non_cfx_candidates))
        sampled_non_cfx = rng.sample(non_cfx_candidates, sample_size) if sample_size > 0 else []

        cf_frame = pd.DataFrame(rows_cf)
        non_cfx_frame = (
            pd.DataFrame(sampled_non_cfx) if sampled_non_cfx else pd.DataFrame(columns=["movie_id", "rating"])
        )
        user_attributions[user_id] = UserAttribution(
            user_id=user_id,
            cfx_interactions=cf_frame,
            non_cfx_interactions=non_cfx_frame,
        )
        valid_user_count += 1
        cfx_search_outcome.n_valid += 1

    cfx_by_user: dict[int, pd.DataFrame] = {uid: attr.cfx_interactions for uid, attr in user_attributions.items()}
    cfx_interactions = summarise_interactions(cfx_by_user)

    all_interactions = (
        pd.DataFrame(all_interactions_rows)
        if all_interactions_rows
        else pd.DataFrame(columns=["user_id", "item_id", "rating", "attribution_score", "is_counterfactual"])
    )

    return AttributionResult(
        recommendations=recommendations,
        user_attributions=user_attributions,
        cfx_interactions=cfx_interactions,
        all_interactions=all_interactions,
        user_histories=user_histories,
        user_targets=user_targets,
        cfx_search_outcome=cfx_search_outcome,
    )


def _collect_all_interactions(
    *,
    user_id: int,
    all_scored_items: list[tuple[int, float]],
    minimal_cf_items: list[tuple[int, float]],
    base_vector: np.ndarray[Any, Any],
    num_items: int,
) -> list[dict[str, object]]:
    """Collect all positive interactions with attribution scores."""
    rows: list[dict[str, object]] = []
    minimal_cf_ids = {int(item_id) for item_id, _ in minimal_cf_items}
    for item_id, score in all_scored_items:
        if item_id < 0 or item_id >= num_items:
            continue
        rating = float(base_vector[item_id])
        if rating <= 0.0:
            continue

        rows.append(
            {
                "user_id": user_id,
                "item_id": int(item_id),
                "rating": rating,
                "attribution_score": float(score),
                "is_counterfactual": int(item_id) in minimal_cf_ids,
            }
        )
    return rows
