"""Command-line entry point to run the RecSys NLE pipeline end-to-end."""

# ruff: noqa: E402

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from recsys_nle.cuda_utils import enable_expandable_segments

enable_expandable_segments()

import argparse
import json
import logging
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path

import wandb

from recsys_nle.core.attribution import AttributionConfig, AttributionMethod
from recsys_nle.nl_explanations.workflow import ExplanationConfig
from recsys_nle.pipeline.config import OutputConfig, PipelineConfig, RecommendationConfig
from recsys_nle.pipeline.datasets_exporter import DatasetsExporter
from recsys_nle.pipeline.reporting import PipelineReporter
from recsys_nle.pipeline.reward import RewardType, apply_reward_composite_to_summary_results
from recsys_nle.pipeline.run_summary import _config_to_dict, build_run_summary
from recsys_nle.pipeline.workflow import PipelineResult, PipelineWorkflow

LOGGER = logging.getLogger(__name__)

SAMPLE_USERS_FOR_LOGGING = 3
DEFAULT_WANDB_PROJECT = "recsys-nle-run-pipeline"

_EVALUATION_METRICS = (
    "plausibility",
    "readability",
    "cfx_match",
    "non_cfx_match",
    "faithfulness_removal",
    "faithfulness_replacement",
)
_EVALUATION_CHOICES = (*_EVALUATION_METRICS, "all")


def _parse_evaluation_selection(raw: object | None) -> tuple[str, ...]:
    """Normalise CLI evaluation selections into a canonical tuple of metric names."""
    if raw is None:
        return _EVALUATION_METRICS

    values: list[str]
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = [str(item) for item in raw]
    else:
        values = [str(raw)]

    selected: list[str] = []
    for value in values:
        name = value.strip().lower()
        if not name:
            continue
        if name == "all":
            return _EVALUATION_METRICS
        if name in _EVALUATION_METRICS and name not in selected:
            selected.append(name)

    return tuple(selected or _EVALUATION_METRICS)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate top-k recommendations and explanations for RecSys NLE.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to generate per user (default: 10).",
    )
    parser.add_argument(
        "--n-cfx-interactions",
        type=int,
        required=True,
        help="Number of CFX interactions to include in generation prompts.",
    )
    parser.add_argument(
        "--n-non-cfx-interactions",
        type=int,
        required=True,
        help="Number of non-CFX interactions to sample for each user.",
    )
    parser.add_argument(
        "--min-cfx-interactions",
        type=int,
        required=True,
        help="Minimum number of CFX interactions required per user.",
    )
    parser.add_argument(
        "--attribution-method",
        type=str,
        required=True,
        choices=[method.value for method in AttributionMethod],
        help="Attribution algorithm to generate counterfactual interactions.",
    )
    parser.add_argument(
        "--max-cfx-removals",
        type=int,
        required=True,
        help="Maximum number of CF items to remove when finding minimal counterfactual set.",
    )
    parser.add_argument(
        "--target-cfx-rank",
        type=int,
        required=True,
        help="Rank threshold for counterfactual validation (recommendation must drop below this).",
    )
    parser.add_argument(
        "--n-judged-interactions",
        type=int,
        help="Number of interactions to randomly sample for judgment in evaluation metrics.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help="Batch size for the explanation generation stage (default: 4).",
    )
    parser.add_argument(
        "--evaluation-user-batch-size",
        type=int,
        default=4,
        help="Batch size for user evaluation batches (default: 4).",
    )
    parser.add_argument(
        "--evaluation-llm-batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation LLM inference (default: 4).",
    )
    parser.add_argument(
        "--model-id-generation",
        type=str,
        required=True,
        help=(
            "Model id for reasoning and explanation generation (Hugging Face id or local adapter path). "
            "EINFRA/... is not supported here (evaluation only)."
        ),
    )
    parser.add_argument(
        "--model-id-evaluation",
        type=str,
        required=True,
        help=(
            "Model id for LLM judge evaluation: Hugging Face id, or EINFRA/<api_model_id> for "
            "e-INFRA OpenAI-compatible chat completions."
        ),
    )
    parser.add_argument(
        "--evaluation",
        choices=_EVALUATION_CHOICES,
        nargs="+",
        required=True,
        help=(
            "Evaluation metrics to compute and report. "
            "Choose one or more of: plausibility, cfx_match, non_cfx_match, or 'all' for every metric."
        ),
    )
    parser.add_argument(
        "--output-datasets-path",
        type=Path,
        help="Path to save the results as Feather dataframes.",
    )
    parser.add_argument(
        "--create-output-datasets-subdirectory",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Create a timestamped subdirectory inside --output-datasets-path for each run "
            "(default: true). Use --no-create-output-datasets-subdirectory to write directly "
            "to --output-datasets-path."
        ),
    )
    parser.add_argument(
        "--n-sampled-distance-pairs",
        type=int,
        required=True,
        help="Number of interaction pairs to sample for distance metrics.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Include all LLM prompts in pipeline log tables.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help=(
            "Disable intermediate chain-of-thought reasoning so that explanations are "
            "generated directly from influential interactions."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        required=True,
        help="Random seed for reproducible user sampling.",
    )
    parser.add_argument(
        "--sample-user-count",
        type=int,
        default=SAMPLE_USERS_FOR_LOGGING,
        help="Maximum number of users to sample for detailed logging (default: 3).",
    )
    parser.add_argument(
        "--target-set",
        choices=("validation", "test"),
        required=True,
        help=("Holdout partition within test_data.csv: validation = middle ~⅓; test = first ~⅓ (final benchmark)."),
    )
    parser.add_argument(
        "--user-pool",
        choices=("train", "eval"),
        required=True,
        help=("Run purpose: train = DPO preference-pair generation; eval = benchmark / metrics."),
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity. Leave unset to use the current W&B default/account.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project (default: {DEFAULT_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--reward-metric",
        choices=[reward.value for reward in RewardType],
        default=None,
        help=("If set, add reward_composite to the run summary using the same linear formula as DPO --reward."),
    )
    # Faithfulness evaluation parameters
    parser.add_argument(
        "--n-sampled-faithfulness-interactions",
        type=int,
        default=20,
        help="Number of interactions/non-interactions to sample for faithfulness scoring (default: 20).",
    )
    parser.add_argument(
        "--faithfulness-match-threshold",
        type=float,
        default=0.5,
        help="Score threshold for considering an interaction as 'similar' to explanation (default: 0.5).",
    )
    parser.add_argument(
        "--n-faithfulness-interactions-min-limit",
        type=int,
        required=True,
        help=(
            "Minimum number of similar/dissimilar interactions required for faithfulness metrics; "
            "otherwise the score is NaN."
        ),
    )
    parser.add_argument(
        "--n-faithfulness-trials",
        type=int,
        required=True,
        help="Number of sampling trials for each faithfulness metric.",
    )
    parser.add_argument(
        "--n-faithfulness-samples",
        type=int,
        required=True,
        help="Number of sampled items per faithfulness trial.",
    )
    return parser.parse_args()


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Translate CLI arguments into a pipeline configuration object."""
    recommendation_config = RecommendationConfig(
        top_k=args.top_k,
    )
    attribution_config = AttributionConfig(
        method=AttributionMethod(args.attribution_method),
        max_cfx_removals=args.max_cfx_removals,
        target_cfx_rank=args.target_cfx_rank,
        min_cfx_interactions=args.min_cfx_interactions,
        recommendation_count=args.top_k,
        n_non_cfx_interactions=args.n_non_cfx_interactions,
    )
    enabled_evaluations = _parse_evaluation_selection(getattr(args, "evaluation", None))
    explanation_config = ExplanationConfig(
        model_id_generation=args.model_id_generation,
        model_id_evaluation=args.model_id_evaluation,
        n_cfx_interactions=args.n_cfx_interactions,
        n_judged_interactions=getattr(args, "n_judged_interactions", None),
        generation_batch_size=args.generation_batch_size,
        evaluation_user_batch_size=args.evaluation_user_batch_size,
        evaluation_llm_batch_size=args.evaluation_llm_batch_size,
        disable_reasoning=getattr(args, "disable_reasoning", False),
        enabled_evaluations=enabled_evaluations,
        n_sampled_faithfulness_interactions=args.n_sampled_faithfulness_interactions,
        faithfulness_match_threshold=args.faithfulness_match_threshold,
        n_faithfulness_interactions_min_limit=args.n_faithfulness_interactions_min_limit,
        n_faithfulness_trials=args.n_faithfulness_trials,
        n_faithfulness_samples=args.n_faithfulness_samples,
    )
    output_config = OutputConfig(
        output_datasets_path=args.output_datasets_path,
        n_sampled_distance_pairs=args.n_sampled_distance_pairs,
        create_output_datasets_subdirectory=args.create_output_datasets_subdirectory,
    )
    return PipelineConfig(
        explanation=explanation_config,
        attribution=attribution_config,
        target_set=args.target_set,
        user_pool=args.user_pool,
        recommendation=recommendation_config,
        outputs=output_config,
        max_users_for_attribution=args.sample_user_count,
        sample_user_count=max(0, int(args.sample_user_count)),
        random_seed=args.random_seed,
        show_prompts=getattr(args, "show_prompts", False),
    )


def _log_summary(pipeline_result: PipelineResult) -> None:
    """Log aggregate statistics about the executed pipeline run."""
    num_users = len(pipeline_result.recommendations["user_id"].unique())
    LOGGER.info(
        "Produced %d recommendations across %d users",
        len(pipeline_result.recommendations),
        num_users,
    )
    LOGGER.info(
        "Computed attributions for %d users",
        len(pipeline_result.user_attributions),
    )
    LOGGER.info(
        "Generated natural-language explanations for %d users",
        len(pipeline_result.explanations.dataset),
    )


def main() -> int:
    """CLI entry point for running the RecSys NLE workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    pipeline_config = _build_pipeline_config(args)
    if args.enable_wandb:
        wandb_kwargs = {
            "project": args.wandb_project,
            "config": _config_to_dict(pipeline_config),
        }
        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_kwargs)
    workflow = PipelineWorkflow()
    pipeline_result = workflow.run(pipeline_config)

    _log_summary(pipeline_result)
    datasets_exporter = DatasetsExporter(pipeline_config.outputs)
    export_dir = datasets_exporter.export(pipeline_result)

    reporter = PipelineReporter(LOGGER)
    reporter.render(pipeline_result=pipeline_result, config=pipeline_config)

    summary = build_run_summary(
        pipeline_result,
        pipeline_config,
        enabled_evaluations=pipeline_config.explanation.enabled_evaluations,
    )
    if args.reward_metric is not None:
        reward_type = RewardType(args.reward_metric)
        missing = apply_reward_composite_to_summary_results(summary["results"], reward_type)
        if missing:
            LOGGER.warning(
                "Cannot compute reward_composite: missing or non-numeric aggregate keys %s "
                "(enable matching --evaluation metrics).",
                sorted(missing),
            )
    LOGGER.info("Run summary: %s", json.dumps(summary, indent=4))
    if args.enable_wandb:
        wandb.log(summary["results"])

    if export_dir:
        (export_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        LOGGER.info(
            "Done, generation and evaluation dataframes written to %s and %s",
            export_dir.resolve() / "generation.feather",
            export_dir.resolve() / "evaluation.feather",
        )

    if args.enable_wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)
