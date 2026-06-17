# From Counterfactual Edits to Human Narratives

Code for the paper **From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations**.

The repository turns recommender-system counterfactual edit lists into short natural-language explanations, then evaluates them with correctness, informativeness, and linguistic-quality metrics.

## Setup

Requirements:
- Python `>=3.11,<3.12`
- [`uv`](https://github.com/astral-sh/uv)

```bash
uv sync --group dev
. .venv/bin/activate
python init_data.py
```

`python init_data.py` downloads the non-committed MovieLens-1M data and model checkpoints.

Some Hugging Face models used in this repository may require authentication and license acceptance before they can be downloaded. If needed, log in using the Hugging Face CLI.

## Human-Labeled Data

Committed human labels live under `datasets/human-feedback/`:

- `validation/readability-human-labeled.csv`: binary human judgments of overall linguistic quality (validation split).
- `validation/interaction-match-human-labeled.csv`: binary human judgments of explanation-interaction match (validation split).
- `test/readability-human-labeled.csv`: binary human judgments of overall linguistic quality (test split).
- `test/interaction-match-human-labeled.csv`: binary human judgments of explanation-interaction match (test split).

The validation split is used for model-selection and prompt-tuning workflows. The test split is the held-out final evaluation set.

## Main Scripts

- `python init_data.py`: download the required non-committed artifacts.
- `./get_all_results.sh`: launcher for the main paper experiment grid. It runs both vanilla and DPO workflows for the seven CFX methods and three models as reported in the paper (Ministral 8B, Gemma 3 12B, Qwen3 8B).
- `./get_smoke_results.sh`: quick smoke-run variant of the main grid for lightweight validation of the environment setup.
- `./train_human_feedback_models.sh`: train the validation and test human-feedback classifier pairs used for local model selection and final reporting.
- `./calculate_human_models_feedback.sh`: score `runs/` with the trained validation and test human-feedback models in the required order.
- `./generate_latex_results_table.sh`: summarize `runs/` into the LaTeX table used in the paper.
- `./gather_eval_datasets.sh`: export paired readability and interaction-match CSV datasets for manual annotation.

Low-level entry points used by the main script:

- `python scripts/run_pipeline.py`: run one vanilla eval pipeline configuration.
- `python scripts/run_dpo.py`: train one DPO model from two prepared dataset directories.
- `python scripts/run_dpo_eval_sweep.py`: run a W&B sweep over repeated DPO-plus-eval trials.
- `bash scripts/run_eval_eval_dpo_eval.sh`: orchestrate one train_a/train_b/DPO/eval workflow.
- `python scripts/train_human_feedback_model.py`: train one readability classifier and one interaction-match classifier from a human-labeled dataset split.
- `python -m scripts.calculate_human_model_feedback`: score run outputs with those classifiers.
- `python scripts/generate_latex_results_table.py`: aggregate `runs/` into the LaTeX results table used in the paper.

## Reproducing The Paper

Run the main experiment sweep:

```bash
./get_all_results.sh
```

By default this runs the experiment grid with fixed DPO hyperparameters and no W&B sweep.

For the paper, this grid was actually executed as scheduled jobs on a GPU cluster with NVIDIA 40GB A100 GPUs, and the full run required many GPU hours. For a quick smoke run instead, use:

```bash
./get_smoke_results.sh
```

To run the W&B hyperparameter sweep used for DPO model selection, enable sweep mode:

```bash
USE_SWEEP=1 ENABLE_WANDB=1 ./get_all_results.sh
```

Outputs are written under `runs/`, with separate experiment roots such as `runs/run_pipeline_*` and `runs/run_eval_eval_dpo_eval_*`. Each timestamped run directory contains a `run_summary.json` plus detailed Feather exports.

The main paper table reports human-calibrated metrics. DPO optimization uses LLM judges, but the reported metrics come from separate ModernBERT classifiers trained on the committed human-labeled validation and test dataset splits.

Train those human-feedback models with:

```bash
./train_human_feedback_models.sh
```

This helper trains and stores the `modernbert_readability_validation`, `modernbert_interaction_validation`, `modernbert_readability_test`, and `modernbert_interaction_test` models.

Then score the run outputs with those models:

```bash
./calculate_human_models_feedback.sh
```

This helper first writes validation human-feedback summaries, then writes test human-feedback summaries. For sweep-based DPO experiments, as reported in the paper, the validation summaries select the best DPO sweep run, and the corresponding test summaries provide the reported human-feedback metrics for that selected run. When no sweep is present, the later table-generation step falls back to the latest non-sweep run.

Finally, generate the LaTeX table:

```bash
./generate_latex_results_table.sh
```

To build fresh human-annotation CSV datasets from the current `runs/` outputs, which is useful for extending the labeled data but not required to reproduce the main table:

```bash
./gather_eval_datasets.sh
```

This helper writes the paired CSV files under `runs/eval_datasets/`.

## Development Checks

```bash
ruff format
ruff check
mypy .
pytest -svv
```
