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

- `validation/readability-human-labeled.csv`: binary human judgments of overall linguistic quality.
- `validation/interaction-match-human-labeled.csv`: binary human judgments of explanation/interaction match.
- `test/readability-human-labeled.csv`: held-out readability labels.
- `test/interaction-match-human-labeled.csv`: held-out interaction-match labels.

The validation split is used for model-selection and prompt-tuning workflows. The test split is the held-out final evaluation set.

## Main Scripts

- `python init_data.py`: download the required non-committed artifacts.
- `./get_all_results.sh`: local batch launcher for the main paper experiment grid. It runs both vanilla and DPO workflows for the seven CFX methods and the three paper models (Ministral 8B, Gemma 3 12B, Qwen3 8B).
- `./generate_latex_results_table.sh`: summarize `runs/` into the LaTeX table used in the paper.
- `./gather_eval_datasets.sh`: export paired readability and interaction-match CSV datasets for manual annotation.
- `./train_human_feedback_model.sh`: example wrapper for training validation or test human-feedback classifiers and saving them under split-specific output directories.
- `./evaluate_llm_judge.sh`: benchmark an LLM judge on the committed test readability labels.

Low-level entry points used by the main script:

- `python scripts/run_pipeline.py`: run one vanilla eval pipeline configuration.
- `python scripts/run_dpo.py`: train one DPO model from two prepared dataset directories.
- `python scripts/run_dpo_eval_sweep.py`: run a W&B sweep over repeated DPO-plus-eval trials.
- `bash scripts/run_eval_eval_dpo_eval.sh`: orchestrate one train_a/train_b/DPO/eval workflow.
- `python -m scripts.calculate_human_model_feedback`: score run outputs with those classifiers.
- `python scripts/generate_latex_results_table.py`: aggregate `runs/` into the LaTeX results table used in the paper.

## Reproducing The Paper

Run the main local experiment sweep:

```bash
./get_all_results.sh
```

By default this runs the experiment grid with fixed DPO hyperparameters and no W&B sweep.

To run the W&B hyperparameter sweep used for DPO model selection, enable sweep mode:

```bash
USE_SWEEP=1 ENABLE_WANDB=1 ./get_all_results.sh
```

Outputs are written under `runs/`, with separate experiment roots such as `runs/run_pipeline_*` and `runs/run_eval_eval_dpo_eval_*`. Each timestamped run directory contains a `run_summary.json` plus detailed Feather exports.

To reproduce the paper table from those run directories, use `scripts/generate_latex_results_table.py`. It reads the run summaries plus human-feedback summaries, selects the validation-best DPO variant, and emits the LaTeX table corresponding to the reported metrics.

For DPO model selection, the best variant is selected with a human-feedback model trained on the validation human-labeled dataset, and the selected run is then reported with a human-feedback model trained on the test human-labeled dataset.

To reproduce that selection/reporting flow locally, train one human-feedback model on the validation labels and one on the test labels, then score the `runs/` outputs separately for each split with `python -m scripts.calculate_human_model_feedback`. The example helpers `./train_human_feedback_model.sh` and `./evaluate_llm_judge.sh` cover the common local entrypoints.

To benchmark the judge model against the committed test labels:

```bash
./evaluate_llm_judge.sh
```

To build fresh human-annotation CSV datasets from the current `runs/` outputs:

```bash
./gather_eval_datasets.sh
```

This helper writes the paired CSV files under `runs/eval_datasets/`.

To train split-specific human-feedback classifiers with the example wrapper:

```bash
./train_human_feedback_model.sh validation
./train_human_feedback_model.sh test
```

This helper writes the models to `modernbert_readability_<split>` and `modernbert_interaction_<split>`.

To reproduce the paper table from the produced run directories:

```bash
./generate_latex_results_table.sh
```

This helper writes `runs/main_results_table.tex`. You can also call `python scripts/generate_latex_results_table.py --outputs-dir runs` directly.

When both validation and test human-feedback summaries are present, the table script uses validation human-feedback scores to choose the best DPO variant and test human-feedback scores for the displayed metrics.

## Notes

- Exact numbers may not match the paper bit-for-bit because LLM generation is not fully deterministic.
- Full reproduction is GPU-intensive.

## Development Checks

```bash
ruff format
ruff check
mypy .
pytest -svv
```
