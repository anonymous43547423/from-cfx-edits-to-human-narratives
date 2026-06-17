# ruff: noqa: S101, TC002, TC003
"""CLI tests for ``run_pipeline`` reward and W&B project overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_pipeline


def test_parse_args_reward_metric_and_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--reward-metric``, ``--wandb-entity``, and ``--wandb-project`` parse correctly."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--top-k",
            "5",
            "--n-cfx-interactions",
            "3",
            "--n-non-cfx-interactions",
            "3",
            "--min-cfx-interactions",
            "1",
            "--attribution-method",
            "jaccard",
            "--max-cfx-removals",
            "5",
            "--target-cfx-rank",
            "3",
            "--n-judged-interactions",
            "5",
            "--model-id-generation",
            "g",
            "--model-id-evaluation",
            "e",
            "--evaluation",
            "cfx_match",
            "--output-datasets-path",
            str(tmp_path / "out"),
            "--n-sampled-distance-pairs",
            "5",
            "--n-faithfulness-interactions-min-limit",
            "1",
            "--n-faithfulness-trials",
            "2",
            "--n-faithfulness-samples",
            "1",
            "--random-seed",
            "0",
            "--target-set",
            "test",
            "--user-pool",
            "eval",
            "--reward-metric",
            "correctness",
            "--wandb-entity",
            "ent",
            "--wandb-project",
            "proj",
        ],
    )
    args = run_pipeline.parse_args()
    assert args.reward_metric == "correctness"
    assert args.wandb_entity == "ent"
    assert args.wandb_project == "proj"
