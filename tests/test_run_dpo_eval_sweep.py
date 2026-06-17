# ruff: noqa: S101, SLF001, PLR2004
"""Tests for the DPO + eval W&B sweep driver."""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts import run_dpo_eval_sweep


def test_split_argv(tmp_path: Path) -> None:
    """Arguments after ``--`` become the eval argv."""
    base = str(tmp_path / "base")
    a, b = run_dpo_eval_sweep._split_argv(["--base-dir", base, "--", "--foo", "1"])
    assert a == ["--base-dir", base]
    assert b == ["--foo", "1"]


def test_validate_ranges_rejects_non_positive() -> None:
    """Log-uniform requires strictly positive bounds."""
    with pytest.raises(ValueError, match="0 < min < max"):
        run_dpo_eval_sweep._validate_ranges(0.0, 1e-3, 0.01, 0.1)


def test_validate_ranges_rejects_inverted_interval() -> None:
    """Min must be strictly less than max."""
    with pytest.raises(ValueError, match="0 < min < max"):
        run_dpo_eval_sweep._validate_ranges(1e-2, 1e-3, 0.01, 0.1)


def _meta(run_id: str, trial_dir: Path, score: float | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "score": score,
        "trial_dir": str(trial_dir),
        "learning_rate": 1e-4,
        "beta": 0.05,
        "lora_r": 8,
    }


def _add_trial(sweep_root: Path, run_id: str, score: float | None) -> Path:
    trial_dir = sweep_root / run_id
    (trial_dir / "dpo").mkdir(parents=True)
    (trial_dir / "dpo" / "marker").write_text("x", encoding="utf-8")
    run_dpo_eval_sweep._update_trial_registry(sweep_root, _meta(run_id, trial_dir, score))
    return trial_dir


def test_update_trial_registry_keeps_top_k(tmp_path: Path) -> None:
    """Only the top-K trials retain a ``dpo/`` directory; all trials are recorded."""
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()

    for run_id, score in [("a", 1.0), ("b", 0.5), ("c", 2.0), ("d", 1.5), ("e", 0.2)]:
        _add_trial(sweep_root, run_id, score)

    retained = {rid for rid in "abcde" if (sweep_root / rid / "dpo").exists()}
    assert retained == {"a", "c", "d"}

    state = json.loads((sweep_root / "trials.json").read_text(encoding="utf-8"))
    assert state["top_k"] == 3
    assert [t["run_id"] for t in state["trials"]] == ["c", "d", "a", "b", "e"]
    assert {t["run_id"]: t["retained"] for t in state["trials"]} == {
        "a": True,
        "b": False,
        "c": True,
        "d": True,
        "e": False,
    }


def test_update_trial_registry_nan_never_retained(tmp_path: Path) -> None:
    """Non-finite scores are recorded but never retain ``dpo/``."""
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir()
    _add_trial(sweep_root, "ok", 1.0)
    trial_dir = _add_trial(sweep_root, "bad", None)

    assert not (trial_dir / "dpo").exists()
    state = json.loads((sweep_root / "trials.json").read_text(encoding="utf-8"))
    bad = next(t for t in state["trials"] if t["run_id"] == "bad")
    assert bad["score"] is None
    assert bad["retained"] is False


def test_execute_sweep_trial_invokes_subprocesses_and_reads_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DPO and pipeline commands run; ``run_summary.json`` drives bookkeeping."""
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "run_dpo.py").write_text("#", encoding="utf-8")
    (repo_root / "scripts" / "run_pipeline.py").write_text("#", encoding="utf-8")

    base = tmp_path / "base"
    (base / "train_a").mkdir(parents=True)
    (base / "train_b").mkdir(parents=True)

    driver = Namespace(
        model_id="m",
        dpo_reward="correctness",
        n_epochs=1,
        eval_dataset_split=None,
        log_level="INFO",
        wandb_entity="e",
        wandb_project="p",
    )

    pipeline_cmds: list[list[str]] = []
    pipeline_envs: list[dict[str, str]] = []
    dpo_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        exe = str(cmd[1])
        if exe.endswith("run_dpo.py"):
            dpo_cmds.append(cmd)
            out_i = cmd.index("--output-dir") + 1
            dpo_out = Path(cmd[out_i])
            (dpo_out / "best_model").mkdir(parents=True)
        elif exe.endswith("run_pipeline.py"):
            out_i = cmd.index("--output-datasets-path") + 1
            trial_dir = Path(cmd[out_i])
            summary = {"results": {"reward_composite": 3.0, "explanation_cfx_pattern_match_mean": 1.0}}
            (trial_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            pipeline_cmds.append(cmd)
            env = kwargs.get("env")
            assert isinstance(env, dict)
            pipeline_envs.append(env)
        return MagicMock(returncode=0)

    monkeypatch.setenv("WANDB_SWEEP_ID", "outer-sweep")
    monkeypatch.setenv("WANDB_RUN_ID", "outer-run")
    results = run_dpo_eval_sweep.execute_sweep_trial(
        repo_root=repo_root,
        base_dir=base,
        driver=driver,
        eval_argv=["--random-seed", "0"],
        run_id="trial1",
        learning_rate=1e-4,
        beta=0.05,
        lora_r=16,
        run_subprocess=fake_run,
    )

    assert results == {"reward_composite": 3.0, "explanation_cfx_pattern_match_mean": 1.0}
    assert dpo_cmds
    i = dpo_cmds[0].index("--lora-r")
    assert dpo_cmds[0][i + 1] == "16"
    state = json.loads((base / "sweep" / "trials.json").read_text(encoding="utf-8"))
    assert state["trials"][0]["run_id"] == "trial1"
    assert state["trials"][0]["score"] == 3.0
    assert state["trials"][0]["lora_r"] == 16
    assert state["trials"][0]["retained"] is True
    assert "--enable-wandb" not in pipeline_cmds[0]
    assert pipeline_envs[0]["WANDB_MODE"] == "disabled"
    assert "WANDB_SWEEP_ID" not in pipeline_envs[0]
    assert "WANDB_RUN_ID" not in pipeline_envs[0]


class _HardExitError(RuntimeError):
    pass


def _fake_exit(codes: list[int]) -> Any:
    def _exit(code: int) -> None:
        codes.append(code)
        raise _HardExitError

    return _exit


def _driver_namespace(base_dir: Path) -> Namespace:
    return Namespace(
        base_dir=base_dir,
        model_id="m",
        dpo_reward="correctness",
        n_epochs=1,
        eval_dataset_split=None,
        log_level="INFO",
        sweep_time_limit_minutes=60.0,
        sweep_dpo_learning_rate_min=1e-5,
        sweep_dpo_learning_rate_max=1e-3,
        sweep_dpo_beta_min=0.01,
        sweep_dpo_beta_max=0.1,
        sweep_lora_r=[8, 16],
        wandb_entity="e",
        wandb_project="p",
    )


def test_trial_fn_skips_new_trials_after_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the deadline has already passed, ``trial_fn`` hard-exits before running anything."""
    captured: list[Any] = []

    def fake_agent(_sweep_id: str, function: object, **_kwargs: object) -> None:
        captured.append(function)

    mock_wandb = MagicMock()
    mock_wandb.sweep = MagicMock(return_value="sweep-id")
    mock_wandb.agent = fake_agent
    monkeypatch.setattr(run_dpo_eval_sweep, "wandb", mock_wandb)

    stop_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        run_dpo_eval_sweep,
        "_stop_sweep_via_cli",
        lambda sid, entity, project: stop_calls.append((sid, entity, project)),
    )

    hard_exit_codes: list[int] = []
    monkeypatch.setattr(os, "_exit", _fake_exit(hard_exit_codes))

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    run_dpo_eval_sweep.run_sweep(
        repo_root=repo,
        driver=_driver_namespace(base_dir),
        eval_argv=[],
        deadline_monotonic=time.monotonic() - 1.0,
    )
    trial = captured[0]
    with pytest.raises(_HardExitError):
        trial()
    assert stop_calls == [("sweep-id", "e", "p")]
    assert hard_exit_codes == [0]


def test_stop_sweep_via_cli_invokes_wandb_module() -> None:
    """``_stop_sweep_via_cli`` runs ``python -m wandb sweep --stop`` with entity, project, sweep id."""
    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured_cmd.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    run_dpo_eval_sweep._stop_sweep_via_cli("sid", "ent", "proj", run_cli=fake_run)
    assert captured_cmd == [
        [
            sys.executable,
            "-m",
            "wandb",
            "sweep",
            "--stop",
            "--entity",
            "ent",
            "--project",
            "proj",
            "sid",
        ]
    ]


def test_stop_sweep_via_cli_nonzero_exit_does_not_raise() -> None:
    """A failing ``wandb sweep --stop`` process is logged and does not propagate."""

    def fake_run(_cmd: list[str], **_kwargs: object) -> MagicMock:
        return MagicMock(returncode=1, stdout="x", stderr="y")

    run_dpo_eval_sweep._stop_sweep_via_cli("sid", "e", "p", run_cli=fake_run)


def test_run_sweep_registers_discrete_lora_r_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``wandb.sweep`` receives categorical ``lora_r`` from the driver."""
    captured: list[dict[str, Any]] = []

    def fake_sweep(cfg: dict[str, Any], **_kwargs: object) -> str:
        captured.append(cfg)
        return "sweep-id"

    mock_wandb = MagicMock()
    mock_wandb.sweep = fake_sweep
    mock_wandb.agent = MagicMock()
    monkeypatch.setattr(run_dpo_eval_sweep, "wandb", mock_wandb)

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    driver = _driver_namespace(base_dir)
    driver.sweep_lora_r = [4, 32]

    run_dpo_eval_sweep.run_sweep(repo_root=tmp_path / "repo", driver=driver, eval_argv=[], deadline_monotonic=None)

    assert captured[0]["parameters"]["lora_r"] == {"values": [4, 32]}
