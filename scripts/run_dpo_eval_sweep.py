"""W&B random sweep over DPO learning rate, beta, and LoRA rank; each trial runs DPO then eval."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import wandb

LOGGER = logging.getLogger(__name__)

DEFAULT_SWEEP_WANDB_PROJECT = "recsys-nle-dpo-sweep"
DEFAULT_WANDB_ENTITY = "your-wandb-entity"
TOP_K_MODELS = 3
_WANDB_SWEEP_STOP_CLI_TIMEOUT_S = 60.0


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at ``--`` into driver args and eval subprocess argv."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _parse_driver_args(argv: list[str]) -> argparse.Namespace:
    """Parse sweep driver flags (everything before ``--`` in ``sys.argv``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Run directory containing train_a/ and train_b/ (from the orchestrator).",
    )
    parser.add_argument("--model-id", type=str, required=True, help="Base model for DPO.")
    parser.add_argument(
        "--dpo-reward",
        type=str,
        required=True,
        help="DPO reward id (must match --reward-metric passed to eval).",
    )
    parser.add_argument("--n-epochs", type=int, required=True, help="DPO training epochs.")
    parser.add_argument("--eval-dataset-split", type=float, default=None, help="Optional DPO eval split.")
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging for this driver and forwarded to DPO.",
    )
    parser.add_argument(
        "--sweep-time-limit-minutes",
        type=float,
        required=True,
        help="Soft wall-clock budget: no new trials after this many minutes.",
    )
    parser.add_argument("--sweep-dpo-learning-rate-min", type=float, required=True)
    parser.add_argument("--sweep-dpo-learning-rate-max", type=float, required=True)
    parser.add_argument("--sweep-dpo-beta-min", type=float, required=True)
    parser.add_argument("--sweep-dpo-beta-max", type=float, required=True)
    parser.add_argument(
        "--sweep-lora-r",
        type=int,
        nargs="+",
        required=True,
        help="Discrete LoRA ranks r for DPO trials (lora_alpha is r*2 in run_dpo.py).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity for the sweep and eval subprocess (default: {DEFAULT_WANDB_ENTITY}).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_SWEEP_WANDB_PROJECT,
        help=f"W&B project for the sweep (default: {DEFAULT_SWEEP_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (parent of scripts/). Defaults to parent of this file's directory.",
    )
    return parser.parse_args(argv)


def _validate_ranges(lr_min: float, lr_max: float, b_min: float, b_max: float) -> None:
    """Ensure sweep bounds are valid for log-uniform sampling."""
    for name, lo, hi in (("learning-rate", lr_min, lr_max), ("beta", b_min, b_max)):
        if lo <= 0 or hi <= 0 or lo >= hi:
            msg = f"--sweep-dpo-{name}-min/max must satisfy 0 < min < max (got {lo}, {hi})."
            raise ValueError(msg)


def _quiet_finish() -> None:
    """Call ``wandb.finish`` swallowing (and logging) any error."""
    try:
        wandb.finish()
    except Exception:
        LOGGER.exception("wandb.finish failed")


def _stop_sweep_via_cli(
    sweep_id: str,
    entity: str,
    project: str,
    *,
    run_cli: Callable[..., Any] = subprocess.run,
) -> None:
    """Finish the sweep on W&B using ``wandb sweep --stop`` (no new runs; in-flight may finish)."""
    cmd = [
        sys.executable,
        "-m",
        "wandb",
        "sweep",
        "--stop",
        "--entity",
        entity,
        "--project",
        project,
        sweep_id,
    ]
    try:
        completed = run_cli(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_WANDB_SWEEP_STOP_CLI_TIMEOUT_S,
            env=os.environ,
        )
        if completed.returncode != 0:
            tail = ((completed.stdout or "") + (completed.stderr or ""))[:2000]
            LOGGER.warning("wandb sweep --stop exited %s: %s", completed.returncode, tail)
        else:
            LOGGER.info("Sweep %s marked stopped via W&B CLI.", sweep_id)
    except subprocess.TimeoutExpired:
        LOGGER.warning(
            "wandb sweep --stop timed out after %s s; exiting anyway.",
            _WANDB_SWEEP_STOP_CLI_TIMEOUT_S,
        )
    except Exception:
        LOGGER.exception("wandb sweep --stop failed for sweep %s", sweep_id)


def _rmtree_if_exists(path: Path) -> None:
    """Remove ``path`` if it is an existing directory."""
    if path.is_dir():
        shutil.rmtree(path)


def _read_summary_results(path: Path) -> dict[str, Any]:
    """Return the ``results`` dict from ``run_summary.json`` (empty if missing/invalid)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8")).get("results")
    return data if isinstance(data, dict) else {}


def _results_score(results: Mapping[str, Any]) -> float:
    """Extract ``reward_composite`` from a ``results`` dict; NaN when unusable."""
    raw = results.get("reward_composite")
    try:
        return float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _trial_sort_key(trial: dict[str, Any]) -> tuple[int, float]:
    """Rank finite scores first (higher is better); non-finite go last."""
    score = trial.get("score")
    if isinstance(score, (int, float)) and math.isfinite(score):
        return (0, -float(score))
    return (1, 0.0)


def _update_trial_registry(
    sweep_root: Path,
    trial_meta: dict[str, Any],
    *,
    top_k: int = TOP_K_MODELS,
) -> None:
    """Append ``trial_meta`` to ``sweep_root/trials.json`` and keep only top-``top_k`` ``dpo/`` dirs."""
    registry_path = sweep_root / "trials.json"
    lock_path = sweep_root / "trials.json.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            state = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
            trials: list[dict[str, Any]] = [
                t for t in state.get("trials", []) if t.get("run_id") != trial_meta.get("run_id")
            ]
            trials.append(trial_meta)
            trials.sort(key=_trial_sort_key)

            keep_dirs = {Path(t["trial_dir"]).resolve() for t in trials[:top_k] if _trial_sort_key(t)[0] == 0}
            for trial in trials:
                trial["retained"] = Path(trial["trial_dir"]).resolve() in keep_dirs
                if not trial["retained"]:
                    _rmtree_if_exists(Path(trial["trial_dir"]) / "dpo")

            registry_path.write_text(json.dumps({"top_k": top_k, "trials": trials}, indent=2), encoding="utf-8")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _subprocess_env() -> dict[str, str]:
    """Return an env that disables wandb in the child without leaking parent run state."""
    env = {**os.environ, "WANDB_MODE": "disabled"}
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        env.pop(key, None)
    return env


def execute_sweep_trial(
    *,
    repo_root: Path,
    base_dir: Path,
    driver: argparse.Namespace,
    eval_argv: list[str],
    run_id: str,
    learning_rate: float,
    beta: float,
    lora_r: int,
    run_subprocess: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run DPO then eval; update the trial registry; return the summary ``results`` dict."""
    sweep_root = base_dir / "sweep"
    trial_dir = sweep_root / run_id
    dpo_out = trial_dir / "dpo"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = _subprocess_env()

    dpo_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_dpo.py"),
        "--model-id", driver.model_id,
        "--datasets-dir-a", str(base_dir / "train_a"),
        "--datasets-dir-b", str(base_dir / "train_b"),
        "--output-dir", str(dpo_out),
        "--reward", driver.dpo_reward,
        "--learning-rate", str(learning_rate),
        "--beta", str(beta),
        "--n-epochs", str(driver.n_epochs),
        "--lora-r", str(lora_r),
        "--log-level", driver.log_level,
    ]  # fmt: skip
    if driver.eval_dataset_split is not None:
        dpo_cmd.extend(["--eval-dataset-split", str(driver.eval_dataset_split)])

    LOGGER.info("Trial %s: DPO lr=%s beta=%s lora_r=%s", run_id, learning_rate, beta, lora_r)
    run_subprocess(dpo_cmd, check=True, cwd=str(repo_root), env=env)

    pipeline_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_pipeline.py"),
        *eval_argv,
        "--model-id-generation", str(dpo_out / "best_model"),
        "--output-datasets-path", str(trial_dir),
        "--no-create-output-datasets-subdirectory",
        "--reward-metric", driver.dpo_reward,
        "--log-level", driver.log_level,
    ]  # fmt: skip
    run_subprocess(pipeline_cmd, check=True, cwd=str(repo_root), env=env)

    results = _read_summary_results(trial_dir / "run_summary.json")
    score = _results_score(results)
    _update_trial_registry(
        sweep_root,
        {
            "run_id": run_id,
            "score": score if math.isfinite(score) else None,
            "trial_dir": str(trial_dir),
            "learning_rate": learning_rate,
            "beta": beta,
            "lora_r": lora_r,
        },
    )
    return results


def run_sweep(
    *,
    repo_root: Path,
    driver: argparse.Namespace,
    eval_argv: list[str],
    deadline_monotonic: float | None,
) -> None:
    """Create the W&B sweep and run the agent until the time limit or exhaustion."""
    _validate_ranges(
        driver.sweep_dpo_learning_rate_min,
        driver.sweep_dpo_learning_rate_max,
        driver.sweep_dpo_beta_min,
        driver.sweep_dpo_beta_max,
    )

    sweep_config: dict[str, Any] = {
        "method": "random",
        "metric": {"name": "reward_composite", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": driver.sweep_dpo_learning_rate_min,
                "max": driver.sweep_dpo_learning_rate_max,
            },
            "beta": {
                "distribution": "log_uniform_values",
                "min": driver.sweep_dpo_beta_min,
                "max": driver.sweep_dpo_beta_max,
            },
            "lora_r": {"values": list(driver.sweep_lora_r)},
        },
    }
    sweep_id = wandb.sweep(sweep_config, entity=driver.wandb_entity, project=driver.wandb_project)

    base_dir = driver.base_dir.resolve()
    (base_dir / "sweep").mkdir(parents=True, exist_ok=True)

    def trial_fn() -> None:
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            LOGGER.info("Sweep time limit reached; not starting a new trial.")
            _stop_sweep_via_cli(sweep_id, driver.wandb_entity, driver.wandb_project)
            os._exit(0)

        run = wandb.init()
        try:
            lora_r_trial = int(wandb.config["lora_r"])
            results = execute_sweep_trial(
                repo_root=repo_root,
                base_dir=base_dir,
                driver=driver,
                eval_argv=eval_argv,
                run_id=str(run.id),
                learning_rate=float(wandb.config["learning_rate"]),
                beta=float(wandb.config["beta"]),
                lora_r=lora_r_trial,
            )
            if results:
                wandb.log(results)
        finally:
            _quiet_finish()

    wandb.agent(sweep_id, function=trial_fn, entity=driver.wandb_entity, project=driver.wandb_project)


def main() -> int:
    """CLI entry point."""
    driver_argv, eval_argv = _split_argv(sys.argv[1:])
    driver = _parse_driver_args(driver_argv)
    logging.basicConfig(
        level=getattr(logging, driver.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    repo_root = (driver.repo_root or Path(__file__).resolve().parent.parent).resolve()
    deadline = time.monotonic() + 60.0 * float(driver.sweep_time_limit_minutes)
    run_sweep(repo_root=repo_root, driver=driver, eval_argv=eval_argv, deadline_monotonic=deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
