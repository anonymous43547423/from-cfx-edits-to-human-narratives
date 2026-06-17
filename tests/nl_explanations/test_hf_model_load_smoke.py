# ruff: noqa: S101
"""Opt-in GPU integration tests: real Hugging Face model load attempts (isolated subprocesses)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = Path(__file__).with_name("hf_model_load_smoke_worker.py")

_HF_MODEL_LOAD_SMOKE_MODEL_IDS = (
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "openai/gpt-oss-20b",
)


def _run_load_subprocess(model_id: str) -> subprocess.CompletedProcess[str]:
    """Run the load helper in a fresh process; return completed process with stdout/stderr."""
    return subprocess.run(  # noqa: S603 — fixed interpreter path and smoke worker; model_id from test table only.
        [sys.executable, str(_WORKER), model_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
        env=os.environ.copy(),
    )


@pytest.fixture(autouse=True)
def _require_cuda_and_opt_in() -> None:
    """Skip unless opt-in env is set and CUDA is available."""
    if os.environ.get("RUN_HF_MODEL_LOAD_SMOKE") != "1":
        pytest.skip("Set RUN_HF_MODEL_LOAD_SMOKE=1 to run HF model load smoke tests.")
    if not torch.cuda.is_available():
        pytest.skip("HF model load smoke tests require CUDA.")


@pytest.mark.gpu
@pytest.mark.hf_integration
@pytest.mark.parametrize("model_id", _HF_MODEL_LOAD_SMOKE_MODEL_IDS)
def test_hf_model_load_smoke_subprocess(model_id: str) -> None:
    """Each model loads in an isolated process; exit code 0 means load succeeded."""
    proc = _run_load_subprocess(model_id)
    detail = f"model_id={model_id!r} rc={proc.returncode}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert proc.returncode == 0, detail
