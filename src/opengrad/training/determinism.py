"""Seeding and kernel determinism for training runs (Study 002 docs 05 and 14).

`14-HETEROGENEITY-POLICY.md` requires every run to declare one of two modes:

* ``DECLARED_DETERMINISTIC`` -- the run asks torch for deterministic kernels, and a REP-A rerun must
  reproduce its metrics exactly; failing that is a defect.
* ``NON_DETERMINISTIC_KERNEL`` -- the run leaves kernel selection alone, and the REP-A residual is
  its noise floor (`05-SEED-AND-REPRODUCIBILITY-POLICY.md`).

A config declares the mode as ``reproducibility.determinism``. A config that declares nothing is
``UNDECLARED`` and keeps the behaviour every Study 001 run had: torch and CUDA seeded, nothing else.
The readiness gate ``determinism_declared`` blocks an undeclared config unless it is one of the
frozen Study 001 configs (`src/opengrad/readiness.py`).

``DECLARED_DETERMINISTIC`` can fail at the first operation that has no deterministic kernel; torch
then raises rather than silently running a non-deterministic one. Qwen3.5's linear-attention
kernels may be such operations; only a GPU smoke run shows it, and a run that hits one declares
``NON_DETERMINISTIC_KERNEL`` instead.
"""

from __future__ import annotations

import os
import random
from collections.abc import MutableMapping
from typing import Any

DECLARED_DETERMINISTIC = "DECLARED_DETERMINISTIC"
NON_DETERMINISTIC_KERNEL = "NON_DETERMINISTIC_KERNEL"
UNDECLARED = "UNDECLARED"
DETERMINISM_MODES = (DECLARED_DETERMINISTIC, NON_DETERMINISTIC_KERNEL)

#: cuBLAS needs a fixed workspace for reproducible results; torch refuses deterministic mode on
#: CUDA >= 10.2 without it. It must be in the environment before the first cuBLAS handle exists.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def resolve_determinism(experiment: dict[str, Any]) -> str:
    """The run's declared mode, or ``UNDECLARED``. An unknown value is an error, not a default."""
    value = (experiment.get("reproducibility") or {}).get("determinism")
    if value is None:
        return UNDECLARED
    if value not in DETERMINISM_MODES:
        raise ValueError(
            f"reproducibility.determinism must be one of {', '.join(DETERMINISM_MODES)}; "
            f"got {value!r}"
        )
    return str(value)


def seed_everything(seed: int, mode: str, torch: Any) -> list[str]:
    """Seed the randomness sources the mode covers, and return their names for the run record.

    ``UNDECLARED`` seeds torch and CUDA only, exactly as Study 001's trainers did, so an undeclared
    run's behaviour is unchanged. A declared mode also seeds Python's ``random`` and numpy (when
    installed), which doc 05 requires every source to name.
    """
    seeded: list[str] = []
    if mode in DETERMINISM_MODES:
        random.seed(seed)
        seeded.append("python.random")
        try:
            import numpy
        except ImportError:
            pass
        else:
            numpy.random.seed(seed)
            seeded.append("numpy.random")
    elif mode != UNDECLARED:
        raise ValueError(f"unknown determinism mode {mode!r}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seeded += ["torch", "torch.cuda"]
    return seeded


def apply_determinism(
    mode: str, torch: Any, environ: MutableMapping[str, str] | None = None
) -> dict[str, Any]:
    """Configure torch for ``mode`` before any model runs, and return what was set.

    Only ``DECLARED_DETERMINISTIC`` changes anything. ``CUBLAS_WORKSPACE_CONFIG`` is set only when
    the environment does not already choose a value, and the record says which happened.
    """
    env = os.environ if environ is None else environ
    if mode not in (*DETERMINISM_MODES, UNDECLARED):
        raise ValueError(f"unknown determinism mode {mode!r}")
    record: dict[str, Any] = {"mode": mode}
    if mode != DECLARED_DETERMINISTIC:
        return record
    if "CUBLAS_WORKSPACE_CONFIG" not in env:
        env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        record["cublas_workspace_config_set"] = True
    else:
        record["cublas_workspace_config_set"] = False
    record["cublas_workspace_config"] = env["CUBLAS_WORKSPACE_CONFIG"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    record["use_deterministic_algorithms"] = True
    record["cudnn_deterministic"] = True
    record["cudnn_benchmark"] = False
    return record
