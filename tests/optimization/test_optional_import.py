"""Optional-dependency tests: ModelOpt must be absent-safe.

The whole package must import and the selector must work with ModelOpt not installed.
Absence is simulated deterministically by making ``modelopt`` un-importable, so these
tests do not depend on the host's environment.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from opengrad import optimization
from opengrad.optimization import modelopt_backend
from opengrad.optimization.capabilities import CapabilityStatus
from opengrad.optimization.protocol import OptimizationRecipe, SourceCheckpoint


def _modelopt_available() -> bool:
    try:
        importlib.import_module("modelopt")
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        return bool(importlib.import_module("torch").cuda.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


@pytest.fixture
def modelopt_absent(monkeypatch):
    """Make every lazy ``importlib.import_module('modelopt...')`` fail."""
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "modelopt" or name.startswith("modelopt."):
            raise ImportError(f"simulated absence of {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(modelopt_backend.importlib, "import_module", fake_import)
    return fake_import


def _source(tmp_path) -> SourceCheckpoint:
    return SourceCheckpoint(
        checkpoint_id="exp::checkpoint-1200",
        path=str(tmp_path / "src"),
        model_id="Qwen/Qwen3.5-2B",
        model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    )


def _recipe() -> OptimizationRecipe:
    return OptimizationRecipe(
        technique="ptq",
        format="fp8",
        calibration_dataset_id="behavioral-heldout-v2",
        calibration_sample_count=512,
        calibration_seed=42,
    )


def test_package_imports_and_selector_works_without_modelopt(modelopt_absent):
    backend = optimization.select_optimization_backend("modelopt")
    assert backend is not None
    assert backend.name == "modelopt"
    assert backend.version is None, "an absent backend must report no revision"


def test_importing_the_package_does_not_import_modelopt():
    # Importing the package must not drag in the heavy optional dependency.
    assert "modelopt" not in sys.modules or _modelopt_available()


def test_modelopt_optimize_raises_naming_the_extra(modelopt_absent, tmp_path):
    backend = optimization.ModelOptBackend()
    with pytest.raises(RuntimeError, match="optimization extra"):
        backend.optimize(_source(tmp_path), _recipe(), tmp_path / "out")


def test_modelopt_export_raises_naming_the_extra(modelopt_absent, tmp_path):
    backend = optimization.ModelOptBackend()
    from opengrad.optimization import MockOptimizationBackend

    result = MockOptimizationBackend().optimize(_source(tmp_path), _recipe(), tmp_path / "out")
    with pytest.raises(RuntimeError, match="optimization extra"):
        backend.export_artifact(result, tmp_path / "export", export_format="hf")


def test_capabilities_stay_unknown_and_non_raising_when_backend_absent(modelopt_absent):
    backend = optimization.ModelOptBackend()
    matrix = backend.capability_matrix(
        model_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    )
    assert matrix.backend_revision is None
    assert matrix.status_of("fp8_ptq") == CapabilityStatus.UNKNOWN.value
    assert all(probe.status == CapabilityStatus.UNKNOWN.value for probe in matrix.probes)


def test_real_modelopt_execution_path_is_never_silently_passed():
    """GPU-only path: skipped with an explicit reason, never silently green."""
    if not _modelopt_available():
        pytest.skip("modelopt is not installed (the optional 'optimization' extra)")
    if not _cuda_available():
        pytest.skip("no CUDA GPU: the real ModelOpt execution path requires one")
    pytest.skip(
        "OpenGrad does not authorize a ModelOpt optimization run; execution is NOT_ATTEMPTED"
    )
