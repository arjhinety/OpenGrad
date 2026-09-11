"""Optional optimization producer layer (NVIDIA ModelOpt integration).

Isolated boundary: a trained checkpoint goes in, an optimized checkpoint comes out. The
existing inference backend executes the result and the existing benchmark/evaluation
system measures it. Nothing here trains, and nothing here is a required dependency.

Importing this package never imports an optimization backend. ``modelopt`` is resolved
lazily inside :mod:`opengrad.optimization.modelopt_backend`, so a host without NVIDIA
packages can still import the protocol, discover capabilities (which report ``UNKNOWN``),
and run the deterministic mock.
"""

from opengrad.optimization.capabilities import (
    DEFAULT_MODEL_KNOWLEDGE,
    TECHNIQUE_SPECS,
    CapabilityMatrix,
    CapabilityProbe,
    CapabilityStatus,
    TechniqueSpec,
    probe_capabilities,
    technique_spec,
)
from opengrad.optimization.mock_backend import MOCK_BACKEND_VERSION, MockOptimizationBackend
from opengrad.optimization.modelopt_backend import OPTIMIZATION_EXTRA, ModelOptBackend
from opengrad.optimization.protocol import (
    ExecutionStatus,
    OptimizationBackend,
    OptimizationRecipe,
    OptimizationResult,
    OptimizationTechnique,
    SourceCheckpoint,
    build_optimization_result,
    ensure_distinct_output,
    validate_recipe_shape,
)

__all__ = [
    "DEFAULT_MODEL_KNOWLEDGE",
    "MOCK_BACKEND_VERSION",
    "OPTIMIZATION_EXTRA",
    "TECHNIQUE_SPECS",
    "CapabilityMatrix",
    "CapabilityProbe",
    "CapabilityStatus",
    "ExecutionStatus",
    "MockOptimizationBackend",
    "ModelOptBackend",
    "OptimizationBackend",
    "OptimizationRecipe",
    "OptimizationResult",
    "OptimizationTechnique",
    "SourceCheckpoint",
    "TechniqueSpec",
    "build_optimization_result",
    "ensure_distinct_output",
    "probe_capabilities",
    "select_optimization_backend",
    "technique_spec",
    "validate_recipe_shape",
]


def select_optimization_backend(name: str) -> OptimizationBackend | None:
    """Hard-coded optimization-backend selector.

    Mirrors ``_select_trainer`` in :mod:`opengrad.agent_cli`: a small, explicit mapping
    from a configured backend name to an implementation, returning ``None`` for anything
    unrecognized. Constructing a backend never imports its optional dependency, so this
    returns a :class:`ModelOptBackend` even when ModelOpt is not installed; the missing
    dependency is only reported when a mutating call is made.
    """
    if name in {"mock", "deterministic"}:
        return MockOptimizationBackend()
    if name == "modelopt":
        return ModelOptBackend()
    return None
