"""Optimization producer protocol, recipe, and provenance contracts.

This module defines the *optimization producer* boundary. An optimization backend
consumes a trained checkpoint and produces a new, separately-identified artifact: a
quantized, pruned, distilled, or otherwise optimized checkpoint. It never trains, and
it never mutates its source.

The boundary mirrors the existing trainer boundary in
:mod:`opengrad.training.protocol`:

    trained checkpoint -> optimization recipe -> optimized checkpoint
        -> existing inference backend -> existing benchmark/evaluation system

Optimization changes artifacts; the existing inference backend executes them and the
existing benchmark/evaluation system measures them. No optimization backend is a
required dependency: this protocol, the recipe, the result contract, and the
deterministic mock all import and work with every optional optimization library absent.

Nothing in this module runs, installs, or schedules an optimization. A backend that
cannot execute records ``ExecutionStatus.NOT_ATTEMPTED`` rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from opengrad.optimization.capabilities import CapabilityMatrix, CapabilityProbe


class OptimizationTechnique(str, Enum):
    """Optimization techniques the producer layer can name.

    Naming a technique is not a claim that any backend supports it; support is
    established separately, per model, by :mod:`opengrad.optimization.capabilities`.
    """

    PTQ = "ptq"
    QAT = "qat"
    QAD = "qad"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
    SPARSITY = "sparsity"
    SPECULATIVE_DECODING = "speculative_decoding"


class ExecutionStatus(str, Enum):
    """How far an optimization attempt actually got.

    Only ``EXECUTED`` is evidence. ``DRY_RUN`` and ``MOCK`` are deterministic CPU
    plumbing, and ``NOT_ATTEMPTED`` means exactly that.
    """

    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    DRY_RUN = "DRY_RUN"
    MOCK = "MOCK"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a mapping deterministically for hashing.

    Sorted keys, no insignificant whitespace, ASCII only: the same logical content
    always produces the same bytes regardless of dict insertion order or locale.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deterministic_hash(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical serialization of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hash_directory(path: Path) -> str:
    """SHA-256 over a directory tree's sorted relative paths and file bytes.

    Deterministic for a deterministic tree: paths are sorted and each is length-safe by
    a NUL separator, so two trees with identical content hash identically regardless of
    filesystem ordering. Used to identify an exported artifact without reading weights
    back into memory.
    """
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class OptimizationRecipe:
    """A declarative, deterministically-hashed optimization recipe.

    The recipe is the complete, model-independent description of *what* should be
    produced. Everything the result needs to be reproducible - the technique, the
    target format, the calibration identity, and any backend parameters - lives here so
    that :attr:`recipe_hash` pins the intent independently of the backend that runs it.
    """

    technique: str
    format: str | None = None
    calibration_dataset_id: str | None = None
    calibration_sample_count: int | None = None
    calibration_seed: int | None = None
    export_format: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique": self.technique,
            "format": self.format,
            "calibration_dataset_id": self.calibration_dataset_id,
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_seed": self.calibration_seed,
            "export_format": self.export_format,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationRecipe:
        return cls(
            technique=str(data["technique"]),
            format=data.get("format"),
            calibration_dataset_id=data.get("calibration_dataset_id"),
            calibration_sample_count=(
                int(data["calibration_sample_count"])
                if data.get("calibration_sample_count") is not None
                else None
            ),
            calibration_seed=(
                int(data["calibration_seed"]) if data.get("calibration_seed") is not None else None
            ),
            export_format=data.get("export_format"),
            parameters=dict(data.get("parameters") or {}),
        )

    @property
    def recipe_hash(self) -> str:
        """Deterministic SHA-256 pin over the recipe's canonical serialization."""
        return deterministic_hash(self.to_dict())


@dataclass(frozen=True)
class SourceCheckpoint:
    """The trained checkpoint an optimization consumes, and its lineage.

    A backend must treat this as read-only. ``path`` is what the overwrite guard
    protects; ``checkpoint_id`` and ``hash`` are what the result records as provenance.
    """

    checkpoint_id: str
    path: str
    model_id: str
    model_revision: str | None = None
    hash: str | None = None
    experiment_id: str = ""
    lineage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": self.path,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hash": self.hash,
            "experiment_id": self.experiment_id,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceCheckpoint:
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            path=str(data["path"]),
            model_id=str(data["model_id"]),
            model_revision=data.get("model_revision"),
            hash=data.get("hash"),
            experiment_id=str(data.get("experiment_id", "")),
            lineage=tuple(str(item) for item in (data.get("lineage") or [])),
        )


@dataclass(frozen=True)
class OptimizationResult:
    """Typed, frozen optimization result carrying full reproducibility provenance.

    Every field is either a primitive or a small typed structure so the result
    round-trips through :meth:`to_dict` / :meth:`from_dict` without loss. ``evidence``
    is ``False`` for any mock or dry-run, and ``execution_status`` is
    ``NOT_ATTEMPTED`` until a backend actually runs.
    """

    optimization_id: str
    source_checkpoint_id: str
    source_checkpoint_hash: str | None
    source_experiment_id: str
    source_experiment_lineage: tuple[str, ...]
    backend: str
    backend_version: str | None
    technique: str
    recipe_hash: str
    recipe: OptimizationRecipe
    calibration_dataset_id: str | None
    calibration_dataset_fingerprint: str | None
    calibration_sample_count: int | None
    calibration_seed: int | None
    quantization_format: str | None
    hardware: dict[str, Any] = field(default_factory=dict)
    cuda_version: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    export_format: str | None = None
    output_artifact_path: str | None = None
    output_artifact_hash: str | None = None
    runtime_used: str | None = None
    benchmark_ids: tuple[str, ...] = ()
    capability_deltas: dict[str, float] = field(default_factory=dict)
    efficiency_deltas: dict[str, float] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()
    execution_status: str = ExecutionStatus.NOT_ATTEMPTED.value
    evidence: bool = False
    created_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "source_experiment_id": self.source_experiment_id,
            "source_experiment_lineage": list(self.source_experiment_lineage),
            "backend": self.backend,
            "backend_version": self.backend_version,
            "technique": self.technique,
            "recipe_hash": self.recipe_hash,
            "recipe": self.recipe.to_dict(),
            "calibration_dataset_id": self.calibration_dataset_id,
            "calibration_dataset_fingerprint": self.calibration_dataset_fingerprint,
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_seed": self.calibration_seed,
            "quantization_format": self.quantization_format,
            "hardware": self.hardware,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "export_format": self.export_format,
            "output_artifact_path": self.output_artifact_path,
            "output_artifact_hash": self.output_artifact_hash,
            "runtime_used": self.runtime_used,
            "benchmark_ids": list(self.benchmark_ids),
            "capability_deltas": self.capability_deltas,
            "efficiency_deltas": self.efficiency_deltas,
            "failures": list(self.failures),
            "unsupported_features": list(self.unsupported_features),
            "execution_status": self.execution_status,
            "evidence": self.evidence,
            "created_timestamp": self.created_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationResult:
        return cls(
            optimization_id=str(data["optimization_id"]),
            source_checkpoint_id=str(data["source_checkpoint_id"]),
            source_checkpoint_hash=data.get("source_checkpoint_hash"),
            source_experiment_id=str(data.get("source_experiment_id", "")),
            source_experiment_lineage=tuple(
                str(item) for item in (data.get("source_experiment_lineage") or [])
            ),
            backend=str(data["backend"]),
            backend_version=data.get("backend_version"),
            technique=str(data["technique"]),
            recipe_hash=str(data["recipe_hash"]),
            recipe=OptimizationRecipe.from_dict(dict(data["recipe"])),
            calibration_dataset_id=data.get("calibration_dataset_id"),
            calibration_dataset_fingerprint=data.get("calibration_dataset_fingerprint"),
            calibration_sample_count=(
                int(data["calibration_sample_count"])
                if data.get("calibration_sample_count") is not None
                else None
            ),
            calibration_seed=(
                int(data["calibration_seed"]) if data.get("calibration_seed") is not None else None
            ),
            quantization_format=data.get("quantization_format"),
            hardware=dict(data.get("hardware") or {}),
            cuda_version=data.get("cuda_version"),
            torch_version=data.get("torch_version"),
            transformers_version=data.get("transformers_version"),
            export_format=data.get("export_format"),
            output_artifact_path=data.get("output_artifact_path"),
            output_artifact_hash=data.get("output_artifact_hash"),
            runtime_used=data.get("runtime_used"),
            benchmark_ids=tuple(str(item) for item in (data.get("benchmark_ids") or [])),
            capability_deltas={
                str(k): float(v) for k, v in (data.get("capability_deltas") or {}).items()
            },
            efficiency_deltas={
                str(k): float(v) for k, v in (data.get("efficiency_deltas") or {}).items()
            },
            failures=tuple(str(item) for item in (data.get("failures") or [])),
            unsupported_features=tuple(
                str(item) for item in (data.get("unsupported_features") or [])
            ),
            execution_status=str(data.get("execution_status", ExecutionStatus.NOT_ATTEMPTED.value)),
            evidence=bool(data.get("evidence", False)),
            created_timestamp=str(data.get("created_timestamp", "")),
        )


_KNOWN_TECHNIQUES = frozenset(
    {"ptq", "qat", "qad", "distillation", "pruning", "sparsity", "speculative_decoding"}
)


def validate_recipe_shape(recipe: OptimizationRecipe) -> None:
    """Backend-independent recipe validation shared by every optimization backend.

    This is deliberately pure: it never imports or runs a backend, so a recipe can be
    checked on a CPU-only host. Whether the *backend* can honour a valid recipe is a
    separate, per-model question answered by capability discovery.
    """
    if recipe.technique not in _KNOWN_TECHNIQUES:
        raise ValueError(
            f"unknown optimization technique {recipe.technique!r}; expected one of "
            f"{', '.join(sorted(_KNOWN_TECHNIQUES))}"
        )
    if recipe.technique in {"ptq", "qat", "qad"}:
        if not recipe.format:
            raise ValueError(f"technique {recipe.technique!r} requires a target format")
        if not recipe.calibration_dataset_id:
            raise ValueError(
                f"technique {recipe.technique!r} requires a calibration dataset identity"
            )
        if recipe.calibration_sample_count is None or recipe.calibration_sample_count <= 0:
            raise ValueError(
                f"technique {recipe.technique!r} requires a positive calibration sample count"
            )


def build_optimization_result(
    *,
    source: SourceCheckpoint,
    backend: str,
    backend_version: str | None,
    recipe: OptimizationRecipe,
    execution_status: str,
    evidence: bool,
    output_artifact_path: str | None = None,
    output_artifact_hash: str | None = None,
    runtime_used: str | None = None,
    benchmark_ids: list[str] | None = None,
    hardware: dict[str, Any] | None = None,
    cuda_version: str | None = None,
    torch_version: str | None = None,
    transformers_version: str | None = None,
    export_format: str | None = None,
    failures: list[str] | None = None,
    unsupported_features: list[str] | None = None,
    capability_deltas: dict[str, float] | None = None,
    efficiency_deltas: dict[str, float] | None = None,
) -> OptimizationResult:
    """Assemble a fully-populated result so every backend carries identical provenance.

    Centralizing the assembly is what makes "provenance completeness" checkable: a
    backend cannot quietly omit a field, because it does not build the result itself.
    """
    optimization_id = (
        f"{source.checkpoint_id}::{recipe.technique}:{recipe.format or 'none'}"
        f"::{backend}::{recipe.recipe_hash[:12]}"
    )
    return OptimizationResult(
        optimization_id=optimization_id,
        source_checkpoint_id=source.checkpoint_id,
        source_checkpoint_hash=source.hash,
        source_experiment_id=source.experiment_id,
        source_experiment_lineage=source.lineage,
        backend=backend,
        backend_version=backend_version,
        technique=recipe.technique,
        recipe_hash=recipe.recipe_hash,
        recipe=recipe,
        calibration_dataset_id=recipe.calibration_dataset_id,
        calibration_dataset_fingerprint=recipe.parameters.get("calibration_dataset_fingerprint"),
        calibration_sample_count=recipe.calibration_sample_count,
        calibration_seed=recipe.calibration_seed,
        quantization_format=recipe.format,
        hardware=dict(hardware or {}),
        cuda_version=cuda_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
        export_format=export_format if export_format is not None else recipe.export_format,
        output_artifact_path=output_artifact_path,
        output_artifact_hash=output_artifact_hash,
        runtime_used=runtime_used,
        benchmark_ids=tuple(benchmark_ids or ()),
        capability_deltas=dict(capability_deltas or {}),
        efficiency_deltas=dict(efficiency_deltas or {}),
        failures=tuple(failures or ()),
        unsupported_features=tuple(unsupported_features or ()),
        execution_status=execution_status,
        evidence=evidence,
    )


def ensure_distinct_output(source_path: str | Path, output_path: str | Path) -> None:
    """Refuse an optimization output that could overwrite its source checkpoint.

    A backend must never write into, over, or above the checkpoint it reads. This is a
    fail-closed guard: it raises rather than silently picking a different location,
    because a silent relocation would hide a caller's containment mistake.
    """
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == source:
        raise ValueError(f"optimization output must not overwrite its source checkpoint: {source}")
    if output.is_relative_to(source):
        raise ValueError(
            f"optimization output {output} is inside the source checkpoint {source}; "
            "writing there could overwrite source weights"
        )
    if source.is_relative_to(output):
        raise ValueError(
            f"the source checkpoint {source} is inside the optimization output {output}; "
            "refusing to write over the directory that contains the source"
        )


class OptimizationBackend(Protocol):
    """Stable interface shared by all optimization producers.

    Mirrors :class:`opengrad.training.protocol.TrainerBackend`: a name, a capability
    probe that discovers rather than assumes, recipe validation, an ``optimize`` call,
    and an ``export_artifact`` step. Implementations must be lazy about optional
    dependencies so importing this module never imports an optimization library.
    """

    name: str

    def capabilities(
        self, *, model_id: str, revision: str | None = None
    ) -> list[CapabilityProbe]: ...

    def capability_matrix(
        self, *, model_id: str, revision: str | None = None
    ) -> CapabilityMatrix: ...

    def validate_recipe(
        self,
        recipe: OptimizationRecipe,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None: ...

    def optimize(
        self,
        source_checkpoint: SourceCheckpoint,
        recipe: OptimizationRecipe,
        output_dir: Path,
        *,
        dry_run: bool = False,
        runtime: str | None = None,
        benchmark_ids: list[str] | None = None,
    ) -> OptimizationResult: ...

    def export_artifact(
        self,
        result: OptimizationResult,
        destination: Path,
        *,
        export_format: str,
    ) -> Path: ...
