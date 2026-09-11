"""Deterministic CPU/mock optimization backend.

This backend exists so the optimization *producer boundary* can be exercised end to end
on a CPU-only host with no GPU, no ModelOpt, and no model weights. It is pure plumbing:
it writes a small, deterministic provenance artifact that stands in for an optimized
checkpoint and reports ``ExecutionStatus.MOCK`` with ``evidence=False``.

It must never be mistaken for a capability claim. Its ``capabilities()`` answers
``SUPPORTED`` only in the narrow sense that the mock can deterministically simulate a
recipe; the evidence string says so explicitly, and the results carry no metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opengrad.optimization.capabilities import (
    TECHNIQUE_SPECS,
    CapabilityMatrix,
    CapabilityProbe,
)
from opengrad.optimization.protocol import (
    ExecutionStatus,
    OptimizationRecipe,
    OptimizationResult,
    SourceCheckpoint,
    build_optimization_result,
    ensure_distinct_output,
    hash_directory,
    validate_recipe_shape,
)

MOCK_BACKEND_VERSION = "mock-1"


class MockOptimizationBackend:
    """Deterministic, GPU-free optimization producer for tests and dry-runs."""

    name = "mock"

    @property
    def version(self) -> str:
        return MOCK_BACKEND_VERSION

    def capabilities(self, *, model_id: str, revision: str | None = None) -> list[CapabilityProbe]:
        return list(self.capability_matrix(model_id=model_id, revision=revision).probes)

    def capability_matrix(self, *, model_id: str, revision: str | None = None) -> CapabilityMatrix:
        probes = tuple(
            CapabilityProbe(
                technique=spec.key,
                model_id=model_id,
                model_revision=revision,
                backend=self.name,
                backend_revision=MOCK_BACKEND_VERSION,
                status="SUPPORTED",
                evidence=(
                    f"mock backend: deterministically simulates {spec.key!r} for tests only; "
                    "this is not a real capability claim and not evidence of upstream support"
                ),
                requires_gpu=spec.requires_gpu,
                upstream_documented=spec.upstream_documented,
            )
            for spec in TECHNIQUE_SPECS
        )
        return CapabilityMatrix(
            model_id=model_id,
            model_revision=revision,
            backend=self.name,
            backend_revision=MOCK_BACKEND_VERSION,
            probes=probes,
            note="MOCK: synthetic capabilities for CPU plumbing. Never evidence.",
        )

    def validate_recipe(
        self,
        recipe: OptimizationRecipe,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        validate_recipe_shape(recipe)

    def optimize(
        self,
        source_checkpoint: SourceCheckpoint,
        recipe: OptimizationRecipe,
        output_dir: Path,
        *,
        dry_run: bool = False,
        runtime: str | None = None,
        benchmark_ids: list[str] | None = None,
    ) -> OptimizationResult:
        self.validate_recipe(recipe, model_id=source_checkpoint.model_id)
        output_dir = Path(output_dir)
        fmt = recipe.format or recipe.technique
        artifact_dir = output_dir / f"{source_checkpoint.checkpoint_id}--{recipe.technique}--{fmt}"
        # Never write over the checkpoint being optimized, even in a mock.
        ensure_distinct_output(source_checkpoint.path, artifact_dir)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "MOCK_OPTIMIZED_ARTIFACT",
            "backend": self.name,
            "backend_version": MOCK_BACKEND_VERSION,
            "source_checkpoint_id": source_checkpoint.checkpoint_id,
            "source_checkpoint_hash": source_checkpoint.hash,
            "source_experiment_id": source_checkpoint.experiment_id,
            "source_experiment_lineage": list(source_checkpoint.lineage),
            "recipe": recipe.to_dict(),
            "recipe_hash": recipe.recipe_hash,
            "evidence": False,
        }
        artifact_file = artifact_dir / "optimized_artifact.json"
        artifact_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        status = ExecutionStatus.MOCK.value if not dry_run else ExecutionStatus.DRY_RUN.value
        return build_optimization_result(
            source=source_checkpoint,
            backend=self.name,
            backend_version=MOCK_BACKEND_VERSION,
            recipe=recipe,
            execution_status=status,
            evidence=False,
            output_artifact_path=str(artifact_dir),
            output_artifact_hash=hash_directory(artifact_dir),
            runtime_used=runtime,
            benchmark_ids=benchmark_ids,
        )

    def export_artifact(
        self,
        result: OptimizationResult,
        destination: Path,
        *,
        export_format: str,
    ) -> Path:
        if result.output_artifact_path is None:
            raise ValueError("cannot export a result with no output artifact")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "MOCK_EXPORT",
            "export_format": export_format,
            "optimization_id": result.optimization_id,
            "source_artifact": result.output_artifact_path,
            "source_artifact_hash": result.output_artifact_hash,
            "evidence": False,
        }
        (destination / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination
