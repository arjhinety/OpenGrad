"""Optional NVIDIA ModelOpt backend for the optimization producer layer.

This is the one real optimization implementation, and it is *optional*: nothing here
runs at import time, and nothing outside this module imports ``modelopt``. Importing
:mod:`opengrad.optimization` (or any unrelated OpenGrad command) must succeed with
ModelOpt absent, so every ModelOpt symbol is resolved through a lazy import performed
inside the call that needs it.

When ModelOpt is missing, the mutating calls fail closed with a ``RuntimeError`` that
names the extra to install. Capability discovery still works and reports ``UNKNOWN``:
an absent backend is not evidence of support and not evidence of its absence.

No ModelOpt execution has been performed or validated in this repository. The dispatch
below is written against ModelOpt's documented surface, but every path that would touch
weights or a GPU is marked UNVERIFIED and has not been run.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from opengrad.optimization.capabilities import CapabilityMatrix, CapabilityProbe, probe_capabilities
from opengrad.optimization.protocol import (
    ExecutionStatus,
    OptimizationRecipe,
    OptimizationResult,
    SourceCheckpoint,
    artifact_directory_name,
    build_optimization_result,
    ensure_distinct_output,
    hash_directory,
    validate_recipe_shape,
)

#: The optional-dependency extra that provides ``modelopt``.
OPTIMIZATION_EXTRA = "optimization"

#: Textbook mapping from a recipe's ``format`` to a ModelOpt named config. Only the
#: entries ModelOpt is known to publish are listed; anything else falls back to the
#: convention ``<FORMAT>_DEFAULT_CFG``, and a missing attribute is a hard error rather
#: than a silently-skipped quantization.
_FORMAT_CONFIG_NAMES: dict[str, str] = {
    "fp8": "FP8_DEFAULT_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "int8": "INT8_DEFAULT_CFG",
    "int4": "INT4_DEFAULT_CFG",
    "awq": "INT4_AWQ_CFG",
}


def _import_modelopt() -> Any:
    """Import ModelOpt lazily, or raise naming the extra that provides it."""
    try:
        return importlib.import_module("modelopt")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RuntimeError(
            "the modelopt optimization backend requires the optimization extra: "
            f"pip install '.[{OPTIMIZATION_EXTRA}]'"
        ) from exc


def _modelopt_revision() -> str | None:
    """ModelOpt's installed version, or ``None`` when it is absent.

    Capability discovery must not raise when the backend is missing, so this is the
    non-raising counterpart of :func:`_import_modelopt`.
    """
    try:
        module = importlib.import_module("modelopt")
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _hardware_snapshot() -> dict[str, Any] | None:
    """Best-effort hardware description; never raises (probe catches its own errors)."""
    from opengrad.hardware.probe import probe_hardware

    return probe_hardware().to_dict()


class ModelOptBackend:
    """NVIDIA ModelOpt optimization producer.

    The class is importable and constructible with ModelOpt absent; only its mutating
    methods require the dependency.
    """

    name = "modelopt"

    def _require_modelopt(self) -> Any:
        return _import_modelopt()

    @property
    def version(self) -> str | None:
        return _modelopt_revision()

    def capabilities(self, *, model_id: str, revision: str | None = None) -> list[CapabilityProbe]:
        return list(self.capability_matrix(model_id=model_id, revision=revision).probes)

    def capability_matrix(self, *, model_id: str, revision: str | None = None) -> CapabilityMatrix:
        """Discover the capability matrix for one exact model against this backend.

        Uses the installed ModelOpt revision when present and ``None`` when absent; in
        both cases the answer is ``UNKNOWN`` for any fact that is not explicitly
        recorded for this exact model.
        """
        return probe_capabilities(
            model_id,
            model_revision=revision,
            backend=self.name,
            backend_revision=self.version,
            hardware=_hardware_snapshot(),
        )

    def validate_recipe(
        self,
        recipe: OptimizationRecipe,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        """Validate a recipe's shape. Pure: never imports or runs ModelOpt."""
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
        """Produce an optimized artifact.

        Requiring ModelOpt even for ``dry_run`` keeps this backend honest: a real
        optimization plan needs the backend that would execute it. Deterministic,
        GPU-free plumbing lives in :class:`~opengrad.optimization.mock_backend.MockOptimizationBackend`.
        """
        module = self._require_modelopt()  # raises naming the extra when absent
        self.validate_recipe(recipe, model_id=source_checkpoint.model_id)
        output_dir = Path(output_dir)
        plan_name = _artifact_name(source_checkpoint, recipe)
        output_path = output_dir / plan_name
        ensure_distinct_output(source_checkpoint.path, output_path)

        if dry_run:
            # A plan only: it lists what would be produced and touches no weights.
            plan_path = output_dir / f"{plan_name}.plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                _plan_payload(source_checkpoint, recipe, self.name, self.version),
                encoding="utf-8",
            )
            return build_optimization_result(
                source=source_checkpoint,
                backend=self.name,
                backend_version=self.version,
                recipe=recipe,
                execution_status=ExecutionStatus.DRY_RUN.value,
                evidence=False,
                output_artifact_path=str(plan_path),
                output_artifact_hash=None,
                runtime_used=runtime,
                benchmark_ids=benchmark_ids,
                failures=[],
                unsupported_features=[],
            )

        return self._execute(source_checkpoint, recipe, output_path, runtime, module)

    def _execute(
        self,
        source_checkpoint: SourceCheckpoint,
        recipe: OptimizationRecipe,
        output_path: Path,
        runtime: str | None,
        module: Any,
    ) -> OptimizationResult:
        """Real quantization path. UNVERIFIED: never executed in this repository.

        Written against ModelOpt's documented ``mtq.quantize`` surface. It is deliberately
        never reached here (no GPU, no ModelOpt, and no optimization experiment is
        authorized), so it must not be read as evidence that any of it works.
        """
        mtq = importlib.import_module("modelopt.torch.quantization")
        config_attr = _FORMAT_CONFIG_NAMES.get(
            (recipe.format or "").lower(), f"{(recipe.format or '').upper()}_DEFAULT_CFG"
        )
        quant_config = getattr(mtq, config_attr, None)
        if quant_config is None:
            raise RuntimeError(
                f"the installed modelopt revision does not expose {config_attr!r} for "
                f"format {recipe.format!r}"
            )

        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = transformers.AutoModelForCausalLM.from_pretrained(
            source_checkpoint.path,
            trust_remote_code=False,
            torch_dtype="auto",
        ).to(device)
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            source_checkpoint.path, trust_remote_code=False
        )

        calibration_texts = list(recipe.parameters.get("calibration_texts") or [])
        if not calibration_texts:
            raise RuntimeError(
                "modelopt PTQ requires calibration samples; none were provided in the recipe"
            )

        def forward_loop(inner: Any) -> None:
            for text in calibration_texts:
                batch = tokenizer(text, return_tensors="pt").to(device)
                inner(**batch)

        mtq.quantize(model, quant_config, forward_loop)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return build_optimization_result(
            source=source_checkpoint,
            backend=self.name,
            backend_version=self.version,
            recipe=recipe,
            execution_status=ExecutionStatus.EXECUTED.value,
            evidence=True,
            output_artifact_path=str(output_path),
            output_artifact_hash=hash_directory(output_path),
            runtime_used=runtime,
            hardware=_hardware_snapshot(),
            cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
            torch_version=getattr(torch, "__version__", None),
            transformers_version=getattr(transformers, "__version__", None),
        )

    def export_artifact(
        self,
        result: OptimizationResult,
        destination: Path,
        *,
        export_format: str,
    ) -> Path:
        """Export an optimized artifact for a deployment runtime.

        UNVERIFIED: requires ModelOpt and a produced artifact, neither of which exists
        here. Raises naming the extra when ModelOpt is absent.
        """
        self._require_modelopt()
        if result.output_artifact_path is None:
            raise ValueError("cannot export a result with no output artifact")
        exporter = importlib.import_module("modelopt.torch.export")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        export_hf = getattr(exporter, "export_hf_checkpoint", None)
        if export_hf is None:
            raise RuntimeError(
                "the installed modelopt revision does not expose export_hf_checkpoint"
            )
        export_hf(result.output_artifact_path, export_dir=str(destination))
        return destination


def _artifact_name(source: SourceCheckpoint, recipe: OptimizationRecipe) -> str:
    fmt = recipe.format or recipe.technique
    return artifact_directory_name(source.checkpoint_id, recipe.technique, fmt)


def _plan_payload(
    source: SourceCheckpoint,
    recipe: OptimizationRecipe,
    backend: str,
    backend_version: str | None,
) -> str:
    import json

    payload = {
        "plan": True,
        "backend": backend,
        "backend_version": backend_version,
        "source_checkpoint_id": source.checkpoint_id,
        "source_checkpoint_hash": source.hash,
        "recipe": recipe.to_dict(),
        "recipe_hash": recipe.recipe_hash,
        "execution_status": ExecutionStatus.DRY_RUN.value,
        "evidence": False,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
