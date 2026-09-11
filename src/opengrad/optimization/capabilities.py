"""Per-model, per-technique capability discovery for the optimization producer layer.

The rule this module exists to enforce: **discover, never assume**. Support for an
optimization technique is a property of an exact model (and revision), a technique, a
backend, and the hardware it runs on. None of those may be inferred from a sibling
model size, from a model family, or from a library-level default.

Concretely, the probe for ``Qwen/Qwen3.5-2B`` must not be filled in because some other
``Qwen3.5`` size is believed to work, and a technique ModelOpt documents at the library
level is not thereby supported for a specific checkpoint. When a question cannot be
settled without executing the backend, the answer is :data:`CapabilityStatus.UNKNOWN`.

This module only *records* and *looks up* capability facts. It never imports a backend,
never runs one, and never downloads a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityStatus(str, Enum):
    """The only four answers a probe may give."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TechniqueSpec:
    """Static facts about a technique that do not depend on a model.

    ``requires_gpu`` and ``upstream_documented`` describe the *technique*, not any
    model's support for it. They explain why an answer is unknown; they are never used
    to promote a status.
    """

    key: str
    label: str
    requires_gpu: bool
    upstream_documented: bool
    upstream_note: str


TECHNIQUE_SPECS: tuple[TechniqueSpec, ...] = (
    TechniqueSpec(
        key="fp8_ptq",
        label="FP8 post-training quantization",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="ModelOpt documents FP8 PTQ (mtq.quantize) but does not declare "
        "which concrete checkpoints it supports without running.",
    ),
    TechniqueSpec(
        key="nvfp4",
        label="NVFP4 quantization",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="NVFP4 is documented upstream and is arch-gated to Blackwell-class "
        "devices; per-model support is not declared without running.",
    ),
    TechniqueSpec(
        key="ptq_other",
        label="Other PTQ formats (INT8/INT4/AWQ/GPTQ)",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="Several PTQ formats are documented upstream; the applicable format "
        "for a given checkpoint is established by execution, not by list membership.",
    ),
    TechniqueSpec(
        key="qat",
        label="Quantization-aware training",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="QAT is a training-time method; it is out of scope for a producer "
        "layer that must not train and has not been attempted here.",
    ),
    TechniqueSpec(
        key="qad",
        label="Quantization-aware distillation",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="QAD combines quantization with distillation; neither has been "
        "attempted here and no per-model fact is recorded.",
    ),
    TechniqueSpec(
        key="distillation",
        label="Distillation-based optimization",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="ModelOpt exposes distillation entry points; no per-model support "
        "is declared without running, and OpenGrad already has a separate OPD path.",
    ),
    TechniqueSpec(
        key="pruning_sparsity",
        label="Pruning and sparsity",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="Pruning/sparsity is documented upstream; the reachable sparsity "
        "for a specific checkpoint is established by execution.",
    ),
    TechniqueSpec(
        key="speculative_decoding",
        label="Speculative-decoding training methods (EAGLE/DFlash-style)",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="Speculative-decoding training methods are exposed upstream only "
        "where the installed revision provides them; none has been attempted here.",
    ),
    TechniqueSpec(
        key="hf_export",
        label="Hugging Face export",
        requires_gpu=False,
        upstream_documented=True,
        upstream_note="ModelOpt documents an Hugging Face exporter; whether a given "
        "optimized checkpoint round-trips is verified by executing the export.",
    ),
    TechniqueSpec(
        key="vllm_deployment",
        label="vLLM-compatible deployment",
        requires_gpu=True,
        upstream_documented=True,
        upstream_note="Deployment compatibility with vLLM is a runtime property of the "
        "produced artifact and must be measured by the existing inference backend.",
    ),
)

TECHNIQUE_KEYS: tuple[str, ...] = tuple(spec.key for spec in TECHNIQUE_SPECS)

#: The single source of truth for what is actually established, keyed by the exact
#: ``(model_id, technique)`` pair. It is empty by design: no capability for any model
#: has been verified in this repository, and an empty table returns UNKNOWN rather than
#: inventing an answer. A fact is added only with the evidence that established it.
DEFAULT_MODEL_KNOWLEDGE: Mapping[tuple[str, str], tuple[str, str]] = {}


def technique_spec(key: str) -> TechniqueSpec:
    for spec in TECHNIQUE_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown optimization technique: {key!r}")


@dataclass(frozen=True)
class CapabilityProbe:
    """One (model, technique) capability answer with its evidence and backend revision."""

    technique: str
    model_id: str
    model_revision: str | None
    backend: str
    backend_revision: str | None
    status: str
    evidence: str
    requires_gpu: bool = False
    upstream_documented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique": self.technique,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "status": self.status,
            "evidence": self.evidence,
            "requires_gpu": self.requires_gpu,
            "upstream_documented": self.upstream_documented,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityProbe:
        return cls(
            technique=str(data["technique"]),
            model_id=str(data["model_id"]),
            model_revision=data.get("model_revision"),
            backend=str(data["backend"]),
            backend_revision=data.get("backend_revision"),
            status=str(data["status"]),
            evidence=str(data.get("evidence", "")),
            requires_gpu=bool(data.get("requires_gpu", False)),
            upstream_documented=bool(data.get("upstream_documented", False)),
        )


@dataclass(frozen=True)
class CapabilityMatrix:
    """The full set of probes for one model against one backend."""

    model_id: str
    model_revision: str | None
    backend: str
    backend_revision: str | None
    probes: tuple[CapabilityProbe, ...] = field(default_factory=tuple)
    note: str = ""

    def status_of(self, technique: str) -> str:
        for probe in self.probes:
            if probe.technique == technique:
                return probe.status
        raise KeyError(f"no probe recorded for technique: {technique!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "note": self.note,
            "probes": [probe.to_dict() for probe in self.probes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityMatrix:
        return cls(
            model_id=str(data["model_id"]),
            model_revision=data.get("model_revision"),
            backend=str(data["backend"]),
            backend_revision=data.get("backend_revision"),
            probes=tuple(CapabilityProbe.from_dict(row) for row in (data.get("probes") or [])),
            note=str(data.get("note", "")),
        )


def _unknown_evidence(
    spec: TechniqueSpec,
    *,
    model_id: str,
    model_revision: str | None,
    backend: str,
    backend_revision: str | None,
    hardware: Mapping[str, Any] | None,
) -> str:
    revision = backend_revision if backend_revision is not None else "absent"
    parts = [
        f"no per-model capability fact is recorded for model_id={model_id!r}"
        + (f" revision={model_revision!r}" if model_revision else "")
        + f" technique={spec.key!r}",
        f"backend={backend!r} revision={revision!r}",
        (
            "support is not inferred from another model size or family and not from a "
            "library-level default; it can only be established by executing the backend"
        ),
    ]
    if spec.upstream_documented:
        parts.append(f"upstream: {spec.upstream_note}")
    if hardware is not None:
        gpu = bool(hardware.get("gpu_available"))
        parts.append(
            "hardware: accelerator available"
            if gpu
            else "hardware: no accelerator detected, so execution is impossible here"
        )
    return "; ".join(parts)


def probe_capabilities(
    model_id: str,
    *,
    model_revision: str | None = None,
    backend: str = "modelopt",
    backend_revision: str | None = None,
    knowledge: Mapping[tuple[str, str], tuple[str, str]] | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> CapabilityMatrix:
    """Build the capability matrix for one exact model against one backend.

    ``knowledge`` defaults to :data:`DEFAULT_MODEL_KNOWLEDGE` (empty). A fact is looked
    up by the exact ``(model_id, technique)`` key: there is deliberately no family or
    size fallback, so a fact recorded for one model can never leak onto another.
    """
    table = DEFAULT_MODEL_KNOWLEDGE if knowledge is None else knowledge
    probes: list[CapabilityProbe] = []
    for spec in TECHNIQUE_SPECS:
        recorded = table.get((model_id, spec.key))
        if recorded is not None:
            status, evidence = recorded
        else:
            status = CapabilityStatus.UNKNOWN.value
            evidence = _unknown_evidence(
                spec,
                model_id=model_id,
                model_revision=model_revision,
                backend=backend,
                backend_revision=backend_revision,
                hardware=hardware,
            )
        probes.append(
            CapabilityProbe(
                technique=spec.key,
                model_id=model_id,
                model_revision=model_revision,
                backend=backend,
                backend_revision=backend_revision,
                status=status,
                evidence=evidence,
                requires_gpu=spec.requires_gpu,
                upstream_documented=spec.upstream_documented,
            )
        )
    return CapabilityMatrix(
        model_id=model_id,
        model_revision=model_revision,
        backend=backend,
        backend_revision=backend_revision,
        probes=tuple(probes),
        note=(
            "UNKNOWN means the question cannot be settled without running the backend. "
            "It is never a synonym for UNSUPPORTED and never a synonym for SUPPORTED."
        ),
    )
