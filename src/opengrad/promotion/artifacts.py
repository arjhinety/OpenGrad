"""Structural guards on deployment artifacts, separate from the behavioural gate.

`quantization.py` answers "did this artifact preserve the behaviour". This module answers the
questions that come before that and can invalidate the answer regardless of what the metrics say:
was it quantized from a high-precision source, does it descend from the model it claims to, and is
it allowed to be released at all.

These are deliberately cheap, deterministic checks with no I/O, so they can sit directly in the
quantize and publish paths rather than in a report nobody reruns.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

M1_V2_EXPERIMENT = "m1_dpo_canonical_v2_final_v2"
M1_V2_CHECKPOINT = "dpo-checkpoint-30"

# Precision types a k-quant may legitimately be produced from. Anything else is either already
# quantized or an unknown format, and both are refused rather than guessed about.
HIGH_PRECISION_TYPES = frozenset({"bf16", "f16", "fp16", "f32", "fp32"})

# GGUF/quantized format markers, matched case-insensitively against the artifact name. Requantizing
# one of these compounds two lossy roundings and produces an artifact whose error budget cannot be
# attributed to a single step.
_LOW_BIT = re.compile(
    r"(?:^|[^a-z0-9])(?:q\d+(?:_[01k])?(?:_[sml])?|iq\d+[a-z_]*|8da4w|16a4w|8a8w|int4|int8|nvfp4|mxfp4)"
    r"(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

RELEASABLE_FROM = frozenset({"PTQ_ACCEPTED", "QAD_ACCEPTED"})


class ArtifactLineageError(ValueError):
    """An artifact does not descend from the model it claims to descend from."""


class RequantizationError(ValueError):
    """A quantization was requested from an already-quantized source."""


def is_low_bit_name(name: str) -> bool:
    """Whether an artifact name advertises an already-quantized format."""
    return bool(_LOW_BIT.search(str(name)))


def assert_quantizable_source(source_name: str, *, source_type: str | None = None) -> None:
    """Refuse to quantize from anything that is not high precision.

    The study's ladder produces four artifacts from one BF16 source. Producing Q4_K_M from Q8_0
    instead would be cheaper and would look identical in a directory listing, so the refusal has to
    live in the code path rather than in the procedure.
    """
    if source_type is not None and source_type.lower() not in HIGH_PRECISION_TYPES:
        raise RequantizationError(
            f"source type {source_type!r} is not high precision "
            f"(expected one of {sorted(HIGH_PRECISION_TYPES)})"
        )
    if is_low_bit_name(source_name):
        raise RequantizationError(
            f"refusing to quantize from an already-quantized source: {source_name!r}. "
            "Every artifact in this study is produced from the high-precision conversion."
        )


def assert_m1_v2_lineage(metadata: Mapping[str, Any]) -> None:
    """Every artifact in this study must resolve to the promoted M1-v2 checkpoint.

    Two shapes satisfy that, and they are checked separately rather than through a fallback chain.
    The reference artifact *is* M1-v2, so it declares `experiment_id`/`selected_checkpoint` and its
    `parent_experiment_id` correctly names M0. A quantized descendant is *derived from* M1-v2, so it
    declares `parent_experiment_id`/`parent_checkpoint`. Collapsing the two would silently accept a
    descendant of M0 as if it were a descendant of M1-v2.
    """
    if str(metadata.get("experiment_id") or "") == M1_V2_EXPERIMENT:
        checkpoint = str(metadata.get("selected_checkpoint") or "")
        if M1_V2_CHECKPOINT not in checkpoint:
            raise ArtifactLineageError(
                f"{M1_V2_EXPERIMENT} artifact names checkpoint {checkpoint!r}, "
                f"not {M1_V2_CHECKPOINT!r}"
            )
        return

    parent = str(metadata.get("parent_experiment_id") or "")
    if parent != M1_V2_EXPERIMENT:
        raise ArtifactLineageError(
            f"artifact is neither {M1_V2_EXPERIMENT!r} nor derived from it "
            f"(parent_experiment_id={parent!r})"
        )
    checkpoint = str(metadata.get("parent_checkpoint") or "")
    if M1_V2_CHECKPOINT not in checkpoint:
        raise ArtifactLineageError(
            f"artifact parent checkpoint {checkpoint!r} does not name {M1_V2_CHECKPOINT!r}"
        )


def can_release(status: str, verdict: Mapping[str, Any] | None) -> bool:
    """Whether an artifact may be published as a preserving release.

    Requires both an accepting status and a gate verdict with no failed dimension. A status without
    its verdict is treated as unproven, because the status is a label and the verdict is evidence.
    """
    if status not in RELEASABLE_FROM:
        return False
    if not verdict:
        return False
    if verdict.get("decision") == "REJECTED_ACCURACY":
        return False
    return not list(verdict.get("failed_dimensions") or [])


def assert_releasable(status: str, verdict: Mapping[str, Any] | None) -> None:
    if not can_release(status, verdict):
        failed = list((verdict or {}).get("failed_dimensions") or [])
        raise ValueError(
            f"artifact with status {status!r} cannot be released"
            + (f"; failed dimensions: {', '.join(failed)}" if failed else "")
        )
