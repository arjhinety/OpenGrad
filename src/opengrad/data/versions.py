"""The single authoritative source of the versions that identify canonical artifacts.

Why this module exists
----------------------
The published Canonical-v2 corpus carries **two different adapter versions inside one artifact**: every
record's ``metadata.adapter_version`` is ``"1.0.0"`` (written from ``opengrad.data.adapters``) while the
materialization manifest's ``config.adapter_version`` is the literal ``"1.0.2"`` hard-coded in
``opengrad.data.materialize``. A reader therefore cannot answer "which normalizer produced this record?"
from the artifact, which is exactly the provenance question a corpus release exists to answer.

That is a *provenance* defect, not a data defect, and it is **not** repaired by editing history: the
published corpora keep the versions they recorded, and their manifests keep their hashes. From this
module onward one constant per versioned concern is declared once and every consumer derives from it --
record metadata, normalization manifests, release manifests, fingerprints, and the validation
invariants that refuse to let the layers disagree.

``ADAPTER_VERSION`` is deliberately a new major version. The bump is not cosmetic: decision derivation
changes from message-shape inference to a versioned prose classifier, so records produced before and
after are not comparable and must not share a version string.
"""

from __future__ import annotations

from typing import Any, Mapping

# ── The authoritative versions ──────────────────────────────────────────────────────────────────
# Bump a constant here and every consumer follows. Do not literal a version string anywhere else.

#: Adapter that turns a source record into the canonical IR.
#: 2.0.0 -- behavior is derived by a versioned classifier instead of message shape alone.
#: 2.1.0 -- ToolACE moves to `adapt_toolace_v2`, which stops leaving the parsed call duplicated as
#: text beside the structured call (9,785 of 9,786 call turns under 2.0.0). The message text of those
#: records changes, so records built before and after are not interchangeable.
#: 2.2.0 -- ToolACE moves to `adapt_toolace_v3`: call-final records of the shape validated in
#: reports/normalization-v3/toolace-call-final-shape.json declare CALL_PREDICTION instead of
#: COMPLETE_TRAJECTORY (an OpenGrad structural inference; ToolACE does not declare it). Their messages are
#: unchanged, but they now pass the trajectory gate that quarantined them under 2.1.0, so records built
#: before and after are not interchangeable. Canonical-v2 does not use this adapter and is unchanged.
ADAPTER_VERSION = "2.2.0"

#: Source-scoped translation of upstream tool-schema type vocabulary into the canonical one.
SCHEMA_NORMALIZATION_VERSION = "source-schema-normalization-v1"

#: Deterministic prose classifier that infers a decision where the source provides no label.
DECISION_CLASSIFIER_VERSION = "prose-decision-classifier-v1"

#: Under development (docs/research/study-002/37): reads contract prose-decision-input-v2 (first replies). v1 above
#: stays frozen and is still what every existing artifact names.
DECISION_CLASSIFIER_V2_VERSION = "prose-decision-classifier-v2"

#: Versioned concerns owned elsewhere but named here so one call reports the whole set.
BEHAVIOR_TAXONOMY_VERSION = "tool-use-behavior-taxonomy-v1"
CANONICAL_SCHEMA_VERSION = "tool_use_ir_v1"
SUPERVISION_CONTRACT_VERSION = "supervision_contract_v1"

#: Version of the invariant that record-level and manifest-level versions must agree.
PROVENANCE_INVARIANT_VERSION = "provenance-version-agreement-v1"

#: The pre-classifier normalization artifact built from the canonical-v3 source-and-adapter manifest
#: (``configs/releases/toolpolicy_canonical_v3_sources.yaml``). It carries messages, tools, structured
#: calls, structural facts and provenance, and deliberately no behaviour label.
NORMALIZATION_VERSION = "normalization-v3"

#: What the future prose decision classifier may read (docs/research/study-002/32).
CLASSIFIER_INPUT_CONTRACT_VERSION = "prose-decision-input-v1"

#: Contract v2 (docs/research/study-002/36 §2): the first assistant reply of any record, whatever follows it.
#: v1 stays in force for everything built under it.
CLASSIFIER_INPUT_CONTRACT_V2_VERSION = "prose-decision-input-v2"

#: Manifest/config field -> the constant it must equal. Used by the invariant below and by
#: ``opengrad.data.provenance_gate``. Keys are the field names actually written today, so a manifest
#: that already exists can be checked without being rewritten.
MANIFEST_VERSION_FIELDS: Mapping[str, str] = {
    "adapter_version": ADAPTER_VERSION,
    "schema_normalization_version": SCHEMA_NORMALIZATION_VERSION,
    "behavior_taxonomy_version": BEHAVIOR_TAXONOMY_VERSION,
    "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
    "supervision_contract_version": SUPERVISION_CONTRACT_VERSION,
}

#: Record-metadata field -> the constant it must equal.
RECORD_VERSION_FIELDS: Mapping[str, str] = {
    "adapter_version": ADAPTER_VERSION,
    "schema_normalization_version": SCHEMA_NORMALIZATION_VERSION,
    "behavior_taxonomy_version": BEHAVIOR_TAXONOMY_VERSION,
}


class ProvenanceVersionMismatch(ValueError):
    """A record and its manifest disagree about which normalizer produced them."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"PROV_VERSION_MISMATCH: {detail}")


def provenance_versions() -> dict[str, str]:
    """Every version that identifies a canonical-v3 artifact, in one dict."""
    return {
        "adapter_version": ADAPTER_VERSION,
        "schema_normalization_version": SCHEMA_NORMALIZATION_VERSION,
        "decision_classifier_version": DECISION_CLASSIFIER_VERSION,
        "behavior_taxonomy_version": BEHAVIOR_TAXONOMY_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "supervision_contract_version": SUPERVISION_CONTRACT_VERSION,
        "provenance_invariant_version": PROVENANCE_INVARIANT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
    }


def record_versions() -> dict[str, str]:
    """The subset stamped into every canonical record's metadata."""
    return {field: value for field, value in RECORD_VERSION_FIELDS.items()}


def check_version_agreement(
    record_metadata: Mapping[str, Any],
    manifest_config: Mapping[str, Any],
) -> None:
    """Fail if a record and the manifest that describes it disagree about their versions.

    Only fields present on both sides are compared, so a manifest that predates a field is not
    retroactively broken -- absence is not disagreement. The check is also deliberately tolerant of
    *historical* versions: two layers that agree on an old version are internally consistent, and the
    published v1/v2 corpora must stay readable rather than fail a check written after them. Whether an
    artifact was produced by the *current* code is a different question, answered by
    :func:`check_artifact_matches_authoritative_versions`.
    """
    problems: list[str] = []
    for field in RECORD_VERSION_FIELDS:
        record = record_metadata.get(field)
        manifest = manifest_config.get(field)
        if record and manifest and str(record) != str(manifest):
            problems.append(f"{field}: record={record!r} manifest={manifest!r}")
    if problems:
        raise ProvenanceVersionMismatch("; ".join(problems))


def check_artifact_matches_authoritative_versions(manifest_config: Mapping[str, Any]) -> None:
    """Fail if a manifest claims a version other than the one this code produces.

    Applied to newly built artifacts -- the canonical-v3 release and its normalization manifests --
    so that a corpus can always be traced to the code that made it. Historical manifests are not run
    through this check, because they were produced by code that no longer exists and editing them
    would falsify the record.
    """
    problems: list[str] = []
    for field, expected in MANIFEST_VERSION_FIELDS.items():
        observed = manifest_config.get(field)
        if observed and str(observed) != expected:
            problems.append(f"{field}: manifest={observed!r} authoritative={expected!r}")
    if problems:
        raise ProvenanceVersionMismatch("; ".join(problems))
