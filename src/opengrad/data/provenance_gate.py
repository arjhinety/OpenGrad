"""The C1 pre-GPU provenance gate (docs/research/study-002/21-C1-IMPLEMENTATION-STATUS.md, phase 6).

This is the module :mod:`opengrad.data.versions` names as a consumer of its version fields. It is the
machine-checkable provenance/identity gate over the C1 artifacts that exist today -- canonical-v3,
its decision balance and its behaviour labels -- and it authorises nothing: no training is launched,
no artifact is written except by ``--record``.

What it checks (each an independent :class:`~opengrad.verification.accounting.ValidationResult`,
assembled into one :class:`~opengrad.verification.accounting.VerificationReport`):

1. ``versions_authoritative`` -- every C1 manifest's version block equals the authoritative constants;
2. ``classifier_identity`` -- the applied classifier is byte-for-byte the frozen
   ``prose-decision-classifier-v2``, through a known input contract, and where a manifest declares a
   ``versions`` block beside its ``labels.classifier`` the two must name the same classifier;
3. ``renderer_identity`` -- the renderer identity canonical-v3 recorded equals the pinned Study 001
   contract (the phase-7 equality proof expressed as a gate);
4. ``authorisation_recorded`` -- each C1 artifact names the 38 authorisation document;
5. ``record_version_agreement`` -- per record, metadata and manifest versions agree (the phase-1
   invariant applied to the artifact);
6. ``selection_plan_agreement`` -- canonical-v3 selects exactly the ids the balance plan chose.

Two properties are deliberate, because a gate that cannot fail is worth nothing (doc 16):

* it reads **committed** artifacts, so it runs on a clean checkout;
* where an input is a **git-ignored local build** the check reports ``BLOCKED_INPUT_MISSING``, never
  a pass -- but a real disagreement is ``FAIL``.

This is a new check set, so it carries its own contract integer (``PROVENANCE_GATE_CONTRACT``) and
does not bump the shared ``opengrad.verification.VERIFIER_CONTRACT`` that publication verification
uses; if this gate is later folded into that report, the shared contract is what bumps.

    python -m opengrad.data.provenance_gate --verify
    python -m opengrad.data.provenance_gate --record
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from opengrad.data import (
    behaviour_labels,
    canonical_v3,
    canonical_v3_balance,
    normalization_v3,
    versions,
)
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.verification.accounting import (
    BLOCKED_INPUT_MISSING,
    CONDITIONALLY_REQUIRED,
    PASS,
    REQUIRED_NONEMPTY,
    Skip,
    ValidationResult,
    VerificationReport,
)

ROOT = Path(__file__).resolve().parents[3]

#: The gate's own contract. Bump when the check set, its discovery rules or its non-vacuity
#: requirements change, so a PASS under one contract is never read as a PASS under another.
PROVENANCE_GATE_CONTRACT = 1

#: Where a recorded run is written. A later run is a new record, never an edit of this one.
RECORD = Path("reports/canonical-v3/provenance-gate-v1.json")

#: The study-owner authorisation every C1 artifact must name (38 §3, condition 2).
AUTHORISATION = "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md"

#: The frozen Study 001 renderer contract and the committed renderer snapshot, the "before" side of
#: phase 7's equality proof.
HELDOUT_V1 = Path("reports/evaluation/behavioral-heldout-v1.manifest.json")
RENDERER_SNAPSHOT = Path("tests/fixtures/rendered/qwen35_2b_metadata.json")

#: The input contracts the classifier may be run through (32, 36 §2).
INPUT_CONTRACTS = frozenset({"prose-decision-input-v1", "prose-decision-input-v2"})

FAIL_CLASSIFIER_VERSION = "FAIL_CLASSIFIER_VERSION"
FAIL_RENDERER_IDENTITY = "FAIL_RENDERER_IDENTITY"
FAIL_AUTHORISATION = "FAIL_AUTHORISATION"
FAIL_SELECTION_PLAN = "FAIL_SELECTION_PLAN"

#: The C1 artifacts that carry a classifier block, in a fixed order.
CLASSIFIED_ARTIFACTS = (
    ("canonical-v3", canonical_v3.REPORT),
    ("balance", canonical_v3_balance.REPORT),
    ("labels", behaviour_labels.REPORT),
)


class ProvenanceGateError(RuntimeError):
    """The gate cannot run."""


# ── the census helper ────────────────────────────────────────────────────────────────────────────


def _result(
    name: str,
    policy: str,
    ids: Sequence[str],
    errors: Sequence[str],
    *,
    blocked_ids: Sequence[str] = (),
    skipped: Sequence[Skip] = (),
    blocked_status: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> ValidationResult:
    """Build one result whose counters add up (``discovered == checked + blocked + skipped``).

    Errors are attributed to a candidate by the ``"<id>: "`` prefix -- the convention
    :func:`opengrad.registry.validate.result_from` uses. An error that names no candidate is
    attributed to a per-gate sentinel so it still fails the gate rather than being lost.
    """
    discovered = list(ids)
    for extra in (*blocked_ids, *(skip.id for skip in skipped)):
        if extra not in discovered:
            discovered.append(extra)
    blocked = [candidate for candidate in blocked_ids if candidate in discovered]
    skip_ids = {skip.id for skip in skipped}
    blocked_set = set(blocked)
    checked = [
        candidate for candidate in discovered if candidate not in blocked_set and candidate not in skip_ids
    ]

    attributed: dict[str, list[str]] = {}
    unattributed: list[str] = []
    for error in errors:
        match = next((candidate for candidate in checked if error.startswith(f"{candidate}:")), None)
        if match is None:
            unattributed.append(error)
        else:
            attributed.setdefault(match, []).append(error)
    if unattributed:
        sentinel = f"<{name}>"
        discovered.append(sentinel)
        checked.append(sentinel)
        attributed[sentinel] = unattributed

    passed = sum(1 for candidate in checked if candidate not in attributed)
    failed = len(checked) - passed
    return ValidationResult(
        name=name,
        policy=policy,
        discovered=len(discovered),
        checked=len(checked),
        passed=passed,
        failed=failed,
        blocked=len(blocked),
        skipped=list(skipped),
        errors=list(errors),
        blocked_reasons=[f"{candidate}: not present in this checkout" for candidate in blocked],
        detail=dict(detail or {}),
        blocked_status=blocked_status if blocked else None,
    )


def _json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _document(root: Path, path: Path) -> dict[str, Any]:
    text = (root / path).read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _classifier_block(document: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """The classifier block of an artifact, whether nested under ``labels`` or declared flat."""
    labels = document.get("labels")
    if isinstance(labels, Mapping):
        return dict(labels.get("classifier") or {}), labels.get("input_contract")
    return dict(document.get("classifier") or {}), document.get("input_contract")


# ── the checks ───────────────────────────────────────────────────────────────────────────────────


def versions_authoritative(root: Path) -> ValidationResult:
    """Every C1 manifest's declared versions must equal the authoritative constants."""
    candidates = (
        ("canonical-v3", canonical_v3.REPORT),
        ("source-manifest", normalization_v3.SOURCE_MANIFEST),
    )
    ids: list[str] = []
    errors: list[str] = []
    detail: dict[str, int] = {}
    for name, path in candidates:
        ids.append(name)
        if not (root / path).is_file():
            errors.append(f"{name}: artifact missing at {path}")
            continue
        declared = dict(_document(root, path).get("versions") or {})
        if not declared:
            errors.append(f"{name}: no versions block")
            continue
        try:
            versions.check_artifact_matches_authoritative_versions(declared)
        except versions.ProvenanceVersionMismatch as exc:
            errors.append(f"{name}: {exc}")
        detail[name] = sum(1 for field in versions.MANIFEST_VERSION_FIELDS if field in declared)
    return _result("versions_authoritative", REQUIRED_NONEMPTY, ids, errors, detail=detail)


def classifier_identity(root: Path) -> ValidationResult:
    """The applied classifier must be the frozen one, and must agree with any declared version."""
    ids: list[str] = []
    errors: list[str] = []
    for name, path in CLASSIFIED_ARTIFACTS:
        ids.append(name)
        if not (root / path).is_file():
            errors.append(f"{name}: artifact missing at {path}")
            continue
        document = _json(root, path)
        classifier, contract = _classifier_block(document)
        tag = classifier.get("tag")
        if tag != behaviour_labels.FROZEN_TAG:
            errors.append(
                f"{name}: classifier tag {tag!r} is not the frozen {behaviour_labels.FROZEN_TAG!r}"
            )
        if classifier.get("source_sha256_lf") != behaviour_labels.FROZEN_SOURCE_SHA256_LF:
            errors.append(f"{name}: classifier source hash is not the frozen one")
        if contract not in INPUT_CONTRACTS:
            errors.append(f"{name}: input contract {contract!r} is not a known contract")
        declared = (document.get("versions") or {}).get("decision_classifier_version")
        applied = classifier.get("version")
        if declared and applied and declared != applied:
            errors.append(
                f"{name}: {FAIL_CLASSIFIER_VERSION}: versions.decision_classifier_version "
                f"{declared!r} != labels.classifier.version {applied!r}"
            )
    return _result("classifier_identity", REQUIRED_NONEMPTY, ids, errors)


def renderer_identity(root: Path) -> ValidationResult:
    """The renderer canonical-v3 recorded must equal the pinned Study 001 contract (phase 7)."""
    after = (_json(root, canonical_v3.REPORT).get("gates") or {}).get("renderability", {}).get(
        "identity", {}
    )
    before = _json(root, HELDOUT_V1).get("model_renderer_contract", {})
    snapshot = _json(root, RENDERER_SNAPSHOT)
    witnesses: dict[str, dict[str, Any]] = {
        "renderer": {
            "before": before.get("renderer"),
            "after": after.get("renderer"),
            "snapshot": snapshot.get("renderer"),
            "code": Qwen35_2BRenderer.renderer_version,
        },
        "model_revision": {
            "before": before.get("model_revision"),
            "after": after.get("model_revision"),
            "snapshot": snapshot.get("model_revision"),
            "code": Qwen35_2BRenderer.model_revision,
        },
        "template_hash": {
            "before": before.get("template_hash"),
            "after": after.get("template_hash"),
            "snapshot": snapshot.get("chat_template_hash"),
        },
    }
    ids: list[str] = []
    errors: list[str] = []
    for field, values in witnesses.items():
        ids.append(field)
        observed = set(values.values())
        if len(observed) != 1 or None in observed:
            errors.append(f"{field}: {FAIL_RENDERER_IDENTITY}: {values}")
    return _result("renderer_identity", REQUIRED_NONEMPTY, ids, errors)


def authorisation_recorded(root: Path) -> ValidationResult:
    """Every C1 artifact must name the study-owner authorisation it was built under."""
    ids: list[str] = []
    errors: list[str] = []
    for name, path in CLASSIFIED_ARTIFACTS:
        ids.append(name)
        if not (root / path).is_file():
            errors.append(f"{name}: artifact missing at {path}")
            continue
        recorded = _json(root, path).get("authorisation")
        if recorded != AUTHORISATION:
            errors.append(f"{name}: {FAIL_AUTHORISATION}: authorisation is {recorded!r}")
    return _result("authorisation_recorded", REQUIRED_NONEMPTY, ids, errors)


def record_version_agreement(root: Path) -> ValidationResult:
    """Per record, metadata and manifest versions must agree (the phase-1 invariant)."""
    if not (root / canonical_v3.REPORT).is_file():
        return _result(
            "record_version_agreement",
            CONDITIONALLY_REQUIRED,
            ["canonical-v3"],
            [],
            blocked_ids=["canonical-v3"],
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": "the canonical-v3 manifest is missing"},
        )
    manifest = _json(root, canonical_v3.REPORT)
    config = manifest.get("versions") or {}
    directory = root / canonical_v3.OUTPUT_DIR
    shards = [shard["file"] for shard in manifest.get("shards", [])]
    missing = [shard for shard in shards if not (directory / shard).is_file()]
    if missing or not shards:
        return _result(
            "record_version_agreement",
            CONDITIONALLY_REQUIRED,
            shards or ["canonical-v3 shards"],
            [],
            blocked_ids=missing or (shards if shards else ["canonical-v3 shards"]),
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": "canonical-v3 shards are git-ignored; build them with "
            "python -m opengrad.data.canonical_v3 --build"},
        )

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from opengrad.data.normalization_v3 import decode_row

    ids: list[str] = []
    errors: list[str] = []
    rows_checked = 0
    for shard in shards:
        ids.append(shard)
        problems = 0
        first = ""
        for batch in pq.ParquetFile(directory / shard).iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                row = decode_row(raw)
                rows_checked += 1
                metadata = row.get("metadata") or {}
                try:
                    versions.check_version_agreement(metadata, config)
                    versions.check_version_agreement(metadata.get("supervision") or {}, config)
                except versions.ProvenanceVersionMismatch as exc:
                    problems += 1
                    first = first or f"{row.get('id')}: {exc}"
        if problems:
            errors.append(f"{shard}: PROV_VERSION_MISMATCH on {problems} rows; first {first}")
    return _result(
        "record_version_agreement",
        CONDITIONALLY_REQUIRED,
        ids,
        errors,
        detail={"shards": len(shards), "rows": rows_checked},
    )


def selection_plan_agreement(root: Path) -> ValidationResult:
    """canonical-v3 must select exactly the ids the balance plan chose."""
    ids = ["selected_ids_sha256"]
    errors: list[str] = []
    detail: dict[str, Any] = {}
    canonical = _json(root, canonical_v3.REPORT)
    expected = (_json(root, canonical_v3_balance.REPORT).get("selected_ids_sha256"))
    observed = (canonical.get("selection_plan") or {}).get("selected_ids_sha256")
    if observed != expected:
        errors.append(
            f"selected_ids_sha256: {FAIL_SELECTION_PLAN}: canonical-v3 {observed!r} != plan {expected!r}"
        )
    # The plan's full reproduction needs the git-ignored labels artifact; run it when present.
    if (root / canonical_v3_balance.OUTPUT_DIR / "manifest.json").is_file():
        verification = canonical_v3_balance.verify(root)
        detail["balance_verify"] = verification["status"]
        if verification["status"] != PASS:
            errors.append(f"selected_ids_sha256: {FAIL_SELECTION_PLAN}: {verification['problems']}")
    else:
        detail["balance_verify"] = "NOT_RUN"
    return _result("selection_plan_agreement", CONDITIONALLY_REQUIRED, ids, errors, detail=detail)


CHECKS = (
    versions_authoritative,
    classifier_identity,
    renderer_identity,
    authorisation_recorded,
    record_version_agreement,
    selection_plan_agreement,
)


def gate(root: Path = ROOT) -> VerificationReport:
    """Run every check and assemble the report."""
    report = VerificationReport(contract=PROVENANCE_GATE_CONTRACT)
    for check in CHECKS:
        report.add(check(root))
    return report


def report_document(report: VerificationReport) -> dict[str, Any]:
    """The machine-readable form written by ``--record`` and printed by ``--verify``."""
    return {
        "contract": report.contract,
        "overall": report.overall,
        "totals": report.totals(),
        "results": [
            {
                "name": result.name,
                "status": result.status,
                "policy": result.policy,
                "counts": result.counts(),
                "errors": result.all_errors(),
                "detail": result.detail,
            }
            for result in report.results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--record", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    report = gate(args.root)
    document = report_document(report)
    if args.record:
        destination = args.root / RECORD
        if destination.exists():
            raise ProvenanceGateError(
                f"{destination} exists: a recorded run is evidence, never overwritten"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if report.overall == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
