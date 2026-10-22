"""Behaviour labels for normalization-v3 from the frozen ``prose-decision-classifier-v3``, labels-v2.

v3 names one case apart: a decline followed by delivered content is ``DECLINE_WITH_CONTENT``, where v2 said
``DIRECT``. Everything else is v2's decision, step for step (see ``decision_classifier_v3.py``). This module
builds a second, separate label set. It does not replace ``behaviour-labels-v1``, which stays as it is.

Weights: no row in labels-v2 carries a balancing weight. The v2 balancing authorisation (38) is void for any
new classifier version until it is tested on a new untouched population (38 §5 item 3, 22 §6). Until that test
passes, ``weight_permitted`` is False for every row, and the manifest says so.

    python -m opengrad.data.behaviour_labels_v3 --build
    python -m opengrad.data.behaviour_labels_v3 --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.behaviour_labels import UNLABELLED, BehaviourLabelError, _rows_bytes, summarise
from opengrad.data.classifier_input import (
    CONTRACT_V2_VERSION,
    HeldoutIndex,
    IneligibleRecord,
    build_first_reply_input,
)
from opengrad.data.decision_classifier_v3 import CLASSIFIER_VERSION, classify
from opengrad.data.normalization_v3 import iter_rows
from opengrad.hashing import sha256_bytes as _sha256

ROOT = Path(__file__).resolve().parents[3]
LABELS_VERSION = "behaviour-labels-v2"
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier_v3.py")
#: The freeze this pass is pinned to: the LF-normalised sha256 of ``decision_classifier_v3.py``.
FROZEN_SOURCE_SHA256_LF = "b3ea0e12b49c984c00f73d5dfaa343912308c59d9fd55cdb58f55b8721d31fb3"
FROZEN_TAG = CLASSIFIER_VERSION

CORPUS_DIR = Path("data/processed/normalization-v3")
CORPUS_MANIFEST = Path("reports/normalization-v3/manifests/normalization-v3.manifest.json")
OUTPUT_DIR = Path("data/processed/behaviour-labels-v2")
REPORT = Path("reports/canonical-v3/behaviour-labels-v2.counts.json")


def check_frozen(root: Path) -> None:
    observed = _sha256((root / CLASSIFIER_MODULE).read_bytes().replace(b"\r\n", b"\n"))
    if observed != FROZEN_SOURCE_SHA256_LF:
        raise BehaviourLabelError(
            f"{CLASSIFIER_MODULE.as_posix()} is not the frozen {FROZEN_TAG} (sha256 LF {observed})"
        )


def weight_permitted(label: str, source: str) -> bool:
    """Always False in labels-v2: no balancing weight until the 22 §6 retest of this version passes."""
    del label, source
    return False


def label_record(record: Mapping[str, Any], source: str, heldout: HeldoutIndex) -> dict[str, Any]:
    """One record's label under v3. Never raises on an ineligible record: it records why."""
    try:
        built = build_first_reply_input(record, heldout)
    except IneligibleRecord as ineligible:
        reasons = tuple(ineligible.result.reasons)
        label = "CALL_BY_STRUCTURE" if reasons == ("FIRST_REPLY_IS_STRUCTURAL_CALL",) else UNLABELLED
        return {
            "id": record["id"],
            "source": source,
            "canonical_hash": record["canonical_hash"],
            "label": label,
            "step": None,
            "unit_kind": None,
            "contract": CONTRACT_V2_VERSION,
            "reasons": list(reasons),
            "weight_permitted": False,
        }
    decision = classify(built.features)
    return {
        "id": record["id"],
        "source": source,
        "canonical_hash": record["canonical_hash"],
        "label": decision.label,
        "step": decision.step,
        "unit_kind": getattr(built.provenance, "unit_kind", None),
        "contract": built.contract_version,
        "reasons": [],
        "weight_permitted": weight_permitted(decision.label, source),
    }


def label_source(root: Path, source: str, heldout: HeldoutIndex) -> Iterator[dict[str, Any]]:
    for record in iter_rows(root / CORPUS_DIR, source):
        yield label_record(record, source, heldout)


def build(root: Path = ROOT) -> dict[str, Any]:
    """Label every accepted record of every source under v3. Deterministic; overwrites only identical bytes."""
    check_frozen(root)
    corpus_manifest = json.loads((root / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    if corpus_manifest.get("behavior_labels") != "ABSENT":
        raise BehaviourLabelError("the normalization-v3 manifest already declares behaviour labels")
    heldout = HeldoutIndex.load(root)
    out = root / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, Any] = {}
    totals: list[Mapping[str, Any]] = []
    for source in sorted(corpus_manifest["sources"]):
        rows = list(label_source(root, source, heldout))
        payload = _rows_bytes(rows)
        path = out / f"{source}.labels.jsonl"
        if path.exists() and path.read_bytes() != payload:
            raise BehaviourLabelError(f"{path} exists with different labels; a labelling pass is deterministic")
        path.write_bytes(payload)
        per_source[source] = {"file": path.name, "sha256": _sha256(payload), **summarise(rows)}
        totals.extend(rows)
    manifest = {
        "artifact_kind": "BEHAVIOUR_LABELS",
        "labels_version": LABELS_VERSION,
        "statement": (
            "Behaviour labels for normalization-v3 from the frozen prose-decision-classifier-v3. Labels only: no "
            "mixture, no sampling weight, no training corpus. Identical to behaviour-labels-v1 except that a decline "
            "with delivered content is DECLINE_WITH_CONTENT, not DIRECT. weight_authorised is false for every row "
            "until the 22 §6 retest of this classifier version passes. Never human gold, never evidence of accuracy."
        ),
        "classifier": {"version": CLASSIFIER_VERSION, "tag": FROZEN_TAG, "source_sha256_lf": FROZEN_SOURCE_SHA256_LF},
        "supersedes_labels": "behaviour-labels-v1",
        "input_contract": CONTRACT_V2_VERSION,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "corpus_fingerprint": corpus_manifest["fingerprint"],
        "weight_authorised": False,
        "weight_authorised_reason": "38 §5 item 3 and 22 §6: a classifier change voids the authorisation until retested",
        "sources": per_source,
        "totals": summarise(totals),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out / "manifest.json").write_bytes(manifest_bytes)
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(manifest_bytes)
    return manifest


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Re-read the written labels and check them against their manifest. Classifies nothing."""
    check_frozen(root)
    manifest = json.loads((root / OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest["classifier"]["source_sha256_lf"] != FROZEN_SOURCE_SHA256_LF:
        problems.append("the manifest names another classifier freeze")
    corpus_manifest = json.loads((root / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    if manifest["corpus_fingerprint"] != corpus_manifest["fingerprint"]:
        problems.append("the labels were built on another normalization-v3 artifact")
    for source, entry in manifest["sources"].items():
        payload = (root / OUTPUT_DIR / entry["file"]).read_bytes()
        if _sha256(payload) != entry["sha256"]:
            problems.append(f"{source}: label file does not match its manifest hash")
            continue
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
        if summarise(rows) != {
            k: entry[k] for k in ("records", "labels", "weight_permitted", "unit_kind", "unlabelled_reasons")
        }:
            problems.append(f"{source}: counts do not match the labels")
        if any(row["weight_permitted"] for row in rows):
            problems.append(f"{source}: a row carries a weight, which labels-v2 must not do before the retest")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "totals": manifest["totals"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        print(json.dumps(build(args.root)["totals"], indent=2, sort_keys=True))
        return 0
    result = verify(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
