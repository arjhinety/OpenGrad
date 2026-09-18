"""Behaviour labels for normalization-v3, from the frozen prose decision classifier (21 phase 3; 38 §3).

This is the C1 labelling pass: every accepted normalization-v3 record gets one behaviour label for its first
assistant reply, from the frozen ``prose-decision-classifier-v2`` through the input contract
``prose-decision-input-v2`` (36 §2). It produces labels and counts. It builds no mixture, writes no training
corpus and changes no existing artifact.

What a record gets:

* **A mode** -- ``DIRECT``, ``CLARIFY``, ``UNSUPPORTED`` or ``CALL`` -- when its first reply is prose the
  contract admits and the classifier decides;
* ``ABSTAIN`` when the classifier will not decide;
* ``CALL_BY_STRUCTURE`` when the first reply *is* a structured tool call: layer A by structure, never
  classified (30 §4);
* ``UNLABELLED`` with the contract's reasons otherwise -- held-out or evaluation material, a malformed record,
  a first turn that is not one user message then one assistant reply, or an empty first reply.

**Balancing weight** is a separate field from the label, and follows 38 §2 exactly: permitted for
``UNSUPPORTED`` and ``CLARIFY``; for ``DIRECT`` only on ``glaive``, because ToolACE's post-stratified DIRECT
precision row was ``NOT_EVALUABLE`` (37 §8); never for ``CALL``, ``CALL_BY_STRUCTURE``, ``ABSTAIN`` or
``UNLABELLED``. A label without permission is still recorded: it may be retained, it may not carry weight
(22 §6).

Safeguards:

* it refuses to run unless the classifier source is byte-for-byte the frozen one;
* it reads normalization-v3 only through :func:`opengrad.data.normalization_v3.iter_rows` and never modifies it;
* held-out and evaluation material is excluded by the contract itself, not by a rule here;
* every output records the classifier tag and hash, both contract versions, the corpus fingerprint, and that
  the qualifications behind the permission are provisional ``MODEL_REFERENCE`` ones (37 §8, 38 §2).

    python -m opengrad.data.behaviour_labels --build
    python -m opengrad.data.behaviour_labels --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import (
    CONTRACT_V2_VERSION,
    HeldoutIndex,
    IneligibleRecord,
    build_first_reply_input,
)
from opengrad.data.decision_classifier_v2 import CLASSIFIER_VERSION, classify
from opengrad.data.normalization_v3 import iter_rows

ROOT = Path(__file__).resolve().parents[3]
LABELS_VERSION = "behaviour-labels-v1"
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier_v2.py")
#: The freeze this pass is pinned to (37 §7). ``tests`` assert it equals the one-shot runner's constant.
FROZEN_SOURCE_SHA256_LF = "47436ca990bd4c1846113dd8d28e2c8755cff50837581acaa39654d0d2f2b382"
FROZEN_TAG = "prose-decision-classifier-v2"

CORPUS_DIR = Path("data/processed/normalization-v3")
CORPUS_MANIFEST = Path("reports/normalization-v3/manifests/normalization-v3.manifest.json")
OUTPUT_DIR = Path("data/processed/behaviour-labels-v1")
REPORT = Path("reports/canonical-v3/behaviour-labels-v1.counts.json")

CALL_BY_STRUCTURE = "CALL_BY_STRUCTURE"
UNLABELLED = "UNLABELLED"
#: 38 §2: DIRECT carries weight only where its post-stratified precision row passed.
DIRECT_PERMITTED_SOURCES = frozenset({"glaive"})
WEIGHTED_MODES = frozenset({"UNSUPPORTED", "CLARIFY"})


class BehaviourLabelError(RuntimeError):
    """The labelling pass cannot run, or its output does not match its inputs."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_frozen(root: Path) -> None:
    observed = _sha256((root / CLASSIFIER_MODULE).read_bytes().replace(b"\r\n", b"\n"))
    if observed != FROZEN_SOURCE_SHA256_LF:
        raise BehaviourLabelError(f"{CLASSIFIER_MODULE.as_posix()} is not the frozen {FROZEN_TAG} (sha256 LF {observed})")


def weight_permitted(label: str, source: str) -> bool:
    """38 §2, as a pure function of the label and the source."""
    if label in WEIGHTED_MODES:
        return True
    if label == "DIRECT":
        return source in DIRECT_PERMITTED_SOURCES
    return False


def label_record(record: Mapping[str, Any], source: str, heldout: HeldoutIndex) -> dict[str, Any]:
    """One record's behaviour label. Never raises on an ineligible record: it records why."""
    try:
        built = build_first_reply_input(record, heldout)
    except IneligibleRecord as ineligible:
        reasons = tuple(ineligible.result.reasons)
        label = CALL_BY_STRUCTURE if reasons == ("FIRST_REPLY_IS_STRUCTURAL_CALL",) else UNLABELLED
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
    unit_kind = getattr(built.provenance, "unit_kind", None)
    return {
        "id": record["id"],
        "source": source,
        "canonical_hash": record["canonical_hash"],
        "label": decision.label,
        "step": decision.step,
        "unit_kind": unit_kind,
        "contract": built.contract_version,
        "reasons": [],
        "weight_permitted": weight_permitted(decision.label, source),
    }


def label_source(root: Path, source: str, heldout: HeldoutIndex) -> Iterator[dict[str, Any]]:
    for record in iter_rows(root / CORPUS_DIR, source):
        yield label_record(record, source, heldout)


def summarise(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        labels[row["label"]] += 1
        if row["weight_permitted"]:
            weighted[row["label"]] += 1
        if row["unit_kind"]:
            kinds[row["unit_kind"]] += 1
        for reason in row["reasons"]:
            reasons[reason] += 1
    return {
        "records": len(rows),
        "labels": dict(sorted(labels.items())),
        "weight_permitted": dict(sorted(weighted.items())),
        "unit_kind": dict(sorted(kinds.items())),
        "unlabelled_reasons": dict(sorted(reasons.items())),
    }


def _rows_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def build(root: Path = ROOT) -> dict[str, Any]:
    """Label every accepted record of every source. Deterministic; overwrites only identical bytes."""
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
            "Behaviour labels for normalization-v3 from the frozen prose-decision-classifier-v2 (21 phase 3). "
            "Labels only: no mixture, no sampling weight, no training corpus. The qualifications behind "
            "weight_permitted are provisional MODEL_REFERENCE ones (37 §8, 38 §2), never human gold, and never "
            "evidence of accuracy."
        ),
        "classifier": {
            "version": CLASSIFIER_VERSION,
            "tag": FROZEN_TAG,
            "source_sha256_lf": FROZEN_SOURCE_SHA256_LF,
        },
        "input_contract": CONTRACT_V2_VERSION,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "corpus_fingerprint": corpus_manifest["fingerprint"],
        "authorisation": "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md",
        "weight_rule": {
            "permitted_modes": sorted(WEIGHTED_MODES),
            "direct_permitted_sources": sorted(DIRECT_PERMITTED_SOURCES),
            "never_weighted": ["CALL", CALL_BY_STRUCTURE, "ABSTAIN", UNLABELLED],
        },
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
        if summarise(rows) != {k: entry[k] for k in ("records", "labels", "weight_permitted", "unit_kind", "unlabelled_reasons")}:
            problems.append(f"{source}: counts do not match the labels")
        if any(row["weight_permitted"] != weight_permitted(row["label"], row["source"]) for row in rows):
            problems.append(f"{source}: a weight permission does not follow 38 §2")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "totals": manifest["totals"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        manifest = build(args.root)
        print(json.dumps({"totals": manifest["totals"], "sources": {s: v["labels"] for s, v in manifest["sources"].items()}}, indent=2))
        return 0
    print(json.dumps(verify(args.root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
