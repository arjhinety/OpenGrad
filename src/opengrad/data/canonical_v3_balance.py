"""The materialized decision balance for canonical-v3 (21 phase 4), specified in 39 before it was computed.

Takes the behaviour labels of :mod:`opengrad.data.behaviour_labels` and selects a balanced set of records: equal
shares over the four decisions of :data:`opengrad.data.behavior.DECISIONS`, each stratum contributing
``min(supply)`` records, chosen deterministically by ``sha256(seed | record id)``.

Stratum membership (39 §1), and why they differ:

* ``CALL`` -- **structural**: the record's first assistant reply *is* a structured tool call
  (``CALL_BY_STRUCTURE``, 30 §4 layer A). No classifier judgment enters it, which is why it may define a
  stratum although a classifier ``CALL`` label may not carry weight (38 §2).
* ``ANSWER`` / ``CLARIFY`` / ``UNSUPPORTED`` -- the classifier's ``DIRECT`` / ``CLARIFY`` / ``UNSUPPORTED``
  labels, and only where ``weight_permitted`` is true: ``ANSWER`` therefore comes from glaive alone.

This is a **selection plan, not a corpus**: it renders nothing, writes no shard, changes no arm and authorises
no training (38 §4, 39 §4). Materializing canonical-v3 must still re-check contamination, renderability and the
supervision contracts.

    python -m opengrad.data.canonical_v3_balance --build
    python -m opengrad.data.canonical_v3_balance --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from opengrad.data import behaviour_labels as labels_pass
from opengrad.data.mixture import load_mixture

ROOT = Path(__file__).resolve().parents[3]
BALANCE_VERSION = "canonical-v3-decision-balance-v1"
CONFIG = Path("configs/data/tool_calling/decision_balance_v1.yaml")
SPEC = "docs/research/study-002/39-CANONICAL-V3-DECISION-BALANCE-SPEC.md"
OUTPUT_DIR = Path("data/processed/canonical-v3-balance-v1")
REPORT = Path("reports/canonical-v3/decision-balance-v1.json")

#: decision stratum -> the label that puts a record in it.
STRATUM_LABEL = {
    "CALL": labels_pass.CALL_BY_STRUCTURE,
    "ANSWER": "DIRECT",
    "CLARIFY": "CLARIFY",
    "UNSUPPORTED": "UNSUPPORTED",
}
#: CALL membership is structural, so ``weight_permitted`` (a statement about classifier labels) does not apply.
STRUCTURAL_STRATA = frozenset({"CALL"})


class BalanceError(RuntimeError):
    """The balance cannot be computed, or the written plan does not match its inputs."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stratum_of(row: Mapping[str, Any]) -> str | None:
    """The decision stratum a labelled record belongs to, or None when it carries no weight (39 §1)."""
    for stratum, label in STRATUM_LABEL.items():
        if row["label"] != label:
            continue
        if stratum in STRUCTURAL_STRATA or row["weight_permitted"]:
            return stratum
        return None
    return None


def rank_key(seed: str, record_id: str) -> str:
    return _sha256(f"{seed}|{record_id}".encode("utf-8"))


def select(rows: list[Mapping[str, Any]], seed: str) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    """Equal shares, supply-limited: every stratum contributes ``min(supply)`` records, deterministically."""
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = stratum_of(row)
        if stratum:
            strata[stratum].append(row)
    if set(strata) != set(STRATUM_LABEL):
        missing = sorted(set(STRATUM_LABEL) - set(strata))
        raise BalanceError(f"no record supplies the strata {missing}; a balanced set cannot be built")
    supply = {stratum: len(items) for stratum, items in sorted(strata.items())}
    per_stratum = min(supply.values())
    chosen = {
        stratum: sorted(items, key=lambda row: rank_key(seed, row["id"]))[:per_stratum]
        for stratum, items in sorted(strata.items())
    }
    counts = {
        "supply": supply,
        "per_stratum": per_stratum,
        "selected": per_stratum * len(chosen),
        "left_out": {stratum: supply[stratum] - per_stratum for stratum in supply},
    }
    return chosen, counts


def composition(chosen: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    by_source: dict[str, Counter[str]] = {stratum: Counter() for stratum in chosen}
    by_kind: dict[str, Counter[str]] = {stratum: Counter() for stratum in chosen}
    for stratum, rows in chosen.items():
        for row in rows:
            by_source[stratum][row["source"]] += 1
            by_kind[stratum][row["unit_kind"] or "not_applicable"] += 1
    return {
        "per_source": {stratum: dict(sorted(counter.items())) for stratum, counter in sorted(by_source.items())},
        "per_unit_kind": {stratum: dict(sorted(counter.items())) for stratum, counter in sorted(by_kind.items())},
    }


def load_labels(root: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    manifest_path = root / labels_pass.OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise BalanceError("behaviour labels do not exist yet: run opengrad.data.behaviour_labels --build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[Mapping[str, Any]] = []
    for source, entry in sorted(manifest["sources"].items()):
        payload = (root / labels_pass.OUTPUT_DIR / entry["file"]).read_bytes()
        if _sha256(payload) != entry["sha256"]:
            raise BalanceError(f"{source}: label file does not match the labels manifest")
        rows.extend(json.loads(line) for line in payload.decode("utf-8").splitlines() if line)
    return rows, manifest


def build(root: Path = ROOT) -> dict[str, Any]:
    config = load_mixture(root / CONFIG)
    if config["selection"]["rule"] != "equal_shares_supply_limited":
        raise BalanceError("this builder implements 39 §2's equal-shares rule only")
    rows, labels_manifest = load_labels(root)
    seed = config["selection"]["seed"]
    chosen, counts = select(rows, seed)
    selected_ids = {stratum: [row["id"] for row in items] for stratum, items in sorted(chosen.items())}
    payload = (json.dumps(selected_ids, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out = root / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected-ids.json").write_bytes(payload)
    plan = {
        "artifact_kind": "CANONICAL_V3_DECISION_BALANCE",
        "balance_version": BALANCE_VERSION,
        "statement": (
            "A selection plan for canonical-v3: which records a decision-balanced corpus would draw, and in what "
            "proportion. No record is rendered, no shard is written, no arm changes and no training is "
            "authorised (38 §4, 39 §4). Every stratum but CALL rests on classifier labels whose qualifications "
            "are provisional MODEL_REFERENCE ones (37 §8)."
        ),
        "specification": SPEC,
        "authorisation": "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md",
        "config": {"path": CONFIG.as_posix(), "sha256": _sha256((root / CONFIG).read_bytes().replace(b"\r\n", b"\n"))},
        "labels": {
            "labels_version": labels_manifest["labels_version"],
            "classifier": labels_manifest["classifier"],
            "input_contract": labels_manifest["input_contract"],
            "corpus_fingerprint": labels_manifest["corpus_fingerprint"],
        },
        "seed": seed,
        "stratum_membership": {
            "CALL": "structural: the first reply is a structured tool call (30 §4); never a classifier CALL label",
            "ANSWER": "classifier DIRECT where 38 §2 permits weight (glaive only)",
            "CLARIFY": "classifier CLARIFY where 38 §2 permits weight",
            "UNSUPPORTED": "classifier UNSUPPORTED where 38 §2 permits weight",
        },
        "decision_weights": config["decision_weights"],
        "counts": counts,
        "composition": composition(chosen),
        "selected_ids_file": "selected-ids.json",
        "selected_ids_sha256": _sha256(payload),
    }
    plan_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out / "manifest.json").write_bytes(plan_bytes)
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(plan_bytes)
    return plan


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Recompute the selection from the same labels and config and compare it byte for byte."""
    plan = json.loads((root / OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    rows, labels_manifest = load_labels(root)
    if labels_manifest["corpus_fingerprint"] != plan["labels"]["corpus_fingerprint"]:
        problems.append("the labels were rebuilt on another normalization-v3 artifact")
    config_hash = _sha256((root / CONFIG).read_bytes().replace(b"\r\n", b"\n"))
    if config_hash != plan["config"]["sha256"]:
        problems.append("the mixture config changed since the plan was written")
    chosen, counts = select(rows, plan["seed"])
    selected_ids = {stratum: [row["id"] for row in items] for stratum, items in sorted(chosen.items())}
    payload = (json.dumps(selected_ids, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _sha256(payload) != plan["selected_ids_sha256"]:
        problems.append("the selection does not reproduce")
    if _sha256((root / OUTPUT_DIR / plan["selected_ids_file"]).read_bytes()) != plan["selected_ids_sha256"]:
        problems.append("the written selection does not match its hash")
    if counts != plan["counts"]:
        problems.append("the counts do not reproduce")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "counts": plan["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        plan = build(args.root)
        print(json.dumps({"counts": plan["counts"], "composition": plan["composition"]}, indent=2))
        return 0
    print(json.dumps(verify(args.root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
