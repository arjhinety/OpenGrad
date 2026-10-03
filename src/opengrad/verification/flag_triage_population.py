"""The populations the flag-set triage labels (46 §5, 47 §3): the flag set itself, its trial, and the recall sample.

* **The triage population** holds every record of the flag set `F` (reports/study-002/flag-set/), one item each:
  the offered tools, the user's first message and the first reply, read from the same verified release under the
  same first-reply rules (contract `prose-decision-input-v2`) the flag set was computed with.
* **The trial** is 100 of those items, drawn per source by :func:`flag_triage.draw_trial` with the adopted minimum
  (ToolACE at least 20). They are marked by a blinded field, `in_trial`, so one file serves both tasks; trial
  records are labelled again in the full triage, and only those labels count (47 §3 E).
* **The recall population** is 400 `ANSWER`-labelled first replies that classifier v2 did not flag, drawn per
  source with the adopted minimums (ToolACE and When2Call at least 80 each, 47 §3 F), in ascending order of
  `sha256(RECALL_SEED | record id)`. Each carries its predicted label, blinded, for the per-label report.

Every item's id is opaque (`sha256` of a fixed salt and the record id, prefixed `r1:` or `r2:`), because the
labelling models see item ids; the prefixes, like the task ids, say nothing about flags or recall.
The source, record id, canonical hash, unit kind, trial mark and predicted label are blinded fields. Items are
written in ascending order of their opaque id, so a batch mixes sources.

Counts and hashes only: nothing here prints item text.

    python -m opengrad.verification.flag_triage_population --dry-run   # counts, writes nothing
    python -m opengrad.verification.flag_triage_population --build     # writes reports/study-002/flag-triage/
    python -m opengrad.verification.flag_triage_population --verify
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.decision_classifier_v2 import UNSUPPORTED
from opengrad.hashing import sha256_bytes, sha256_text
from opengrad.verification import flag_set as fs
from opengrad.verification import flag_triage as ft

ROOT = fs.ROOT
OUTPUT_DIR = Path("reports/study-002/flag-triage")
TRIAGE_NAME = "flag-triage.population.jsonl"
RECALL_NAME = "flag-recall.population.jsonl"
MANIFEST_NAME = "flag-triage.manifest.json"
ITEM_SALT = "opengrad-flag-triage-item-v15"
RECALL_ITEM_SALT = "opengrad-flag-recall-item-v15"
CODE_MODULES = (
    "src/opengrad/verification/flag_triage_population.py",
    "src/opengrad/verification/flag_triage.py",
)
#: Never shown to a labeller (46 §5: source, record id and label are hidden).
BLINDED = (
    "opengrad_id",
    "canonical_hash",
    "source_dataset",
    "unit_kind",
    "in_trial",
    "predicted_label",
)


class TriagePopulationError(RuntimeError):
    pass


def triage_id(opengrad_id: str) -> str:
    return "r1:" + sha256_text(f"{ITEM_SALT}|{opengrad_id}")[:24]


def recall_id(opengrad_id: str) -> str:
    return "r2:" + sha256_text(f"{RECALL_ITEM_SALT}|{opengrad_id}")[:24]


def _lines(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _flag_set(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8"))
    members_bytes = (root / fs.OUTPUT_DIR / fs.MEMBERS_NAME).read_bytes()
    if sha256_bytes(members_bytes) != manifest["members"]["sha256"]:
        raise TriagePopulationError("the flag set's members file does not match its manifest")
    members = [json.loads(line) for line in members_bytes.decode("utf-8").splitlines() if line]
    return manifest, members


def trial_ids(members: list[dict[str, Any]], flags_by_source: dict[str, int]) -> set[str]:
    allocation = ft.trial_allocation(flags_by_source, ft.TRIAL_SIZE, ft.TRIAL_MINIMUM)
    return set(ft.draw_trial(members, allocation, ft.TRIAL_SEED))


def _item(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    user, reply, _, _ = fs.first_reply(json.loads(record["messages"]))
    assert user is not None and reply is not None
    return {
        "opengrad_id": str(record["opengrad_id"]),
        "canonical_hash": str(record["canonical_hash"]),
        "source_dataset": str(record["source_dataset"]),
        "unit_kind": result["unit_kind"],
        "user_message": user,
        "assistant_response": reply["content"],
        "tools": json.loads(record["tools"]) or [],
    }


def build(root: Path = ROOT) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from opengrad.verification import prose_classifier_v2_oneshot as oneshot

    oneshot.check_frozen(root)
    flag_manifest, members = _flag_set(root)
    member_hash = {m["opengrad_id"]: m["canonical_hash"] for m in members}
    flags_by_source = flag_manifest["flag_set"]["by_source"]
    in_trial = trial_ids(members, flags_by_source)
    _, release = fs.load_release(root)

    triage: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = {}
    for record in fs.iter_records(root):
        result = fs.evaluate(record)
        if result["decision"] != "ANSWER" or result["ineligible"] is not None:
            continue
        opengrad_id = str(record["opengrad_id"])
        if result["label"] == UNSUPPORTED:
            if member_hash.get(opengrad_id) != str(record["canonical_hash"]):
                raise TriagePopulationError("a flagged record is not the flag set's member")
            item = _item(record, result)
            item["triage_id"] = triage_id(opengrad_id)
            item["in_trial"] = "yes" if opengrad_id in in_trial else "no"
            triage.append(item)
        else:
            candidates.setdefault(str(record["source_dataset"]), []).append(
                {**_item(record, result), "predicted_label": result["label"]}
            )
    if len(triage) != len(members):
        raise TriagePopulationError(
            f"rebuilt {len(triage)} flagged records, the flag set holds {len(members)}"
        )
    expected = ft.recall_population(flag_manifest["precision_reporting"])
    found = {source: len(rows) for source, rows in sorted(candidates.items())}
    if found != expected:
        raise TriagePopulationError("the unflagged first replies differ from the flag set's counts")
    allocation = ft.trial_allocation(found, ft.RECALL_SIZE, ft.RECALL_MINIMUM)
    recall: list[dict[str, Any]] = []
    for source, k in sorted(allocation.items()):
        ranked = sorted(
            candidates[source], key=lambda c: sha256_text(f"{ft.RECALL_SEED}|{c['opengrad_id']}")
        )
        for row in ranked[:k]:
            recall.append({**row, "recall_id": recall_id(row["opengrad_id"])})
    triage.sort(key=lambda row: row["triage_id"])
    recall.sort(key=lambda row: row["recall_id"])

    triage_bytes, recall_bytes = _lines(triage), _lines(recall)
    by_label: Counter[tuple[str, str]] = Counter(
        (row["source_dataset"], row["predicted_label"]) for row in recall
    )
    manifest: dict[str, Any] = {
        "artifact_kind": "STUDY_002_FLAG_TRIAGE_POPULATIONS",
        "preregistration": {
            "adoption_amendments": ["study_002_prereg_v14", "study_002_prereg_v15"],
            "documents": [
                "docs/research/study-002/46-READINESS-DESIGN-AMENDMENT-DRAFT.md",
                "docs/research/study-002/47-PER-SOURCE-FLAG-PRECISION-DRAFT.md",
            ],
        },
        "corpus": release,
        "flag_set": {
            "manifest_sha256": sha256_bytes((root / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_bytes()),
            "members_sha256": flag_manifest["members"]["sha256"],
            "n": flag_manifest["flag_set"]["n"],
        },
        "populations": {
            "triage": {
                "file": TRIAGE_NAME,
                "records": len(triage),
                "sha256": sha256_bytes(triage_bytes),
                "by_source": dict(Counter(row["source_dataset"] for row in triage)),
                "trial": {
                    "select": {"field": "in_trial", "equals": "yes"},
                    "records": len(in_trial),
                    "by_source": dict(
                        sorted(
                            Counter(
                                row["source_dataset"] for row in triage if row["in_trial"] == "yes"
                            ).items()
                        )
                    ),
                    "seed": ft.TRIAL_SEED,
                    "minimum": ft.TRIAL_MINIMUM,
                },
            },
            "recall": {
                "file": RECALL_NAME,
                "records": len(recall),
                "sha256": sha256_bytes(recall_bytes),
                "allocation": allocation,
                "population_by_source": found,
                "by_source_and_predicted_label": {
                    source: {label: n for (s, label), n in sorted(by_label.items()) if s == source}
                    for source in sorted(allocation)
                },
                "seed": ft.RECALL_SEED,
                "minimum": ft.RECALL_MINIMUM,
            },
        },
        "id_rule": {
            "triage": f'"r1:" + sha256("{ITEM_SALT}|" + opengrad_id)[:24]',
            "recall": f'"r2:" + sha256("{RECALL_ITEM_SALT}|" + opengrad_id)[:24]',
        },
        "presentation_order": "ascending item id",
        "blinded_fields": list(BLINDED),
        "shown_fields": ["user_message", "assistant_response", "tools"],
        "gold_labels_present": False,
        "code_sha256_lf": {path: fs._lf_sha256(root, path) for path in CODE_MODULES},
        "statement": (
            "Item text from the verified release, first reply only. No label exists. Each labelling run needs "
            "the owner's word."
        ),
    }
    manifest["populations"]["triage"]["by_source"] = dict(
        sorted(manifest["populations"]["triage"]["by_source"].items())
    )
    return manifest, triage, recall


def write(
    root: Path, manifest: dict[str, Any], triage: list[dict[str, Any]], recall: list[dict[str, Any]]
) -> None:
    out = root / OUTPUT_DIR
    if (out / MANIFEST_NAME).exists():
        raise TriagePopulationError(
            f"{OUTPUT_DIR / MANIFEST_NAME} exists; populations are never overwritten"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / TRIAGE_NAME).write_bytes(_lines(triage))
    (out / RECALL_NAME).write_bytes(_lines(recall))
    (out / MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )


def verify(root: Path = ROOT, *, rebuild: bool = True) -> dict[str, Any]:
    """Re-hash the populations, re-derive the trial from the flag set, and rebuild when the release is cached."""
    manifest = json.loads((root / OUTPUT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    errors: list[str] = []
    populations = manifest["populations"]
    rows: dict[str, list[dict[str, Any]]] = {}
    for key, name in (("triage", TRIAGE_NAME), ("recall", RECALL_NAME)):
        data = (root / OUTPUT_DIR / name).read_bytes()
        if sha256_bytes(data) != populations[key]["sha256"]:
            errors.append(f"{name} does not hash to the manifest's sha256")
        rows[key] = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        if len(rows[key]) != populations[key]["records"]:
            errors.append(
                f"{name} has {len(rows[key])} rows, the manifest says {populations[key]['records']}"
            )
    flag_manifest, members = _flag_set(root)
    if manifest["flag_set"]["members_sha256"] != flag_manifest["members"]["sha256"]:
        errors.append("the populations were built from another flag set")
    if {row["opengrad_id"] for row in rows["triage"]} != {m["opengrad_id"] for m in members}:
        errors.append("the triage population is not exactly the flag set")
    expected_trial = trial_ids(members, flag_manifest["flag_set"]["by_source"])
    if {row["opengrad_id"] for row in rows["triage"] if row["in_trial"] == "yes"} != expected_trial:
        errors.append("the trial mark is not the seeded draw")
    if {row["opengrad_id"] for row in rows["recall"]} & {m["opengrad_id"] for m in members}:
        errors.append("a recall item is in the flag set")
    if any(row["predicted_label"] == UNSUPPORTED for row in rows["recall"]):
        errors.append("a recall item is a flagged reply")
    for path, digest in manifest["code_sha256_lf"].items():
        if fs._lf_sha256(root, path) != digest:
            errors.append(f"{path} has changed since the populations were built")
    result: dict[str, Any] = {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "triage": populations["triage"]["records"],
        "trial": populations["triage"]["trial"]["records"],
        "recall": populations["recall"]["records"],
    }
    if not rebuild or errors:
        return result
    try:
        rebuilt, triage, recall = build(root)
    except Exception as exc:  # noqa: BLE001 -- classified: only a cache miss is a missing input
        if fs._input_missing(exc):
            return {**result, "status": "BLOCKED_INPUT_MISSING", "reason": str(exc)[:200]}
        return {**result, "status": "FAIL", "errors": [f"{type(exc).__name__}: {str(exc)[:200]}"]}
    if _lines(triage) != (root / OUTPUT_DIR / TRIAGE_NAME).read_bytes():
        errors.append("rebuilding gives a different triage population")
    if _lines(recall) != (root / OUTPUT_DIR / RECALL_NAME).read_bytes():
        errors.append("rebuilding gives a different recall population")
    if rebuilt["populations"] != populations:
        errors.append("rebuilding gives a different populations block")
    return {**result, "status": "FAIL" if errors else "PASS", "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify(ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("PASS", "BLOCKED_INPUT_MISSING") else 1
    manifest, triage, recall = build(ROOT)
    if args.build:
        write(ROOT, manifest, triage, recall)
    print(json.dumps(manifest["populations"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
