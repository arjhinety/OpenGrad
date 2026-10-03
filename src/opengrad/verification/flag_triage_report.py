"""The flag-set triage's reports (46 §5, 47 §3), from the two models' labels.

Each set is labelled by Gemini 3.8 Flash (High) and deepseek-v4.1-flash under neutral task ids. Their labels are
exported, verified and turned into a reference by `opengrad.verification.pdet_coverage_reference --task <task>`,
which keeps both votes. This module maps each opaque item id back to its record, source and predicted label
through the population, and applies the adopted rules:

* ``trial`` (`first-reply-review-v1-trial`): the per-source precision report only, with no status. The trial guides
  the one revision 46 §5 allows and counts toward no floor (47 §3 E).
* ``triage`` (`first-reply-review-v1`): :func:`flag_triage.triage_decision` over every record of the flag set,
  stop rules 1 and 2 pooled and per source.
* ``recall`` (`first-reply-review-v1-sample`): :func:`flag_triage.recall_report` with the drawn allocation.

A report is written once and never overwritten. Counts only: nothing here reads or prints item text.

    python -m opengrad.verification.flag_triage_report --set trial|triage|recall [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes
from opengrad.verification import flag_triage as ft
from opengrad.verification import flag_triage_population as pop

ROOT = pop.ROOT
REFERENCE_DIR = pop.OUTPUT_DIR / "reference"
ANNOTATORS = ("model.deepseek-v4.1-flash", "model.gemini-3.8-flash-high")
#: The ambiguity reason for a decline although an offered tool could have done it (owner decision, 2026-10-03).
DECLINED_DESPITE_TOOL = "DECLINED_DESPITE_TOOL"
SETS: dict[str, dict[str, str]] = {
    "trial": {"task": "first-reply-review-v1-trial", "population": "triage", "id": "triage_id"},
    "triage": {"task": "first-reply-review-v1", "population": "triage", "id": "triage_id"},
    "recall": {"task": "first-reply-review-v1-sample", "population": "recall", "id": "recall_id"},
}


class FlagTriageReportError(RuntimeError):
    pass


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def load(
    root: Path, set_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The set's population rows, its reference rows, and the populations manifest; every hash checked."""
    spec = SETS[set_name]
    directory = root / pop.OUTPUT_DIR
    manifest = json.loads((directory / pop.MANIFEST_NAME).read_text(encoding="utf-8"))
    population_spec = manifest["populations"][spec["population"]]
    population = (directory / population_spec["file"]).read_bytes()
    if sha256_bytes(population) != population_spec["sha256"]:
        raise FlagTriageReportError(
            f"the {spec['population']} population does not match its manifest"
        )
    task = spec["task"]
    reference_manifest_path = root / REFERENCE_DIR / f"{task}.reference.manifest.json"
    if not reference_manifest_path.is_file():
        raise FlagTriageReportError(
            f"no reference in {REFERENCE_DIR.as_posix()}; build it with "
            f"opengrad.verification.pdet_coverage_reference --task {task}"
        )
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    reference = (root / REFERENCE_DIR / f"{task}.reference.jsonl").read_bytes()
    if sha256_bytes(reference) != reference_manifest["reference_sha256"]:
        raise FlagTriageReportError("the reference does not match its manifest")
    if reference_manifest.get("population_sha256") != population_spec["sha256"]:
        raise FlagTriageReportError("the reference was built on a different population")
    rows = _jsonl(population)
    if set_name == "trial":
        rows = [row for row in rows if row["in_trial"] == "yes"]
    return rows, _jsonl(reference), manifest


def labels_by_record(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], id_field: str
) -> dict[str, tuple[str, str]]:
    """Both models' labels per record id, in `ANNOTATORS` order; every item must carry both votes."""
    record_of = {row[id_field]: row["opengrad_id"] for row in rows}
    labels: dict[str, tuple[str, str]] = {}
    for reference in references:
        item = reference[id_field]
        if item not in record_of:
            raise FlagTriageReportError("a reference item is not in the set's population")
        votes = reference["votes"]
        if set(votes) != set(ANNOTATORS):
            raise FlagTriageReportError(
                f"an item lacks a vote from {sorted(set(ANNOTATORS) - set(votes))}"
            )
        a, b = (votes[annotator] for annotator in ANNOTATORS)
        labels[record_of[item]] = (a, b)
    if len(labels) != len(rows):
        raise FlagTriageReportError(f"{len(labels)} of the set's {len(rows)} items are labelled")
    return labels


def declined_despite_tool(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], id_field: str
) -> dict[str, int]:
    """Per source, the items both models called `UNKNOWN` for a decline although an offered tool fitted.

    They count as not a decline in flag precision like every agreed `UNKNOWN` (47 §4); this only counts them.
    """
    source_of_item = {row[id_field]: row["source_dataset"] for row in rows}
    counts = dict.fromkeys(sorted(set(source_of_item.values())), 0)
    for reference in references:
        if (
            reference.get("reference_label") == "UNKNOWN"
            and reference.get("reference_ambiguity_status") == DECLINED_DESPITE_TOOL
        ):
            counts[source_of_item[reference[id_field]]] += 1
    return counts


def report(root: Path, set_name: str) -> dict[str, Any]:
    spec = SETS[set_name]
    rows, references, manifest = load(root, set_name)
    labels = labels_by_record(rows, references, spec["id"])
    source_of = {row["opengrad_id"]: row["source_dataset"] for row in rows}
    out: dict[str, Any] = {
        "artifact_kind": "STUDY_002_FLAG_TRIAGE_REPORT",
        "set": set_name,
        "task": spec["task"],
        "annotators": list(ANNOTATORS),
        "population_sha256": manifest["populations"][spec["population"]]["sha256"],
        "items": len(labels),
        "preregistration": ["study_002_prereg_v14", "study_002_prereg_v15"],
        "agreed_unknown_declined_despite_tool": declined_despite_tool(rows, references, spec["id"]),
    }
    if set_name == "trial":
        out["status"] = "TRIAL_NO_FLOOR"
        out["precision"] = ft.flag_precision(labels, source_of, sources=ft.SOURCES)
    elif set_name == "triage":
        # The members come from the flag set's own file (hash-checked), not from the population being reported.
        flag_manifest, members = pop._flag_set(root)
        if manifest["flag_set"]["members_sha256"] != flag_manifest["members"]["sha256"]:
            raise FlagTriageReportError("the populations were built from another flag set")
        flag_members = {member["opengrad_id"] for member in members}
        out["decision"] = ft.triage_decision(labels, source_of, flag_members)
        out["status"] = out["decision"]["decision"]
    else:
        recall = manifest["populations"]["recall"]
        flag_manifest, _ = pop._flag_set(root)
        # The weights are the whole population outside the flag set, per source and predicted label.
        population_by_label = ft.recall_population_by_label(flag_manifest["precision_reporting"])
        predicted_of = {row["opengrad_id"]: row["predicted_label"] for row in rows}
        out["recall"] = ft.recall_report(
            labels, source_of, population_by_label, predicted_of, recall["allocation"]
        )
        out["status"] = "REPORTED_NO_FLOOR"
    return out


def write(root: Path, set_name: str, result: dict[str, Any]) -> Path:
    path = root / pop.OUTPUT_DIR / f"flag-triage-{set_name}.report.json"
    if path.exists():
        raise FlagTriageReportError(f"{path.name} exists; a triage report is never overwritten")
    path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = report(ROOT, args.set)
    if args.write:
        write(ROOT, args.set, result)
    print(
        json.dumps({k: v for k, v in result.items() if k in ("set", "items", "status")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
