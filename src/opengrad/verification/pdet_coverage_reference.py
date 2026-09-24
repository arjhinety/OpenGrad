"""P-DET-COVERAGE-v1's reference labels: a two-of-three consensus of three non-Claude models (34).

Amendment ``study_002_prereg_v5`` (docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md)
replaced the human-only reference of 30 §10. Three declared model annotators label every item independently
and blind (``scripts/run_external_annotation.py``). This module turns their labels into the reference:

* **reference label** -- the label (CALL, DIRECT, CLARIFY, UNSUPPORTED or UNKNOWN) at least two annotators
  gave;
* **NO_CONSENSUS** -- all three differ. The item keeps no reference label and is left out of every metric
  (34 §1);
* **agreement** -- how many items were unanimous, two-of-three or split, each pair's agreement, and each
  annotator's label counts. It is *model-model* agreement, never inter-annotator agreement (34 §4).

It reads only a **verified annotation package** (``opengrad-annotate export`` followed by
:func:`opengrad.annotation.export.verify_package`), never the working store, so the reference can be traced
to hashed files. Nothing here is human gold: the manifest says ``MODEL_REFERENCE`` and provisional.

    python -m opengrad.verification.pdet_coverage_reference --task pdet-coverage-v1 \\
        --package reports/pdet-coverage/annotation/wip/pdet-coverage-v1.annotation-manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes as _sha256

ROOT = Path(__file__).resolve().parents[3]
AMENDMENT = "study_002_prereg_v5"
AMENDMENT_DOCUMENT = "docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md"
#: The declared annotators of 34 §1, in a fixed order.
ANNOTATORS = ("model.gemini-3.8-flash-high", "model.gpt-5.6-sol", "model.deepseek-v4.1-flash")
OUTPUT_DIR = Path("reports/pdet-coverage/reference")
#: task -> (authorizing amendment, its document, output directory). P-DET-COVERAGE-v2 (36 §3.7) keeps 34's
#: three-model consensus under amendment study_002_prereg_v6.
TASK_SPECS: dict[str, tuple[str, str, Path]] = {
    "pdet-coverage-v1": (AMENDMENT, AMENDMENT_DOCUMENT, OUTPUT_DIR),
    "pdet-coverage-v1-routing": (AMENDMENT, AMENDMENT_DOCUMENT, OUTPUT_DIR),
    "pdet-coverage-v2": (
        "study_002_prereg_v6",
        "docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md",
        Path("reports/pdet-coverage-v2/reference"),
    ),
    # The ANSWER strata candidates (41 §9) use the same three annotators and the same consensus rule.
    "answer-strata-v1": (
        "study_002_prereg_v9",
        "docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md",
        Path("reports/study-002/answer-strata-v1/reference"),
    ),
}
#: The reference's artifact kind, by task; P-DET-COVERAGE tasks keep the kind their references were built with.
ARTIFACT_KINDS = {"answer-strata-v1": "ANSWER_STRATA_MODEL_CONSENSUS_REFERENCE"}
TASKS = tuple(TASK_SPECS)
UNANIMOUS = "unanimous"
MAJORITY = "two_of_three"
NO_CONSENSUS = "NO_CONSENSUS"
LABEL_KEY = "gold_policy_label"


class ReferenceError(ValueError):
    """The package cannot yield a reference: unverified, incomplete, or not these three annotators."""


def consensus(votes: Mapping[str, str]) -> dict[str, Any]:
    """The reference for one item from exactly the three declared annotators' labels."""
    if set(votes) != set(ANNOTATORS):
        raise ReferenceError(f"expected labels from {list(ANNOTATORS)}, got {sorted(votes)}")
    counts = Counter(votes.values())
    label, top = counts.most_common(1)[0]
    if top == 3:
        return {"reference_label": label, "consensus": UNANIMOUS, "dissenting_annotator": None}
    if top == 2:
        dissent = next(annotator for annotator, vote in votes.items() if vote != label)
        return {"reference_label": label, "consensus": MAJORITY, "dissenting_annotator": dissent}
    return {"reference_label": None, "consensus": NO_CONSENSUS, "dissenting_annotator": None}


def build_reference(
    records: Sequence[Mapping[str, Any]], item_ids: list[str], *, item_key: str = "pdetcov_id"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-item reference records and the summary, from exported session records of the three annotators.

    Refuses unless each declared annotator labelled every item exactly once.
    """
    votes: dict[str, dict[str, str]] = {item_id: {} for item_id in item_ids}
    ambiguity: dict[str, dict[str, Any]] = {item_id: {} for item_id in item_ids}
    for record in records:
        annotator = str(record.get("annotator_id"))
        if annotator not in ANNOTATORS:
            continue
        item_id = str(record.get(item_key))
        if item_id not in votes:
            raise ReferenceError(
                f"{annotator} labelled {item_id!r}, which is not an item of this task"
            )
        if record.get("status") != "labeled" or record.get(LABEL_KEY) is None:
            continue
        if annotator in votes[item_id]:
            raise ReferenceError(f"{annotator} has two records for {item_id}")
        votes[item_id][annotator] = str(record[LABEL_KEY])
        ambiguity[item_id][annotator] = record.get("ambiguity_status")
    missing = {
        annotator: sum(1 for item_id in item_ids if annotator not in votes[item_id])
        for annotator in ANNOTATORS
    }
    if any(missing.values()):
        raise ReferenceError(f"incomplete: items without a label, per annotator: {missing}")

    references = []
    for item_id in item_ids:
        result = consensus(votes[item_id])
        label = result["reference_label"]
        agreeing = [a for a in ANNOTATORS if label is not None and votes[item_id][a] == label]
        statuses = Counter(ambiguity[item_id][a] for a in agreeing)
        references.append(
            {
                item_key: item_id,
                **result,
                "reference_ambiguity_status": (
                    statuses.most_common(1)[0][0]
                    if statuses and statuses.most_common(1)[0][1] >= 2
                    else None
                ),
                "votes": {annotator: votes[item_id][annotator] for annotator in ANNOTATORS},
                "metric_eligible": label is not None,
            }
        )
    summary = {
        "items": len(item_ids),
        "consensus": dict(Counter(ref["consensus"] for ref in references)),
        "reference_labels": dict(
            sorted(Counter(ref["reference_label"] or NO_CONSENSUS for ref in references).items())
        ),
        "pairwise_agreement": {
            f"{a}|{b}": sum(1 for item_id in item_ids if votes[item_id][a] == votes[item_id][b])
            for a, b in combinations(ANNOTATORS, 2)
        },
        "annotator_label_counts": {
            annotator: dict(sorted(Counter(votes[i][annotator] for i in item_ids).items()))
            for annotator in ANNOTATORS
        },
        "dissent_by_annotator": dict(
            Counter(
                ref["dissenting_annotator"] for ref in references if ref["dissenting_annotator"]
            )
        ),
    }
    return references, summary


def load_package(
    task: str, package: Path, root: Path = ROOT
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """A verified export package of ``task``: its manifest, the three annotators' records and the item ids."""
    from opengrad.annotation.config import load_task_config
    from opengrad.annotation.export import verify_package
    from opengrad.annotation.items import load_source

    # verify_package returns the validation result and a status summary, not the manifest itself.
    result, verification = verify_package(package, root, require_source=True)
    errors = result.all_errors()
    if errors or verification.get("status") != "PASS":
        raise ReferenceError(
            "the package failed verification:\n  - " + "\n  - ".join(map(str, errors[:20]))
        )
    manifest = json.loads(package.read_text(encoding="utf-8"))
    if manifest.get("task_id") != task:
        raise ReferenceError(f"the package is for {manifest.get('task_id')!r}, not {task!r}")
    config = load_task_config(root / "configs" / "annotation" / f"{task}.yaml", root=root)
    _digest, items = load_source(config)
    records: list[dict[str, Any]] = []
    sessions = {s["annotator_id"]: s for s in manifest.get("sessions") or []}
    for annotator in ANNOTATORS:
        if annotator not in sessions:
            raise ReferenceError(f"the package has no session of {annotator}")
        path = package.parent / sessions[annotator]["file"]
        records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    return manifest, records, [item.item_id for item in items]


def build(task: str, package: Path, root: Path = ROOT) -> dict[str, Any]:
    """Write ``<task>.reference.jsonl`` and its manifest under the task's output directory
    (:data:`TASK_SPECS`). Never overwrites a different reference: a rebuild must reproduce the same bytes."""
    from opengrad.annotation.config import load_task_config

    manifest, records, item_ids = load_package(task, package, root)
    config = load_task_config(root / "configs" / "annotation" / f"{task}.yaml", root=root)
    references, summary = build_reference(
        records, item_ids, item_key=config.source.id_field or "pdetcov_id"
    )
    declared = {item.annotator_id: item for item in config.model_annotators}
    payload = b"".join(
        (json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for ref in references
    )
    amendment, amendment_document, output_dir = TASK_SPECS[task]
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    reference_path = out / f"{task}.reference.jsonl"
    manifest_path = out / f"{task}.reference.manifest.json"
    reference_manifest = {
        "artifact_kind": ARTIFACT_KINDS.get(task, "PDET_COVERAGE_MODEL_CONSENSUS_REFERENCE"),
        "status": "MODEL_REFERENCE_PROVISIONAL",
        "statement": (
            "Reference labels from a two-of-three consensus of three non-Claude models, each labelling blind. "
            "Model judgments, not human gold; qualifications against it are MODEL_REFERENCE and provisional; "
            "agreement below is model-model agreement."
        ),
        "task_id": task,
        "amendment": amendment,
        "amendment_document": amendment_document,
        "annotators": [
            {
                "annotator_id": annotator,
                "model": declared[annotator].model,
                "procedure": declared[annotator].procedure,
                "procedure_sha256": declared[annotator].procedure_sha256,
            }
            for annotator in ANNOTATORS
        ],
        "population_sha256": manifest.get("source", {}).get("sha256")
        or manifest.get("source_sha256"),
        "package_manifest": package.resolve().relative_to(root.resolve()).as_posix(),
        "package_manifest_sha256": _sha256(package.read_bytes()),
        "reference_file": reference_path.name,
        "reference_sha256": _sha256(payload),
        "summary": summary,
    }
    manifest_bytes = (json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    for path, data in ((reference_path, payload), (manifest_path, manifest_bytes)):
        if path.exists() and path.read_bytes() != data:
            raise ReferenceError(
                f"{path} exists with different content; a reference is never overwritten"
            )
        path.write_bytes(data)
    return reference_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--package", required=True, type=Path, help="a verified annotation package manifest"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    built = build(args.task, args.package, args.root)
    # Counts only: no item id, label per item or item text is printed.
    print(
        json.dumps(
            {"task_id": built["task_id"], "status": built["status"], "summary": built["summary"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
