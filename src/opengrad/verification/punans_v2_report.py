"""P-UNANS-v2's trial report and main-set stratum, from the two-model references (44 §9-§10).

Written before any v2 label exists. Reads a committed set (hash-checked against the manifest) and the reference
that ``python -m opengrad.verification.pdet_coverage_reference --task punans-v2-trial | punans-v2`` wrote, and
reports:

* **agreement:** the share of items on which the two annotators give the same §8 label, Cohen's kappa beside
  it, and both per source and per question author. Three labels make chance agreement higher than v11's six,
  so the report carries 44 §10's caveat wherever it reports the floor;
* **trial set** (44 §9): the agreement and label counts only. No floor applies, and nothing is built;
* **main set** (44 §10, 03 stop rule 2, gate check 5): raw agreement must be at least 0.80. Below it the report
  records ``STOP`` and writes no membership. At or above it, ``P-UNANS-unknowable`` is the items both annotators
  label ``UNKNOWABLE``, sized with the 06 table: below 385 check 4 is ``UNDER_POWERED`` (40 item B).

The membership file lists each stratum item's id, nothing else. Output is counts only.

    python -m opengrad.verification.punans_v2_report --set trial | main
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes
from opengrad.verification import punans_v2
from opengrad.verification.answer_strata_report import balance, sizing_status
from opengrad.verification.punans_report import cohen_kappa
from opengrad.verification.resolvability import resolvable_row

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = punans_v2.OUTPUT_DIR / "reference"
SETS = {
    "trial": {
        "task": "punans-v2-trial",
        "report": "punans-v2-trial.report.json",
    },
    "main": {
        "task": "punans-v2",
        "report": "punans-v2.strata.json",
        "members": "punans-v2.strata-members.jsonl",
    },
}
LABELS = ("UNKNOWABLE", "NOT_UNKNOWABLE", "UNKNOWN")
STRATUM = "P-UNANS-unknowable"
AGREEMENT_FLOOR = 0.80
#: The unknowable stratum is the P-UNANS of check 4 (44 §10); 40 item B's size.
CHECK_4_MIN_N = 385
THREE_LABEL_CAVEAT = (
    "44 §10: the floor is 0.80 over three labels, where v11's was over six. Chance agreement is higher with "
    "fewer labels, so 0.80 here is easier to reach than v11's floor was; kappa is the figure that compares "
    "across the two attempts."
)


class PUnansV2ReportError(ValueError):
    pass


def _pairs(references: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs = []
    for ref in references:
        votes = ref["votes"]
        if len(votes) != 2:
            raise PUnansV2ReportError("each reference must hold exactly two votes")
        first, second = (votes[a] for a in sorted(votes))
        for label in (first, second):
            if label not in LABELS:
                raise PUnansV2ReportError(f"unexpected label {label!r}")
        pairs.append((first, second))
    return pairs


def agreement(references: list[dict[str, Any]], *, floor_applies: bool) -> dict[str, Any]:
    pairs = _pairs(references)
    same = sum(1 for a, b in pairs if a == b)
    raw = same / len(pairs) if pairs else 0.0
    result: dict[str, Any] = {
        "items": len(pairs),
        "same_label": same,
        "raw_agreement": round(raw, 6),
        "cohen_kappa": cohen_kappa(pairs),
        "floor": AGREEMENT_FLOOR,
        "caveat": THREE_LABEL_CAVEAT,
    }
    if floor_applies:
        result["status"] = "PASS" if pairs and raw >= AGREEMENT_FLOOR else "STOP"
        result["rule"] = (
            "03 stop rule 2 as set by 44 §10: below the floor the population is not built and the study "
            "does not proceed to training"
        )
    else:
        result["status"] = "TRIAL_NO_FLOOR"
        result["at_or_above_floor"] = bool(pairs) and raw >= AGREEMENT_FLOOR
        result["rule"] = "44 §9: the trial guides the procedure and never decides stop rule 2"
    return result


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def load(
    root: Path, set_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = SETS[set_name]
    directory = root / punans_v2.OUTPUT_DIR
    manifest = json.loads((directory / punans_v2.MANIFEST_NAME).read_text(encoding="utf-8"))
    population_spec = manifest["populations"][set_name]
    population = (directory / population_spec["file"]).read_bytes()
    if sha256_bytes(population) != population_spec["sha256"]:
        raise PUnansV2ReportError(f"the {set_name} set does not match its manifest")
    task = spec["task"]
    reference_dir = root / REFERENCE_DIR
    manifest_path = reference_dir / f"{task}.reference.manifest.json"
    if not manifest_path.is_file():
        raise PUnansV2ReportError(
            f"no reference in {REFERENCE_DIR.as_posix()}; build it with "
            f"opengrad.verification.pdet_coverage_reference --task {task}"
        )
    reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = (reference_dir / f"{task}.reference.jsonl").read_bytes()
    if sha256_bytes(reference) != reference_manifest["reference_sha256"]:
        raise PUnansV2ReportError("the reference does not match its manifest")
    if reference_manifest.get("population_sha256") != population_spec["sha256"]:
        raise PUnansV2ReportError("the reference was built on a different population")
    return _jsonl(population), _jsonl(reference), reference_manifest


def _grouped_agreement(
    items: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], field: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item[field]), []).append(by_id[str(item["punans_id"])])
    return {
        key: {
            k: v
            for k, v in agreement(refs, floor_applies=False).items()
            if k in ("items", "same_label", "raw_agreement", "cohen_kappa")
        }
        for key, refs in sorted(groups.items())
    }


def build_report(
    items: list[dict[str, Any]], references: list[dict[str, Any]], set_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(ref["punans_id"]): ref for ref in references}
    if set(by_id) != {str(item["punans_id"]) for item in items}:
        raise PUnansV2ReportError("the reference does not cover exactly the set's items")
    is_main = set_name == "main"
    floor = agreement(references, floor_applies=is_main)
    outcome: dict[str, Counter[str]] = {}
    for item in items:
        ref = by_id[str(item["punans_id"])]
        outcome.setdefault(str(item["source_dataset"]), Counter())[
            ref["reference_label"] or ref["consensus"]
        ] += 1
    report: dict[str, Any] = {
        "set": set_name,
        "agreement": floor,
        "agreement_by_source": _grouped_agreement(items, by_id, "source_dataset"),
        "agreement_by_author": _grouped_agreement(items, by_id, "source_author"),
        "reference_labels_by_source": {
            k: dict(sorted(v.items())) for k, v in sorted(outcome.items())
        },
    }
    if not is_main:
        return report, []
    group = [i for i in items if by_id[str(i["punans_id"])]["reference_label"] == "UNKNOWABLE"]
    report["strata"] = {
        STRATUM: {
            "n": len(group),
            "sizing_status": sizing_status(len(group)),
            "resolvability": resolvable_row(len(group)),
            "by_source": dict(sorted(Counter(str(i["source_dataset"]) for i in group).items())),
            "balance": balance(group),
        }
    }
    report["check_4"] = {
        "population": STRATUM,
        "n": len(group),
        "min_n": CHECK_4_MIN_N,
        "status": "EVALUABLE" if len(group) >= CHECK_4_MIN_N else "UNDER_POWERED",
    }
    membership = (
        [
            {"punans_id": item["punans_id"], "stratum": STRATUM}
            for item in sorted(group, key=lambda i: str(i["punans_id"]))
        ]
        if floor["status"] == "PASS"
        else []
    )
    return report, membership


def build(root: Path = ROOT, set_name: str = "main") -> dict[str, Any]:
    items, references, reference_manifest = load(root, set_name)
    report, membership = build_report(items, references, set_name)
    spec = SETS[set_name]
    members_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in membership
    )
    if set_name == "trial":
        status = "TRIAL_REPORT"
    elif report["agreement"]["status"] == "PASS":
        status = "MODEL_REFERENCE_PROVISIONAL"
    else:
        status = "STOPPED_AGREEMENT_FLOOR"
    document = {
        "artifact_kind": "PUNANS_V2_TRIAL_REPORT"
        if set_name == "trial"
        else "PUNANS_V2_STRATA_REPORT",
        "status": status,
        "statement": (
            "P-UNANS-v2's trial set: agreement of two non-Claude models, reported to guide the procedure; it "
            "never enters P-UNANS (44 §9)."
            if set_name == "trial"
            else "P-UNANS-v2's unknowable stratum: the main-set items two non-Claude models both label "
            "UNKNOWABLE (44 §10). Model judgments, not human gold; nothing is redrawn."
        ),
        "amendment": punans_v2.ADOPTION_AMENDMENT,
        "amendment_document": punans_v2.PREREGISTRATION.as_posix(),
        "population_sha256": reference_manifest["population_sha256"],
        "reference_manifest": (
            REFERENCE_DIR / f"{spec['task']}.reference.manifest.json"
        ).as_posix(),
        "reference_sha256": reference_manifest["reference_sha256"],
        **report,
    }
    if set_name == "main":
        document["members_file"] = spec["members"] if membership else None
        document["members_sha256"] = sha256_bytes(members_bytes) if membership else None
    data = (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    directory = root / punans_v2.OUTPUT_DIR
    outputs = [(directory / spec["report"], data)]
    if membership:
        outputs.append((directory / spec["members"], members_bytes))
    for path, payload in outputs:
        if path.exists() and path.read_bytes() != payload:
            raise PUnansV2ReportError(
                f"{path} exists with different content; it is never overwritten"
            )
        path.write_bytes(payload)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    document = build(args.root, args.set)
    # Counts only.
    keys = ("status", "agreement", "agreement_by_source", "reference_labels_by_source", "check_4")
    summary = {key: document[key] for key in keys if key in document}
    if "strata" in document:
        summary["strata"] = {
            name: {k: v for k, v in stratum.items() if k in ("n", "sizing_status", "by_source")}
            for name, stratum in document["strata"].items()
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
