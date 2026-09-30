"""P-UNANS-v2-constructed's floors, stratum and check 4's population (45 §8-§9).

Written before any label. Reads the committed constructed population (hash-checked against its manifest), the
reference that ``python -m opengrad.verification.pdet_coverage_reference --task punans-v2-constructed`` wrote,
and P-UNANS-v2's natural stratum, and reports:

* **the agreement floor:** the share of all 750 items on which the two annotators give the same label must be
  at least 0.80, with Cohen's kappa beside it;
* **the control rule:** at least 90% of the answerable controls must have the reference label
  ``NOT_UNKNOWABLE``, or the labellers did not tell the constructed questions from their answerable twins;
* **the constructed stratum:** the non-control items both annotators label ``UNKNOWABLE``, by family. It is
  built only when both rules pass;
* **check 4's population:** the union of P-UNANS-v2's natural stratum and the constructed stratum, each part
  counted separately. Below 385, check 4 is ``UNDER_POWERED`` (40 item B).

The membership file lists each constructed stratum item's id, nothing else. Output is counts only.

    python -m opengrad.verification.punans_constructed_report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes
from opengrad.verification import punans_constructed as pc
from opengrad.verification import punans_v2
from opengrad.verification.answer_strata_report import balance, sizing_status
from opengrad.verification.punans_report import cohen_kappa
from opengrad.verification.punans_v2_report import LABELS
from opengrad.verification.resolvability import resolvable_row

ROOT = Path(__file__).resolve().parents[3]
TASK = "punans-v2-constructed"
REFERENCE_DIR = pc.OUTPUT_DIR / "reference"
REPORT_NAME = "punans-v2-constructed.strata.json"
MEMBERS_NAME = "punans-v2-constructed.strata-members.jsonl"
NATURAL_REPORT = punans_v2.OUTPUT_DIR / "punans-v2.strata.json"
NATURAL_MEMBERS = punans_v2.OUTPUT_DIR / "punans-v2.strata-members.jsonl"
STRATUM = "P-UNANS-constructed-unknowable"
AGREEMENT_FLOOR = 0.80
CONTROL_FLOOR = 0.90
CHECK_4_MIN_N = 385


class PUnansConstructedReportError(ValueError):
    pass


def _pair(ref: dict[str, Any]) -> tuple[str, str]:
    votes = ref["votes"]
    if len(votes) != 2:
        raise PUnansConstructedReportError("each reference must hold exactly two votes")
    first, second = (votes[a] for a in sorted(votes))
    for label in (first, second):
        if label not in LABELS:
            raise PUnansConstructedReportError(f"unexpected label {label!r}")
    return first, second


def agreement(references: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [_pair(ref) for ref in references]
    same = sum(1 for a, b in pairs if a == b)
    raw = same / len(pairs) if pairs else 0.0
    return {
        "items": len(pairs),
        "same_label": same,
        "raw_agreement": round(raw, 6),
        "cohen_kappa": cohen_kappa(pairs),
        "floor": AGREEMENT_FLOOR,
        "status": "PASS" if pairs and raw >= AGREEMENT_FLOOR else "STOP",
        "rule": "45 §8: below the floor the constructed stratum is not built",
    }


def control_rule(items: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    controls = [i for i in items if i["is_control"]]
    answerable = sum(
        1 for i in controls if by_id[str(i["punans_id"])]["reference_label"] == "NOT_UNKNOWABLE"
    )
    share = answerable / len(controls) if controls else 0.0
    return {
        "controls": len(controls),
        "reference_not_unknowable": answerable,
        "share": round(share, 6),
        "floor": CONTROL_FLOOR,
        "status": "PASS" if controls and share >= CONTROL_FLOOR else "FAIL",
        "rule": "45 §8: below 90% the labellers did not tell the questions from their answerable twins, and "
        "the constructed stratum is not built",
    }


def build_report(
    items: list[dict[str, Any]], references: list[dict[str, Any]], natural_n: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(ref["punans_id"]): ref for ref in references}
    if set(by_id) != {str(item["punans_id"]) for item in items}:
        raise PUnansConstructedReportError(
            "the reference does not cover exactly the population's items"
        )
    floor = agreement(references)
    controls = control_rule(items, by_id)
    built = floor["status"] == "PASS" and controls["status"] == "PASS"
    outcome: dict[str, Counter[str]] = {}
    for item in items:
        ref = by_id[str(item["punans_id"])]
        key = f"{item['family']}:{'control' if item['is_control'] else 'question'}"
        outcome.setdefault(key, Counter())[ref["reference_label"] or ref["consensus"]] += 1
    group = [
        i
        for i in items
        if not i["is_control"] and by_id[str(i["punans_id"])]["reference_label"] == "UNKNOWABLE"
    ]
    constructed_n = len(group) if built else 0
    union = natural_n + constructed_n
    report = {
        "agreement": floor,
        "agreement_by_family": {
            family: {
                k: v
                for k, v in agreement(
                    [by_id[str(i["punans_id"])] for i in items if i["family"] == family]
                ).items()
                if k in ("items", "same_label", "raw_agreement", "cohen_kappa")
            }
            for family in pc.FAMILIES
        },
        "control_rule": controls,
        "constructed_stratum_built": built,
        "reference_labels_by_family": {
            k: dict(sorted(v.items())) for k, v in sorted(outcome.items())
        },
        "strata": {
            STRATUM: {
                "n": len(group),
                "counted_toward_check_4": constructed_n,
                "sizing_status": sizing_status(len(group)),
                "resolvability": resolvable_row(len(group)),
                "by_family": dict(sorted(Counter(str(i["family"]) for i in group).items())),
                "balance": balance(group),
            }
        },
        "check_4": {
            "population": "P-UNANS-v2 natural unknowable + P-UNANS-constructed-unknowable",
            "natural_n": natural_n,
            "constructed_n": constructed_n,
            "n": union,
            "min_n": CHECK_4_MIN_N,
            "status": "EVALUABLE" if union >= CHECK_4_MIN_N else "UNDER_POWERED",
        },
    }
    membership = (
        [
            {"punans_id": item["punans_id"], "stratum": STRATUM}
            for item in sorted(group, key=lambda i: str(i["punans_id"]))
        ]
        if built
        else []
    )
    return report, membership


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def load(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    directory = root / pc.OUTPUT_DIR
    manifest = json.loads((directory / pc.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = (directory / pc.POPULATION_NAME).read_bytes()
    if sha256_bytes(population) != manifest["population_sha256"]:
        raise PUnansConstructedReportError("the population does not match its manifest")
    manifest_path = root / REFERENCE_DIR / f"{TASK}.reference.manifest.json"
    if not manifest_path.is_file():
        raise PUnansConstructedReportError(
            f"no reference in {REFERENCE_DIR.as_posix()}; build it with "
            f"opengrad.verification.pdet_coverage_reference --task {TASK}"
        )
    reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = (root / REFERENCE_DIR / f"{TASK}.reference.jsonl").read_bytes()
    if sha256_bytes(reference) != reference_manifest["reference_sha256"]:
        raise PUnansConstructedReportError("the reference does not match its manifest")
    if reference_manifest.get("population_sha256") != manifest["population_sha256"]:
        raise PUnansConstructedReportError("the reference was built on a different population")
    natural = json.loads((root / NATURAL_REPORT).read_text(encoding="utf-8"))
    members = (root / NATURAL_MEMBERS).read_bytes()
    if sha256_bytes(members) != natural["members_sha256"]:
        raise PUnansConstructedReportError("P-UNANS-v2's natural members do not match its report")
    return _jsonl(population), _jsonl(reference), reference_manifest, len(_jsonl(members))


def build(root: Path = ROOT) -> dict[str, Any]:
    items, references, reference_manifest, natural_n = load(root)
    report, membership = build_report(items, references, natural_n)
    members_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in membership
    )
    if report["agreement"]["status"] != "PASS":
        status = "STOPPED_AGREEMENT_FLOOR"
    elif report["control_rule"]["status"] != "PASS":
        status = "NOT_BUILT_CONTROL_RULE"
    else:
        status = "MODEL_REFERENCE_PROVISIONAL"
    document = {
        "artifact_kind": "PUNANS_CONSTRUCTED_STRATA_REPORT",
        "status": status,
        "statement": (
            "P-UNANS-v2-constructed: the non-control items two non-Claude models both label UNKNOWABLE (45 §8), "
            "and check 4's population, the union with P-UNANS-v2's natural stratum. Model judgments, not human "
            "gold; nothing is redrawn."
        ),
        "amendment": pc.ADOPTION_AMENDMENT,
        "amendment_document": pc.PREREGISTRATION.as_posix(),
        "population_sha256": reference_manifest["population_sha256"],
        "reference_manifest": (REFERENCE_DIR / f"{TASK}.reference.manifest.json").as_posix(),
        "reference_sha256": reference_manifest["reference_sha256"],
        "natural_members_file": NATURAL_MEMBERS.as_posix(),
        "members_file": MEMBERS_NAME if membership else None,
        "members_sha256": sha256_bytes(members_bytes) if membership else None,
        **report,
    }
    data = (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    directory = root / pc.OUTPUT_DIR
    outputs = [(directory / REPORT_NAME, data)]
    if membership:
        outputs.append((directory / MEMBERS_NAME, members_bytes))
    for path, payload in outputs:
        if path.exists() and path.read_bytes() != payload:
            raise PUnansConstructedReportError(
                f"{path} exists with different content; it is never overwritten"
            )
        path.write_bytes(payload)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    document = build(args.root)
    # Counts only.
    keys = (
        "status",
        "agreement",
        "agreement_by_family",
        "control_rule",
        "reference_labels_by_family",
        "check_4",
    )
    summary = {key: document[key] for key in keys}
    summary["strata"] = {
        name: {k: v for k, v in s.items() if k in ("n", "counted_toward_check_4", "by_family")}
        for name, s in document["strata"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
