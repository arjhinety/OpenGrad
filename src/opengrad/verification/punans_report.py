"""The P-UNANS strata of P-UNANS-v1, from its two-model reference (43 §9-§10).

Reads the committed candidate population (hash-checked against its manifest) and the reference that
``python -m opengrad.verification.pdet_coverage_reference --task punans-v1`` wrote, and reports:

* **the agreement floor** (03 stop rule 2, gate check 5): the share of items on which the two annotators give
  the same label must be at least 0.80. Cohen's kappa over the same labels is printed beside it. Below the
  floor the population is not built: the report records ``STOP`` and writes no membership;
* **the strata** by reference label: ``P-UNANS-unknowable`` (``UNKNOWABLE``) and ``P-UNANS-false-premise``
  (``FALSE_PREMISE``). Every other reference label, and every disagreement, is counted by pool and source and
  left out;
* **sizing** with the 06 table: the unknowable stratum is check 4's population, so below 385 check 4 is
  ``UNDER_POWERED`` (40 item B). The false-premise stratum enters no gate;
* **balance:** prompt length and offered tools per stratum.

The membership file lists each stratum item's id and stratum, nothing else. Output is counts only.

    python -m opengrad.verification.punans_report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes
from opengrad.verification import punans
from opengrad.verification.answer_strata_report import balance, sizing_status
from opengrad.verification.resolvability import resolvable_row

ROOT = Path(__file__).resolve().parents[3]
TASK = "punans-v1"
REFERENCE_DIR = punans.OUTPUT_DIR / "reference"
REFERENCE_NAME = f"{TASK}.reference.jsonl"
REFERENCE_MANIFEST_NAME = f"{TASK}.reference.manifest.json"
REPORT_NAME = "punans-v1.strata.json"
MEMBERS_NAME = "punans-v1.strata-members.jsonl"
STRATA = {"UNKNOWABLE": "P-UNANS-unknowable", "FALSE_PREMISE": "P-UNANS-false-premise"}
AGREEMENT_FLOOR = 0.80
#: The unknowable stratum is the P-UNANS of check 4 (43 §9); 40 item B's size.
CHECK_4_MIN_N = 385


class PUnansReportError(ValueError):
    pass


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters' labels on the same items; None when chance agreement is total."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[label] * second[label] for label in first.keys() | second.keys()) / (n * n)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 6)


def agreement(references: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [tuple(ref["votes"][a] for a in sorted(ref["votes"])) for ref in references]
    same = sum(1 for a, b in pairs if a == b)
    raw = same / len(pairs) if pairs else 0.0
    return {
        "items": len(pairs),
        "same_label": same,
        "raw_agreement": round(raw, 6),
        "floor": AGREEMENT_FLOOR,
        "cohen_kappa": cohen_kappa([(a, b) for a, b in pairs]),
        "status": "PASS" if pairs and raw >= AGREEMENT_FLOOR else "STOP",
        "rule": "03 stop rule 2 as set by 43 §9: below the floor the population is not built and the study "
        "does not proceed to training",
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = root / punans.OUTPUT_DIR
    manifest = json.loads((directory / punans.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = (directory / punans.POPULATION_NAME).read_bytes()
    if sha256_bytes(population) != manifest["population_sha256"]:
        raise PUnansReportError("the population does not match its manifest")
    reference_dir = root / REFERENCE_DIR
    if not (reference_dir / REFERENCE_MANIFEST_NAME).is_file():
        raise PUnansReportError(
            f"no reference in {REFERENCE_DIR.as_posix()}; build it with "
            f"opengrad.verification.pdet_coverage_reference --task {TASK}"
        )
    reference_manifest = json.loads(
        (reference_dir / REFERENCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if (
        sha256_bytes((reference_dir / REFERENCE_NAME).read_bytes())
        != reference_manifest["reference_sha256"]
    ):
        raise PUnansReportError("the reference does not match its manifest")
    if reference_manifest.get("population_sha256") != manifest["population_sha256"]:
        raise PUnansReportError("the reference was built on a different population")
    items = [json.loads(line) for line in population.decode("utf-8").splitlines() if line.strip()]
    return items, _jsonl(reference_dir / REFERENCE_NAME), reference_manifest


def build_report(
    items: list[dict[str, Any]], references: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(ref["punans_id"]): ref for ref in references}
    if set(by_id) != {str(item["punans_id"]) for item in items}:
        raise PUnansReportError("the reference does not cover exactly the population's items")
    floor = agreement(references)
    outcome: dict[str, Counter[str]] = {}
    members: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA.values()}
    for item in items:
        ref = by_id[str(item["punans_id"])]
        label = ref["reference_label"] or ref["consensus"]
        outcome.setdefault(f"{item['pool']}:{item['source_dataset']}", Counter())[label] += 1
        if ref["reference_label"] in STRATA:
            members[STRATA[ref["reference_label"]]].append(item)
    strata_report = {
        name: {
            "n": len(group),
            "sizing_status": sizing_status(len(group)),
            "resolvability": resolvable_row(len(group)),
            "by_source": dict(
                sorted(Counter(f"{i['pool']}:{i['source_dataset']}" for i in group).items())
            ),
            "balance": balance(group),
        }
        for name, group in members.items()
    }
    unknowable = len(members["P-UNANS-unknowable"])
    report = {
        "agreement": floor,
        "strata": strata_report,
        "check_4": {
            "population": "P-UNANS-unknowable",
            "n": unknowable,
            "min_n": CHECK_4_MIN_N,
            "status": "EVALUABLE" if unknowable >= CHECK_4_MIN_N else "UNDER_POWERED",
        },
        "false_premise_scoring": "NOT_EVALUABLE until a premise-rejection judge is specified and validated (43 §10)",
        "reference_labels_by_source": {
            k: dict(sorted(v.items())) for k, v in sorted(outcome.items())
        },
    }
    membership = (
        [
            {"punans_id": item["punans_id"], "stratum": name}
            for name, group in members.items()
            for item in sorted(group, key=lambda i: str(i["punans_id"]))
        ]
        if floor["status"] == "PASS"
        else []
    )
    return report, membership


def build(root: Path = ROOT) -> dict[str, Any]:
    items, references, reference_manifest = load(root)
    report, membership = build_report(items, references)
    members_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in membership
    )
    document = {
        "artifact_kind": "PUNANS_STRATA_REPORT",
        "status": "MODEL_REFERENCE_PROVISIONAL"
        if report["agreement"]["status"] == "PASS"
        else "STOPPED_AGREEMENT_FLOOR",
        "statement": (
            "The P-UNANS strata of P-UNANS-v1: the items two non-Claude models both label UNKNOWABLE or "
            "FALSE_PREMISE (43 §9). Model judgments, not human gold; no pool is redrawn."
        ),
        "amendment": punans.ADOPTION_AMENDMENT,
        "amendment_document": punans.PREREGISTRATION.as_posix(),
        "population_sha256": reference_manifest["population_sha256"],
        "reference_manifest": (REFERENCE_DIR / REFERENCE_MANIFEST_NAME).as_posix(),
        "reference_sha256": reference_manifest["reference_sha256"],
        "members_file": MEMBERS_NAME if membership else None,
        "members_sha256": sha256_bytes(members_bytes) if membership else None,
        **report,
    }
    data = (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    directory = root / punans.OUTPUT_DIR
    outputs = [(directory / REPORT_NAME, data)]
    if membership:
        outputs.append((directory / MEMBERS_NAME, members_bytes))
    for path, payload in outputs:
        if path.exists() and path.read_bytes() != payload:
            raise PUnansReportError(
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
    summary = {
        key: document[key]
        for key in ("status", "agreement", "check_4", "reference_labels_by_source")
    }
    summary["strata"] = {
        name: {k: v for k, v in stratum.items() if k in ("n", "sizing_status", "by_source")}
        for name, stratum in document["strata"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
