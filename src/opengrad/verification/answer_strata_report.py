"""The ANSWER strata of ANSWER-STRATA-v1, from its three-model consensus reference (41 §9-§11).

Reads the committed candidate population (hash-checked against its manifest) and the consensus reference that
``python -m opengrad.verification.pdet_coverage_reference --task answer-strata-v1`` wrote, and reports:

* **§9, the strata:** ``ANSWER-natural`` (pool N) and ``ANSWER-constructed`` (pool K), each on its own n. A
  pooled figure is printed only beside them. Pool N's other consensus labels are counted as a by-product;
  pool K items reaching another label are counted as construction failures.
* **§10, sizing:** each stratum's status against 06, with no redraw.
* **§11, balance:** each stratum's prompt length in characters (quartiles) and offered-tool counts. The
  comparison with the other three modes of the confirmatory partition waits for that partition to be built.

The membership file lists each stratum item's id and stratum, nothing else. Output is counts only: no item
text, id or label is printed.

    python -m opengrad.verification.answer_strata_report
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes as _sha256
from opengrad.verification import answer_strata as strata

ROOT = Path(__file__).resolve().parents[3]
TASK = "answer-strata-v1"
REFERENCE_DIR = strata.OUTPUT_DIR / "reference"
REFERENCE_NAME = f"{TASK}.reference.jsonl"
REFERENCE_MANIFEST_NAME = f"{TASK}.reference.manifest.json"
REPORT_NAME = "answer-strata-v1.strata.json"
MEMBERS_NAME = "answer-strata-v1.strata-members.jsonl"
STRATA = {"N": "ANSWER-natural", "K": "ANSWER-constructed"}
#: 41 §10, from 06-SPLIT-SPEC C2: (lower bound, status), highest first.
SIZING = (
    (601, "RESOLVES_8_POINTS"),
    (385, "RESOLVES_10_POINTS"),
    (200, "MEETS_FLOOR_ONLY"),
    (0, "UNDER_POWERED"),
)


class StrataReportError(ValueError):
    pass


def sizing_status(n: int) -> str:
    return next(status for bound, status in SIZING if n >= bound)


def quartiles(values: list[int]) -> dict[str, float] | None:
    """Minimum, the three quartiles (``statistics.quantiles``, inclusive method) and maximum."""
    if not values:
        return None
    if len(values) == 1:
        q1 = q2 = q3 = float(values[0])
    else:
        q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"min": min(values), "q1": q1, "median": q2, "q3": q3, "max": max(values)}


def balance(items: list[dict[str, Any]]) -> dict[str, Any]:
    tools = Counter(len(item.get("tools") or []) for item in items)
    return {
        "n": len(items),
        "prompt_chars": quartiles([len(str(item["user_message"])) for item in items]),
        "offered_tools": {str(k): tools[k] for k in sorted(tools)},
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The population, the reference records and the reference manifest, each checked against its hash."""
    directory = root / strata.OUTPUT_DIR
    manifest = json.loads((directory / strata.MANIFEST_NAME).read_text(encoding="utf-8"))
    population_bytes = (directory / strata.POPULATION_NAME).read_bytes()
    if _sha256(population_bytes) != manifest["population_sha256"]:
        raise StrataReportError("the population does not match its manifest")
    reference_dir = root / REFERENCE_DIR
    if not (reference_dir / REFERENCE_MANIFEST_NAME).is_file():
        raise StrataReportError(
            f"no consensus reference in {REFERENCE_DIR.as_posix()}; build it with "
            f"opengrad.verification.pdet_coverage_reference --task {TASK}"
        )
    reference_manifest = json.loads(
        (reference_dir / REFERENCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    reference_bytes = (reference_dir / REFERENCE_NAME).read_bytes()
    if _sha256(reference_bytes) != reference_manifest["reference_sha256"]:
        raise StrataReportError("the reference does not match its manifest")
    if reference_manifest.get("population_sha256") != manifest["population_sha256"]:
        raise StrataReportError("the reference was built on a different population")
    items = [json.loads(line) for line in population_bytes.decode("utf-8").splitlines() if line]
    return items, _jsonl(reference_dir / REFERENCE_NAME), reference_manifest


def build_report(
    items: list[dict[str, Any]], references: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(ref["answer_id"]): ref for ref in references}
    if set(by_id) != {str(item["answer_id"]) for item in items}:
        raise StrataReportError("the reference does not cover exactly the population's items")
    labels: dict[str, Counter[str]] = {pool: Counter() for pool in STRATA}
    consensus: dict[str, Counter[str]] = {pool: Counter() for pool in STRATA}
    members: dict[str, list[dict[str, Any]]] = {pool: [] for pool in STRATA}
    for item in items:
        pool, ref = str(item["pool"]), by_id[str(item["answer_id"])]
        labels[pool][ref["reference_label"] or ref["consensus"]] += 1
        consensus[pool][ref["consensus"]] += 1
        if ref["reference_label"] == "ANSWER":
            members[pool].append(item)
    strata_report = {
        STRATA[pool]: {
            "pool": pool,
            "candidates": sum(labels[pool].values()),
            "n": len(members[pool]),
            "sizing_status": sizing_status(len(members[pool])),
            "consensus": dict(sorted(consensus[pool].items())),
            "reference_labels": dict(sorted(labels[pool].items())),
            "balance": balance(members[pool]),
        }
        for pool in STRATA
    }
    strata_report["ANSWER-constructed"]["construction_failures"] = sum(
        count for label, count in labels["K"].items() if label not in ("ANSWER", "NO_CONSENSUS")
    )
    strata_report["ANSWER-natural"]["by_product_labels"] = {
        label: labels["N"][label] for label in ("CALL", "CLARIFY", "UNSUPPORTED")
    }
    pooled = len(members["N"]) + len(members["K"])
    report = {
        "strata": strata_report,
        "pooled_beside_strata": {"n": pooled, "sizing_status": sizing_status(pooled)},
        "balance_against_other_modes": {
            "status": "BLOCKED_PCONF_NOT_BUILT",
            "note": (
                "41 §11 compares each stratum with the other three modes of the confirmatory partition; "
                "that partition is built after these strata, so the comparison is made at the P-CONF build."
            ),
        },
    }
    membership = [
        {"answer_id": item["answer_id"], "stratum": STRATA[pool]}
        for pool in STRATA
        for item in sorted(members[pool], key=lambda item: str(item["answer_id"]))
    ]
    return report, membership


def build(root: Path = ROOT) -> dict[str, Any]:
    items, references, reference_manifest = load(root)
    report, membership = build_report(items, references)
    members_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in membership
    )
    document = {
        "artifact_kind": "ANSWER_STRATA_REPORT",
        "status": "MODEL_REFERENCE_PROVISIONAL",
        "statement": (
            "The ANSWER strata of ANSWER-STRATA-v1: the items a two-of-three consensus of three non-Claude "
            "models labels ANSWER, per pool. Model judgments, not human gold. Each stratum is judged on its own "
            "n (41 §10); no pool is redrawn."
        ),
        "amendment": strata.ADOPTION_AMENDMENT,
        "amendment_document": strata.PREREGISTRATION.as_posix(),
        "population_sha256": reference_manifest["population_sha256"],
        "reference_manifest": (REFERENCE_DIR / REFERENCE_MANIFEST_NAME).as_posix(),
        "reference_sha256": reference_manifest["reference_sha256"],
        "members_file": MEMBERS_NAME,
        "members_sha256": _sha256(members_bytes),
        "quartile_method": "statistics.quantiles(n=4, method='inclusive') over prompt characters",
        **report,
    }
    data = (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    directory = root / strata.OUTPUT_DIR
    for path, payload in (
        (directory / MEMBERS_NAME, members_bytes),
        (directory / REPORT_NAME, data),
    ):
        if path.exists() and path.read_bytes() != payload:
            raise StrataReportError(
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
    print(
        json.dumps(
            {key: document[key] for key in ("strata", "pooled_beside_strata", "members_sha256")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
