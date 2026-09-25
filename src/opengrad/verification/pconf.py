"""P-CONF-v1: Study 002's four-mode confirmatory population (06-SPLIT-SPEC, "Building P-CONF for four modes").

P-CONF is the union of:

1. **the existing confirmatory side**: the 1,277 When2Call items of
   ``reports/evaluation/behavioral-heldout-v2-partition.json`` (CALL 453, UNSUPPORTED 453, CLARIFY 371), reused
   unchanged so Study 002 stays comparable with Study 001's frozen numbers. That file is not edited: its bytes
   are pinned elsewhere;
2. **the ANSWER strata set** of 41 (``study_002_prereg_v9``), with the reference of 42 (``study_002_prereg_v10``):
   ``ANSWER-natural`` and ``ANSWER-constructed``, kept apart. Every ANSWER figure is reported per stratum, and a
   pooled figure only beside them (41 §9).

The build writes ``reports/study-002/pconf-v1/``:

* ``pconf-v1.answer-records.jsonl`` -- the ANSWER items as :class:`CanonicalEvaluationExample` rows, the record
  the evaluator loads, one ``stable_json`` line each, sorted by id. Gold is ``direct`` (canonical ``ANSWER``);
  the stratum rides in ``metadata``;
* ``pconf-v1.partition.json`` (+ ``.sha256``) -- the new partition artifact (06, "What this spec does not do"):
  every example id, the fingerprints, the gold-count table with C1 and C2 applied, the disjointness checks and
  the 41 §11 balance report.

A collision between an ANSWER item and the development side, P-DET-COVERAGE or the When2Call held-out text
refuses the build: 41 §10 forbids dropping a labelled item without an amendment.

The When2Call question text lives in git-ignored materialized shards, so ``--verify`` re-derives the balance
report and the text check only where they exist and reports ``BLOCKED_INPUT_MISSING`` otherwise; everything
else is re-derived from tracked files. Output is counts and hashes only.

    python -m opengrad.verification.pconf --build | --verify
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.canonical import CanonicalEvaluationExample, stable_json
from opengrad.evaluation.backends import _decision
from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata as strata
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.answer_strata_report import MEMBERS_NAME, REPORT_NAME, balance
from opengrad.verification.population_validators import v1_mode_coverage
from opengrad.verification.resolvability import resolvable_row

ROOT = Path(__file__).resolve().parents[3]
POPULATION_ID = "P-CONF-v1"
OUTPUT_DIR = Path("reports/study-002/pconf-v1")
RECORDS_NAME = "pconf-v1.answer-records.jsonl"
PARTITION_NAME = "pconf-v1.partition.json"
FROZEN_PARTITION = Path("reports/evaluation/behavioral-heldout-v2-partition.json")
EVALUATION_MANIFEST = Path("reports/evaluation/behavioral-heldout-v2.manifest.json")
PDET_COVERAGE = (
    Path("reports/pdet-coverage/pdet-coverage-v1.population.jsonl"),
    Path("reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl"),
)
MODES = ("CALL", "ANSWER", "CLARIFY", "UNSUPPORTED")
STRATA = ("ANSWER-natural", "ANSWER-constructed")
#: source_dataset -> (upstream revision, upstream split), from the pins of 41 §3.
UPSTREAM = {
    "bfcl_v4_irrelevance": (strata.GORILLA, "irrelevance"),
    "bfcl_v4_live_irrelevance": (strata.GORILLA, "live_irrelevance"),
    "nq_open_validation": (strata.NQ_OPEN, "validation"),
}
#: Fields re-derived from tracked files alone; the rest need the git-ignored When2Call shards.
TRACKED_FIELDS = (
    "example_ids",
    "fingerprints",
    "gold_counts",
    "answer_strata",
    "coverage",
    "resolvability",
    "components",
    "answer_records",
    "disjointness_tracked",
)
LOCAL_FIELDS = ("disjointness_when2call_text", "balance")


class PConfError(ValueError):
    pass


def fingerprint(ids: list[str]) -> str:
    """sha256 over the sorted ids, one per line: the rule of ``scripts/freeze_eval_partition.py``."""
    return sha256_bytes("".join(f"{i}\n" for i in sorted(ids)).encode("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ── Inputs ────────────────────────────────────────────────────────────────────


def confirmatory_side(root: Path) -> dict[str, Any]:
    """The frozen partition's confirmatory ids and gold counts, its fingerprint re-checked."""
    raw = (root / FROZEN_PARTITION).read_bytes()
    partition = json.loads(raw)
    ids = list(partition["example_ids"]["confirmatory"])
    dev = list(partition["example_ids"]["dev"])
    if fingerprint(ids) != partition["confirmatory"]["fingerprint"]:
        raise PConfError("the frozen partition's confirmatory ids no longer match its fingerprint")
    if fingerprint(dev) != partition["dev"]["fingerprint"]:
        raise PConfError("the frozen partition's dev ids no longer match its fingerprint")
    counts: Counter[str] = Counter()
    for raw_label, row in partition["per_class"].items():
        counts[_decision(raw_label)] += int(row["confirmatory"])
    if sum(counts.values()) != len(ids) or "UNKNOWN" in counts:
        raise PConfError("the frozen partition's per-class counts do not map onto its ids")
    return {
        "ids": ids,
        "dev_ids": dev,
        "gold_counts": dict(counts),
        "file_sha256": sha256_bytes(raw),
        "fingerprint": partition["confirmatory"]["fingerprint"],
        "dev_fingerprint": partition["dev"]["fingerprint"],
    }


def answer_side(root: Path) -> dict[str, Any]:
    """The ANSWER strata members and their population items, each file checked against its recorded hash."""
    directory = root / strata.OUTPUT_DIR
    manifest = json.loads((directory / strata.MANIFEST_NAME).read_text(encoding="utf-8"))
    population = (directory / strata.POPULATION_NAME).read_bytes()
    if sha256_bytes(population) != manifest["population_sha256"]:
        raise PConfError("the ANSWER strata population does not match its manifest")
    report = json.loads((directory / REPORT_NAME).read_text(encoding="utf-8"))
    members_bytes = (directory / MEMBERS_NAME).read_bytes()
    if sha256_bytes(members_bytes) != report["members_sha256"]:
        raise PConfError("the ANSWER strata members do not match the strata report")
    reference = json.loads((root / report["reference_manifest"]).read_text(encoding="utf-8"))
    if reference["reference_sha256"] != report["reference_sha256"]:
        raise PConfError("the strata report cites a different reference")
    items = {str(item["answer_id"]): item for item in _jsonl(directory / strata.POPULATION_NAME)}
    members = _jsonl(directory / MEMBERS_NAME)
    by_stratum: dict[str, list[str]] = {name: [] for name in STRATA}
    for member in members:
        by_stratum[str(member["stratum"])].append(str(member["answer_id"]))
    for name in STRATA:
        if len(by_stratum[name]) != report["strata"][name]["n"]:
            raise PConfError(f"{name}: the members file and the strata report disagree on n")
    return {
        "items": items,
        "by_stratum": by_stratum,
        "population_sha256": manifest["population_sha256"],
        "members_sha256": report["members_sha256"],
        "strata_report_sha256": sha256_bytes((directory / REPORT_NAME).read_bytes()),
        "reference_sha256": report["reference_sha256"],
        "reference_status": reference["status"],
        "reference_amendment": reference["amendment"],
    }


def when2call_examples(root: Path) -> list[CanonicalEvaluationExample] | None:
    """The 3,650 frozen held-out examples, or None when their git-ignored shards are absent."""
    from opengrad.evaluation.runner import load_evaluation_examples

    try:
        return load_evaluation_examples(root, root / EVALUATION_MANIFEST)
    except (FileNotFoundError, ValueError):
        return None


# ── Records and checks ────────────────────────────────────────────────────────


def answer_record(item: dict[str, Any], stratum: str) -> CanonicalEvaluationExample:
    """One ANSWER strata item as the evaluator's record. Only the question and tools reach a model."""
    revision, split = UPSTREAM[str(item["source_dataset"])]
    metadata: dict[str, Any] = {
        "eligibility": "evaluation_only",
        "population": POPULATION_ID,
        "stratum": stratum,
        "pool": item["pool"],
        "gold_status": "MODEL_REFERENCE_PROVISIONAL",
    }
    if item["pool"] == "N":
        metadata["upstream_id"] = item["source_id"]
    if item.get("reference_answers") is not None:
        metadata["reference_answers"] = item["reference_answers"]
    record = CanonicalEvaluationExample(
        example_id=str(item["answer_id"]),
        source={"dataset_id": item["source_dataset"], "split": split, "revision": revision},
        question=str(item["user_message"]),
        tools=list(item["tools"]),
        expected_decision="direct",
        candidates={},
        metadata=metadata,
    )
    record.validate()
    if _decision(record.expected_decision) != "ANSWER":
        raise PConfError("the ANSWER gold label does not map to ANSWER")
    return record


def records_bytes(records: list[CanonicalEvaluationExample]) -> bytes:
    rows = sorted((dataclasses.asdict(r) for r in records), key=lambda row: str(row["example_id"]))
    return "".join(stable_json(row) + "\n" for row in rows).encode("utf-8")


def text_collisions(questions: dict[str, str], others: list[str]) -> int:
    seen = {strata.normalise(text) for text in others}
    return sum(1 for text in questions.values() if strata.normalise(text) in seen)


def _pdet_coverage_messages(root: Path) -> list[str]:
    return [str(row["user_message"]) for path in PDET_COVERAGE for row in _jsonl(root / path)]


def assemble(root: Path, *, local: bool) -> tuple[dict[str, Any], bytes]:
    """The partition document and the ANSWER records bytes. ``local`` also derives the fields that need the
    When2Call shards; it raises when they are absent."""
    conf = confirmatory_side(root)
    ans = answer_side(root)
    records = [
        answer_record(ans["items"][answer_id], stratum)
        for stratum in STRATA
        for answer_id in ans["by_stratum"][stratum]
    ]
    payload = records_bytes(records)
    questions = {r.example_id: r.question for r in records}
    answer_ids = sorted(questions)

    id_overlap = len(set(answer_ids) & (set(conf["ids"]) | set(conf["dev_ids"])))
    pdet_collisions = text_collisions(questions, _pdet_coverage_messages(root))
    if id_overlap or pdet_collisions:
        raise PConfError(
            f"ANSWER items collide with the held-out ids ({id_overlap}) or P-DET-COVERAGE text "
            f"({pdet_collisions}); 41 §10 forbids dropping them without an amendment"
        )

    gold = dict(conf["gold_counts"])
    gold["ANSWER"] = len(records)
    gold = {mode: gold.get(mode, 0) for mode in MODES}
    coverage = v1_mode_coverage(gold, enforce_floor=True)
    stratum_n = {name: len(ans["by_stratum"][name]) for name in STRATA}
    all_ids = sorted(conf["ids"]) + answer_ids
    document: dict[str, Any] = {
        "artifact_kind": "EVAL_PARTITION_FOUR_MODE",
        "schema_version": 1,
        "partition_id": POPULATION_ID,
        "role": "confirmatory: scored once per arm, on a checkpoint chosen in advance (06, one-shot discipline)",
        "preregistration": {
            "specification": "docs/research/study-002/06-SPLIT-SPEC.md",
            "answer_source": {
                "amendment": strata.ADOPTION_AMENDMENT,
                "document": strata.PREREGISTRATION.as_posix(),
            },
            "answer_reference": {
                "amendment": ans["reference_amendment"],
                "document": "docs/research/study-002/42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md",
            },
        },
        "components": {
            "confirmatory_when2call": {
                "partition": FROZEN_PARTITION.as_posix(),
                "partition_sha256": conf["file_sha256"],
                "evaluation_manifest": EVALUATION_MANIFEST.as_posix(),
                "fingerprint": conf["fingerprint"],
                "count": len(conf["ids"]),
                "gold_status": "upstream When2Call gold, reused unchanged from Study 001",
            },
            "answer_strata": {
                "population_sha256": ans["population_sha256"],
                "members": (strata.OUTPUT_DIR / MEMBERS_NAME).as_posix(),
                "members_sha256": ans["members_sha256"],
                "strata_report": (strata.OUTPUT_DIR / REPORT_NAME).as_posix(),
                "strata_report_sha256": ans["strata_report_sha256"],
                "reference_sha256": ans["reference_sha256"],
                "gold_status": ans["reference_status"],
                "count": len(records),
            },
        },
        "answer_records": {
            "file": RECORDS_NAME,
            "content_hash": sha256_bytes(payload),
            "records": len(records),
            "encoding": "opengrad.data.canonical.stable_json per CanonicalEvaluationExample row + LF, sorted by example_id",
            "expected_decision": "direct (canonical ANSWER)",
        },
        "example_ids": {
            "confirmatory_when2call": sorted(conf["ids"]),
            **{name: sorted(ans["by_stratum"][name]) for name in STRATA},
        },
        "fingerprints": {
            "partition": fingerprint(all_ids),
            "confirmatory_when2call": conf["fingerprint"],
            **{name: fingerprint(ans["by_stratum"][name]) for name in STRATA},
        },
        "gold_counts": gold,
        "answer_strata": {
            **stratum_n,
            "rule": "every ANSWER figure is reported per stratum; a pooled figure only beside them (41 §9)",
        },
        "coverage": {"status": coverage.status, "errors": coverage.all_errors(), "floor": 200},
        "resolvability": {
            **{mode: resolvable_row(gold[mode]) for mode in MODES},
            **{name: resolvable_row(stratum_n[name]) for name in STRATA},
        },
        "disjointness_tracked": {
            "answer_vs_heldout_ids": id_overlap,
            "answer_vs_pdet_coverage_text": pdet_collisions,
            "pdet_coverage_populations": [p.as_posix() for p in PDET_COVERAGE],
            "training_corpora": "screened before labelling (41 §6); see the ANSWER strata manifest",
            "p_unans_and_p_sealed": "not built; whichever is built later must exclude P-CONF",
        },
        "one_shot": {
            "scored": False,
            "rule": "a second scoring pass by the same arm on this partition is FAIL_ONE_SHOT; a forced re-score is "
            "a recorded replacement with both run ids kept (06)",
        },
        "statement": (
            "Study 002's confirmatory population: the When2Call confirmatory side unchanged, plus the ANSWER strata. "
            "The ANSWER gold is a provisional model reference (two non-Claude models agreeing), not human gold. "
            + " ".join(
                f"{name} (n = {stratum_n[name]}) resolves margins of "
                f"{100 * resolvable_row(stratum_n[name])['resolvable_margin']:.1f} points or more."  # type: ignore[operator]
                for name in STRATA
            )
        ),
    }
    if local:
        examples = when2call_examples(root)
        if examples is None:
            raise PConfError(
                "the When2Call held-out shards are required for the build; they are git-ignored"
            )
        conf_ids = set(conf["ids"])
        w2c_collisions = text_collisions(questions, [e.question for e in examples])
        if w2c_collisions:
            raise PConfError(f"{w2c_collisions} ANSWER items repeat a When2Call held-out question")
        by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES if mode != "ANSWER"}
        for example in examples:
            if example.example_id in conf_ids:
                by_mode[_decision(example.expected_decision)].append(
                    {"user_message": example.question, "tools": example.tools}
                )
        document["disjointness_when2call_text"] = {
            "answer_vs_when2call_heldout_questions": w2c_collisions,
            "heldout_questions": len(examples),
        }
        document["balance"] = {
            "method": "prompt length in characters of the user message (quartiles, statistics.quantiles inclusive) and offered tools, per stratum and per other mode (41 §11)",
            **{
                name: balance(
                    [
                        {"user_message": q.question, "tools": q.tools}
                        for q in records
                        if q.metadata["stratum"] == name
                    ]
                )
                for name in STRATA
            },
            **{mode: balance(rows) for mode, rows in by_mode.items()},
            "note": "a difference is a stated limitation of any ANSWER-mode comparison, never corrected by reweighting after scores exist (41 §11)",
        }
    return document, payload


def document_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def build(root: Path = ROOT) -> dict[str, Any]:
    directory = root / OUTPUT_DIR
    targets = (directory / RECORDS_NAME, directory / PARTITION_NAME)
    if any(path.exists() for path in targets):
        raise PConfError(
            f"{OUTPUT_DIR.as_posix()} already holds P-CONF; a partition is never overwritten"
        )
    document, payload = assemble(root, local=True)
    if document["coverage"]["status"] != PASS:
        raise PConfError(f"C1 coverage fails: {document['coverage']['errors']}")
    directory.mkdir(parents=True, exist_ok=True)
    data = document_bytes(document)
    (directory / RECORDS_NAME).write_bytes(payload)
    (directory / PARTITION_NAME).write_bytes(data)
    (directory / (PARTITION_NAME + ".sha256")).write_bytes(f"{sha256_bytes(data)}\n".encode())
    return document


def verify(root: Path = ROOT) -> dict[str, Any]:
    directory = root / OUTPUT_DIR
    partition_path, records_path = directory / PARTITION_NAME, directory / RECORDS_NAME
    if not partition_path.is_file() or not records_path.is_file():
        return {"status": FAIL, "errors": [f"{POPULATION_ID} artifacts are missing"], "blocked": []}
    data = partition_path.read_bytes()
    committed = json.loads(data)
    errors: list[str] = []
    sidecar = directory / (PARTITION_NAME + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_bytes(data):
        errors.append("FAIL_HASH: the partition does not match its .sha256 sidecar")
    local = when2call_examples(root) is not None
    try:
        fresh, payload = assemble(root, local=local)
    except PConfError as exc:
        return {"status": FAIL, "errors": [f"FAIL_REBUILD: {exc}"], "blocked": []}
    fresh = json.loads(document_bytes(fresh))
    if records_path.read_bytes() != payload:
        errors.append("FAIL_REPRODUCIBILITY: the ANSWER records differ from a fresh build")
    fields = TRACKED_FIELDS + (LOCAL_FIELDS if local else ())
    errors += [
        f"FAIL_PROVENANCE: {key!r} differs from a fresh build"
        for key in fields
        if committed.get(key) != fresh.get(key)
    ]
    blocked = (
        []
        if local
        else [
            f"{BLOCKED_INPUT_MISSING}: When2Call shards absent; {', '.join(LOCAL_FIELDS)} not re-derived"
        ]
    )
    return {
        "status": FAIL if errors else BLOCKED_INPUT_MISSING if blocked else PASS,
        "partition_id": POPULATION_ID,
        "fingerprint": committed.get("fingerprints", {}).get("partition"),
        "gold_counts": committed.get("gold_counts"),
        "answer_strata": {name: committed.get("answer_strata", {}).get(name) for name in STRATA},
        "errors": errors,
        "blocked": blocked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        document = build(args.root)
        summary = {
            key: document[key]
            for key in ("gold_counts", "answer_strata", "coverage", "fingerprints")
        }
        summary["fingerprints"] = {"partition": document["fingerprints"]["partition"]}
        print(json.dumps(summary, indent=2))
        return 0
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in (PASS, BLOCKED_INPUT_MISSING) else 1


if __name__ == "__main__":
    sys.exit(main())
