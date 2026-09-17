"""Counts-only supply audit for the population that must make DIRECT measurable (35 §2, step 1).

Doc 35 decided that DIRECT, the one mode blocking C1, is made measurable by ``prose-decision-classifier-v2``
and a new untouched population. Before that population is preregistered, this audit answers one question:
**can the unused part of the eligible pool supply at least 50 reference DIRECT items (30 of them in the
challenge strata R, Q, M and X), and any textual CALL at all?**

The pool is built exactly as the classifier's development and check sets build theirs
(:mod:`opengrad.verification.classifier_devcheck`): normalization-v3 layer B candidates eligible under
``prose-decision-input-v1``, minus P-DET-COVERAGE-v1's own draw exclusions (sentinel prompts, the QAD
recovery set, P-DET-v1), minus every P-DET-COVERAGE-v1 item, minus every item of ``prose-classifier-dev-v1``,
``-devcheck-v1`` and ``-devcheck-v2``, each by identity, normalized prompt and normalized response. It is then
deduplicated by raw hash, response, prompt and per-stratum response skeleton.

DIRECT yield per stratum and source is estimated from labels that already exist, reported separately:

* the P-DET-COVERAGE-v1 three-model consensus reference (34), and
* the Claude model labels of the development and check sets (33 §3-§5a).

Both populations are already development-exposed for v2 (35 §2). Only aggregate counts are read: no item
text, id, tool or rationale is printed or written. The projection multiplies deduplicated supply by yield;
it is an estimate for sizing, not a draw, and dedup survivors depend slightly on draw order.

    python scripts/audit_pdet_coverage_v2_supply.py            # print
    python scripts/audit_pdet_coverage_v2_supply.py --write    # also write reports/pdet-coverage-v2/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opengrad.data.classifier_input import HeldoutIndex
from opengrad.data.normalization_v3 import load_source_manifest
from opengrad.verification import classifier_devcheck as devcheck
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/pdet-coverage-v2/pdet-coverage-v2.supply-audit.json")
DEV_DIR = devset.OUTPUT_DIR
USED_SETS = devcheck.CheckSpec(
    check_id="pdet-coverage-v2-supply-audit",
    seed="unused",
    quota=0,
    id_prefix="unused",
    excludes=(
        *devcheck.CHECKS["v2"].excludes,
        (DEV_DIR / "prose-classifier-devcheck-v2.population.jsonl", "2f4eb1354903661c7d998dc4920b33110a70281b2eb17380247feeab26708c1f"),
    ),
    statement="",
)
COVERAGE_REFERENCE = Path("reports/pdet-coverage/reference/pdet-coverage-v1.reference.jsonl")
MODEL_LABELS = {
    "prose-classifier-dev-v1": (
        DEV_DIR / "prose-classifier-dev-v1.population.jsonl",
        DEV_DIR / "annotation/wip/prose-classifier-dev-v1.annotations.model.claude-opus-5.model-dev.jsonl",
    ),
    "prose-classifier-devcheck-v1": (
        DEV_DIR / "prose-classifier-devcheck-v1.population.jsonl",
        Path("reports/prose-classifier/devcheck/annotation/wip/prose-classifier-devcheck-v1.annotations.model.claude-opus-5.model-devcheck.jsonl"),
    ),
    "prose-classifier-devcheck-v2": (
        DEV_DIR / "prose-classifier-devcheck-v2.population.jsonl",
        Path("reports/prose-classifier/devcheck-v2/annotation/wip/prose-classifier-devcheck-v2.annotations.model.claude-opus-5.model-devcheck-v2.jsonl"),
    ),
}
CHALLENGE = ("X", "M", "R", "Q")
MIN_DIRECT = 50
MIN_DIRECT_CHALLENGE = 30
LABELS = ("CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def supply(root: Path) -> dict[str, Any]:
    coverage.check_input(root)
    roles = {source: {coverage.LAYER_A: False, coverage.LAYER_B: True} for source in coverage.sampling_roles(load_source_manifest(root))}
    candidates = coverage.collect_candidates(coverage._iter_input(root, roles), HeldoutIndex.load(root), roles)
    layer_b = [unit for unit in candidates.units if unit["layer"] == coverage.LAYER_B]
    kept, _stats = coverage.apply_draw_exclusions(layer_b, coverage.load_draw_exclusions(root))
    coverage_exclusion = devset.load_coverage_exclusion(root)
    used_exclusion = devcheck.load_devset_exclusion(root, USED_SETS)
    not_coverage = [unit for unit in kept if not coverage_exclusion.hits(unit)]
    unused = [unit for unit in not_coverage if not used_exclusion.hits(unit)]
    survivors, dedup_stats = devset.deduplicate(unused)
    before = Counter((unit["stratum"], unit["source_name"]) for unit in unused)
    after = Counter((unit["stratum"], unit["source_name"]) for unit in survivors)
    sources = sorted({source for _stratum, source in before})
    manifest_roles = coverage.sampling_roles(load_source_manifest(root))
    return {
        # 30 §6: only these sources may feed a layer B population; the others (When2Call, covered by P-DET-v1)
        # are listed because the development sets drew from them, and are left out of the projection.
        "layer_b_candidate_sources": sorted(s for s, role in manifest_roles.items() if role[coverage.LAYER_B]),
        "counts": {
            "layer_b_candidates": len(layer_b),
            "after_construction_exclusions": len(kept),
            "pdet_coverage_v1_overlap_removed": len(kept) - len(not_coverage),
            "development_and_check_set_overlap_removed": len(not_coverage) - len(unused),
            "dedup": dedup_stats,
            "after_dedup": len(survivors),
        },
        "sources": sources,
        "by_stratum": {
            name: {source: {"records": before[(name, source)], "after_dedup": after[(name, source)]} for source in sources}
            for name in coverage.STRATA
        },
    }


def _tally(pairs: list[tuple[str, str, str]]) -> dict[str, dict[str, dict[str, int]]]:
    table: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for stratum, source, label in pairs:
        table[stratum][source][label] += 1
    return {
        name: {source: {label: counts[label] for label in LABELS} | {"items": sum(counts.values())} for source, counts in sorted(table[name].items())}
        for name in coverage.STRATA
        if name in table
    }


def yields(root: Path) -> dict[str, Any]:
    population = {item["pdetcov_id"]: item for item in _lines(root / devset.COVERAGE_POPULATION)}
    reference = [
        (population[row["pdetcov_id"]]["stratum"], population[row["pdetcov_id"]]["source_name"], row["reference_label"])
        for row in _lines(root / COVERAGE_REFERENCE)
    ]
    model: list[tuple[str, str, str]] = []
    for population_path, labels_path in MODEL_LABELS.values():
        items = {item["dev_id"]: item for item in _lines(root / population_path)}
        model.extend((items[r["dev_id"]]["stratum"], items[r["dev_id"]]["source_name"], r["gold_policy_label"]) for r in _lines(root / labels_path))
    return {
        "pdet_coverage_v1_consensus_reference": _tally(reference),
        "development_and_check_set_model_labels": _tally(model),
    }


def _rate(cell: dict[str, int] | None, label: str) -> tuple[float | None, int]:
    if not cell or not cell["items"]:
        return None, 0
    return cell[label] / cell["items"], cell["items"]


def projection(supplied: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Expected reference DIRECT (and CALL) if the whole deduplicated supply were drawn, per yield basis."""
    out: dict[str, Any] = {}
    for basis, table in observed.items():
        cells = {}
        totals = Counter()
        for name in coverage.STRATA:
            for source in supplied["layer_b_candidate_sources"]:
                available = supplied["by_stratum"][name][source]["after_dedup"]
                cell = table.get(name, {}).get(source)
                direct_rate, n = _rate(cell, "DIRECT")
                call_rate, _ = _rate(cell, "CALL")
                expected_direct = round(available * direct_rate, 1) if direct_rate is not None else None
                expected_call = round(available * call_rate, 1) if call_rate is not None else None
                cells[f"{name}/{source}"] = {
                    "available_after_dedup": available,
                    "yield_items_observed": n,
                    "direct_yield": round(direct_rate, 3) if direct_rate is not None else None,
                    "expected_direct_if_all_drawn": expected_direct,
                    "expected_call_if_all_drawn": expected_call,
                }
                if expected_direct is not None:
                    totals["direct"] += expected_direct
                    if name in CHALLENGE:
                        totals["direct_challenge"] += expected_direct
                    if source == "toolace":
                        totals["direct_toolace"] += expected_direct
                if expected_call is not None:
                    totals["call"] += expected_call
                if direct_rate is None and available:
                    totals["available_without_yield_estimate"] += available
        out[basis] = {
            "cells": cells,
            "expected_direct_if_all_drawn": round(totals["direct"], 1),
            "expected_direct_in_challenge_strata_if_all_drawn": round(totals["direct_challenge"], 1),
            "expected_direct_from_toolace_if_all_drawn": round(totals["direct_toolace"], 1),
            "expected_textual_call_if_all_drawn": round(totals["call"], 1),
            "available_items_without_yield_estimate": totals["available_without_yield_estimate"],
            "minimum_direct": MIN_DIRECT,
            "minimum_direct_challenge": MIN_DIRECT_CHALLENGE,
        }
    return out


def audit(root: Path = ROOT) -> dict[str, Any]:
    supplied = supply(root)
    observed = yields(root)
    return {
        "artifact_kind": "PDET_COVERAGE_V2_SUPPLY_AUDIT",
        "authorization": "docs/research/study-002/35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md §2 step 1",
        "status": "COUNTS ONLY. Not a draw, not a preregistration, not gold. Yields come from model labels.",
        "excluded_sets": {
            "pdet_coverage_v1_population_sha256": devset.COVERAGE_POPULATION_SHA256,
            "development_and_check_sets": {path.as_posix(): sha for path, sha in USED_SETS.excludes},
            "construction_exclusions": "as P-DET-COVERAGE-v1's draw (30 §9), including P-DET-v1",
        },
        "input_sha256": {
            COVERAGE_REFERENCE.as_posix(): hashlib.sha256((root / COVERAGE_REFERENCE).read_bytes()).hexdigest(),
            **{labels.as_posix(): hashlib.sha256((root / labels).read_bytes()).hexdigest() for _p, labels in MODEL_LABELS.values()},
        },
        "supply": supplied,
        "label_counts_by_stratum_and_source": observed,
        "projection": projection(supplied, observed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = audit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        path = args.root / OUTPUT
        if path.exists():
            raise SystemExit(f"{OUTPUT.as_posix()} already exists; it is never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
