"""The prose decision classifier's development set (docs/research/study-002/33, §3).

A development set exists so the classifier's rules can be improved against labelled examples **without**
touching either validation population (22 §6-§7, 30 §9). It is drawn from normalization-v3 records eligible
under ``prose-decision-input-v1`` and excludes, before anything is drawn:

* everything P-DET-COVERAGE-v1's own draw excludes (sentinel prompts, the QAD recovery set, P-DET-v1 by
  identity, prompt or response; 30 §9), through :func:`opengrad.verification.pdet_coverage.load_draw_exclusions`;
* every **P-DET-COVERAGE-v1** item, by id, ``raw_record_hash``, normalized prompt or normalized response.

It then keeps one record per raw hash, per normalized response, per normalized prompt, and per response
skeleton within a stratum, and takes up to 50 per stratum of 30 §7.2, split equally across sources, in
the order of ``sha256(seed | stratum | id)`` with its own seed. Its labels are model judgments, used only
for development and never as gold (33 §4). Nothing here prints item text.

    python -m opengrad.verification.classifier_devset --build
    python -m opengrad.verification.classifier_devset --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import HeldoutIndex, normalize_prompt
from opengrad.data.normalization_v3 import lf_sha256, load_source_manifest
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[3]
DEVSET_ID = "prose-classifier-dev-v1"
SEED = "opengrad-prose-classifier-dev-v1"
PLAN = Path("docs/research/study-002/33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md")
QUOTAS = {name: 50 for name in coverage.STRATA}
OUTPUT_DIR = Path("reports/prose-classifier/dev")
POPULATION_NAME = f"{DEVSET_ID}.population.jsonl"
MANIFEST_NAME = f"{DEVSET_ID}.manifest.json"
COVERAGE_POPULATION = coverage.OUTPUT_DIR / coverage.POPULATION_NAME
#: The drawn P-DET-COVERAGE-v1 population this set must not overlap (30 §13).
COVERAGE_POPULATION_SHA256 = "755bc16e79ceb1cb9e461c0fe8b9125628c16f263ed24f9e008f55b613a7158c"
COVERAGE_OVERLAP = "PDET_COVERAGE_V1_OVERLAP"
LABEL_FIELDS = ("gold_policy_label", "ambiguity_status", "annotator_id", "annotator_rationale", "boundary_rule_cited", "annotation_version")


class DevsetError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank_key(stratum: str, item_id: str) -> str:
    return _sha256(f"{SEED}|{stratum}|{item_id}".encode())


def order_key(item_id: str) -> str:
    return _sha256(f"{SEED}:order:{item_id}".encode())


@dataclass(frozen=True)
class CoverageExclusion:
    """P-DET-COVERAGE-v1's items, as identities, normalized prompts and normalized responses."""

    identities: frozenset[str] = frozenset()
    prompts: frozenset[str] = frozenset()
    responses: frozenset[str] = frozenset()

    def hits(self, unit: Mapping[str, Any]) -> bool:
        identities = {unit["pdetcov_id"], unit["raw_record_hash"], unit["record_id"], unit["canonical_hash"]}
        response = unit.get("assistant_response")
        return bool(
            identities & self.identities
            or normalize_prompt(unit["user_message"]) in self.prompts
            or (isinstance(response, str) and normalize_prompt(response) in self.responses)
        )


def load_coverage_exclusion(root: Path) -> CoverageExclusion:
    path = root / COVERAGE_POPULATION
    data = path.read_bytes() if path.is_file() else b""
    if _sha256(data) != COVERAGE_POPULATION_SHA256:
        raise DevsetError(f"{COVERAGE_POPULATION.as_posix()} is missing or not the drawn P-DET-COVERAGE-v1 population")
    identities: set[str] = set()
    prompts: set[str] = set()
    responses: set[str] = set()
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        identities.update(
            str(item[key]) for key in ("pdetcov_id", "raw_record_hash", "record_id", "canonical_hash") if item.get(key)
        )
        prompts.add(normalize_prompt(item["user_message"]))
        if isinstance(item.get("assistant_response"), str):
            responses.add(normalize_prompt(item["assistant_response"]))
    return CoverageExclusion(frozenset(identities), frozenset(prompts), frozenset(responses))


def dedup_order(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strata scarcest first (ties in stratum order), each in this set's own rank order."""
    ranked = sorted(units, key=lambda unit: (rank_key(unit["stratum"], unit["pdetcov_id"]), unit["pdetcov_id"]), reverse=True)
    supply = Counter(unit["stratum"] for unit in ranked)
    strata = sorted(supply, key=lambda name: (supply[name], coverage.STRATA.index(name)))
    return [unit for name in strata for unit in ranked if unit["stratum"] == name]


def deduplicate(units: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = Counter({"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0})
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors = []
    for unit in dedup_order(units):
        keys = [
            ("raw_record_hash", unit["raw_record_hash"]),
            ("response_text", normalize_prompt(unit["assistant_response"])),
            ("user_prompt", normalize_prompt(unit["user_message"])),
            ("skeleton", (unit["stratum"], coverage.skeleton(unit["assistant_response"], unit["tools"]))),
        ]
        duplicate = next((name for name, key in keys if key in seen[name]), None)
        if duplicate is not None:
            stats[duplicate] += 1
            continue
        for name, key in keys:
            seen[name].add(key)
        survivors.append(unit)
    return survivors, dict(stats)


def draw(
    units: Iterable[dict[str, Any]], exclusions: coverage.DrawExclusions, coverage_exclusion: CoverageExclusion
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure: the development set from layer B candidate units."""
    layer_b = [unit for unit in units if unit["layer"] == coverage.LAYER_B]
    kept, exclusion_stats = coverage.apply_draw_exclusions(layer_b, exclusions)
    not_coverage = [unit for unit in kept if not coverage_exclusion.hits(unit)]
    survivors, dedup_stats = deduplicate(not_coverage)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in dedup_order(survivors):
        pools[(unit["stratum"], unit["source_name"])].append(unit)
    sources = sorted({source for _stratum, source in pools})
    chosen: list[dict[str, Any]] = []
    per_stratum: dict[str, Any] = {}
    for name in coverage.STRATA:
        supply = {source: len(pools.get((name, source), [])) for source in sources}
        allocation = coverage.allocate(QUOTAS[name], supply)
        for source, count in allocation.items():
            chosen.extend(pools.get((name, source), [])[:count])
        realized = sum(allocation.values())
        per_stratum[name] = {
            "quota": QUOTAS[name],
            "supply": sum(supply.values()),
            "realized": realized,
            "shortage": QUOTAS[name] - realized,
            "per_source": {s: {"supply": supply[s], "realized": allocation[s]} for s in sources},
        }
    population = sorted(chosen, key=lambda unit: (order_key(unit["pdetcov_id"]), unit["pdetcov_id"]))
    for index, unit in enumerate(population):
        unit["dev_id"] = "dev:" + unit["pdetcov_id"].split(":", 1)[1]
        unit["dev_index"] = index
        for name in LABEL_FIELDS:
            unit[name] = None
    counts = {
        "layer_b_candidates": len(layer_b),
        "construction_exclusions": exclusion_stats[coverage.LAYER_B],
        "pdet_coverage_v1_overlap_removed": len(kept) - len(not_coverage),
        "dedup": dedup_stats,
        "after_dedup": len(survivors),
        "strata": per_stratum,
        "realized": len(population),
    }
    return population, counts


def population_bytes(population: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for unit in population)


def build(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_record = coverage.check_input(root)
    roles = {source: {coverage.LAYER_A: False, coverage.LAYER_B: True} for source in coverage.sampling_roles(load_source_manifest(root))}
    heldout = HeldoutIndex.load(root)
    candidates = coverage.collect_candidates(coverage._iter_input(root, roles), heldout, roles)
    population, counts = draw(candidates.units, coverage.load_draw_exclusions(root), load_coverage_exclusion(root))
    payload = population_bytes(population)
    manifest = {
        "artifact_kind": "PROSE_CLASSIFIER_DEVELOPMENT_SET",
        "devset_id": DEVSET_ID,
        "plan": PLAN.as_posix(),
        "status": "DEVELOPMENT_ONLY_NEVER_GOLD",
        "seed": SEED,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending; strata scarcest first for dedup",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "input": input_record,
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "excludes": {
            "construction_exclusions": "as P-DET-COVERAGE-v1's draw (30 §9), including P-DET-v1",
            "pdet_coverage_v1_population_sha256": COVERAGE_POPULATION_SHA256,
        },
        "quotas": QUOTAS,
        "counts": counts,
        "population_file": POPULATION_NAME,
        "population_sha256": _sha256(payload),
        "code_sha256_lf": {
            module: lf_sha256(root / module)
            for module in ("src/opengrad/verification/classifier_devset.py", "src/opengrad/verification/pdet_coverage.py")
        },
        "statement": (
            "A development set for the prose decision classifier. Its labels are model judgments used only to "
            "develop the rules, never as gold and never as evidence of accuracy (33). It shares no item, prompt "
            "or response with P-DET-v1 or P-DET-COVERAGE-v1."
        ),
    }
    return population, manifest


def write(root: Path, population: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    out = root / OUTPUT_DIR
    if (out / POPULATION_NAME).exists() or (out / MANIFEST_NAME).exists():
        raise DevsetError(f"{out} already holds a development set; it is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / POPULATION_NAME).write_bytes(population_bytes(population))
    (out / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Re-hash the written set, re-derive it, and check it overlaps neither validation population."""
    out = root / OUTPUT_DIR
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    written = (out / POPULATION_NAME).read_bytes()
    population, rebuilt = build(root)
    overlap = sum(1 for unit in population if load_coverage_exclusion(root).hits(unit))
    checks = {
        "file_hash_matches_manifest": _sha256(written) == manifest["population_sha256"],
        "rederived_bytes_equal": population_bytes(population) == written,
        "counts_equal": rebuilt["counts"] == manifest["counts"],
        "pdet_coverage_v1_overlap": overlap,
    }
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k != "pdet_coverage_v1_overlap") and overlap == 0 else "FAIL"
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        population, manifest = build(args.root)
        write(args.root, population, manifest)
        print(json.dumps({"population_sha256": manifest["population_sha256"], "counts": manifest["counts"]}, indent=2))
        return 0
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
