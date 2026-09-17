"""A held-out development check for the prose decision classifier (33 §5a).

The development set ``prose-classifier-dev-v1`` shaped the classifier's rules, so agreement on it is in-sample
and overstates how well the rules carry to new text. Before the rules are frozen, they are scored **once** on
this second, disjoint set, drawn the same way with its own seed. It excludes, beyond everything the development
set excludes (P-DET-v1, the QAD set, sentinels, P-DET-COVERAGE-v1), every development-set item by identity,
normalized prompt and normalized response.

Its labels are model judgments made the same audited way. Agreement on it is still agreement with a model,
never accuracy. Nothing is tuned against it: a change after scoring is reported with both results.

    python -m opengrad.verification.classifier_devcheck --build
    python -m opengrad.verification.classifier_devcheck --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import HeldoutIndex, normalize_prompt
from opengrad.data.normalization_v3 import lf_sha256, load_source_manifest
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[3]
CHECK_ID = "prose-classifier-devcheck-v1"
SEED = "opengrad-prose-classifier-devcheck-v1"
QUOTAS = {name: 25 for name in coverage.STRATA}
OUTPUT_DIR = devset.OUTPUT_DIR
POPULATION_NAME = f"{CHECK_ID}.population.jsonl"
MANIFEST_NAME = f"{CHECK_ID}.manifest.json"
DEVSET_POPULATION = devset.OUTPUT_DIR / devset.POPULATION_NAME
DEVSET_POPULATION_SHA256 = "abbdfc497dca61821bcdf35624988e82aa12ba5b562564c9aa2fe36fb0c72456"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank_key(stratum: str, item_id: str) -> str:
    return _sha256(f"{SEED}|{stratum}|{item_id}".encode())


def order_key(item_id: str) -> str:
    return _sha256(f"{SEED}:order:{item_id}".encode())


def load_devset_exclusion(root: Path) -> devset.CoverageExclusion:
    """The development set's items, in the same shape as the P-DET-COVERAGE-v1 exclusion."""
    path = root / DEVSET_POPULATION
    data = path.read_bytes() if path.is_file() else b""
    if _sha256(data) != DEVSET_POPULATION_SHA256:
        raise devset.DevsetError(f"{DEVSET_POPULATION.as_posix()} is missing or not prose-classifier-dev-v1")
    identities: set[str] = set()
    prompts: set[str] = set()
    responses: set[str] = set()
    for line in data.decode("utf-8").splitlines():
        item = json.loads(line)
        identities.update(str(item[key]) for key in ("pdetcov_id", "raw_record_hash", "record_id", "canonical_hash"))
        prompts.add(normalize_prompt(item["user_message"]))
        responses.add(normalize_prompt(item["assistant_response"]))
    return devset.CoverageExclusion(frozenset(identities), frozenset(prompts), frozenset(responses))


def _ranked(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(units, key=lambda unit: (rank_key(unit["stratum"], unit["pdetcov_id"]), unit["pdetcov_id"]), reverse=True)
    supply = Counter(unit["stratum"] for unit in ranked)
    strata = sorted(supply, key=lambda name: (supply[name], coverage.STRATA.index(name)))
    return [unit for name in strata for unit in ranked if unit["stratum"] == name]


def draw(
    units: Iterable[dict[str, Any]],
    exclusions: coverage.DrawExclusions,
    coverage_exclusion: devset.CoverageExclusion,
    devset_exclusion: devset.CoverageExclusion,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure: the check set from layer B candidate units."""
    layer_b = [unit for unit in units if unit["layer"] == coverage.LAYER_B]
    kept, exclusion_stats = coverage.apply_draw_exclusions(layer_b, exclusions)
    not_coverage = [unit for unit in kept if not coverage_exclusion.hits(unit)]
    not_devset = [unit for unit in not_coverage if not devset_exclusion.hits(unit)]
    stats = Counter({"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0})
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors = []
    for unit in _ranked(not_devset):
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
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in _ranked(survivors):
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
        unit["dev_id"] = "devcheck:" + unit["pdetcov_id"].split(":", 1)[1]
        unit["dev_index"] = index
        for name in devset.LABEL_FIELDS:
            unit[name] = None
    counts = {
        "layer_b_candidates": len(layer_b),
        "construction_exclusions": exclusion_stats[coverage.LAYER_B],
        "pdet_coverage_v1_overlap_removed": len(kept) - len(not_coverage),
        "devset_overlap_removed": len(not_coverage) - len(not_devset),
        "dedup": dict(stats),
        "after_dedup": len(survivors),
        "strata": per_stratum,
        "realized": len(population),
    }
    return population, counts


def build(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_record = coverage.check_input(root)
    roles = {source: {coverage.LAYER_A: False, coverage.LAYER_B: True} for source in coverage.sampling_roles(load_source_manifest(root))}
    candidates = coverage.collect_candidates(coverage._iter_input(root, roles), HeldoutIndex.load(root), roles)
    population, counts = draw(
        candidates.units,
        coverage.load_draw_exclusions(root),
        devset.load_coverage_exclusion(root),
        load_devset_exclusion(root),
    )
    payload = devset.population_bytes(population)
    manifest = {
        "artifact_kind": "PROSE_CLASSIFIER_DEVELOPMENT_CHECK_SET",
        "devset_id": CHECK_ID,
        "plan": devset.PLAN.as_posix(),
        "status": "DEVELOPMENT_ONLY_NEVER_GOLD",
        "seed": SEED,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending; strata scarcest first for dedup",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "input": input_record,
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "excludes": {
            "construction_exclusions": "as P-DET-COVERAGE-v1's draw (30 §9), including P-DET-v1",
            "pdet_coverage_v1_population_sha256": devset.COVERAGE_POPULATION_SHA256,
            "devset_population_sha256": DEVSET_POPULATION_SHA256,
        },
        "quotas": QUOTAS,
        "counts": counts,
        "population_file": POPULATION_NAME,
        "population_sha256": _sha256(payload),
        "code_sha256_lf": {
            module: lf_sha256(root / module)
            for module in (
                "src/opengrad/verification/classifier_devcheck.py",
                "src/opengrad/verification/classifier_devset.py",
                "src/opengrad/verification/pdet_coverage.py",
            )
        },
        "statement": (
            "A held-out development check for the prose decision classifier (33 §5a): scored once before the rules "
            "are frozen, to see how much of the development agreement carries to unseen items. Its labels are "
            "model judgments, never gold. It shares no item, prompt or response with prose-classifier-dev-v1, "
            "P-DET-v1 or P-DET-COVERAGE-v1."
        ),
    }
    return population, manifest


def write(root: Path, population: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    out = root / OUTPUT_DIR
    if (out / POPULATION_NAME).exists() or (out / MANIFEST_NAME).exists():
        raise devset.DevsetError(f"{out / POPULATION_NAME} already exists; it is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / POPULATION_NAME).write_bytes(devset.population_bytes(population))
    (out / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def verify(root: Path = ROOT) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    written = (out / POPULATION_NAME).read_bytes()
    population, rebuilt = build(root)
    coverage_exclusion = devset.load_coverage_exclusion(root)
    devset_exclusion = load_devset_exclusion(root)
    checks: dict[str, Any] = {
        "file_hash_matches_manifest": _sha256(written) == manifest["population_sha256"],
        "rederived_bytes_equal": devset.population_bytes(population) == written,
        "counts_equal": rebuilt["counts"] == manifest["counts"],
        "pdet_coverage_v1_overlap": sum(1 for unit in population if coverage_exclusion.hits(unit)),
        "devset_overlap": sum(1 for unit in population if devset_exclusion.hits(unit)),
    }
    checks["status"] = (
        "PASS"
        if all(checks[k] is True for k in ("file_hash_matches_manifest", "rederived_bytes_equal", "counts_equal"))
        and checks["pdet_coverage_v1_overlap"] == 0
        and checks["devset_overlap"] == 0
        else "FAIL"
    )
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
