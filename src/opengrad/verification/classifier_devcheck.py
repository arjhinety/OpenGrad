"""Held-out development checks for the prose decision classifier (33 §5a).

The development set ``prose-classifier-dev-v1`` shaped the classifier's rules, so agreement on it is in-sample
and overstates how well the rules carry to new text. Before the rules are frozen, they are scored **once** on a
check set the developer has not read, drawn like the development set with its own seed. A check set excludes,
beyond everything the development set excludes (P-DET-v1, the QAD set, sentinels, P-DET-COVERAGE-v1), every item
of each earlier development or check set, by identity, normalized prompt and normalized response.

* ``prose-classifier-devcheck-v1`` scored the round-1 rules; the developer read its disagreements in round 2,
  so it is exposed and now development data.
* ``prose-classifier-devcheck-v2`` scores the round-2 rules and excludes both earlier sets.

Their labels are model judgments made the same audited way. Agreement on them is agreement with a model, never
accuracy.

    python -m opengrad.verification.classifier_devcheck --check v2 --build
    python -m opengrad.verification.classifier_devcheck --check v2 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import HeldoutIndex, normalize_prompt
from opengrad.data.normalization_v3 import lf_sha256, load_source_manifest
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = devset.OUTPUT_DIR
DEVSET_POPULATION = devset.OUTPUT_DIR / devset.POPULATION_NAME
DEVSET_POPULATION_SHA256 = "abbdfc497dca61821bcdf35624988e82aa12ba5b562564c9aa2fe36fb0c72456"


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    seed: str
    quota: int
    id_prefix: str
    #: Earlier development or check sets this one must not overlap: (population path, pinned sha256).
    excludes: tuple[tuple[Path, str], ...]
    statement: str

    @property
    def population_name(self) -> str:
        return f"{self.check_id}.population.jsonl"

    @property
    def manifest_name(self) -> str:
        return f"{self.check_id}.manifest.json"

    @property
    def quotas(self) -> dict[str, int]:
        return {name: self.quota for name in coverage.STRATA}


CHECKS = {
    "v1": CheckSpec(
        check_id="prose-classifier-devcheck-v1",
        seed="opengrad-prose-classifier-devcheck-v1",
        quota=25,
        id_prefix="devcheck",
        excludes=((DEVSET_POPULATION, DEVSET_POPULATION_SHA256),),
        statement=(
            "A held-out development check for the prose decision classifier (33 §5a): scored once before the rules "
            "are frozen, to see how much of the development agreement carries to unseen items. Its labels are "
            "model judgments, never gold. It shares no item, prompt or response with prose-classifier-dev-v1, "
            "P-DET-v1 or P-DET-COVERAGE-v1."
        ),
    ),
    "v2": CheckSpec(
        check_id="prose-classifier-devcheck-v2",
        seed="opengrad-prose-classifier-devcheck-v2",
        quota=25,
        id_prefix="devcheck-v2",
        excludes=(
            (DEVSET_POPULATION, DEVSET_POPULATION_SHA256),
            (
                OUTPUT_DIR / "prose-classifier-devcheck-v1.population.jsonl",
                "56564d85fa0db4abbc971b16bac5557edea8391620ccc3c3816a144cea651348",
            ),
        ),
        statement=(
            "The second held-out development check for the prose decision classifier (33 §5a): scored once on the "
            "round-2 rules, because the first check set was read in round 2. Its labels are model judgments, never "
            "gold. It shares no item, prompt or response with prose-classifier-dev-v1, "
            "prose-classifier-devcheck-v1, P-DET-v1 or P-DET-COVERAGE-v1."
        ),
    ),
}
V1 = CHECKS["v1"]
# The v1 names, kept for callers written before v2 existed.
CHECK_ID, SEED, QUOTAS = V1.check_id, V1.seed, V1.quotas
POPULATION_NAME, MANIFEST_NAME = V1.population_name, V1.manifest_name


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank_key(stratum: str, item_id: str, seed: str = SEED) -> str:
    return _sha256(f"{seed}|{stratum}|{item_id}".encode())


def order_key(item_id: str, seed: str = SEED) -> str:
    return _sha256(f"{seed}:order:{item_id}".encode())


def load_devset_exclusion(root: Path, spec: CheckSpec = V1) -> devset.CoverageExclusion:
    """Every earlier development or check set's items, in the shape of the P-DET-COVERAGE-v1 exclusion."""
    identities: set[str] = set()
    prompts: set[str] = set()
    responses: set[str] = set()
    for relative, expected in spec.excludes:
        path = root / relative
        data = path.read_bytes() if path.is_file() else b""
        if _sha256(data) != expected:
            raise devset.DevsetError(f"{relative.as_posix()} is missing or not the pinned population")
        for line in data.decode("utf-8").splitlines():
            item = json.loads(line)
            identities.update(str(item[key]) for key in ("pdetcov_id", "raw_record_hash", "record_id", "canonical_hash"))
            prompts.add(normalize_prompt(item["user_message"]))
            responses.add(normalize_prompt(item["assistant_response"]))
    return devset.CoverageExclusion(frozenset(identities), frozenset(prompts), frozenset(responses))


def _ranked(units: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    ranked = sorted(units, key=lambda unit: (rank_key(unit["stratum"], unit["pdetcov_id"], seed), unit["pdetcov_id"]), reverse=True)
    supply = Counter(unit["stratum"] for unit in ranked)
    strata = sorted(supply, key=lambda name: (supply[name], coverage.STRATA.index(name)))
    return [unit for name in strata for unit in ranked if unit["stratum"] == name]


def draw(
    units: Iterable[dict[str, Any]],
    exclusions: coverage.DrawExclusions,
    coverage_exclusion: devset.CoverageExclusion,
    devset_exclusion: devset.CoverageExclusion,
    spec: CheckSpec = V1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure: the check set from layer B candidate units."""
    layer_b = [unit for unit in units if unit["layer"] == coverage.LAYER_B]
    kept, exclusion_stats = coverage.apply_draw_exclusions(layer_b, exclusions)
    not_coverage = [unit for unit in kept if not coverage_exclusion.hits(unit)]
    not_devset = [unit for unit in not_coverage if not devset_exclusion.hits(unit)]
    stats = Counter({"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0})
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors = []
    for unit in _ranked(not_devset, spec.seed):
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
    for unit in _ranked(survivors, spec.seed):
        pools[(unit["stratum"], unit["source_name"])].append(unit)
    sources = sorted({source for _stratum, source in pools})
    chosen: list[dict[str, Any]] = []
    per_stratum: dict[str, Any] = {}
    for name in coverage.STRATA:
        supply = {source: len(pools.get((name, source), [])) for source in sources}
        allocation = coverage.allocate(spec.quota, supply)
        for source, count in allocation.items():
            chosen.extend(pools.get((name, source), [])[:count])
        realized = sum(allocation.values())
        per_stratum[name] = {
            "quota": spec.quota,
            "supply": sum(supply.values()),
            "realized": realized,
            "shortage": spec.quota - realized,
            "per_source": {s: {"supply": supply[s], "realized": allocation[s]} for s in sources},
        }
    population = sorted(chosen, key=lambda unit: (order_key(unit["pdetcov_id"], spec.seed), unit["pdetcov_id"]))
    for index, unit in enumerate(population):
        unit["dev_id"] = f"{spec.id_prefix}:" + unit["pdetcov_id"].split(":", 1)[1]
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


def build(root: Path = ROOT, spec: CheckSpec = V1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_record = coverage.check_input(root)
    roles = {source: {coverage.LAYER_A: False, coverage.LAYER_B: True} for source in coverage.sampling_roles(load_source_manifest(root))}
    candidates = coverage.collect_candidates(coverage._iter_input(root, roles), HeldoutIndex.load(root), roles)
    population, counts = draw(
        candidates.units,
        coverage.load_draw_exclusions(root),
        devset.load_coverage_exclusion(root),
        load_devset_exclusion(root, spec),
        spec,
    )
    payload = devset.population_bytes(population)
    excludes: dict[str, Any] = {
        "construction_exclusions": "as P-DET-COVERAGE-v1's draw (30 §9), including P-DET-v1",
        "pdet_coverage_v1_population_sha256": devset.COVERAGE_POPULATION_SHA256,
    }
    if spec is V1:
        excludes["devset_population_sha256"] = DEVSET_POPULATION_SHA256
    else:
        excludes["earlier_sets"] = {relative.as_posix(): sha for relative, sha in spec.excludes}
    manifest = {
        "artifact_kind": "PROSE_CLASSIFIER_DEVELOPMENT_CHECK_SET",
        "devset_id": spec.check_id,
        "plan": devset.PLAN.as_posix(),
        "status": "DEVELOPMENT_ONLY_NEVER_GOLD",
        "seed": spec.seed,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending; strata scarcest first for dedup",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "input": input_record,
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "excludes": excludes,
        "quotas": spec.quotas,
        "counts": counts,
        "population_file": spec.population_name,
        "population_sha256": _sha256(payload),
        "code_sha256_lf": {
            module: lf_sha256(root / module)
            for module in (
                "src/opengrad/verification/classifier_devcheck.py",
                "src/opengrad/verification/classifier_devset.py",
                "src/opengrad/verification/pdet_coverage.py",
            )
        },
        "statement": spec.statement,
    }
    return population, manifest


def write(root: Path, population: list[dict[str, Any]], manifest: dict[str, Any], spec: CheckSpec = V1) -> None:
    out = root / OUTPUT_DIR
    if (out / spec.population_name).exists() or (out / spec.manifest_name).exists():
        raise devset.DevsetError(f"{out / spec.population_name} already exists; it is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / spec.population_name).write_bytes(devset.population_bytes(population))
    (out / spec.manifest_name).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def verify(root: Path = ROOT, spec: CheckSpec = V1) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    manifest = json.loads((out / spec.manifest_name).read_text(encoding="utf-8"))
    written = (out / spec.population_name).read_bytes()
    population, rebuilt = build(root, spec)
    coverage_exclusion = devset.load_coverage_exclusion(root)
    devset_exclusion = load_devset_exclusion(root, spec)
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
    parser.add_argument("--check", choices=sorted(CHECKS), default="v1")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    spec = CHECKS[args.check]
    if args.build:
        population, manifest = build(args.root, spec)
        write(args.root, population, manifest, spec)
        print(json.dumps({"population_sha256": manifest["population_sha256"], "counts": manifest["counts"]}, indent=2))
        return 0
    result = verify(args.root, spec)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
