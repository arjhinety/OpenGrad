"""Development and check sets of first replies for ``prose-decision-classifier-v2`` (37 §3-§4).

Drawn from the unit of contract ``prose-decision-input-v2`` (36 §2) over the layer B sources, like the v1
development sets (:mod:`opengrad.verification.classifier_devcheck`), with the exclusions plan 37 fixes:

* 30 §9's construction exclusions and held-out material (through the contract and
  :func:`opengrad.verification.pdet_coverage.load_draw_exclusions`);
* every item of **P-DET-COVERAGE-v2** and of everything it excludes: P-DET-COVERAGE-v1 and the three v1 development
  and check sets (:func:`opengrad.verification.pdet_coverage_v2.load_used_items`);
* every item of each earlier v2 development or check set listed in its spec, pinned by file hash;

each by identity, normalized prompt and normalized response. Then 30 §9's dedup in rank order, a quota per stratum
split equally across sources, no backfill. Labels are Claude model judgments, used only for development, never
gold (37 §3). Nothing here prints item text.

    python -m opengrad.verification.classifier_devset_v2 --set dev --build
    python -m opengrad.verification.classifier_devset_v2 --set dev --verify
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
from opengrad.data.normalization_v3 import lf_sha256
from opengrad.verification import pdet_coverage as coverage
from opengrad.verification import pdet_coverage_v2 as coverage_v2

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path("docs/research/study-002/37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md")
OUTPUT_DIR = Path("reports/prose-classifier/dev-v2")
LABEL_FIELDS = ("gold_policy_label", "ambiguity_status", "annotator_id", "annotator_rationale", "boundary_rule_cited", "annotation_version")
COVERAGE_V2_POPULATION = (
    "reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl",
    "8fa4868c64845a93b0627cd4463887b03181e6479f2e9f48d75e8c80f626f013",
)


class DevsetV2Error(ValueError):
    pass


@dataclass(frozen=True)
class SetSpec:
    set_id: str
    seed: str
    quota: int
    id_prefix: str
    #: Earlier v2 development or check sets this one must not overlap: (population path, pinned sha256).
    excludes: tuple[tuple[str, str], ...]
    statement: str

    @property
    def population_name(self) -> str:
        return f"{self.set_id}.population.jsonl"

    @property
    def manifest_name(self) -> str:
        return f"{self.set_id}.manifest.json"


SETS = {
    "dev": SetSpec(
        set_id="prose-classifier-dev-v2",
        seed="opengrad-prose-classifier-dev-v2",
        quota=50,
        id_prefix="dev-v2",
        excludes=(),
        statement=(
            "The development set of first replies for prose-decision-classifier-v2 (37 §3). Its labels are model "
            "judgments used only to develop the rules, never gold and never evidence of accuracy. It shares no item, "
            "prompt or response with P-DET-COVERAGE-v2, P-DET-COVERAGE-v1 or the v1 development and check sets."
        ),
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank_key(seed: str, stratum: str, identifier: str) -> str:
    return _sha256(f"{seed}|{stratum}|{identifier}".encode())


def order_key(seed: str, identifier: str) -> str:
    return _sha256(f"{seed}:order:{identifier}".encode())


def _items_of(root: Path, relative: str, expected: str) -> list[dict[str, Any]]:
    path = root / relative
    data = path.read_bytes() if path.is_file() else b""
    if _sha256(data) != expected:
        raise DevsetV2Error(f"{relative} is missing or not the pinned population")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def load_exclusion(root: Path, spec: SetSpec) -> coverage_v2.UsedItems:
    """P-DET-COVERAGE-v2, everything it excludes, and the spec's earlier v2 sets, as one exclusion."""
    base = coverage_v2.load_used_items(root)
    identities, prompts, responses = set(base.identities), set(base.prompts), set(base.responses)
    for relative, expected in (COVERAGE_V2_POPULATION, *spec.excludes):
        for item in _items_of(root, relative, expected):
            identities.update(str(item[key]) for key in ("raw_record_hash", "record_id", "canonical_hash") if item.get(key))
            prompts.add(normalize_prompt(item["user_message"]))
            responses.add(normalize_prompt(item["assistant_response"]))
    return coverage_v2.UsedItems(frozenset(identities), frozenset(prompts), frozenset(responses))


def _ranked(units: Iterable[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(units, key=lambda unit: (rank_key(seed, unit["stratum"], unit["pdetcov_id"]), unit["pdetcov_id"]), reverse=True)


def draw(
    units: list[dict[str, Any]],
    exclusions: coverage.DrawExclusions,
    used: coverage_v2.UsedItems,
    sources: Iterable[str],
    spec: SetSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure: the set from first-reply units."""
    kept, exclusion_stats = coverage.apply_draw_exclusions(units, exclusions)
    unused = [unit for unit in kept if not used.hits(unit)]
    stats = Counter({"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0})
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors: list[dict[str, Any]] = []
    for unit in _ranked(unused, spec.seed):
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
    names = sorted(sources)
    chosen: list[dict[str, Any]] = []
    per_stratum: dict[str, Any] = {}
    for stratum in coverage.STRATA:
        supply = {source: len(pools.get((stratum, source), [])) for source in names}
        allocation = coverage.allocate(spec.quota, supply)
        for source, count in allocation.items():
            chosen.extend(pools.get((stratum, source), [])[:count])
        realized = sum(allocation.values())
        per_stratum[stratum] = {
            "quota": spec.quota,
            "supply": sum(supply.values()),
            "realized": realized,
            "shortage": spec.quota - realized,
            "per_source": {source: {"supply": supply[source], "realized": allocation[source]} for source in names},
        }
    population = sorted(chosen, key=lambda unit: (order_key(spec.seed, unit["pdetcov_id"]), unit["pdetcov_id"]))
    for index, unit in enumerate(population):
        unit["dev_id"] = f"{spec.id_prefix}:" + unit["pdetcov_id"].split(":", 1)[1]
        unit["dev_index"] = index
        for name in LABEL_FIELDS:
            unit[name] = None
    counts = {
        "first_reply_units": len(units),
        "construction_exclusions": exclusion_stats[coverage.LAYER_B],
        "used_items_removed": len(kept) - len(unused),
        "dedup": dict(stats),
        "after_dedup": len(survivors),
        "strata": per_stratum,
        "realized": len(population),
        "realized_unit_kind": dict(sorted(Counter(unit["unit_kind"] for unit in population).items())),
    }
    return population, counts


def population_bytes(population: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for unit in population)


def build(root: Path, spec: SetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_record = coverage.check_input(root)
    heldout = HeldoutIndex.load(root)
    units, _dispositions = coverage_v2.collect(root, heldout)
    population, counts = draw(units, coverage.load_draw_exclusions(root), load_exclusion(root, spec), coverage_v2.layer_b_sources(root), spec)
    manifest = {
        "artifact_kind": "PROSE_CLASSIFIER_V2_DEVELOPMENT_SET",
        "set_id": spec.set_id,
        "plan": PLAN.as_posix(),
        "status": "DEVELOPMENT_ONLY_NEVER_GOLD",
        "seed": spec.seed,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "input": input_record,
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_V2_VERSION,
        "excludes": {
            "construction_exclusions": "30 §9, including P-DET-v1, and held-out material",
            "used_populations": [list(item) for item in (*coverage_v2.USED_POPULATIONS, COVERAGE_V2_POPULATION, *spec.excludes)],
        },
        "quota_per_stratum": spec.quota,
        "counts": counts,
        "population_file": spec.population_name,
        "population_sha256": _sha256(population_bytes(population)),
        "code_sha256_lf": {
            module: lf_sha256(root / module)
            for module in (
                "src/opengrad/verification/classifier_devset_v2.py",
                "src/opengrad/verification/pdet_coverage_v2.py",
                "src/opengrad/verification/pdet_coverage.py",
                "src/opengrad/data/classifier_input.py",
            )
        },
        "statement": spec.statement,
    }
    return population, manifest


def write(root: Path, population: list[dict[str, Any]], manifest: dict[str, Any], spec: SetSpec) -> None:
    out = root / OUTPUT_DIR
    if (out / spec.population_name).exists() or (out / spec.manifest_name).exists():
        raise DevsetV2Error(f"{out / spec.population_name} already exists; it is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / spec.population_name).write_bytes(population_bytes(population))
    (out / spec.manifest_name).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def verify(root: Path, spec: SetSpec) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    manifest = json.loads((out / spec.manifest_name).read_text(encoding="utf-8"))
    written = (out / spec.population_name).read_bytes()
    population, rebuilt = build(root, spec)
    used = load_exclusion(root, spec)
    checks: dict[str, Any] = {
        "file_hash_matches_manifest": _sha256(written) == manifest["population_sha256"],
        "rederived_bytes_equal": population_bytes(population) == written,
        "counts_equal": rebuilt["counts"] == manifest["counts"],
        "used_item_overlap": sum(1 for unit in population if used.hits(unit)),
    }
    checks["status"] = (
        "PASS"
        if all(checks[key] is True for key in ("file_hash_matches_manifest", "rederived_bytes_equal", "counts_equal"))
        and checks["used_item_overlap"] == 0
        else "FAIL"
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    spec = SETS[args.set]
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
