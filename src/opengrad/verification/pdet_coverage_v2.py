"""Build and verify P-DET-COVERAGE-v2 (Study 002, protocol ``pdet-coverage-002-v2``).

The validation population of ``prose-decision-classifier-v2``, adopted as ``study_002_prereg_v6`` in
``docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md`` ("36" below). It is drawn like
P-DET-COVERAGE-v1 (30 §7–§9), reusing that module's stratum predicates, exclusions, skeleton and allocation, with
three differences (36 §3):

* **the unit** is the first assistant reply of a record under contract ``prose-decision-input-v2``, whatever
  follows it; layer B only;
* **its own seed and quotas**, and ids ``pdetcov2:`` + ``sha256(source_dataset : raw_record_hash)``;
* **it excludes every item already used**: P-DET-COVERAGE-v1 and the three classifier development and check sets,
  by identity, normalized prompt and normalized response, besides 30 §9's exclusions.

``--dry-run --output-dir DIR`` writes counts and hashes only; ``--build`` writes the population, never over an
existing one; ``--verify`` re-hashes and re-derives it.

It labels nothing, and it selects nothing by any classifier. ``prose-decision-classifier-v1`` exists and is
frozen; the quotas were sized from its aggregate predictions (36 §3, disclosed), but no prediction is computed
here. Whether a record continues after its first reply (``unit_kind``) is kept on the item for reporting and is
blinded from annotators (36 §3.6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import (
    HeldoutIndex,
    build_first_reply_input,
    first_reply_eligibility,
    normalize_prompt,
)
from opengrad.data.normalization_v3 import OUTPUT_DIR as NORMALIZATION_V3_DIR
from opengrad.data.normalization_v3 import iter_rows, lf_sha256, load_source_manifest
from opengrad.verification import pdet_coverage as v1
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

POPULATION_ID = "P-DET-COVERAGE-v2"
PROTOCOL_VERSION = "pdet-coverage-002-v2"
SEED = "opengrad-pdet-coverage-002-v2"
PREREGISTRATION = Path("docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md")
PREREGISTRATION_STATUS = "ADOPTED"
ADOPTION_AMENDMENT = "study_002_prereg_v6"
OUTPUT_DIR = Path("reports/pdet-coverage-v2")
POPULATION_NAME = "pdet-coverage-v2.population.jsonl"
MANIFEST_NAME = "pdet-coverage-v2.manifest.json"
DRY_RUN_NAME = "pdet-coverage-v2.dry-run.json"
ID_PREFIX = "pdetcov2:"
#: 36 §3, proposed quotas, adopted as written.
QUOTAS = {"X": 40, "M": 100, "R": 80, "Q": 100, "P1": 60, "P2": 40}
FIRST_REPLY_UNIT = "FIRST_REPLY_UNIT"
CLASSIFIER_STATUS_AT_SELECTION = "NOT_CONSULTED"
#: Populations whose items this one must not share (36 §3.3), pinned by the sha256 of their files.
USED_POPULATIONS: tuple[tuple[str, str], ...] = (
    ("reports/pdet-coverage/pdet-coverage-v1.population.jsonl", "755bc16e79ceb1cb9e461c0fe8b9125628c16f263ed24f9e008f55b613a7158c"),
    ("reports/prose-classifier/dev/prose-classifier-dev-v1.population.jsonl", "abbdfc497dca61821bcdf35624988e82aa12ba5b562564c9aa2fe36fb0c72456"),
    ("reports/prose-classifier/dev/prose-classifier-devcheck-v1.population.jsonl", "56564d85fa0db4abbc971b16bac5557edea8391620ccc3c3816a144cea651348"),
    ("reports/prose-classifier/dev/prose-classifier-devcheck-v2.population.jsonl", "2f4eb1354903661c7d998dc4920b33110a70281b2eb17380247feeab26708c1f"),
)


class CoverageV2Error(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def item_id(source_dataset: str, raw_record_hash: str) -> str:
    return ID_PREFIX + _sha256(f"{source_dataset}:{raw_record_hash}".encode())


def rank_key(stratum: str, identifier: str) -> str:
    return _sha256(f"{SEED}|{stratum}|{identifier}".encode())


def order_key(identifier: str) -> str:
    return _sha256(f"{SEED}:order:{identifier}".encode())


# ── what is already used ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UsedItems:
    """Every item of an earlier population or development set: identities, normalized prompts and responses."""

    identities: frozenset[str] = frozenset()
    prompts: frozenset[str] = frozenset()
    responses: frozenset[str] = frozenset()

    def hits(self, unit: Mapping[str, Any]) -> bool:
        identities = {unit["raw_record_hash"], unit["record_id"], unit["canonical_hash"]}
        return bool(
            identities & self.identities
            or normalize_prompt(unit["user_message"]) in self.prompts
            or normalize_prompt(unit["assistant_response"]) in self.responses
        )


def load_used_items(root: Path) -> UsedItems:
    identities: set[str] = set()
    prompts: set[str] = set()
    responses: set[str] = set()
    for relative, expected in USED_POPULATIONS:
        data = (root / relative).read_bytes() if (root / relative).is_file() else b""
        if _sha256(data) != expected:
            raise CoverageV2Error(f"{relative} is missing or not the pinned population")
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            identities.update(str(item[key]) for key in ("raw_record_hash", "record_id", "canonical_hash") if item.get(key))
            prompts.add(normalize_prompt(item["user_message"]))
            if isinstance(item.get("assistant_response"), str):
                responses.add(normalize_prompt(item["assistant_response"]))
    return UsedItems(frozenset(identities), frozenset(prompts), frozenset(responses))


# ── candidates ──────────────────────────────────────────────────────────────────────────────────────


def classify_record(record: Mapping[str, Any], heldout: HeldoutIndex) -> tuple[str, dict[str, Any] | None]:
    """The record's disposition under contract v2 and, when eligible, its first-reply unit."""
    result = first_reply_eligibility(record, heldout)
    if not result.eligible:
        return str(result.primary_reason), None
    built = build_first_reply_input(record, heldout)
    base = v1._base_fields(record)
    unit = {
        **base,
        "pdetcov_id": item_id(base["source_dataset"], base["raw_record_hash"]),
        "layer": v1.LAYER_B,
        "user_message": built.features.user_message,
        "assistant_response": built.features.assistant_response,
        "tools": [dict(tool) for tool in built.features.tools],
        "features_sha256": built.features_sha256(),
        "classifier_input_contract": built.contract_version,
        "unit_kind": built.provenance.unit_kind,  # type: ignore[attr-defined]
    }
    unit["stratum"] = v1.stratum(unit["assistant_response"], unit["tools"])
    return FIRST_REPLY_UNIT, unit


def layer_b_sources(root: Path) -> list[str]:
    return sorted(name for name, role in v1.sampling_roles(load_source_manifest(root)).items() if role[v1.LAYER_B])


def collect(root: Path, heldout: HeldoutIndex) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    units: list[dict[str, Any]] = []
    dispositions: dict[str, Counter[str]] = {}
    for source in layer_b_sources(root):
        counts = dispositions.setdefault(source, Counter())
        for record in iter_rows(root / NORMALIZATION_V3_DIR, source):
            disposition, unit = classify_record(record, heldout)
            counts[disposition] += 1
            if unit is not None:
                units.append(unit)
    return units, dispositions


# ── the draw (pure) ─────────────────────────────────────────────────────────────────────────────────


def _ranked(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(units, key=lambda unit: (rank_key(unit["stratum"], unit["pdetcov_id"]), unit["pdetcov_id"]), reverse=True)


def dedup_order(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strata scarcest first (ties in stratum order), each in rank order (30 §8.7)."""
    ranked = _ranked(units)
    supply = Counter(unit["stratum"] for unit in ranked)
    strata = sorted(supply, key=lambda name: (supply[name], v1.STRATA.index(name)))
    return [unit for name in strata for unit in ranked if unit["stratum"] == name]


def deduplicate(units: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """30 §9: raw hash, normalized response, one per user prompt, one per response skeleton per stratum."""
    stats = Counter({"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0})
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors: list[dict[str, Any]] = []
    for unit in dedup_order(units):
        keys = [
            ("raw_record_hash", unit["raw_record_hash"]),
            ("response_text", normalize_prompt(unit["assistant_response"])),
            ("user_prompt", normalize_prompt(unit["user_message"])),
            ("skeleton", (unit["stratum"], v1.skeleton(unit["assistant_response"], unit["tools"]))),
        ]
        duplicate = next((name for name, key in keys if key in seen[name]), None)
        if duplicate is not None:
            stats[duplicate] += 1
            continue
        for name, key in keys:
            seen[name].add(key)
        survivors.append(unit)
    return survivors, dict(stats)


def select(survivors: Iterable[dict[str, Any]], sources: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each stratum's quota split equally across sources with supply (30 §7.3), in rank order. No backfill."""
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in _ranked(survivors):
        pools[(unit["stratum"], unit["source_name"])].append(unit)
    names = sorted(sources)
    chosen: list[dict[str, Any]] = []
    strata: dict[str, Any] = {}
    for stratum in v1.STRATA:
        supply = {source: len(pools.get((stratum, source), [])) for source in names}
        allocation = v1.allocate(QUOTAS[stratum], supply)
        for source, count in allocation.items():
            chosen.extend(pools.get((stratum, source), [])[:count])
        realized = sum(allocation.values())
        strata[stratum] = {
            "quota": QUOTAS[stratum],
            "supply": sum(supply.values()),
            "realized": realized,
            "shortage": QUOTAS[stratum] - realized,
            "per_source": {source: {"supply": supply[source], "realized": allocation[source]} for source in names},
        }
    return chosen, strata


def finalize(chosen: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    population = sorted(chosen, key=lambda unit: (order_key(unit["pdetcov_id"]), unit["pdetcov_id"]))
    for index, unit in enumerate(population):
        unit["pdetcov_index"] = index
        unit["structured_calls"] = None
        unit["trajectory_gate"] = None
        for name in v1.ANNOTATION_FIELDS:
            unit[name] = None
        unit["classifier_version_at_selection"] = CLASSIFIER_STATUS_AT_SELECTION
    return population


def draw(
    units: list[dict[str, Any]],
    dispositions: Mapping[str, Counter[str]],
    exclusions: v1.DrawExclusions,
    used: UsedItems,
    sources: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The deterministic draw over collected units. Pure: no IO, no clock, no RNG."""
    kept, exclusion_stats = v1.apply_draw_exclusions(units, exclusions)
    unused = [unit for unit in kept if not used.hits(unit)]
    survivors, dedup_stats = deduplicate(unused)
    chosen, strata = select(survivors, sources)
    population = finalize(chosen)
    m_kinds = {
        unit["pdetcov_id"]: v1.m_match_kind(unit["assistant_response"], unit["tools"])
        for unit in population
        if unit["stratum"] == "M"
    }
    pool: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        pool[unit["source_name"]][unit["stratum"]] += 1
    counts = {
        "dispositions": {source: dict(sorted(counter.items())) for source, counter in sorted(dispositions.items())},
        "first_reply_units": len(units),
        "unit_kind_of_units": dict(sorted(Counter(unit["unit_kind"] for unit in units).items())),
        # The eligible first-reply pool per source and stratum, before exclusions and sampling: the weights of the
        # post-stratified DIRECT precision (pdet_coverage_metrics).
        "pool_strata": {source: dict(sorted(counter.items())) for source, counter in sorted(pool.items())},
        "draw_exclusions": exclusion_stats[v1.LAYER_B],
        "used_items_removed": len(kept) - len(unused),
        "dedup": dedup_stats,
        "after_dedup": len(survivors),
        "layer_b": strata,
        "m_match_kind": dict(sorted(Counter(m_kinds.values()).items())),
        "m_match_kind_by_id": dict(sorted(m_kinds.items())),
        "realized": len(population),
        "realized_unit_kind": dict(sorted(Counter(unit["unit_kind"] for unit in population).items())),
        "shortages": {name: item["shortage"] for name, item in strata.items() if item["shortage"]},
    }
    return population, counts


# ── building, writing, verifying ────────────────────────────────────────────────────────────────────


CODE_MODULES = (
    "src/opengrad/verification/pdet_coverage_v2.py",
    "src/opengrad/verification/pdet_coverage.py",
    "src/opengrad/data/classifier_input.py",
)


def build_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_record = v1.check_input(root)
    heldout = HeldoutIndex.load(root)
    exclusions = v1.load_draw_exclusions(root)
    used = load_used_items(root)
    sources = layer_b_sources(root)
    units, dispositions = collect(root, heldout)
    population, counts = draw(units, dispositions, exclusions, used, sources)
    manifest: dict[str, Any] = {
        "artifact_kind": "PDET_COVERAGE_POPULATION",
        "schema_version": 1,
        "population_id": POPULATION_ID,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": PREREGISTRATION_STATUS,
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "id_rule": "pdetcov2: + sha256(source_dataset : raw_record_hash)",
        "unit": "the first assistant reply of a record, under prose-decision-input-v2 (36 §2)",
        "input": {**input_record, "source_manifest_sha256_lf": v1.SOURCE_MANIFEST_SHA256},
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_V2_VERSION,
        "heldout_inputs": [list(item) for item in heldout.inputs],
        "draw_exclusion_inputs": [list(item) for item in exclusions.inputs],
        "draw_exclusion_order": list(v1.DRAW_EXCLUSION_ORDER),
        "used_populations_excluded": [list(item) for item in USED_POPULATIONS],
        "later_evaluation_sets": v1.LATER_EVALUATION_SETS,
        "sources": sources,
        "stratum_predicates": {"source": v1.STRATA_SOURCE.as_posix(), "priority": list(v1.STRATA)},
        "quotas": {"layer_b": QUOTAS},
        "backfill": "none: an undersupplied stratum takes all it has and reports its shortage",
        "min_usable_gold_per_mode_or_boundary": v1.MIN_USABLE_GOLD,
        "counts": counts,
        "dedup_order": "strata scarcest first (units after exclusions), each in rank order",
        "code_sha256_lf": {module: lf_sha256(root / module) for module in CODE_MODULES},
        "classifier_status_at_selection": CLASSIFIER_STATUS_AT_SELECTION,
        "selection_used_classifier": False,
        "quota_sizing_disclosure": (
            "The quotas (36 §3) were sized by the classifier's developer from aggregate predictions of the frozen "
            "prose-decision-classifier-v1 on the unused pool, per stratum and source. No prediction is computed "
            "here and none selects an item."
        ),
        "gold_labels_present": False,
        "annotation_fields": list(v1.ANNOTATION_FIELDS),
        "blinded_fields": ["source_name", "source_dataset", "stratum", "unit_kind", "layer", "m_match_kind"],
        "statement": (
            "A constructed boundary-coverage population of first replies. It estimates no natural prevalence and "
            "its class mix is not a property of any corpus. Items were selected from the preregistered stratum "
            "predicates and structural facts only. Strata are sampling strata, not labels."
        ),
    }
    return population, manifest


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    for forbidden in (v1.FORBIDDEN_OUTPUT_DIR, v1.OUTPUT_DIR):
        blocked = (root / forbidden).resolve()
        if resolved == blocked or blocked in resolved.parents:
            raise CoverageV2Error(f"P-DET-COVERAGE-v2 is never written under {forbidden.as_posix()}/")
    return resolved


def write_population(root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    if (directory / POPULATION_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise CoverageV2Error(f"a population already exists in {directory}; it is never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    payload = v1.population_bytes(population)
    stamped = {**manifest, "population_file": POPULATION_NAME, "population_records": len(population), "population_sha256": _sha256(payload)}
    data = v1.manifest_bytes(stamped)
    (directory / POPULATION_NAME).write_bytes(payload)
    (directory / MANIFEST_NAME).write_bytes(data)
    (directory / (MANIFEST_NAME + ".sha256")).write_bytes((_sha256(data) + "\n").encode())
    return stamped


def write_dry_run(root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Counts and hashes only: no item, no item id (the draw is reproducible, so a population is the blind sample)."""
    directory = resolve_output_dir(root, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    counts = {key: value for key, value in manifest["counts"].items() if key != "m_match_kind_by_id"}
    record = {
        **manifest,
        "artifact_kind": "PDET_COVERAGE_DRY_RUN",
        "counts": counts,
        "population_written": False,
        "population_records": len(population),
        "population_sha256": _sha256(v1.population_bytes(population)),
        "statement": "Counts-only dry run. No population file was written, and no item or item id is recorded here. "
        + manifest["statement"],
    }
    (directory / DRY_RUN_NAME).write_bytes(v1.manifest_bytes(record))
    return record


REPRODUCED_FIELDS = ("counts", "input", "heldout_inputs", "draw_exclusion_inputs", "used_populations_excluded", "quotas", "seed", "protocol_version")


def verify_population(root: Path, output_dir: Path) -> dict[str, Any]:
    """Re-hash the written population and manifest, re-derive the draw, and check no item is contaminated."""
    directory = output_dir if output_dir.is_absolute() else root / output_dir
    population_path, manifest_path = directory / POPULATION_NAME, directory / MANIFEST_NAME
    if not population_path.is_file() or not manifest_path.is_file():
        return {"status": FAIL, "errors": ["P-DET-COVERAGE-v2 artifacts are missing"]}
    populated, manifest_text = population_path.read_bytes(), manifest_path.read_bytes()
    manifest = json.loads(manifest_text)
    items = [json.loads(line) for line in populated.decode("utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if manifest.get("population_sha256") != _sha256(populated):
        errors.append("FAIL_HASH: population bytes do not match manifest population_sha256")
    sidecar = directory / (MANIFEST_NAME + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != _sha256(manifest_text):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")
    blocked: list[str] = []
    missing = v1.derivation_inputs_missing(root)
    if missing:
        blocked.append(f"{BLOCKED_INPUT_MISSING}: derivation inputs absent {missing}")
    else:
        rebuilt, rebuilt_manifest = build_population(root)
        if v1.population_bytes(rebuilt) != populated:
            errors.append("FAIL_REPRODUCIBILITY: a fresh draw did not reproduce the population bytes")
        errors.extend(
            f"FAIL_PROVENANCE: manifest field {key!r} differs from a fresh draw"
            for key in REPRODUCED_FIELDS
            if manifest.get(key) != rebuilt_manifest.get(key)
        )
        exclusions, used, heldout = v1.load_draw_exclusions(root), load_used_items(root), HeldoutIndex.load(root)
        contaminated = [
            item
            for item in items
            if exclusions.hits(item)
            or used.hits(item)
            or normalize_prompt(item["user_message"]) in heldout.prompts
            or {item["record_id"], item["upstream_id"]} & heldout.record_ids
        ]
        if contaminated:
            errors.append(f"FAIL_CONTAMINATION: {len(contaminated)} items match excluded or used material")
    labelled = sum(1 for item in items if item.get("gold_policy_label") not in (None, ""))
    if labelled:
        errors.append(f"FAIL_ANNOTATION: {labelled} items carry a gold label in the population file")
    if manifest.get("selection_used_classifier") or manifest.get("classifier_status_at_selection") != CLASSIFIER_STATUS_AT_SELECTION:
        errors.append("FAIL_METHOD: manifest does not record that no classifier selected items")
    status = FAIL if errors else BLOCKED_INPUT_MISSING if blocked else PASS
    return {
        "status": status,
        "population_id": manifest.get("population_id"),
        "population_sha256": manifest.get("population_sha256"),
        "items": len(items),
        "errors": errors,
        "blocked": blocked,
    }


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Counts only."""
    counts = record["counts"]
    return {
        "population_sha256": record.get("population_sha256"),
        "realized": counts["realized"],
        "realized_unit_kind": counts["realized_unit_kind"],
        "layer_b": {
            name: {"quota": item["quota"], "supply": item["supply"], "realized": item["realized"], "shortage": item["shortage"],
                   "per_source": {source: value["realized"] for source, value in item["per_source"].items()}}
            for name, item in counts["layer_b"].items()
        },
        "first_reply_units": counts["first_reply_units"],
        "draw_exclusions": counts["draw_exclusions"],
        "used_items_removed": counts["used_items_removed"],
        "dedup": counts["dedup"],
        "after_dedup": counts["after_dedup"],
        "m_match_kind": counts["m_match_kind"],
        "shortages": counts["shortages"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=OUTPUT_DIR.as_posix())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if args.verify:
        report = verify_population(root, output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == PASS else 1
    resolve_output_dir(root, output_dir)
    population, manifest = build_population(root)
    record = write_dry_run(root, output_dir, population, manifest) if args.dry_run else write_population(root, output_dir, population, manifest)
    print(json.dumps(summary(record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
