"""Build and verify the frozen P-DET validation population (Study 002, protocol ``pdet-002-v1``).

P-DET is the independent validation protocol for the *future* deterministic decision classifier. This
module does two and only two things:

``--build``
    Draw the frozen sample deterministically from the candidate population and write
    ``reports/pdet/pdet-v1.population.jsonl`` plus ``reports/pdet/pdet-v1.manifest.json``.

``--verify``
    Re-derive the sample and re-hash the frozen artifacts, then report whether the population and manifest
    are unchanged. This exists so that "frozen" is checkable rather than asserted.

What this module deliberately does **not** do: it does not label anything, does not import or emulate any
classifier, and does not write gold labels. Gold labels are human annotations (`gold_policy_label` is left
null); the population is frozen so that annotation can proceed against a fixed target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from opengrad.verification.accounting import (
    FAIL,
    PASS,
    REQUIRED_NONEMPTY,
    ValidationResult,
)

PROTOCOL_VERSION = "pdet-002-v1"
SEED = "opengrad-pdet-002-v1"
PREVALENCE_SIZE = 400
CHALLENGE_SIZE = 200
MIN_PER_FAMILY = 15

POPULATION_DIR = Path("data/processed/normalization-v1/when2call-sft")
PARTITION_PATH = Path("reports/evaluation/behavioral-heldout-v2-partition.json")
MCQ_DIR = Path("data/processed/normalization-v1/when2call-mcq")
OUTPUT_DIR = Path("reports/pdet")
POPULATION_NAME = "pdet-v1.population.jsonl"
MANIFEST_NAME = "pdet-v1.manifest.json"

CLASSIFIER_STATUS_AT_FREEZE = "NOT_IMPLEMENTED"


def keyed_rank(pdet_id: str) -> str:
    """Deterministic ranking key: ``sha256(seed || id)``, mirroring the eval-partition idiom."""
    return hashlib.sha256(f"{SEED}|{pdet_id}".encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_load(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def load_population(root: Path) -> list[dict[str, Any]]:
    """Load the candidate population with its provenance intact.

    Candidate population: prose-based training-side records whose upstream format carries no authoritative
    decision label. This is the material the future classifier will actually be applied to.
    """
    import pyarrow.parquet as pq

    records: list[dict[str, Any]] = []
    for shard in sorted((root / POPULATION_DIR).glob("*.parquet")):
        table = pq.read_table(shard)
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        for index in range(table.num_rows):
            metadata = _json_load(columns["metadata"][index]) or {}
            messages = _json_load(columns["messages"][index]) or []
            if not messages:
                continue
            source = metadata.get("source") or {}
            raw_hash = str(metadata.get("raw_record_hash") or "")
            if not raw_hash:
                # A record without a stable upstream hash cannot be deduplicated or re-identified, so it
                # is not eligible for a frozen validation population.
                continue
            records.append(
                {
                    "pdet_id": f"when2call-sft:{raw_hash}",
                    "source_dataset": str(source.get("dataset_id") or "when2call"),
                    "source_split": str(source.get("original_split") or "train_sft"),
                    "source_revision": source.get("revision"),
                    "upstream_id": source.get("upstream_id"),
                    "raw_record_hash": raw_hash,
                    "canonical_hash": columns["canonical_hash"][index],
                    "prompt": str((messages[0] or {}).get("content") or ""),
                    "response": str((messages[-1] or {}).get("content") or ""),
                    "tools": _json_load(columns["tools"][index]) or [],
                    "shard": shard.name,
                }
            )
    return records


def load_exclusions(root: Path) -> dict[str, Any]:
    """Everything P-DET must not touch: model-evaluation ids, quarantined ids, held-out prompts."""
    import pyarrow.parquet as pq

    partition = json.loads((root / PARTITION_PATH).read_text(encoding="utf-8"))
    evaluation_ids: set[str] = set()
    for side in ("confirmatory", "dev"):
        evaluation_ids.update(str(item) for item in partition["example_ids"][side])
    heldout_prompts: set[str] = set()
    for shard in sorted((root / MCQ_DIR).glob("*.parquet")):
        heldout_prompts.update(
            normalize_text(question)
            for question in pq.read_table(shard).column("question").to_pylist()
        )
    quarantine_path = root / "reports/evaluation/behavioral-heldout-v2-quarantine.json"
    quarantined: set[str] = set()
    if quarantine_path.exists():
        payload = json.loads(quarantine_path.read_text(encoding="utf-8"))
        quarantined.update(str(item["example_id"]) for item in payload.get("excluded", []))
    return {
        "evaluation_ids": evaluation_ids,
        "heldout_prompts": heldout_prompts,
        "quarantined_ids": quarantined,
    }


def _length_bucket(text: str) -> str:
    size = len(text)
    if size < 160:
        return "short"
    return "medium" if size <= 400 else "long"


def _has_call_payload(response: str) -> bool:
    return '{"name"' in response or "<TOOLCALL>" in response


def strata(record: dict[str, Any], duplicate_responses: Counter) -> dict[str, str]:
    """Observable sampling strata. None of these is a label and none is used as one."""
    from opengrad.evaluation.capability import detect_refusal

    response = record["response"]
    return {
        "has_tools": "tools" if record["tools"] else "no_tools",
        "refusal_signal": "refusal_shaped"
        if detect_refusal(response).is_refusal
        else "not_refusal_shaped",
        "length": _length_bucket(response),
        "response_repeated": "repeated"
        if duplicate_responses[normalize_text(response)] > 1
        else "unique",
    }


#: Challenge families: predicates over *observable* characteristics of the frozen record only. The frozen
#: specification and the dataset decide membership; no classifier exists to consult, and consulting one
#: would be a protocol violation.
QUESTION_CUES = (
    "could you",
    "can you please",
    "please provide",
    "please specify",
    "please tell me",
    "which ",
    "what is the",
    "do you have",
    "would you like",
)
HEDGE_CUES = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i don't know",
    "i'm not able",
    "i do not have",
)
EXTERNAL_SERVICE_CUES = (
    "you may want to",
    "check a reliable",
    "i recommend using",
    "i suggest using",
    "you can use",
    "please check",
)
COURTESY_CUES = (
    "does that help",
    "would you like me to",
    "let me know if",
    "feel free to ask",
    "is there anything else",
)


def challenge_families(record: dict[str, Any]) -> list[str]:
    from opengrad.evaluation.capability import detect_refusal

    response = record["response"]
    lowered = response.casefold()
    refusal = detect_refusal(response).is_refusal
    families: list[str] = []
    if response.rstrip().endswith("?"):
        families.append("question_mark")
    if any(cue in lowered for cue in QUESTION_CUES) and not response.rstrip().endswith("?"):
        families.append("question_without_mark")
    if refusal and "?" in response:
        families.append("refusal_with_question")
    if refusal and "?" not in response:
        families.append("refusal_plain")
    offered = {
        str(tool.get("name")) for tool in record["tools"] if isinstance(tool, dict) and tool.get("name")
    }
    if offered and any(name in response for name in offered) and not _has_call_payload(response):
        families.append("tool_mentioned_no_payload")
    if any(cue in lowered for cue in HEDGE_CUES) and " but " in lowered:
        families.append("caveat_then_content")
    if len(response) > 400 and "?" in response:
        families.append("multi_clause_question")
    if not record["tools"]:
        families.append("no_tools_offered")
    if len(response) < 120 and "?" not in response and not refusal:
        families.append("short_plain")
    if _has_call_payload(response):
        families.append("serialized_call_shape")
    if any(cue in lowered for cue in EXTERNAL_SERVICE_CUES):
        families.append("advice_external_service")
    if any(cue in lowered for cue in COURTESY_CUES):
        families.append("polite_followup")
    return families


def family_quotas(families: Iterable[str], total: int) -> dict[str, int]:
    """Even split with the remainder to families in fixed order; shortfalls are never backfilled."""
    ordered = sorted(set(families))
    if not ordered:
        return {}
    base, remainder = divmod(total, len(ordered))
    return {family: base + (1 if index < remainder else 0) for index, family in enumerate(ordered)}


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Dedup by ``raw_record_hash`` then normalized response text, keeping the top-ranked survivor."""
    ranked = sorted(records, key=lambda item: keyed_rank(item["pdet_id"]), reverse=True)
    seen_hashes: set[str] = set()
    seen_responses: dict[str, str] = {}
    survivors: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = defaultdict(list)
    stats = {"dropped_duplicate_raw_record_hash": 0, "dropped_duplicate_response_text": 0}
    for record in ranked:
        if record["raw_record_hash"] in seen_hashes:
            stats["dropped_duplicate_raw_record_hash"] += 1
            continue
        seen_hashes.add(record["raw_record_hash"])
        key = normalize_text(record["response"])
        if key in seen_responses:
            groups[seen_responses[key]].append(record["pdet_id"])
            stats["dropped_duplicate_response_text"] += 1
            continue
        seen_responses[key] = record["pdet_id"]
        survivors.append(record)
    for record in survivors:
        group = groups.get(record["pdet_id"])
        record["response_duplicate_group"] = record["pdet_id"] if group else None
        record["response_duplicate_group_size"] = (len(group) + 1) if group else 1
    stats["response_duplicate_groups"] = len(groups)
    return survivors, stats


def apply_exclusions(
    records: list[dict[str, Any]], exclusions: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop evaluation ids, quarantined ids, and any record whose prompt is a held-out question."""
    heldout = exclusions["heldout_prompts"]
    blocked_ids = exclusions["evaluation_ids"] | exclusions["quarantined_ids"]
    kept: list[dict[str, Any]] = []
    stats = {"dropped_by_evaluation_id": 0, "dropped_by_prompt_matching_heldout": 0}
    for record in records:
        if record["upstream_id"] in blocked_ids:
            stats["dropped_by_evaluation_id"] += 1
            continue
        if normalize_text(record["prompt"]) in heldout:
            stats["dropped_by_prompt_matching_heldout"] += 1
            continue
        kept.append(record)
    return kept, stats


def _allocate(stratum_counts: Counter, total: int) -> dict[Any, int]:
    """Proportional allocation with a deterministic remainder rule (largest count, then key order)."""
    if not stratum_counts:
        return {}
    population = sum(stratum_counts.values())
    quotas: dict[Any, int] = {}
    order: list[tuple[int, str, Any]] = []
    for key, count in stratum_counts.items():
        quota = int(total * count / population)
        quotas[key] = quota
        order.append((count, str(key), key))
    for _count, _key, key in sorted(order, reverse=True)[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _select_prevalence(
    ranked: list[dict[str, Any]], response_counts: Counter
) -> tuple[list[dict[str, Any]], dict[Any, int], dict[Any, int]]:
    stratum_counts: Counter = Counter(
        tuple(sorted(strata(record, response_counts).items())) for record in ranked
    )
    quotas = _allocate(stratum_counts, PREVALENCE_SIZE)
    taken: Counter = Counter()
    selected: list[dict[str, Any]] = []
    for record in ranked:
        key = tuple(sorted(strata(record, response_counts).items()))
        if taken[key] < quotas.get(key, 0):
            taken[key] += 1
            record["pdet_component"] = "prevalence"
            selected.append(record)
    return selected, quotas, taken


def _select_challenge(
    ranked: list[dict[str, Any]], already: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, int]]]:
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ranked:
        if record["pdet_id"] in already:
            continue
        for family in challenge_families(record):
            members[family].append(record)
    quotas = family_quotas(members, CHALLENGE_SIZE)
    selected: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    counts: dict[str, int] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    for family in sorted(members):
        quota = quotas[family]
        chosen = [r for r in members[family] if r["pdet_id"] not in chosen_ids][:quota]
        counts[family] = len(chosen)
        if len(chosen) < quota:
            shortfalls[family] = {
                "quota": quota,
                "available": len(members[family]),
                "shortfall": quota - len(chosen),
            }
        for record in chosen:
            chosen_ids.add(record["pdet_id"])
            record["pdet_component"] = "challenge"
            record["challenge_families"] = challenge_families(record)
            selected.append(record)
    return selected, counts, shortfalls


def build_sample(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically draw the frozen P-DET sample. Pure function of repository state + constants."""
    population = load_population(root)
    exclusions = load_exclusions(root)
    eligible, exclusion_stats = apply_exclusions(population, exclusions)
    survivors, dedup_stats = deduplicate(eligible)
    response_counts = Counter(normalize_text(record["response"]) for record in survivors)
    ranked = sorted(survivors, key=lambda item: keyed_rank(item["pdet_id"]), reverse=True)

    prevalence, strata_quotas, strata_taken = _select_prevalence(ranked, response_counts)
    challenge, family_counts, family_shortfalls = _select_challenge(
        ranked, {record["pdet_id"] for record in prevalence}
    )

    sample = sorted(prevalence + challenge, key=lambda item: item["pdet_id"])
    for index, record in enumerate(sample):
        record["pdet_index"] = index
        record["gold_policy_label"] = None
        record["ambiguity_status"] = None
        record["annotator_id"] = None
        record["annotator_rationale"] = None
        record["boundary_rule_cited"] = None
        record["annotation_version"] = None
        record["classifier_version_at_selection"] = CLASSIFIER_STATUS_AT_FREEZE

    shortfall_total = sum(item["shortfall"] for item in family_shortfalls.values())
    manifest: dict[str, Any] = {
        "artifact_kind": "PDET_FROZEN_POPULATION",
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_document": "docs/research/study-002/22-PDET-PROTOCOL.md",
        "seed": SEED,
        "ranking": "sha256(seed || pdet_id) descending",
        "candidate_population": {
            "path": str(POPULATION_DIR),
            "records": len(population),
            "eligible_after_exclusions": len(eligible),
            "after_dedup": len(survivors),
        },
        "sizes": {
            "prevalence_target": PREVALENCE_SIZE,
            "challenge_target": CHALLENGE_SIZE,
            "total_target": PREVALENCE_SIZE + CHALLENGE_SIZE,
            "prevalence_realized": len(prevalence),
            "challenge_realized": len(challenge),
            "total_realized": len(sample),
            "challenge_shortfall_total": shortfall_total,
        },
        "exclusions": exclusion_stats,
        "dedup": dedup_stats,
        "strata_quotas": {str(key): count for key, count in sorted(strata_quotas.items(), key=str)},
        "strata_realized": {str(key): count for key, count in sorted(strata_taken.items(), key=str)},
        "challenge_family_counts": family_counts,
        "challenge_family_shortfalls": family_shortfalls,
        "challenge_assignment_rule": (
            "An item may satisfy several boundary families. Quotas are applied in alphabetical family "
            "order and an item is counted once, in the first family that still had quota. Family counts "
            "are therefore assignment counts, not the total number of items that exhibit the boundary."
        ),
        "coverage_requirement_per_mode": 50,
        "excluded_populations": [
            "Study 002 confirmatory partition (1,277 example ids)",
            "Study 002 DEV partition (2,373 example ids)",
            "quarantined held-out example ids",
            "any candidate record whose prompt matches a held-out question",
        ],
        "classifier_status_at_selection": CLASSIFIER_STATUS_AT_FREEZE,
        "selection_used_classifier": False,
        "statement": (
            "Examples were selected from the frozen annotation specification and observable dataset "
            "characteristics only. No decision classifier had been implemented, and none was consulted, "
            "when these examples were chosen."
        ),
        "gold_labels_present": False,
        "blocked_on": "human annotation under docs/research/study-002/22-PDET-PROTOCOL.md",
        "annotation_fields": [
            "gold_policy_label",
            "ambiguity_status",
            "annotator_id",
            "annotator_rationale",
            "boundary_rule_cited",
            "annotation_version",
        ],
    }
    return sample, manifest


def population_bytes(sample: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in sample
    )


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_frozen(root: Path, sample: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Write the frozen artifacts, stamping hashes computed over the exact bytes written."""
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    payload = population_bytes(sample)
    manifest = dict(manifest)
    manifest["population_sha256"] = _sha256_bytes(payload)
    manifest["population_records"] = len(sample)
    manifest["population_file"] = POPULATION_NAME
    stamped = manifest_bytes(manifest)
    # The manifest hash covers the manifest text, so it is recorded beside it rather than inside it.
    (output / POPULATION_NAME).write_bytes(payload)
    (output / MANIFEST_NAME).write_bytes(stamped)
    (output / (MANIFEST_NAME + ".sha256")).write_text(
        _sha256_bytes(stamped) + "\n", encoding="utf-8"
    )
    return manifest


def verify_frozen(root: Path) -> tuple[ValidationResult, dict[str, Any]]:
    """Prove the frozen population and manifest have not changed, and that they are reproducible.

    Checks, all of them blocking:

    1. both artifacts exist and are non-empty;
    2. the population file matches ``population_sha256`` in the manifest;
    3. the manifest text matches the recorded ``.sha256`` sidecar;
    4. re-deriving the sample from the repository reproduces the frozen bytes exactly;
    5. contamination: no frozen item's prompt is a held-out question, and no frozen item carries an
       evaluation id;
    6. no frozen item carries a gold label or an annotation (annotation has not happened yet);
    7. the manifest records that no classifier was used to select the examples.
    """
    output = root / OUTPUT_DIR
    population_path = output / POPULATION_NAME
    manifest_path = output / MANIFEST_NAME
    digest_path = output / (MANIFEST_NAME + ".sha256")
    errors: list[str] = []
    detail: dict[str, int] = {}

    if not population_path.exists() or not manifest_path.exists():
        return (
            ValidationResult(
                name="pdet freeze",
                policy=REQUIRED_NONEMPTY,
                discovered=0,
                checked=0,
                failed=1,
                errors=["FAIL_NONVACUOUS: frozen P-DET artifacts are missing"],
            ),
            {"status": "MISSING"},
        )

    populated = population_path.read_bytes()
    manifest_text = manifest_path.read_bytes()
    manifest = json.loads(manifest_text.decode("utf-8"))
    frozen = [json.loads(line) for line in populated.decode("utf-8").splitlines() if line.strip()]

    if manifest.get("population_sha256") != _sha256_bytes(populated):
        errors.append("FAIL_HASH: population bytes do not match manifest population_sha256")
    if digest_path.exists() and digest_path.read_text(encoding="utf-8").strip() != _sha256_bytes(
        manifest_text
    ):
        errors.append("FAIL_HASH: manifest bytes do not match the recorded .sha256 sidecar")

    rebuilt, rebuilt_manifest = build_sample(root)
    if population_bytes(rebuilt) != populated:
        errors.append("FAIL_REPRODUCIBILITY: re-deriving the sample did not reproduce the frozen bytes")
    for key in ("sizes", "exclusions", "dedup", "challenge_family_counts"):
        if manifest.get(key) != rebuilt_manifest.get(key):
            errors.append(f"FAIL_PROVENANCE: manifest field {key!r} differs from a fresh derivation")

    exclusions = load_exclusions(root)
    heldout = exclusions["heldout_prompts"]
    blocked = exclusions["evaluation_ids"] | exclusions["quarantined_ids"]
    contaminating = [
        record["pdet_id"]
        for record in frozen
        if normalize_text(record.get("prompt", "")) in heldout or record.get("upstream_id") in blocked
    ]
    if contaminating:
        errors.append(f"FAIL_CONTAMINATION: {len(contaminating)} frozen items overlap held-out material")
    detail["frozen_items"] = len(frozen)

    labelled = [r["pdet_id"] for r in frozen if r.get("gold_policy_label") not in (None, "")]
    detail["labelled_items"] = len(labelled)
    claims_labels = bool(manifest.get("gold_labels_present"))
    if claims_labels and len(labelled) != len(frozen):
        errors.append(
            f"FAIL_ANNOTATION: manifest claims gold labels but only {len(labelled)} of "
            f"{len(frozen)} items carry one"
        )
    if labelled and not claims_labels:
        errors.append(
            f"FAIL_ANNOTATION: {len(labelled)} items carry a gold label while the manifest records "
            "gold_labels_present = false"
        )

    if manifest.get("selection_used_classifier") or (
        manifest.get("classifier_status_at_selection") != CLASSIFIER_STATUS_AT_FREEZE
    ):
        errors.append("FAIL_METHOD: manifest does not record that no classifier was used to select")

    components = Counter(record.get("pdet_component") for record in frozen)
    detail["prevalence"] = components.get("prevalence", 0)
    detail["challenge"] = components.get("challenge", 0)
    if set(components) - {"prevalence", "challenge"}:
        errors.append(f"FAIL_COMPONENT: unexpected component values {sorted(set(components))}")

    result = ValidationResult(
        name="pdet freeze",
        policy=REQUIRED_NONEMPTY,
        discovered=len(frozen),
        checked=len(frozen),
        passed=len(frozen) if not errors else 0,
        failed=0 if not errors else len(frozen),
        errors=errors,
        detail=detail,
    )
    status = PASS if not errors and not result.accounting_errors() else FAIL
    summary = {
        "status": status,
        "protocol_version": manifest.get("protocol_version"),
        "population_sha256": manifest.get("population_sha256"),
        "frozen_items": len(frozen),
        "prevalence": detail["prevalence"],
        "challenge": detail["challenge"],
        "labelled_items": detail["labelled_items"],
        "gold_labels_present": manifest.get("gold_labels_present"),
        "errors": errors + result.accounting_errors(),
    }
    return result, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--build", action="store_true", help="draw and write the frozen P-DET sample")
    parser.add_argument("--verify", action="store_true", help="verify the frozen artifacts are unchanged")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.build:
        sample, manifest = build_sample(root)
        written = write_frozen(root, sample, manifest)
        print(json.dumps(written, indent=2, sort_keys=True))
        return 0
    if args.verify:
        result, summary = verify_frozen(root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == PASS else 1
    parser.error("choose --build or --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())