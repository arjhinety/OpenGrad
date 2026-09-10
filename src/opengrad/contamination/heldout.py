"""Held-out vs training contamination screening for the frozen behavioral manifest.

Levels 1-4 of the benchmark contamination policy (docs/evaluation/BENCHMARK_STRATEGY.md
section 4) are implemented here; level 5 is the human audit of the queue this module
produces and is deliberately left to a person.

Because held-out evaluation examples and training trajectories do not share a
conversation schema, levels 1-2 compare the user-visible prompt text rather than a
whole-conversation hash. This is recorded in the report's ``methodology`` field so the
narrower scope is never implied to be a whole-conversation guarantee.

Level 3 is an exact 5-gram Jaccard/containment screen using an inverted index built from
the held-out shingle universe. Shingles that appear in more than ``max_df`` training
records are non-discriminative boilerplate; they are pruned and the pruned count is
reported rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.contamination.audit import (
    AUDIT_PATH,
    QUARANTINE_PATH,
    benchmark_fingerprint,
    finding_fingerprint,
    load_quarantine,
    training_corpus_fingerprint,
)
from opengrad.contamination.scanner import edit_similarity, ngrams, normalize

MANIFEST_ID = "behavioral-heldout-v2"
LEVEL_1 = "1_exact_canonical_conversation_hash"
LEVEL_2 = "2_normalized_prompt_hash"
LEVEL_3 = "3_near_duplicate_ngram_minhash"
LEVEL_4 = "4_semantic_similarity"
LEVEL_5 = "5_manual_audit"

TRAINING_SOURCE_IDS = {
    "xlam-function-calling-60k",
    "toolace",
    "looptool-23k",
    "glaive-function-calling-v2",
    "button",
    "when2call-sft",
}

# The canonical release labels the when2call SFT split as plain "when2call".
SOURCE_LABEL_ALIASES = {"when2call": "when2call-sft"}

HELDOUT_DIRS = ("when2call-mcq", "when2call-llm-judge")
RELEASE_DIR = Path(".release/hf/toolpolicy-canonical-v1")
OUTPUT = Path("reports/data/behavioral-heldout-v2-contamination.json")


@dataclass(frozen=True)
class Record:
    record_id: str
    source: str
    text: str
    shingles: frozenset[str] = field(default_factory=frozenset)
    #: Held-out only: expected decision and candidate answers, for the reviewer's context.
    detail: dict[str, Any] = field(default_factory=dict)
    #: Training only: the matched prompt, assistant behaviour, and tool calls.
    evidence: dict[str, Any] = field(default_factory=dict)


def _exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _user_prompt(messages: Any) -> str:
    value = messages
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    if not isinstance(value, list):
        return ""
    return " ".join(
        str(item.get("content", ""))
        for item in value
        if isinstance(item, dict) and item.get("role") == "user"
    )


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_heldout(root: Path, exclude: set[str] | None = None) -> list[Record]:
    """Load the materialized, hash-verified held-out evaluation prompts.

    ``exclude`` holds record ids (``"<split>:<example_id>"``) that have been quarantined
    after a CONTAMINATED Level-5 verdict; they are removed from the benchmark.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    excluded = exclude or set()
    records: list[Record] = []
    for directory in HELDOUT_DIRS:
        base = root / "data/processed/normalization-v1" / directory
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("finalized") is not True:
            raise ValueError(f"held-out split is not finalized: {directory}")
        for shard in manifest["shards"]:
            for batch in pq.ParquetFile(base / shard).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    record_id = f"{directory}:{row.get('example_id')}"
                    if record_id in excluded:
                        continue
                    question = str(row.get("question", ""))
                    candidates = _maybe_json(row.get("candidates"))
                    tools = _maybe_json(row.get("tools"))
                    records.append(
                        Record(
                            record_id=record_id,
                            source=directory,
                            text=question,
                            shingles=frozenset(ngrams(question)),
                            detail={
                                "split": directory,
                                "expected_decision": row.get("expected_decision"),
                                "candidates": candidates if isinstance(candidates, dict) else {},
                                "tools": [
                                    tool.get("name")
                                    for tool in tools
                                    if isinstance(tool, dict) and tool.get("name")
                                ]
                                if isinstance(tools, list)
                                else [],
                            },
                        )
                    )
    return records


def _training_evidence(record_id: str, source: str, row: dict[str, Any]) -> dict[str, Any]:
    """Human-readable behaviour of one matched training record."""
    messages = _maybe_json(row.get("messages")) or []
    assistant_parts: list[str] = []
    tool_calls: list[Any] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                assistant_parts.append(content)
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                tool_calls.extend(calls)
    return {
        "record_id": record_id,
        "source": source,
        "behavior_decision": row.get("behavior_decision"),
        "prompt": _user_prompt(messages),
        "assistant": " ".join(assistant_parts)[:1500],
        "tool_calls": tool_calls[:5],
    }


def iter_training(
    root: Path,
    release_dir: Path | None = None,
    *,
    with_evidence: bool = False,
    needed: set[str] | None = None,
) -> Iterator[Record]:
    """Stream canonical training records from the built release shards.

    ``opengrad_id`` and ``source_record_id`` are the literal string "unknown" for
    every LoopTool row, so they cannot identify a record. ``canonical_hash`` is the
    only column populated for all rows; it is made unique here because a handful of
    canonical duplicates legitimately share a hash, and a shared bucket would inflate
    overlap counts into impossible Jaccard values.

    ``with_evidence`` attaches prompt/assistant/tool-call detail, and ``needed``
    restricts the stream to specific record ids (used for a targeted evidence pass so
    the full corpus is never held in memory).
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    base = (root / (release_dir or RELEASE_DIR)).resolve()
    columns = ["canonical_hash", "source_dataset", "messages"]
    if with_evidence:
        columns += ["behavior_decision"]
    seen: Counter[str] = Counter()
    for shard in sorted(base.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                label = SOURCE_LABEL_ALIASES.get(
                    str(row.get("source_dataset")), str(row.get("source_dataset"))
                )
                digest = str(row.get("canonical_hash") or "")
                if not digest:
                    raise ValueError(f"training row without canonical_hash in {shard.name}")
                seen[digest] += 1
                suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
                record_id = f"{label}:{digest}{suffix}"
                if needed is not None and record_id not in needed:
                    continue
                yield Record(
                    record_id=record_id,
                    source=label,
                    text=_user_prompt(row.get("messages")),
                    evidence=_training_evidence(record_id, label, row) if with_evidence else {},
                )


def _hash_index(records: Iterable[Record], normalised: bool) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = _stable_hash(record.text) if normalised else _exact_hash(record.text)
        index[digest].append(record.record_id)
    return index


def screen(
    root: Path,
    *,
    release_dir: Path | None = None,
    heldout_records: list[Record] | None = None,
    training_records: list[Record] | None = None,
    exclude_ids: set[str] | None = None,
    max_df: int = 1000,
    min_shingles: int = 25,
    jaccard_threshold: float = 0.5,
    containment_threshold: float = 0.6,
    level4_jaccard_floor: float = 0.2,
    edit_threshold: float = 0.8,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run levels 1-4 and return a machine-generated report.

    ``heldout_records``/``training_records`` may be supplied to score an explicit
    corpus (used by tests); otherwise the materialized held-out splits and the built
    canonical release are read from disk. ``exclude_ids`` are quarantined record ids.
    """
    excluded = exclude_ids or set()
    heldout = (
        heldout_records
        if heldout_records is not None
        else load_heldout(root, exclude=excluded)
    )
    if not heldout:
        raise ValueError("no materialized held-out records found")

    def train_source(*, with_evidence: bool = False, needed: set[str] | None = None) -> Iterator[Record]:
        if training_records is not None:
            return iter(
                record
                for record in training_records
                if needed is None or record.record_id in needed
            )
        return iter_training(root, release_dir, with_evidence=with_evidence, needed=needed)

    # Level 1/2 need full hash indexes; level 3 needs the held-out shingle universe.
    heldout_universe: dict[str, list[int]] = defaultdict(list)
    for position, record in enumerate(heldout):
        for shingle in record.shingles:
            heldout_universe[shingle].append(position)

    train_exact = _hash_index(train_source(), normalised=False)
    train_normalised = _hash_index(train_source(), normalised=True)

    level1 = [
        {"heldout": r.record_id, "training": train_exact[_exact_hash(r.text)]}
        for r in heldout
        if train_exact.get(_exact_hash(r.text))
    ]
    level2 = [
        {"heldout": r.record_id, "training": train_normalised[_stable_hash(r.text)]}
        for r in heldout
        if train_normalised.get(_stable_hash(r.text))
    ]

    # Pass 1: document frequency of held-out-universe shingles across training.
    df: Counter[str] = Counter()
    train_shingle_count: dict[str, int] = {}
    sources_seen: set[str] = set()
    for record in train_source():
        sources_seen.add(record.source)
        shingles = ngrams(record.text)
        train_shingle_count[record.record_id] = len(shingles)
        for shingle in shingles:
            if shingle in heldout_universe:
                df[shingle] += 1

    pruned = {shingle for shingle, count in df.items() if count > max_df}
    postings: dict[str, list[str]] = defaultdict(list)
    for record in train_source():
        for shingle in ngrams(record.text):
            if shingle in df and shingle not in pruned:
                postings[shingle].append(record.record_id)

    # Pass 2: accumulate overlaps and score.
    overlaps: dict[str, Counter[str]] = defaultdict(Counter)
    for position, record in enumerate(heldout):
        for shingle in record.shingles:
            for training_id in postings.get(shingle, ()):
                overlaps[str(position)][training_id] += 1

    findings = []
    level4_candidates: list[tuple[int, str, float]] = []
    for position, record in enumerate(heldout):
        size = len(record.shingles)
        if not size:
            continue
        # Containment is undefined-ish for very short prompts: a 3-shingle prompt
        # ("ok thanks") is trivially contained in almost anything. Require enough
        # signal before containment can flag a candidate.
        short = size < min_shingles
        best = []
        for training_id, shared in overlaps.get(str(position), {}).items():
            if shared > size:
                # A training record cannot share more shingles than the held-out
                # record contains. Guard instead of emitting an impossible Jaccard.
                raise AssertionError(
                    f"overlap {shared} exceeds held-out shingle count {size} for {training_id}"
                )
            union = size + train_shingle_count.get(training_id, 0) - shared
            jaccard = shared / union if union > 0 else 0.0
            containment = shared / size
            if jaccard >= jaccard_threshold or (not short and containment >= containment_threshold):
                best.append((shared, training_id, jaccard, containment))
        best.sort(reverse=True)
        for shared, training_id, jaccard, containment in best[:top_k]:
            findings.append(
                {
                    "heldout": record.record_id,
                    "training": training_id,
                    "shared_5grams": shared,
                    "jaccard": round(jaccard, 6),
                    "containment": round(containment, 6),
                }
            )
        for _shared, training_id, jaccard, _containment in best:
            if jaccard >= level4_jaccard_floor:
                level4_candidates.append((position, training_id, jaccard))

    # Level 4: order-sensitive semantic similarity over the level-3 candidates.
    train_text = {r.record_id: r.text for r in train_source()}
    level4: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for position, training_id, jaccard in level4_candidates:
        key = (heldout[position].record_id, training_id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        ratio = edit_similarity(heldout[position].text, train_text.get(training_id, ""))
        if ratio >= edit_threshold:
            level4.append(
                {
                    "heldout": key[0],
                    "training": training_id,
                    "sequence_matcher_ratio": round(ratio, 6),
                    "jaccard": round(jaccard, 6),
                }
            )

    blocked = bool(level1 or level2 or findings or level4)
    present = set(sources_seen) & TRAINING_SOURCE_IDS

    # One queue entry per distinct held-out record, so a human reviewer sees each
    # question once with all of its evidence instead of per-level duplicates.
    queue: dict[str, dict[str, Any]] = {}

    def enqueue(heldout_id: str, level: str, match: dict[str, Any]) -> None:
        entry = queue.setdefault(
            heldout_id, {"heldout": heldout_id, "levels": [], "matches": []}
        )
        if level not in entry["levels"]:
            entry["levels"].append(level)
        entry["matches"].append({"level": level, **match})

    for item in level1:
        enqueue(item["heldout"], LEVEL_1, {"training": item["training"]})
    for item in level2:
        enqueue(item["heldout"], LEVEL_2, {"training": item["training"]})
    for item in findings:
        enqueue(item["heldout"], LEVEL_3, item)
    for item in level4:
        enqueue(item["heldout"], LEVEL_4, item)

    # Attach reviewer context and matched-training behaviour. Only the referenced
    # records are read back, so the corpus is never held in memory in full.
    heldout_by_id = {record.record_id: record for record in heldout}
    needed: set[str] = set()
    for entry in queue.values():
        for match in entry["matches"]:
            training = match.get("training")
            if isinstance(training, list):
                needed.update(str(value) for value in training)
            elif training:
                needed.add(str(training))
    evidence_map: dict[str, dict[str, Any]] = {}
    for record in train_source(with_evidence=True, needed=needed):
        if record.evidence:
            evidence_map[record.record_id] = record.evidence
    train_text_fn = {r.record_id: r.text for r in train_source(needed=needed)}

    for entry in queue.values():
        source = heldout_by_id.get(entry["heldout"])
        detail = source.detail if source else {}
        entry["split"] = detail.get("split", "")
        entry["text"] = source.text if source else ""
        entry["expected_decision"] = detail.get("expected_decision")
        entry["candidates"] = detail.get("candidates", {})
        entry["tools"] = detail.get("tools", [])
        evidence: list[dict[str, Any]] = []
        seen_evidence: set[str] = set()
        for match in entry["matches"]:
            training = match.get("training")
            ids = training if isinstance(training, list) else ([training] if training else [])
            for training_id in ids:
                training_id = str(training_id)
                if training_id in seen_evidence:
                    continue
                seen_evidence.add(training_id)
                if training_id in evidence_map:
                    evidence.append(evidence_map[training_id])
                elif training_id in train_text_fn:
                    evidence.append({"record_id": training_id, "prompt": train_text_fn[training_id]})
        entry["training_evidence"] = evidence
        entry["finding_fingerprint"] = finding_fingerprint(entry)

    audit_queue = sorted(queue.values(), key=lambda entry: entry["heldout"])
    scan_fingerprint = _sha256_text(
        _canonical([entry["finding_fingerprint"] for entry in audit_queue])
    )
    return {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "status": "REVIEW_REQUIRED_LEVEL_5_PENDING" if blocked else "LEVELS_1_4_MEASURED_LEVEL_5_PENDING",
        "scan_fingerprint": scan_fingerprint,
        "benchmark_fingerprint": benchmark_fingerprint(root),
        "training_corpus_fingerprint": training_corpus_fingerprint(root, release_dir),
        "quarantined_record_ids": sorted(excluded),
        "training_sources_checked": sorted(present),
        "levels": {
            LEVEL_1: "MEASURED",
            LEVEL_2: "MEASURED",
            LEVEL_3: "MEASURED",
            LEVEL_4: "MEASURED",
            LEVEL_5: "NOT_RUN",
        },
        "methodology": {
            "levels_1_2_scope": (
                "user-visible prompt text; held-out evaluation examples and training "
                "trajectories do not share a conversation schema, so no whole-conversation "
                "hash equality is claimed"
            ),
            "level_3": (
                f"exact 5-gram Jaccard/containment over an inverted index of the held-out "
                f"shingle universe; shingles with training document frequency > {max_df} are "
                f"pruned as non-discriminative, and containment cannot flag a held-out prompt "
                f"shorter than {min_shingles} shingles"
            ),
            "level_4": (
                "SequenceMatcher (difflib) over level-3 candidates; candidate generation is "
                "prefiltered, so this is not an exhaustive semantic search and no embedding "
                "similarity was computed"
            ),
            "level_5": "human audit of the emitted queue; not performed by this tool",
            "thresholds": {
                "max_df": max_df,
                "min_shingles": min_shingles,
                "jaccard": jaccard_threshold,
                "containment": containment_threshold,
                "edit_similarity": edit_threshold,
            },
        },
        "counts": {
            "heldout_records": len(heldout),
            "training_records": len(train_shingle_count),
            "heldout_universe_shingles": len(heldout_universe),
            "pruned_shingles": len(pruned),
            "scored_pairs": sum(len(c) for c in overlaps.values()),
        },
        "training_source_labels_seen": sorted(sources_seen),
        "findings": {
            "level_1_exact_prompt_matches": level1,
            "level_2_normalized_prompt_matches": level2,
            "level_3_near_duplicates": findings,
            "level_4_semantic_matches": level4,
        },
        "audit_queue": audit_queue,
        "audit_queue_size": len(queue),
        "audit_artifact": str(AUDIT_PATH),
        "interpretation": (
            "Levels 1-4 are machine-measured. Level 5 is a human adjudication of audit_queue "
            "recorded in the separate audit artifact; this generated report is an input to "
            "that review and must never be hand-edited. No clean or generalization claim is "
            "made until Level 5 completes."
        ),
    }


def screen_and_write(root: Path, **kwargs: Any) -> tuple[dict[str, Any], Path]:
    """Run the scan and write the generated report. Never touches the human audit file."""
    root = root.resolve()
    if "exclude_ids" not in kwargs:
        kwargs["exclude_ids"] = load_quarantine(root / QUARANTINE_PATH).record_ids()
    report = screen(root, **kwargs)
    path = root / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, path
