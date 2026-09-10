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


def _exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def load_heldout(root: Path) -> list[Record]:
    """Load the materialized, hash-verified held-out evaluation prompts."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    records: list[Record] = []
    for directory in HELDOUT_DIRS:
        base = root / "data/processed/normalization-v1" / directory
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("finalized") is not True:
            raise ValueError(f"held-out split is not finalized: {directory}")
        for shard in manifest["shards"]:
            for batch in pq.ParquetFile(base / shard).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    question = str(row.get("question", ""))
                    records.append(
                        Record(
                            record_id=f"{directory}:{row.get('example_id')}",
                            source=directory,
                            text=question,
                            shingles=frozenset(ngrams(question)),
                        )
                    )
    return records


def iter_training(root: Path, release_dir: Path | None = None) -> Iterator[Record]:
    """Stream canonical training records from the built release shards.

    ``opengrad_id`` and ``source_record_id`` are the literal string "unknown" for
    every LoopTool row, so they cannot identify a record. ``canonical_hash`` is the
    only column populated for all rows; it is made unique here because a handful of
    canonical duplicates legitimately share a hash, and a shared bucket would inflate
    overlap counts into impossible Jaccard values.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    base = (root / (release_dir or RELEASE_DIR)).resolve()
    seen: Counter[str] = Counter()
    for shard in sorted(base.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=2048, columns=["canonical_hash", "source_dataset", "messages"]
        ):
            for row in batch.to_pylist():
                label = SOURCE_LABEL_ALIASES.get(
                    str(row.get("source_dataset")), str(row.get("source_dataset"))
                )
                digest = str(row.get("canonical_hash") or "")
                if not digest:
                    raise ValueError(f"training row without canonical_hash in {shard.name}")
                seen[digest] += 1
                suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
                yield Record(
                    record_id=f"{label}:{digest}{suffix}",
                    source=label,
                    text=_user_prompt(row.get("messages")),
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
    canonical release are read from disk.
    """
    heldout = heldout_records if heldout_records is not None else load_heldout(root)
    if not heldout:
        raise ValueError("no materialized held-out records found")

    def train_source() -> Iterator[Record]:
        return iter(training_records) if training_records is not None else iter_training(root, release_dir)

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

    audit_queue = sorted(queue.values(), key=lambda entry: entry["heldout"])
    return {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "status": "REVIEW_REQUIRED_LEVEL_5_PENDING" if blocked else "LEVELS_1_4_MEASURED_LEVEL_5_PENDING",
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
        "interpretation": (
            "Levels 1-4 are machine-measured. Level 5 is a human audit of audit_queue and is "
            "deliberately NOT_RUN; no clean or generalization claim is made until it completes."
        ),
    }


def screen_and_write(root: Path, **kwargs: Any) -> tuple[dict[str, Any], Path]:
    report = screen(root, **kwargs)
    path = root / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, path
