"""Contamination levels 1-4: the one screening engine every OpenGrad contamination scan uses.

The policy (`docs/evaluation/BENCHMARK_STRATEGY.md` section 4) defines five levels. This module
measures levels 1-4 and builds the level-5 queue; level 5 itself is a human adjudication
(`docs/evaluation/CONTAMINATION_ADJUDICATION.md`) that no code here completes.

* **Level 1** -- the prompt text is byte-identical (a sha256 of the raw text).
* **Level 2** -- the prompt text is identical after whitespace collapsing and casefolding.
* **Level 3** -- an exact 5-gram Jaccard/containment screen over an inverted index of the screened
  records' shingles. Shingles in more than ``max_df`` training records are pruned as boilerplate.
* **Level 4** -- ``difflib.SequenceMatcher`` over the level-3 candidates.

Two layers call it: the behavioural held-out screen (`opengrad.contamination.heldout`) and the
benchmark scan (`opengrad.benchmarks.contamination.scanner`). Until 2026-09-24 the benchmark scan
had its own implementation, which normalised text before its "exact" level 1 and marked level 5
complete by itself (`reports/ERRATA.md` §25); one engine means one meaning per level.

The code was moved here verbatim from ``heldout.screen``, whose report on a fixed fixture is pinned
byte-for-byte by `tests/contamination/test_levels.py`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from opengrad.contamination.scanner import edit_similarity, ngrams, normalize
from opengrad.hashing import sha256_text

LEVEL_1 = "1_exact_canonical_conversation_hash"
LEVEL_2 = "2_normalized_prompt_hash"
LEVEL_3 = "3_near_duplicate_ngram_minhash"
LEVEL_4 = "4_semantic_similarity"
LEVEL_5 = "5_manual_audit"

#: What a scan may say about each level. Level 5 is ``NOT_RUN`` in every machine-generated report;
#: it becomes ``COMPLETE`` only through the human audit workflow.
MEASURED = "MEASURED"
NOT_RUN = "NOT_RUN"

#: Report status when levels 1-4 found something, and when they found nothing. Neither is a clean
#: claim: level 5 is pending either way.
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED_LEVEL_5_PENDING"
STATUS_MEASURED = "LEVELS_1_4_MEASURED_LEVEL_5_PENDING"


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


@dataclass(frozen=True)
class Thresholds:
    max_df: int = 1000
    min_shingles: int = 25
    jaccard_threshold: float = 0.5
    containment_threshold: float = 0.6
    level4_jaccard_floor: float = 0.2
    edit_threshold: float = 0.8
    top_k: int = 5


@dataclass
class LevelMatches:
    """Levels 1-4 over one screened set, and the counts that show how much was compared."""

    level1: list[dict[str, Any]]
    level2: list[dict[str, Any]]
    level3: list[dict[str, Any]]
    level4: list[dict[str, Any]]
    heldout_universe_shingles: int
    pruned_shingles: int
    scored_pairs: int
    training_records: int
    sources_seen: set[str]

    @property
    def any_match(self) -> bool:
        return bool(self.level1 or self.level2 or self.level3 or self.level4)


def exact_hash(text: str) -> str:
    """Level 1: sha256 of the raw prompt text, no normalisation."""
    return sha256_text(text)


def normalized_hash(text: str) -> str:
    """Level 2: sha256 of the whitespace-collapsed, casefolded prompt text."""
    return sha256_text(normalize(text))


def hash_index(records: Iterable[Record], normalised: bool) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = normalized_hash(record.text) if normalised else exact_hash(record.text)
        index[digest].append(record.record_id)
    return index


def match_levels(
    heldout: Sequence[Record],
    train_source: Callable[[], Iterable[Record]],
    thresholds: Thresholds | None = None,
) -> LevelMatches:
    """Run levels 1-4 of ``heldout`` against the training records ``train_source`` streams.

    ``train_source`` is called once per pass (the training corpus may be too large to hold), so it
    must return a fresh iterable each time.
    """
    thresholds = thresholds or Thresholds()
    max_df, min_shingles = thresholds.max_df, thresholds.min_shingles
    jaccard_threshold = thresholds.jaccard_threshold
    containment_threshold = thresholds.containment_threshold
    level4_jaccard_floor, edit_threshold = (
        thresholds.level4_jaccard_floor,
        thresholds.edit_threshold,
    )
    top_k = thresholds.top_k

    # Level 1/2 need full hash indexes; level 3 needs the held-out shingle universe.
    heldout_universe: dict[str, list[int]] = defaultdict(list)
    for position, record in enumerate(heldout):
        for shingle in record.shingles:
            heldout_universe[shingle].append(position)

    train_exact = hash_index(train_source(), normalised=False)
    train_normalised = hash_index(train_source(), normalised=True)

    # Heterogeneous value types on purpose: `heldout` is a record id and `training` is every
    # training record id sharing that hash. Without the explicit annotation mypy joins `str` and
    # `list[str]` at `Sequence[str]`, which then mis-types every `item["heldout"]` downstream.
    level1: list[dict[str, Any]] = [
        {"heldout": r.record_id, "training": train_exact[exact_hash(r.text)]}
        for r in heldout
        if train_exact.get(exact_hash(r.text))
    ]
    level2: list[dict[str, Any]] = [
        {"heldout": r.record_id, "training": train_normalised[normalized_hash(r.text)]}
        for r in heldout
        if train_normalised.get(normalized_hash(r.text))
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

    findings: list[dict[str, Any]] = []
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

    return LevelMatches(
        level1=level1,
        level2=level2,
        level3=findings,
        level4=level4,
        heldout_universe_shingles=len(heldout_universe),
        pruned_shingles=len(pruned),
        scored_pairs=sum(len(c) for c in overlaps.values()),
        training_records=len(train_shingle_count),
        sources_seen=sources_seen,
    )


def build_queue(matches: LevelMatches) -> dict[str, dict[str, Any]]:
    """One level-5 queue entry per distinct screened record, with every level that flagged it.

    A human reviewer then sees each question once with all of its evidence instead of per-level
    duplicates.
    """
    queue: dict[str, dict[str, Any]] = {}

    def enqueue(heldout_id: str, level: str, match: dict[str, Any]) -> None:
        entry = queue.setdefault(heldout_id, {"heldout": heldout_id, "levels": [], "matches": []})
        if level not in entry["levels"]:
            entry["levels"].append(level)
        entry["matches"].append({"level": level, **match})

    for item in matches.level1:
        enqueue(item["heldout"], LEVEL_1, {"training": item["training"]})
    for item in matches.level2:
        enqueue(item["heldout"], LEVEL_2, {"training": item["training"]})
    for item in matches.level3:
        enqueue(item["heldout"], LEVEL_3, item)
    for item in matches.level4:
        enqueue(item["heldout"], LEVEL_4, item)
    return queue


def level_status() -> dict[str, str]:
    """The level record every machine-generated report carries: 1-4 measured, 5 not run."""
    return {
        LEVEL_1: MEASURED,
        LEVEL_2: MEASURED,
        LEVEL_3: MEASURED,
        LEVEL_4: MEASURED,
        LEVEL_5: NOT_RUN,
    }


def thresholds_record(thresholds: Thresholds) -> dict[str, Any]:
    return {
        "max_df": thresholds.max_df,
        "min_shingles": thresholds.min_shingles,
        "jaccard": thresholds.jaccard_threshold,
        "containment": thresholds.containment_threshold,
        "edit_similarity": thresholds.edit_threshold,
    }
