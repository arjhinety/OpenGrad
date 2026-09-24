"""Build and verify the ANSWER strata candidate population (Study 002, ``study_002_prereg_v9``).

The amendment is ``docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md`` ("41" below). Two pools:

* **pool N** (natural): BFCL's no-call files at the gorilla revision the benchmark registry pins, every item with
  one user turn, no system message and at least one tool (41 §4);
* **pool K** (constructed): ``nq_open`` validation questions, in seeded order, skipping time cues and short
  questions, each paired with 1-3 BFCL ``simple_python`` tool schemas that cannot serve it (41 §5).

Both pools are deduplicated and screened against every user turn of OpenGrad's normalized corpora, by the
contamination engine at levels 1-4 and by the word 8-gram probe (41 §6). What survives is the population the
three external annotators label (41 §7-§9); nothing here labels anything.

``--dry-run`` writes counts and hashes only; ``--build`` writes the population, never over an existing one;
``--verify`` re-hashes it and, when the inputs are present, re-derives it. Nothing here prints an item: the draw
is reproducible, so the written population is the blind sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from opengrad.contamination.levels import Record, Thresholds, match_levels, thresholds_record
from opengrad.contamination.scanner import ngrams
from opengrad.data.normalization_v3 import lf_sha256
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

POPULATION_ID = "ANSWER-STRATA-v1"
PROTOCOL_VERSION = "answer-strata-001-v1"
SEED = "opengrad-answer-strata-001-v1"
PREREGISTRATION = Path("docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md")
ADOPTION_AMENDMENT = "study_002_prereg_v9"
OUTPUT_DIR = Path("reports/study-002/answer-strata-v1")
POPULATION_NAME = "answer-strata-v1.population.jsonl"
MANIFEST_NAME = "answer-strata-v1.manifest.json"
DRY_RUN_NAME = "answer-strata-v1.dry-run.json"
ID_PREFIX = "ans1:"
CACHE = Path(".cache/answer-strata")
POOL_K_SIZE = 850

GORILLA = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
_BFCL = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{GORILLA}/berkeley-function-call-leaderboard/bfcl_eval/data/"
NQ_OPEN = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
INPUTS: dict[str, dict[str, str]] = {
    "bfcl_irrelevance": {
        "url": _BFCL + "BFCL_v4_irrelevance.json",
        "sha256": "2b6ed4c2e992cdcf5f1678a701851f944bef7550ee026ed1ddb89efed5be01a6",
    },
    "bfcl_live_irrelevance": {
        "url": _BFCL + "BFCL_v4_live_irrelevance.json",
        "sha256": "6559fda2beaceb609a2cd2e504c65b4a56cb448e1ef88fddfd199e163d163349",
    },
    "bfcl_simple_python": {
        "url": _BFCL + "BFCL_v4_simple_python.json",
        "sha256": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    },
    "nq_open_validation": {
        "url": (
            f"https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/{NQ_OPEN}/"
            "nq_open/validation-00000-of-00001.parquet"
        ),
        "sha256": "b074bed0bccb56fa1551a8ac1c9c51ce89bc11c7fbb6a9c713b2c33a98531e12",
    },
}

CORPORA_ROOT = Path("data/processed/normalization-v1")
CORPORA = (
    "button.jsonl",
    "glaive",
    "looptool",
    "toolace",
    "xlam",
    "when2call-sft",
    "when2call-preference",
    "when2call-mcq",
    "when2call-llm-judge",
)
PROBE_N = 8

TIME_CUES = re.compile(
    r"\b(current|currently|latest|now|today|this year|recent|recently|new|newest|next|upcoming|still)\b"
)
INFORMATION_ROUTE = re.compile(
    r"search|lookup|look_up|query|wiki|knowledge|encyclop|fact|trivia|news|web|google|browse|retriev|"
    r"question|answer|definition|dictionary|biograph|histor|database"
)
ENTITY_DOMAIN = re.compile(
    r"music|song|album|singer|movie|film|actor|celebrity|sport|game|team|player|book|author|artist|award|"
    r"election|president|country|capital|population|geograph|city|state"
)
STOP_WORDS = frozenset(
    [
        "that",
        "this",
        "with",
        "from",
        "what",
        "when",
        "where",
        "which",
        "whose",
        "were",
        "have",
        "does",
        "into",
        "their",
        "there",
        "about",
        "after",
        "before",
        "first",
        "last",
        "most",
        "many",
        "much",
        "more",
        "than",
        "they",
        "them",
        "then",
        "also",
        "been",
        "being",
        "over",
        "under",
        "between",
        "during",
    ]
)
CODE_MODULES = (
    "src/opengrad/verification/answer_strata.py",
    "src/opengrad/contamination/levels.py",
)


class AnswerStrataError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rank(*parts: str) -> str:
    return _sha256("|".join((SEED, *parts)).encode("utf-8"))


def item_id(pool: str, source: str, source_id: str) -> str:
    return ID_PREFIX + _sha256(f"{pool}:{source}:{source_id}".encode())


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ── Inputs ───────────────────────────────────────────────────────────────────


def fetch_inputs(root: Path, *, download: bool) -> dict[str, bytes] | None:
    """Every pinned input's bytes, from the cache or (when allowed) downloaded; None if one is missing."""
    cache = root / CACHE
    payloads: dict[str, bytes] = {}
    for key, spec in INPUTS.items():
        path = cache / key
        if not path.is_file():
            if not download:
                return None
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(str(spec["url"]), timeout=300) as response:
                path.write_bytes(response.read())
        payload = path.read_bytes()
        if _sha256(payload) != spec["sha256"]:
            raise AnswerStrataError(
                f"{key}: sha256 {_sha256(payload)} is not the pinned {spec['sha256']}"
            )
        payloads[key] = payload
    return payloads


def _jsonl(payload: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def _turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    question = row["question"]
    return [m for turn in question for m in (turn if isinstance(turn, list) else [turn])]


def _nq_rows(payload: bytes) -> list[dict[str, Any]]:
    import io

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pq.read_table(io.BytesIO(payload))
    return [
        {"question": q, "answers": list(a)}
        for q, a in zip(
            table.column("question").to_pylist(), table.column("answer").to_pylist(), strict=True
        )
    ]


# ── Pools ────────────────────────────────────────────────────────────────────


def pool_n(payloads: dict[str, bytes], counts: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for key, dataset in (
        ("bfcl_irrelevance", "bfcl_v4_irrelevance"),
        ("bfcl_live_irrelevance", "bfcl_v4_live_irrelevance"),
    ):
        for row in _jsonl(payloads[key]):
            turns = _turns(row)
            users = [m for m in turns if m.get("role") == "user"]
            if any(m.get("role") == "system" for m in turns):
                reasons["system_message"] += 1
                continue
            if len(users) != 1:
                reasons["not_one_user_turn"] += 1
                continue
            if not row.get("function"):
                reasons["no_tool"] += 1
                continue
            items.append(
                {
                    "answer_id": item_id("N", dataset, str(row["id"])),
                    "pool": "N",
                    "source_dataset": dataset,
                    "source_id": str(row["id"]),
                    "user_message": str(users[0]["content"]),
                    "tools": row["function"],
                }
            )
    counts["pool_n"] = {"eligible": len(items), "ineligible": dict(sorted(reasons.items()))}
    return items


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", text.lower()) if word not in STOP_WORDS}


def tool_pool(payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in _jsonl(payloads["bfcl_simple_python"]):
        for function in row.get("function") or []:
            seen.setdefault(str(function["name"]), function)
    return [seen[name] for name in sorted(seen)]


def _tool_text(function: dict[str, Any]) -> str:
    return f"{function.get('name', '')} {function.get('description', '')}".lower()


def distractors(question: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1-3 tools that cannot serve the question (41 §5), in seeded order."""
    k = 1 + bytes.fromhex(_rank("k", question))[0] % 3
    words = _content_words(question)
    chosen = []
    for function in sorted(tools, key=lambda f: _rank("tool", question, str(f["name"]))):
        text = _tool_text(function)
        if INFORMATION_ROUTE.search(text) or ENTITY_DOMAIN.search(text):
            continue
        if words & _content_words(text.replace("_", " ").replace(".", " ")):
            continue
        chosen.append(function)
        if len(chosen) == k:
            break
    return chosen


def pool_k_candidates(payloads: dict[str, bytes], counts: dict[str, Any]) -> list[dict[str, Any]]:
    """Every nq_open question in seeded order, with its skip reason or its tools; screening comes later."""
    tools = tool_pool(payloads)
    counts["tool_pool"] = {"functions": len(tools)}
    candidates = []
    for row in sorted(
        _nq_rows(payloads["nq_open_validation"]), key=lambda r: _rank("K", r["question"])
    ):
        question = str(row["question"])
        candidate = {
            "answer_id": item_id("K", "nq_open_validation", question),
            "pool": "K",
            "source_dataset": "nq_open_validation",
            "source_id": question,
            "user_message": question,
            "reference_answers": row["answers"],
        }
        if TIME_CUES.search(question.lower()):
            candidate["skip"] = "time_cue"
        elif len(question.split()) < 4:
            candidate["skip"] = "under_four_words"
        else:
            candidate["tools"] = distractors(question, tools)
            if not candidate["tools"]:
                candidate["skip"] = "no_eligible_tool"
        candidates.append(candidate)
    return candidates


# ── Screen ───────────────────────────────────────────────────────────────────


def corpus_user_turns(root: Path, name: str) -> Iterator[str]:
    path = root / CORPORA_ROOT / name
    files = (
        [path]
        if name.endswith(".jsonl")
        else sorted([*path.rglob("*.jsonl"), *path.rglob("*.parquet")])
    )
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            names = pq.read_schema(file).names
            column = next(c for c in ("messages", "context", "question") if c in names)
            values: Iterable[Any] = pq.read_table(file, columns=[column]).column(column).to_pylist()
        else:
            values = (
                json.loads(line).get("messages", [])
                for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        for value in values:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    yield value
                    continue
            if isinstance(value, str):
                yield value
                continue
            for message in value if isinstance(value, list) else []:
                if (
                    isinstance(message, dict)
                    and message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                ):
                    yield message["content"]


def corpora_identity(root: Path) -> dict[str, str] | None:
    """The sha256 of each corpus's manifest, or None if a corpus is absent (then the screen is BLOCKED)."""
    identity = {}
    for name in CORPORA:
        path = root / CORPORA_ROOT / name
        manifest = (
            path.with_name(name.removesuffix(".jsonl") + ".manifest.json")
            if name.endswith(".jsonl")
            else path / "manifest.json"
        )
        if not path.exists() or not manifest.is_file():
            return None
        identity[name] = _sha256(manifest.read_bytes())
    return identity


def screen(root: Path, items: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    """41 §6: ids flagged by the engine (levels 1-4) or by the word 8-gram probe."""
    training: list[Record] = []
    for name in CORPORA:
        for index, turn in enumerate(corpus_user_turns(root, name)):
            training.append(Record(record_id=f"{name}:{index}", source=name, text=turn))
    heldout = [
        Record(
            record_id=i["answer_id"],
            source=i["pool"],
            text=i["user_message"],
            shingles=frozenset(ngrams(i["user_message"])),
        )
        for i in items
    ]
    thresholds = Thresholds()
    matches = match_levels(heldout, lambda: iter(training), thresholds)
    engine: dict[str, set[str]] = defaultdict(set)
    for level, found in (
        ("1", matches.level1),
        ("2", matches.level2),
        ("3", matches.level3),
        ("4", matches.level4),
    ):
        for match in found:
            engine[level].add(str(match["heldout"]))

    texts = {i["answer_id"]: normalise(i["user_message"]) for i in items}
    owners: dict[str, set[str]] = defaultdict(set)
    exact_index: dict[str, set[str]] = defaultdict(set)
    for answer_id, text in texts.items():
        words = text.split()
        for k in range(len(words) - PROBE_N + 1):
            owners[" ".join(words[k : k + PROBE_N])].add(answer_id)
        if len(text) >= 12:
            exact_index[text].add(answer_id)
    exact: set[str] = set()
    positions: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for record in training:
        text = normalise(record.text)
        exact |= exact_index.get(text, set())
        words = text.split()
        for k in range(len(words) - PROBE_N + 1):
            for answer_id in owners.get(" ".join(words[k : k + PROBE_N]), ()):
                positions[answer_id][record.source].add(k)
    contained = {
        answer_id
        for answer_id, per_corpus in positions.items()
        if max(len(v) for v in per_corpus.values())
        / max(1, len(texts[answer_id].split()) - PROBE_N + 1)
        >= 0.5
    }
    probe = exact | contained
    flagged = set().union(*engine.values(), probe)
    by_pool = {
        pool: sum(1 for i in items if i["pool"] == pool and i["answer_id"] in flagged)
        for pool in ("N", "K")
    }
    report = {
        "training_user_turns": len(training),
        "engine_thresholds": thresholds_record(thresholds),
        "engine_level_counts": {
            f"level_{level}": len(ids) for level, ids in sorted(engine.items())
        },
        "probe_exact": len(exact),
        "probe_half_contained": len(contained),
        "flagged": len(flagged),
        "flagged_by_pool": by_pool,
    }
    return flagged, report


# ── Draw ─────────────────────────────────────────────────────────────────────


def draw(root: Path, payloads: dict[str, bytes]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: dict[str, Any] = {}
    natural = pool_n(payloads, counts)
    constructed = pool_k_candidates(payloads, counts)
    counts["pool_k_skipped_before_screen"] = dict(
        sorted(Counter(c["skip"] for c in constructed if "skip" in c).items())
    )
    usable_k = [c for c in constructed if "skip" not in c]

    # Duplicates (41 §4): within and across pools, pool N first, each pool in ascending id order.
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed: Counter[str] = Counter()
    for pool, items in (("N", natural), ("K", usable_k)):
        for item in sorted(items, key=lambda i: i["answer_id"]):
            key = normalise(item["user_message"])
            if key in seen:
                removed[pool] += 1
                continue
            seen.add(key)
            kept.append(item)
    counts["duplicates_removed"] = {pool: removed.get(pool, 0) for pool in ("N", "K")}

    flagged, report = screen(root, kept)
    counts["screen"] = report
    survivors = [item for item in kept if item["answer_id"] not in flagged]
    natural_final = [i for i in survivors if i["pool"] == "N"]
    # Pool K keeps the first POOL_K_SIZE survivors in the seeded order (41 §5).
    order = {c["answer_id"]: position for position, c in enumerate(constructed)}
    constructed_final = sorted(
        (i for i in survivors if i["pool"] == "K"), key=lambda i: order[i["answer_id"]]
    )[:POOL_K_SIZE]
    counts["pool_k_available_after_screen"] = sum(1 for i in survivors if i["pool"] == "K")
    if len(constructed_final) < POOL_K_SIZE:
        counts["pool_k_shortage"] = POOL_K_SIZE - len(constructed_final)
    population = sorted(
        natural_final + constructed_final, key=lambda i: _rank("order", i["answer_id"])
    )
    counts["realized"] = {
        "N": len(natural_final),
        "K": len(constructed_final),
        "total": len(population),
    }
    counts["realized_by_source"] = dict(
        sorted(Counter(i["source_dataset"] for i in population).items())
    )
    counts["tools_per_item"] = {
        pool: dict(
            sorted(Counter(len(i["tools"]) for i in population if i["pool"] == pool).items())
        )
        for pool in ("N", "K")
    }
    return population, counts


def population_bytes(population: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in population
    ).encode("utf-8")


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def build_population(
    root: Path, *, download: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payloads = fetch_inputs(root, download=download)
    if payloads is None:
        raise AnswerStrataError(
            "inputs are not cached; run --build or --dry-run once with network access"
        )
    identity = corpora_identity(root)
    if identity is None:
        raise AnswerStrataError(
            f"the normalized corpora under {CORPORA_ROOT.as_posix()} are required for the screen"
        )
    population, counts = draw(root, payloads)
    inputs = {
        key: {"url": spec["url"], "sha256": _sha256(payloads[key])} for key, spec in INPUTS.items()
    }
    manifest = {
        "artifact_kind": "ANSWER_STRATA_POPULATION",
        "schema_version": 1,
        "population_id": POPULATION_ID,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": "ADOPTED",
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "inputs": inputs,
        "screen_corpora_manifest_sha256": identity,
        "pool_k_size": POOL_K_SIZE,
        "counts": counts,
        "id_rule": "ans1: + sha256(pool : source_dataset : source_id)",
        "presentation_order": "sha256(seed | order | answer_id) ascending",
        "code_sha256_lf": {module: lf_sha256(root / module) for module in CODE_MODULES},
        "gold_labels_present": False,
        "blinded_fields": ["pool", "source_dataset", "source_id", "reference_answers"],
        "licences": {
            "pool_N": "Apache-2.0 (BFCL)",
            "pool_K": "CC-BY-SA-3.0 (Natural Questions via nq_open) with Apache-2.0 tool schemas (BFCL)",
        },
        "statement": (
            "Candidate items for Study 002's ANSWER strata set, before labelling. Pool N is natural (BFCL no-call "
            "items); pool K is constructed (Natural Questions questions with tools that cannot serve them). "
            "The strata are the items the three-model consensus labels ANSWER (41 §9)."
        ),
    }
    return population, manifest


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    reports = (root / "reports").resolve()
    for forbidden in ("pdet", "pdet-coverage", "pdet-coverage-v2", "evaluation"):
        blocked = reports / forbidden
        if resolved == blocked or blocked in resolved.parents:
            raise AnswerStrataError(f"{POPULATION_ID} is never written under reports/{forbidden}/")
    return resolved


def write_population(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    if (directory / POPULATION_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise AnswerStrataError(
            f"a population already exists in {directory}; it is never overwritten"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = population_bytes(population)
    stamped = {
        **manifest,
        "population_file": POPULATION_NAME,
        "population_records": len(population),
        "population_sha256": _sha256(payload),
    }
    data = manifest_bytes(stamped)
    (directory / POPULATION_NAME).write_bytes(payload)
    (directory / MANIFEST_NAME).write_bytes(data)
    (directory / (MANIFEST_NAME + ".sha256")).write_bytes((_sha256(data) + "\n").encode())
    return stamped


def write_dry_run(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **manifest,
        "artifact_kind": "ANSWER_STRATA_DRY_RUN",
        "population_written": False,
        "population_records": len(population),
        "population_sha256": _sha256(population_bytes(population)),
    }
    (directory / DRY_RUN_NAME).write_bytes(manifest_bytes(record))
    return record


REPRODUCED_FIELDS = (
    "counts",
    "inputs",
    "seed",
    "protocol_version",
    "pool_k_size",
    "screen_corpora_manifest_sha256",
)


def verify_population(root: Path, output_dir: Path) -> dict[str, Any]:
    directory = output_dir if output_dir.is_absolute() else root / output_dir
    population_path, manifest_path = directory / POPULATION_NAME, directory / MANIFEST_NAME
    if not population_path.is_file() or not manifest_path.is_file():
        return {"status": FAIL, "errors": [f"{POPULATION_ID} artifacts are missing"]}
    populated, manifest_text = population_path.read_bytes(), manifest_path.read_bytes()
    manifest = json.loads(manifest_text)
    items = [json.loads(line) for line in populated.decode("utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if manifest.get("population_sha256") != _sha256(populated):
        errors.append("FAIL_HASH: population bytes do not match manifest population_sha256")
    sidecar = directory / (MANIFEST_NAME + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != _sha256(
        manifest_text
    ):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")
    if len(items) != manifest.get("population_records"):
        errors.append("FAIL_COUNT: population_records does not match the file")
    if any(item.get("gold_policy_label") not in (None, "") for item in items):
        errors.append("FAIL_ANNOTATION: an item carries a gold label in the population file")
    blocked: list[str] = []
    payloads = fetch_inputs(root, download=False)
    if payloads is None or corpora_identity(root) is None:
        blocked.append(
            f"{BLOCKED_INPUT_MISSING}: cached inputs or normalized corpora absent; not re-derived"
        )
    else:
        rebuilt, fresh = build_population(root, download=False)
        # Compare as written: JSON turns the integer keys of the tool counts into strings.
        rebuilt_manifest = json.loads(manifest_bytes(fresh))
        if population_bytes(rebuilt) != populated:
            errors.append(
                "FAIL_REPRODUCIBILITY: a fresh draw did not reproduce the population bytes"
            )
        errors += [
            f"FAIL_PROVENANCE: manifest field {key!r} differs from a fresh draw"
            for key in REPRODUCED_FIELDS
            if manifest.get(key) != rebuilt_manifest.get(key)
        ]
    status = FAIL if errors else BLOCKED_INPUT_MISSING if blocked else PASS
    return {
        "status": status,
        "population_id": POPULATION_ID,
        "population_sha256": manifest.get("population_sha256"),
        "items": len(items),
        "errors": errors,
        "blocked": blocked,
    }


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """Counts and hashes only."""
    return {
        "population_sha256": record.get("population_sha256"),
        "population_records": record.get("population_records"),
        "counts": record["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    record = (
        write_dry_run(root, output_dir, population, manifest)
        if args.dry_run
        else write_population(root, output_dir, population, manifest)
    )
    print(json.dumps(summary(record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
