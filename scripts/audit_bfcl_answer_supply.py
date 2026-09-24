"""Size and screen BFCL's no-call items as a source for Study 002's held-out ANSWER stratum.

Two measurements back the `bfcl-*` candidates in registry/source_screening.yaml:

1. **Sizing.** A seeded random sample of `BFCL_v4_irrelevance.json` (50 of 240) and
   `BFCL_v4_live_irrelevance.json` (60 of 884), read at the gorilla revision registry/benchmarks.yaml
   pins. Each sampled item has a reading in `answer-sizing-readings.json`: ANSWER (answerable from
   general knowledge, no offered tool fits), BORDERLINE, or NOT_ANSWER. Those readings are one
   reader's sizing judgement, not gold labels (the annotation protocol is the model-annotation pass).
   The estimate is the ANSWER share scaled to the file, with a 95% Wilson interval whose low end
   counts ANSWER only and whose high end counts ANSWER + BORDERLINE.
2. **Overlap.** Every BFCL question is normalised (lowercase, punctuation to space, whitespace
   collapsed) and compared with every user turn in OpenGrad's local normalized corpora: an exact
   match, and 8-gram containment of at least one half. The corpora are git-ignored; without them the
   probe reports BLOCKED_LOCAL_DATA rather than a result.

Usage:
    python scripts/audit_bfcl_answer_supply.py            # rewrite the two JSON artifacts
    python scripts/audit_bfcl_answer_supply.py --check    # fail if the sizing artifact is stale
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.hashing import sha256_bytes

GORILLA_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
FILES = {
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
}
SAMPLE_SIZES = {"irrelevance": 50, "live_irrelevance": 60}
SEED = 20260924
OUT = ROOT / "reports/source-screening/study-002-answer-heldout"
READINGS = OUT / "answer-sizing-readings.json"
SIZING = OUT / "bfcl-answer-sizing.json"
OVERLAP = OUT / "bfcl-training-overlap.json"
CACHE = ROOT / ".cache/source-screening"
CORPORA = [
    "button.jsonl",
    "glaive",
    "looptool",
    "toolace",
    "xlam",
    "when2call-sft",
    "when2call-preference",
    "when2call-mcq",
    "when2call-llm-judge",
]
SHINGLE = 8


def _url(name: str) -> str:
    return (
        f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{GORILLA_REVISION}/"
        f"berkeley-function-call-leaderboard/bfcl_eval/data/{name}"
    )


def load_bfcl() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Both files at the pinned revision, cached under .cache/, with their sha256."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for key, name in FILES.items():
        path = CACHE / f"{GORILLA_REVISION[:12]}-{name}"
        if not path.exists():
            with urllib.request.urlopen(_url(name), timeout=120) as response:
                path.write_bytes(response.read())
        payload = path.read_bytes()
        digests[name] = sha256_bytes(payload)
        rows[key] = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    return rows, digests


def user_text(row: dict[str, Any]) -> str:
    question = row["question"]
    turns = question[0] if isinstance(question[0], list) else question
    return " || ".join(m["content"] for m in turns if m.get("role") == "user")


def draw_sample(rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """One seeded generator, irrelevance first, then live -- the order the readings were made in."""
    rng = random.Random(SEED)
    return {
        key: rng.sample(rows[key], SAMPLE_SIZES[key]) for key in ("irrelevance", "live_irrelevance")
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def sizing(rows: dict[str, list[dict[str, Any]]], digests: dict[str, str]) -> dict[str, Any]:
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    sample = draw_sample(rows)
    per_file = {}
    total = {"point": 0, "low": 0, "high": 0}
    for key, drawn in sample.items():
        recorded = readings["readings"][key]
        drawn_ids = [row["id"] for row in drawn]
        if [item["id"] for item in recorded] != drawn_ids:
            raise SystemExit(
                f"{key}: the readings are not for this sample (seed or revision changed)"
            )
        counts = defaultdict(int)
        for item in recorded:
            counts[item["reading"]] += 1
        n, population = len(drawn), len(rows[key])
        answer, borderline = counts["ANSWER"], counts["BORDERLINE"]
        low = wilson(answer, n)[0] * population
        high = wilson(answer + borderline, n)[1] * population
        point = answer / n * population
        per_file[key] = {
            "file": FILES[key],
            "population": population,
            "sampled": n,
            "readings": dict(sorted(counts.items())),
            "answer_point_estimate": round(point),
            "answer_95_low": round(low),
            "answer_95_high": round(high),
        }
        total["point"] += round(point)
        total["low"] += round(low)
        total["high"] += round(high)
    return {
        "artifact": "bfcl-answer-sizing",
        "status": "SIZING_ESTIMATE_NOT_GOLD",
        "source": {
            "repository": "ShishirPatil/gorilla",
            "revision": GORILLA_REVISION,
            "sha256": digests,
        },
        "seed": SEED,
        "readings_file": READINGS.relative_to(ROOT).as_posix(),
        "method": (
            "ANSWER share of a seeded sample scaled to the file; 95% Wilson interval, low end from "
            "ANSWER readings, high end from ANSWER + BORDERLINE"
        ),
        "per_file": per_file,
        "total": total,
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _corpus_user_turns(path: Path):
    files = (
        [path]
        if path.suffix == ".jsonl"
        else sorted(
            [Path(p) for p in glob.glob(str(path / "**" / "*.jsonl"), recursive=True)]
            + [Path(p) for p in glob.glob(str(path / "**" / "*.parquet"), recursive=True)]
        )
    )
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            names = pq.read_schema(file).names
            column = next(c for c in ("messages", "context", "question") if c in names)
            values = pq.read_table(file, columns=[column]).column(column).to_pylist()
        else:
            values = [
                json.loads(line).get("messages", [])
                for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        for value in values:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    yield value
                    continue
            if isinstance(value, str):
                yield value
            for message in value if isinstance(value, list) else []:
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    yield message["content"]


def overlap(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = ROOT / "data/processed/normalization-v1"
    missing = [name for name in CORPORA if not (base / name).exists()]
    if missing:
        return {
            "artifact": "bfcl-training-overlap",
            "status": "BLOCKED_LOCAL_DATA",
            "missing": missing,
        }
    questions = {row["id"]: _norm(user_text(row)) for key in FILES for row in rows[key]}
    owners: dict[str, set[str]] = defaultdict(set)
    for item_id, text in questions.items():
        words = text.split()
        for k in range(len(words) - SHINGLE + 1):
            owners[" ".join(words[k : k + SHINGLE])].add(item_id)
    exact_index: dict[str, set[str]] = defaultdict(set)
    for item_id, text in questions.items():
        if len(text) >= 12:
            exact_index[text].add(item_id)
    exact: dict[str, set[str]] = defaultdict(set)
    shingles: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    corpus_turns = {}
    for name in CORPORA:
        count = 0
        for turn in _corpus_user_turns(base / name):
            count += 1
            text = _norm(turn)
            for item_id in exact_index.get(text, ()):
                exact[item_id].add(name)
            words = text.split()
            for k in range(len(words) - SHINGLE + 1):
                for item_id in owners.get(" ".join(words[k : k + SHINGLE]), ()):
                    shingles[item_id][name].add(k)
        corpus_turns[name] = count
    contained = {}
    for item_id, per_corpus in shingles.items():
        total = max(1, len(questions[item_id].split()) - SHINGLE + 1)
        share = min(1.0, max(len(v) for v in per_corpus.values()) / total)
        if share >= 0.5:
            contained[item_id] = sorted(per_corpus)
    flagged = sorted(set(exact) | set(contained))
    return {
        "artifact": "bfcl-training-overlap",
        "status": "MEASURED",
        "method": (
            "normalised exact match of each BFCL question against every corpus user turn, plus "
            f"{SHINGLE}-gram containment >= 0.5 in one corpus"
        ),
        "corpora_root": "data/processed/normalization-v1",
        "corpus_user_turns": corpus_turns,
        "questions": len(questions),
        "exact_matches": len(exact),
        "half_contained": len(contained),
        "flagged": len(flagged),
        "flagged_by_file": {
            key: sum(1 for i in flagged if i.startswith(f"{key}_")) for key in FILES
        },
        "items": {
            item_id: {
                "exact": sorted(exact.get(item_id, ())),
                "contained": contained.get(item_id, []),
            }
            for item_id in flagged
        },
    }


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="fail if the sizing artifact is stale")
    args = parser.parse_args()
    rows, digests = load_bfcl()
    size = _dump(sizing(rows, digests))
    if args.check:
        current = SIZING.read_text(encoding="utf-8") if SIZING.exists() else ""
        if current != size:
            print(f"{SIZING.relative_to(ROOT).as_posix()} is stale")
            return 1
        print("CURRENT")
        return 0
    SIZING.write_text(size, encoding="utf-8", newline="\n")
    result = overlap(rows)
    if result["status"] == "MEASURED" or not OVERLAP.exists():
        OVERLAP.write_text(_dump(result), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "sizing": json.loads(size)["total"],
                "overlap": result.get("flagged", result["status"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
