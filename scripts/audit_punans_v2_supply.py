"""Size the sources screened for Study 002's second P-UNANS attempt (screening study-002-punans-v2).

Every number in reports/source-screening/study-002-punans-v2/REPORT.md comes from the artifact this writes.
It reuses the first screening's download, pin and overlap code (scripts/audit_punans_supply.py) and adds:

1. **Sources the first screening did not read:** KUQ's `unknowns_all.jsonl` (the same pinned revision as
   the first attempt's file, with 6,363 unknowns where that file has 3,437) and KUQP's future questions.
   BIG-bench Known Unknowns is read at the first screening's pin.
2. **Freshness.** A question the first attempt drew (any of P-UNANS-v1's 1,063 candidates, by normalised
   text) is not fresh: its labels exist, and 44 keeps them out.
3. **Supply after exclusions:** distinct normalised questions in the unknowable categories, minus those the
   first attempt drew, those naming a year up to 2026, and exact-text collisions with training, held-out
   or evaluation sets. Counted by the source's own category and by who wrote the question (KUQ's `source`:
   GPT, crowdworkers or the web).

Counts only: no question text is printed or written.

Usage:
    python scripts/audit_punans_v2_supply.py            # rewrite the artifact
    python scripts/audit_punans_v2_supply.py --check    # fail if the artifact is stale
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata

_SPEC = importlib.util.spec_from_file_location(
    "audit_punans_supply", ROOT / "scripts/audit_punans_supply.py"
)
assert _SPEC and _SPEC.loader
v1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v1)

OUT = ROOT / "reports/source-screening/study-002-punans-v2/punans-v2-supply.json"
CACHE = ROOT / ".cache/punans-screening"
V1_POPULATION = ROOT / "reports/study-002/punans-v1/punans-v1.population.jsonl"
SOURCES: dict[str, dict[str, str]] = {
    "kuq": {
        "url": "https://huggingface.co/datasets/amayuelas/KUQ/resolve/"
        "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41/unknowns_all.jsonl",
        "revision": "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41",
        "sha256": "8469ab010ce1142bfe3d330635fb58c1e537253568b417b4435a39d658b10855",
    },
    "kuqp": {
        "url": "https://raw.githubusercontent.com/zhaoy777/kuqp-dataset/"
        "596472f31f73acfdcb95741c277413fd500f8b35/KUQP%20Dataset/future_questions.json",
        "revision": "596472f31f73acfdcb95741c277413fd500f8b35",
        "sha256": "25ee426e7881b3cf950bffb1b6d0ebc6c644eb0c1a2c5939662cce4a01ce40d1",
    },
    "bigbench-known-unknowns": v1.SOURCES["bigbench-known-unknowns"],
}
#: KUQ's categories that fit 44 §4 (no one can answer the question now).
KUQ_UNKNOWABLE = ("future unknown", "unsolved problem/mistery")


def fetch(name: str) -> bytes:
    spec = SOURCES[name]
    path = CACHE / f"v2-{name}{Path(spec['url']).suffix}"
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if sha256_bytes(data) != spec["sha256"]:
        raise SystemExit(f"{name}: the downloaded file does not match its pinned sha256")
    return data


def questions(name: str, data: bytes) -> list[dict[str, str]]:
    """Every item the source marks unknowable: category, author and question."""
    if name == "kuq":
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return [
            {
                "category": str(r["category"]),
                "author": str(r["source"]),
                "question": str(r["question"]),
            }
            for r in rows
            if r["category"] in KUQ_UNKNOWABLE
        ]
    if name == "kuqp":
        rows = json.loads(data)
        rows = rows if isinstance(rows, list) else next(iter(rows.values()))
        return [{"category": "future", "author": "gpt", "question": str(r["u"])} for r in rows]
    return [
        {"category": category, "author": "task authors", "question": text}
        for category, text in v1.questions(name, data)
    ]


def first_attempt() -> set[str]:
    return {
        answer_strata.normalise(str(json.loads(line)["user_message"]))
        for line in V1_POPULATION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build() -> dict[str, Any]:
    references, blocked = v1.reference_texts()
    drawn = first_attempt()
    sources: dict[str, Any] = {}
    for name, spec in SOURCES.items():
        items = questions(name, fetch(name))
        distinct: dict[str, dict[str, str]] = {}
        for item in items:
            distinct.setdefault(answer_strata.normalise(item["question"]), item)
        in_v1 = {k for k in distinct if k in drawn}
        dated = {k for k, item in distinct.items() if v1.names_past_year(item["question"])}
        overlap = {key: sum(1 for k in distinct if k in seen) for key, seen in references.items()}
        colliding = {k for k in distinct if any(k in seen for seen in references.values())}
        fresh = {
            k: item
            for k, item in distinct.items()
            if k not in in_v1 and k not in dated and k not in colliding
        }
        sources[name] = {
            **spec,
            "unknowable_rows": len(items),
            "distinct_questions": len(distinct),
            "drawn_by_the_first_attempt": len(in_v1),
            "names_a_year_up_to_screening_year": len(dated),
            "exact_text_overlap": overlap,
            "fresh_after_exclusions": len(fresh),
            "fresh_by_category": dict(
                sorted(Counter(i["category"] for i in fresh.values()).items())
            ),
            "fresh_by_author": dict(sorted(Counter(i["author"] for i in fresh.values()).items())),
        }
    return {
        "screening": "study-002-punans-v2",
        "built_by": "scripts/audit_punans_v2_supply.py",
        "screening_year": v1.SCREENING_YEAR,
        "first_attempt_population": V1_POPULATION.relative_to(ROOT).as_posix(),
        "first_attempt_distinct_questions": len(drawn),
        "normalisation": "opengrad.verification.answer_strata.normalise (lowercase, punctuation to space, whitespace collapsed)",
        "overlap_blocked": sorted(blocked),
        "overlap_status": "BLOCKED_LOCAL_DATA" if blocked else "COMPLETE",
        "sources": sources,
        "fresh_total": sum(s["fresh_after_exclusions"] for s in sources.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = (json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != data:
            print("STALE: punans-v2-supply.json does not match a fresh audit")
            return 1
        print("CURRENT")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
