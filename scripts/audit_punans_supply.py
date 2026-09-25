"""Size and probe the public sources screened for Study 002's P-UNANS (screening study-002-punans).

Every number in reports/source-screening/study-002-punans/REPORT.md comes from the artifact this writes.

1. **Provenance.** Each source file is downloaded at a pinned revision and refused unless its sha256 equals
   the pin below; the pins are recorded in the artifact and in registry/source_screening.yaml.
2. **Supply.** Items are counted by the source's own labels: KUQ's `unknown` flag and `category`, SelfAware's
   `answerable` flag, CoCoNot's `subcategory` (evaluation split only), BIG-bench Known Unknowns' target.
   These are the sources' labels, not OpenGrad's: P-UNANS items are relabelled blind by two models.
3. **Time.** A question naming a year up to the current one may ask about an event that has since
   happened, which would make it answerable. Such questions are counted per category.
4. **Overlap.** Each candidate question is normalised (lowercase, punctuation to space, whitespace collapsed)
   and matched exactly against every user turn of OpenGrad's normalized training corpora, the When2Call
   held-out questions, the ANSWER strata candidates and the P-DET-COVERAGE populations. The corpora and
   the held-out shards are git-ignored; without them that part is BLOCKED_LOCAL_DATA.

Counts only: no question text is printed or written.

Usage:
    python scripts/audit_punans_supply.py            # rewrite the artifact
    python scripts/audit_punans_supply.py --check    # fail if the artifact is stale
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata

OUT = ROOT / "reports/source-screening/study-002-punans/punans-supply.json"
CACHE = ROOT / ".cache/punans-screening"
#: The latest year a question can name and still ask about a past event, at the screening date.
SCREENING_YEAR = 2026
YEAR = re.compile(r"\b(19|20)\d\d\b")
SOURCES: dict[str, dict[str, str]] = {
    "kuq": {
        "url": "https://huggingface.co/datasets/amayuelas/KUQ/resolve/"
        "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41/knowns_unknowns.jsonl",
        "revision": "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41",
        "sha256": "798d1677f962d11d069f77d1e3db91ad2ddb483a94b697e46bc8ca62ad0aedf6",
    },
    "selfaware": {
        "url": "https://raw.githubusercontent.com/yinzhangyue/SelfAware/"
        "f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c/data/SelfAware.json",
        "revision": "f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c",
        "sha256": "32929585ffdd4048f35f7f167720722eb8ddcf59908ffc958da2f84ea634dc54",
    },
    "coconot": {
        "url": "https://huggingface.co/datasets/allenai/coconot/resolve/"
        "2cbe16aabf9069f17e48c8daad8aeabc29469eb7/original/test-00000-of-00001.parquet",
        "revision": "2cbe16aabf9069f17e48c8daad8aeabc29469eb7",
        "sha256": "e84a986e967c19ea2613e3b2abfd76385d977ad0bf5114e3994cc8304be49573",
    },
    "bigbench-known-unknowns": {
        "url": "https://raw.githubusercontent.com/google/BIG-bench/"
        "124892ccf54f85402852d68c93736a4fa57bf009/bigbench/benchmark_tasks/known_unknowns/task.json",
        "revision": "124892ccf54f85402852d68c93736a4fa57bf009",
        "sha256": "6061bdd796f2225fee6ef9c389e1b1e19f4d83da905dc7b512e0d1ca0e44c557",
    },
}
#: CoCoNot evaluation subcategories that ask for something no one can know (the rest are safety,
#: underspecification, subjectivity, style or modality requests).
COCONOT_UNKNOWABLE = ("universal unknowns", "temporal limitations")


def fetch(name: str) -> bytes:
    spec = SOURCES[name]
    path = CACHE / f"{name}{Path(spec['url']).suffix}"
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if sha256_bytes(data) != spec["sha256"]:
        raise SystemExit(f"{name}: the downloaded file does not match its pinned sha256")
    return data


def questions(name: str, data: bytes) -> list[tuple[str, str]]:
    """(category, question) for every item the source itself marks unanswerable or unknowable."""
    if name == "kuq":
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return [(str(r["category"]), str(r["question"])) for r in rows if r["unknown"]]
    if name == "selfaware":
        rows = json.loads(data)["example"]
        return [
            ("unanswerable (no category)", str(r["question"]))
            for r in rows
            if r["answerable"] is False
        ]
    if name == "coconot":
        import pyarrow.parquet as pq

        rows = pq.read_table(io.BytesIO(data)).to_pylist()
        return [
            (str(r["subcategory"]), str(r["prompt"]))
            for r in rows
            if r["subcategory"] in COCONOT_UNKNOWABLE
        ]
    examples = json.loads(data)["examples"]
    return [
        ("unknown", str(e["input"]))
        for e in examples
        if max(e["target_scores"], key=e["target_scores"].get) == "Unknown"
    ]


def names_past_year(text: str) -> bool:
    return any(int(match.group(0)) <= SCREENING_YEAR for match in YEAR.finditer(text))


def reference_texts() -> tuple[dict[str, set[str]], list[str]]:
    """Normalised texts of everything P-UNANS must stay disjoint from, and what could not be read."""
    texts: dict[str, set[str]] = {}
    blocked: list[str] = []
    turns: set[str] = set()
    for name in answer_strata.CORPORA:
        try:
            turns.update(
                answer_strata.normalise(t) for t in answer_strata.corpus_user_turns(ROOT, name)
            )
        except (FileNotFoundError, OSError):
            blocked.append(f"training corpus {name}")
    texts["training_corpora"] = turns
    try:
        from opengrad.evaluation.runner import load_evaluation_examples

        examples = load_evaluation_examples(
            ROOT, ROOT / "reports/evaluation/behavioral-heldout-v2.manifest.json"
        )
        texts["when2call_heldout"] = {answer_strata.normalise(e.question) for e in examples}
    except (FileNotFoundError, ValueError):
        blocked.append("When2Call held-out shards")
    tracked = {
        "answer_strata_candidates": (
            "reports/study-002/answer-strata-v1/answer-strata-v1.population.jsonl",
        ),
        "pdet_coverage": (
            "reports/pdet-coverage/pdet-coverage-v1.population.jsonl",
            "reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl",
        ),
    }
    for key, paths in tracked.items():
        texts[key] = {
            answer_strata.normalise(str(json.loads(line)["user_message"]))
            for path in paths
            for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return texts, blocked


def build() -> dict[str, Any]:
    references, blocked = reference_texts()
    sources: dict[str, Any] = {}
    for name, spec in SOURCES.items():
        items = questions(name, fetch(name))
        by_category = Counter(category for category, _ in items)
        past_year = Counter(category for category, text in items if names_past_year(text))
        overlap = {
            key: sum(1 for _, text in items if answer_strata.normalise(text) in seen)
            for key, seen in references.items()
        }
        sources[name] = {
            **spec,
            "unknowable_items": len(items),
            "by_category": dict(sorted(by_category.items())),
            "names_a_year_up_to_screening_year": dict(sorted(past_year.items())),
            "exact_text_overlap": overlap,
        }
    return {
        "screening": "study-002-punans",
        "built_by": "scripts/audit_punans_supply.py",
        "screening_year": SCREENING_YEAR,
        "normalisation": "opengrad.verification.answer_strata.normalise (lowercase, punctuation to space, whitespace collapsed)",
        "overlap_blocked": sorted(blocked),
        "overlap_status": "BLOCKED_LOCAL_DATA" if blocked else "COMPLETE",
        "sources": sources,
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
            print("STALE: punans-supply.json does not match a fresh audit")
            return 1
        print("CURRENT")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
