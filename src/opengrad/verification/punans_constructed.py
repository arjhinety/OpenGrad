"""P-UNANS-v2-constructed: constructed unknowable questions for check 4 (``study_002_prereg_v13``, doc 45).

Two families of question, each asking for a specific outcome no one can know yet (45 §4), with answerable
controls (45 §5), all built from one pinned Wikidata snapshot (45 §3, CC0-1.0):

* **F1, a future winner.** A competition that named a winner for an edition in 2021 or later, in a year from
  2035 to 2060 that fits its own cycle (the most common gap between its recent editions), so the question
  carries no false premise. Control: the winner of one of its past editions (1990-2020), in a year with exactly
  one edition.
* **F2, a future dated measurement.** A city of over a million people (one country, a unique name), its highest
  temperature on a date from 2035 to 2060. Control: its population as of a year Wikidata records a figure for.

Each family draws its entities in ascending order of ``sha256(seed | family | entity id)``: the first 300
eligible entities give the questions and the next 75 control-eligible ones give the controls. The screen of 44 §7
runs first, with exact collisions against every P-UNANS-v1 and v2 item. Nothing is redrawn to reach a size.

``--snapshot`` queries Wikidata once and writes the snapshot; every other action reads only the snapshot.
Counts and hashes only are printed: never an item's text or id.

    python -m opengrad.verification.punans_constructed --snapshot | --dry-run | --build | --verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from opengrad.data.normalization_v3 import lf_sha256
from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata as strata
from opengrad.verification import punans as v1
from opengrad.verification import punans_v2 as v2
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

POPULATION_ID = "P-UNANS-v2-constructed"
PROTOCOL_VERSION = "punans-003-v13"
SEED = "opengrad-punans-003-v13"
PREREGISTRATION = Path("docs/research/study-002/45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md")
ADOPTION_AMENDMENT = "study_002_prereg_v13"
OUTPUT_DIR = Path("reports/study-002/punans-v2-constructed")
SNAPSHOT_NAME = "wikidata-snapshot.json"
POPULATION_NAME = "punans-v2-constructed.population.jsonl"
MANIFEST_NAME = "punans-v2-constructed.manifest.json"
DRY_RUN_NAME = "punans-v2-constructed.dry-run.json"
ID_PREFIX = "unc:"
SOURCE = "wikidata-constructed"
FAMILIES = ("F1", "F2")
SIZES = {"F1": 300, "F2": 300}
CONTROLS = {"F1": 75, "F2": 75}
FUTURE = (2035, 2060)
F1_ACTIVE_SINCE = 2021
F1_CONTROL_YEARS = (1990, 2020)
F2_CONTROL_YEARS = (1950, 2023)
MIN_EDITION_YEARS = 3
CYCLE_WINDOW = 7
YEARISH = re.compile(r"\b(?:1[0-9]|20|21)\d\d\b")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "OpenGrad-punans-constructed/1.0 (research; https://github.com/arjhinety/OpenGrad)"
QUERIES = {
    "competition_labels": (
        "SELECT DISTINCT ?series ?label WHERE { ?ed wdt:P3450 ?series ; wdt:P1346 ?w ; wdt:P585 ?t . "
        'FILTER(YEAR(?t) >= 2021) ?series rdfs:label ?label . FILTER(LANG(?label) = "en") }'
    ),
    "competition_editions": (
        "SELECT ?series ?y (COUNT(DISTINCT ?ed) AS ?n) WHERE { ?ed wdt:P3450 ?series ; wdt:P1346 ?w ; "
        "wdt:P585 ?t . BIND(YEAR(?t) AS ?y) } GROUP BY ?series ?y"
    ),
    "cities": (
        "SELECT ?c ?label ?country WHERE { ?c wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?p . FILTER(?p > 1000000) "
        '?c wdt:P17 ?k . ?c rdfs:label ?label . FILTER(LANG(?label) = "en") ?k rdfs:label ?country . '
        'FILTER(LANG(?country) = "en") }'
    ),
    "city_population_years": (
        "SELECT DISTINCT ?c ?y WHERE { ?c wdt:P31/wdt:P279* wd:Q515 ; wdt:P1082 ?p . FILTER(?p > 1000000) "
        "?c p:P1082 ?st . ?st pq:P585 ?t . BIND(YEAR(?t) AS ?y) }"
    ),
}
#: Four phrasings per family and kind (45 §4-§5); the control phrasings parallel the questions' in the past tense.
PHRASINGS = {
    ("F1", "question"): (
        "Who will win the {year} {name}?",
        "Which team or competitor will win the {name} in {year}?",
        "Who is going to be the winner of the {year} {name}?",
        "In {year}, who will win the {name}?",
    ),
    ("F1", "control"): (
        "Who won the {year} {name}?",
        "Which team or competitor won the {name} in {year}?",
        "Who was the winner of the {year} {name}?",
        "In {year}, who won the {name}?",
    ),
    ("F2", "question"): (
        "What will the highest temperature recorded in {name} be on {date}?",
        "What will the maximum temperature in {name} be on {date}?",
        "How hot will it get in {name} on {date}?",
        "On {date}, what high temperature will {name} record?",
    ),
    ("F2", "control"): (
        "What was the population of {name} as of {year}?",
        "How many people lived in {name} in {year}?",
        "What population did {name} have in {year}?",
        "In {year}, what was the population of {name}?",
    ),
}
#: Every tracked item P-UNANS-v2-constructed must not share a question with (45 §6).
DISJOINT_FROM = (
    *v2.DISJOINT_FROM,
    v2.OUTPUT_DIR / v2.TRIAL_NAME,
    v2.OUTPUT_DIR / v2.MAIN_NAME,
)
CODE_MODULES = (
    "src/opengrad/verification/punans_constructed.py",
    "src/opengrad/verification/punans_v2.py",
    "src/opengrad/verification/punans.py",
    "src/opengrad/verification/answer_strata.py",
    "src/opengrad/contamination/levels.py",
)


class PUnansConstructedError(ValueError):
    pass


def _rank(*parts: str) -> str:
    return sha256_bytes("|".join((SEED, *parts)).encode("utf-8"))


def _pick(options: list[Any], *parts: str) -> Any:
    return options[int(_rank(*parts), 16) % len(options)]


def item_id(family: str, kind: str, entity: str) -> str:
    return ID_PREFIX + sha256_bytes(f"{family}:{kind}:{entity}".encode())


def distractors(question: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1-3 tools that cannot serve the question: the rule of 41 §5, 43 §6 and 44 §6 under this seed."""
    k = 1 + bytes.fromhex(_rank("k", question))[0] % 3
    words = strata._content_words(question)
    chosen: list[dict[str, Any]] = []
    for function in sorted(tools, key=lambda f: _rank("tool", question, str(f["name"]))):
        text = strata._tool_text(function)
        if strata.INFORMATION_ROUTE.search(text) or strata.ENTITY_DOMAIN.search(text):
            continue
        if words & strata._content_words(text.replace("_", " ").replace(".", " ")):
            continue
        chosen.append(function)
        if len(chosen) == k:
            break
    return chosen


# ── Snapshot ──────────────────────────────────────────────────────────────────


def _query(query: str) -> list[dict[str, str]]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': query})}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        rows = json.loads(response.read())["results"]["bindings"]
    return [{key: value["value"] for key, value in row.items()} for row in rows]


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def fetch_snapshot() -> dict[str, Any]:
    """The entities 45 draws on, from one set of Wikidata queries, in a canonical order."""
    labels: dict[str, set[str]] = {}
    for row in _query(QUERIES["competition_labels"]):
        labels.setdefault(_qid(row["series"]), set()).add(row["label"])
    editions: dict[str, dict[str, int]] = {}
    for row in _query(QUERIES["competition_editions"]):
        series = _qid(row["series"])
        if series in labels:
            editions.setdefault(series, {})[str(int(row["y"]))] = int(row["n"])
    cities: dict[str, dict[str, set[str]]] = {}
    for row in _query(QUERIES["cities"]):
        city = cities.setdefault(_qid(row["c"]), {"labels": set(), "countries": set()})
        city["labels"].add(row["label"])
        city["countries"].add(row["country"])
    years: dict[str, set[int]] = {}
    for row in _query(QUERIES["city_population_years"]):
        years.setdefault(_qid(row["c"]), set()).add(int(row["y"]))
    return {
        "source": "Wikidata",
        "licence": "CC0-1.0",
        "endpoint": ENDPOINT,
        "retrieved": dt.datetime.now(tz=dt.UTC).date().isoformat(),
        "queries": QUERIES,
        "competitions": {
            series: {
                "labels": sorted(labels[series]),
                "editions": dict(sorted(editions[series].items())),
            }
            for series in sorted(editions)
        },
        "cities": {
            city: {
                "labels": sorted(value["labels"]),
                "countries": sorted(value["countries"]),
                "population_years": sorted(years.get(city, set())),
            }
            for city, value in sorted(cities.items())
        },
    }


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return (json.dumps(snapshot, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def load_snapshot(root: Path) -> tuple[dict[str, Any], str] | None:
    path = root / OUTPUT_DIR / SNAPSHOT_NAME
    if not path.is_file():
        return None
    data = path.read_bytes()
    return json.loads(data), sha256_bytes(data)


# ── Construction ──────────────────────────────────────────────────────────────


def cycle(years: list[int]) -> int | None:
    """The most common gap between consecutive recent editions (the smallest on a tie)."""
    recent = sorted(set(years))[-CYCLE_WINDOW:]
    gaps = Counter(b - a for a, b in pairwise(recent))
    if not gaps:
        return None
    top = max(gaps.values())
    return min(gap for gap, n in gaps.items() if n == top)


def f1_entities(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    entities = []
    for series, record in snapshot["competitions"].items():
        if len(record["labels"]) != 1:
            continue
        name = record["labels"][0]
        if YEARISH.search(name) or len(name) > 80:
            continue
        years = sorted(int(y) for y in record["editions"])
        if len(years) < MIN_EDITION_YEARS or years[-1] < F1_ACTIVE_SINCE:
            continue
        period = cycle(years)
        if period is None or period < 1:
            continue
        future = [
            y
            for y in range(years[-1] + period, FUTURE[1] + 1, period)
            if FUTURE[0] <= y <= FUTURE[1]
        ]
        if not future:
            continue
        past = [
            int(y)
            for y, n in record["editions"].items()
            if n == 1 and F1_CONTROL_YEARS[0] <= int(y) <= F1_CONTROL_YEARS[1]
        ]
        entities.append(
            {
                "family": "F1",
                "entity_id": series,
                "name": name,
                "future_year": _pick(future, "year", series),
                "control_year": _pick(sorted(past), "control-year", series) if past else None,
            }
        )
    return entities


def f2_entities(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    names: Counter[str] = Counter()
    usable = {}
    for city, record in snapshot["cities"].items():
        if len(record["labels"]) != 1 or len(record["countries"]) != 1:
            continue
        name = f"{record['labels'][0]}, {record['countries'][0]}"
        if any(ch.isdigit() for ch in name):
            continue
        usable[city] = (name, record)
        names[name] += 1
    entities = []
    first, last = dt.date(FUTURE[0], 1, 1), dt.date(FUTURE[1], 12, 31)
    span = (last - first).days + 1
    for city, (name, record) in usable.items():
        if names[name] != 1:
            continue
        day = first + dt.timedelta(days=int(_rank("date", city), 16) % span)
        past = [
            y for y in record["population_years"] if F2_CONTROL_YEARS[0] <= y <= F2_CONTROL_YEARS[1]
        ]
        entities.append(
            {
                "family": "F2",
                "entity_id": city,
                "name": name,
                "future_date": f"{day.day} {MONTHS[day.month - 1]} {day.year}",
                "control_year": _pick(past, "control-year", city) if past else None,
            }
        )
    return entities


def question(entity: dict[str, Any], kind: str) -> tuple[str, int]:
    family = entity["family"]
    phrasing = int(_rank("phrasing", kind, entity["entity_id"]), 16) % 4
    template = PHRASINGS[(family, kind)][phrasing]
    if kind == "control":
        text = template.format(name=entity["name"], year=entity["control_year"])
    elif family == "F1":
        text = template.format(name=entity["name"], year=entity["future_year"])
    else:
        text = template.format(name=entity["name"], date=entity["future_date"])
    return text, phrasing


def _texts(root: Path) -> set[str]:
    return {
        strata.normalise(str(json.loads(line)["user_message"]))
        for path in DISJOINT_FROM
        for line in (root / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def draw(
    root: Path, snapshot: dict[str, Any], bfcl: bytes
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: dict[str, Any] = {}
    entities = {"F1": f1_entities(snapshot), "F2": f2_entities(snapshot)}
    counts["eligible"] = {family: len(group) for family, group in entities.items()}
    counts["control_eligible"] = {
        family: sum(1 for e in group if e["control_year"] is not None)
        for family, group in entities.items()
    }
    candidates: list[dict[str, Any]] = []
    for family, group in entities.items():
        for entity in group:
            for kind in ("question", "control"):
                if kind == "control" and entity["control_year"] is None:
                    continue
                text, phrasing = question(entity, kind)
                candidates.append(
                    {
                        "punans_id": item_id(family, kind, entity["entity_id"]),
                        "source_dataset": SOURCE,
                        "family": family,
                        "is_control": kind == "control",
                        "entity_id": entity["entity_id"],
                        "phrasing": phrasing,
                        "user_message": text,
                    }
                )
    others = _texts(root)
    colliding = {
        c["punans_id"] for c in candidates if strata.normalise(c["user_message"]) in others
    }
    counts["evaluation_set_collisions_removed"] = len(colliding)
    shaped = [
        {"answer_id": c["punans_id"], "pool": "U", "user_message": c["user_message"]}
        for c in candidates
        if c["punans_id"] not in colliding
    ]
    flagged, report = strata.screen(root, shaped)
    report.pop("flagged_by_pool", None)
    counts["screen"] = report
    removed = colliding | flagged
    by_id = {c["punans_id"]: c for c in candidates if c["punans_id"] not in removed}

    tools = strata.tool_pool({"bfcl_simple_python": bfcl})
    counts["tool_pool"] = {"functions": len(tools)}
    chosen: list[dict[str, Any]] = []
    shortage: dict[str, int] = {}
    for family in FAMILIES:
        ordered = sorted(entities[family], key=lambda e: _rank(family, e["entity_id"]))
        questions: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        for entity in ordered:
            if len(questions) < SIZES[family]:
                item = by_id.get(item_id(family, "question", entity["entity_id"]))
                if item is not None:
                    questions.append(item)
                continue
            if len(controls) < CONTROLS[family] and entity["control_year"] is not None:
                item = by_id.get(item_id(family, "control", entity["entity_id"]))
                if item is not None:
                    controls.append(item)
        for kind, group, size in (
            ("question", questions, SIZES[family]),
            ("control", controls, CONTROLS[family]),
        ):
            if len(group) < size:
                shortage[f"{family}:{kind}"] = size - len(group)
        chosen += questions + controls
    counts["shortage"] = shortage
    for item in chosen:
        item["tools"] = distractors(item["user_message"], tools)
    no_tool = {i["punans_id"] for i in chosen if not i["tools"]}
    counts["no_eligible_tool_removed"] = len(no_tool)
    population = sorted(
        (i for i in chosen if i["punans_id"] not in no_tool),
        key=lambda i: _rank("order", i["punans_id"]),
    )
    counts["realized"] = dict(
        sorted(
            Counter(
                f"{i['family']}:{'control' if i['is_control'] else 'question'}" for i in population
            ).items()
        )
    )
    counts["phrasings"] = dict(
        sorted(Counter(f"{i['family']}:{i['phrasing']}" for i in population).items())
    )
    return population, counts


def population_bytes(population: list[dict[str, Any]]) -> bytes:
    return v1.population_bytes(population)


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return v1.manifest_bytes(manifest)


def build_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded = load_snapshot(root)
    if loaded is None:
        raise PUnansConstructedError(
            f"no snapshot at {(OUTPUT_DIR / SNAPSHOT_NAME).as_posix()}; run --snapshot first"
        )
    snapshot, snapshot_sha = loaded
    payloads = v2.fetch_inputs(root, download=True)
    if payloads is None:
        raise PUnansConstructedError("the BFCL tool file is not cached")
    identity = strata.corpora_identity(root)
    if identity is None:
        raise PUnansConstructedError(
            f"the normalized corpora under {strata.CORPORA_ROOT.as_posix()} are required for the screen"
        )
    population, counts = draw(root, snapshot, payloads["bfcl_simple_python"])
    manifest = {
        "artifact_kind": "PUNANS_CONSTRUCTED_CANDIDATE_POPULATION",
        "schema_version": 1,
        "population_id": POPULATION_ID,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": "ADOPTED",
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "sizes": SIZES,
        "controls": CONTROLS,
        "future_years": list(FUTURE),
        "snapshot": {
            "file": SNAPSHOT_NAME,
            "sha256": snapshot_sha,
            "retrieved": snapshot["retrieved"],
            "licence": snapshot["licence"],
        },
        "tools": {
            k: v2.INPUTS["bfcl_simple_python"][k] for k in ("url", "revision", "sha256", "licence")
        },
        "screen_corpora_manifest_sha256": identity,
        "disjoint_from": [p.as_posix() for p in DISJOINT_FROM],
        "counts": counts,
        "phrasings": {f"{family}:{kind}": list(t) for (family, kind), t in PHRASINGS.items()},
        "id_rule": "unc: + sha256(family : question|control : Wikidata entity id)",
        "draw_order": "sha256(seed | family | entity id) ascending; the first 300 eligible entities give the "
        "questions, the next 75 control-eligible ones the controls",
        "presentation_order": "sha256(seed | order | punans_id) ascending",
        "code_sha256_lf": {module: lf_sha256(root / module) for module in CODE_MODULES},
        "gold_labels_present": False,
        "blinded_fields": ["source_dataset", "family", "is_control", "entity_id", "phrasing"],
        "statement": (
            "Constructed candidates for Study 002's P-UNANS (45): questions asking for a specific future outcome, "
            "with answerable controls, before labelling. The constructed stratum is the non-control items two "
            "non-Claude models both label UNKNOWABLE (45 §8)."
        ),
    }
    return population, manifest


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = v2.resolve_output_dir(root, output_dir)
    blocked = (root / v2.OUTPUT_DIR).resolve()
    if resolved == blocked or blocked in resolved.parents:
        raise PUnansConstructedError("P-UNANS-v2-constructed is never written under punans-v2/")
    return resolved


def write_population(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    if (directory / POPULATION_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise PUnansConstructedError(
            f"a population already exists in {directory}; never overwritten"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = population_bytes(population)
    stamped = {
        **manifest,
        "population_file": POPULATION_NAME,
        "population_records": len(population),
        "population_sha256": sha256_bytes(payload),
    }
    data = manifest_bytes(stamped)
    (directory / POPULATION_NAME).write_bytes(payload)
    (directory / MANIFEST_NAME).write_bytes(data)
    (directory / (MANIFEST_NAME + ".sha256")).write_bytes((sha256_bytes(data) + "\n").encode())
    return stamped


REPRODUCED_FIELDS = (
    "counts",
    "seed",
    "protocol_version",
    "sizes",
    "controls",
    "snapshot",
    "screen_corpora_manifest_sha256",
)


def verify_population(root: Path, output_dir: Path) -> dict[str, Any]:
    directory = output_dir if output_dir.is_absolute() else root / output_dir
    population_path, manifest_path = directory / POPULATION_NAME, directory / MANIFEST_NAME
    if not population_path.is_file() or not manifest_path.is_file():
        return {"status": FAIL, "errors": [f"{POPULATION_ID} artifacts are missing"], "blocked": []}
    populated, manifest_text = population_path.read_bytes(), manifest_path.read_bytes()
    manifest = json.loads(manifest_text)
    items = [json.loads(line) for line in populated.decode("utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if manifest.get("population_sha256") != sha256_bytes(populated):
        errors.append("FAIL_HASH: population bytes do not match manifest population_sha256")
    sidecar = directory / (MANIFEST_NAME + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_bytes(
        manifest_text
    ):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")
    loaded = load_snapshot(root)
    if loaded is None or loaded[1] != manifest["snapshot"]["sha256"]:
        errors.append("FAIL_HASH: the Wikidata snapshot is missing or does not match the manifest")
    if len(items) != manifest.get("population_records"):
        errors.append("FAIL_COUNT: population_records does not match the file")
    if any(item.get("gold_policy_label") not in (None, "") for item in items):
        errors.append("FAIL_ANNOTATION: an item carries a label in the population file")
    blocked: list[str] = []
    if v2.fetch_inputs(root, download=False) is None or strata.corpora_identity(root) is None:
        blocked.append(
            f"{BLOCKED_INPUT_MISSING}: cached inputs or normalized corpora absent; not re-derived"
        )
    elif loaded is not None:
        rebuilt, fresh = build_population(root)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.snapshot:
        path = root / OUTPUT_DIR / SNAPSHOT_NAME
        if path.exists():
            raise PUnansConstructedError(f"{path} exists; a snapshot is never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = snapshot_bytes(fetch_snapshot())
        path.write_bytes(data)
        print(json.dumps({"snapshot_sha256": sha256_bytes(data)}))
        return 0
    if args.verify:
        result = verify_population(root, args.output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in (PASS, BLOCKED_INPUT_MISSING) else 1
    population, manifest = build_population(root)
    if args.build:
        record = write_population(root, args.output_dir, population, manifest)
    else:
        directory = resolve_output_dir(root, args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            **manifest,
            "artifact_kind": "PUNANS_CONSTRUCTED_DRY_RUN",
            "population_records": len(population),
            "population_sha256": sha256_bytes(population_bytes(population)),
        }
        (directory / DRY_RUN_NAME).write_bytes(manifest_bytes(record))
    print(
        json.dumps(
            {k: record[k] for k in ("population_records", "population_sha256", "counts")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
