"""P-UNANS-v1 candidates: Study 002's unanswerable population (``study_002_prereg_v11``, doc 43).

Two pools, each item paired with tools that cannot serve it (43 §6):

* **pool U (unknowable):** KUQ's ``future unknown`` and ``unsolved problem`` unknowns, and SelfAware's
  unanswerable questions, 350 from each (43 §5);
* **pool F (false premise):** every KUQ ``false assumption`` unknown that survives.

Before the draw, a question naming a year from 1900 to the screening year is dropped (the time rule, 43 §5).
Duplicates are removed, and the 41 §6 screen runs against every training corpus and the When2Call
evaluation splits. Exact normalised-text collisions with the ANSWER strata candidates (and so ``P-CONF-v1``'s
ANSWER items) and with P-DET-COVERAGE are also removed (43 §7). The labels come later (43 §8-§9); this module
writes candidates only, before any label exists.

Counts and hashes only are printed: never an item's text or id.

    python -m opengrad.verification.punans --dry-run | --build | --verify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.normalization_v3 import lf_sha256
from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata as strata
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

POPULATION_ID = "P-UNANS-v1"
PROTOCOL_VERSION = "punans-001-v1"
SEED = "opengrad-punans-001-v1"
PREREGISTRATION = Path("docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md")
ADOPTION_AMENDMENT = "study_002_prereg_v11"
OUTPUT_DIR = Path("reports/study-002/punans-v1")
POPULATION_NAME = "punans-v1.population.jsonl"
MANIFEST_NAME = "punans-v1.manifest.json"
DRY_RUN_NAME = "punans-v1.dry-run.json"
ID_PREFIX = "un1:"
CACHE = Path(".cache/punans")
PER_SOURCE_U = 350
SCREENING_YEAR = 2026
YEAR = re.compile(r"\b(?:19|20)\d\d\b")
KUQ_U = ("future unknown", "unsolved problem")
KUQ_F = ("false assumption",)
INPUTS: dict[str, dict[str, str]] = {
    "kuq": {
        "url": "https://huggingface.co/datasets/amayuelas/KUQ/resolve/"
        "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41/knowns_unknowns.jsonl",
        "revision": "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41",
        "sha256": "798d1677f962d11d069f77d1e3db91ad2ddb483a94b697e46bc8ca62ad0aedf6",
        "licence": "MIT",
    },
    "selfaware": {
        "url": "https://raw.githubusercontent.com/yinzhangyue/SelfAware/"
        "f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c/data/SelfAware.json",
        "revision": "f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c",
        "sha256": "32929585ffdd4048f35f7f167720722eb8ddcf59908ffc958da2f84ea634dc54",
        "licence": "CC-BY-SA-4.0",
    },
    "bfcl_simple_python": {
        "url": strata.INPUTS["bfcl_simple_python"]["url"],
        "revision": strata.GORILLA,
        "sha256": strata.INPUTS["bfcl_simple_python"]["sha256"],
        "licence": "Apache-2.0",
    },
}
#: Tracked populations P-UNANS must not share a question with (43 §7). P-CONF-v1's ANSWER items are the
#: ANSWER strata members, drawn from the first file.
DISJOINT_FROM = (
    Path("reports/study-002/answer-strata-v1/answer-strata-v1.population.jsonl"),
    Path("reports/pdet-coverage/pdet-coverage-v1.population.jsonl"),
    Path("reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl"),
)
CODE_MODULES = (
    "src/opengrad/verification/punans.py",
    "src/opengrad/verification/answer_strata.py",
    "src/opengrad/contamination/levels.py",
)


class PUnansError(ValueError):
    pass


def _rank(*parts: str) -> str:
    return sha256_bytes("|".join((SEED, *parts)).encode("utf-8"))


def item_id(pool: str, source: str, source_id: str) -> str:
    return ID_PREFIX + sha256_bytes(f"{pool}:{source}:{source_id}".encode())


def names_past_year(text: str) -> bool:
    """43 §5: the question names a year from 1900 to the screening year, so its event may have happened."""
    return any(int(match.group(0)) <= SCREENING_YEAR for match in YEAR.finditer(text))


def distractors(question: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1-3 tools that cannot serve the question: the rule of 41 §5 under this document's seed (43 §6)."""
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


# ── Inputs ────────────────────────────────────────────────────────────────────


def fetch_inputs(root: Path, *, download: bool) -> dict[str, bytes] | None:
    """Every pinned input's bytes, from the cache or (when allowed) downloaded; None if one is missing."""
    payloads: dict[str, bytes] = {}
    for key, spec in INPUTS.items():
        path = root / CACHE / key
        if not path.is_file():
            if not download:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(spec["url"], timeout=300) as response:
                path.write_bytes(response.read())
        payload = path.read_bytes()
        if sha256_bytes(payload) != spec["sha256"]:
            raise PUnansError(f"{key}: the file does not match its pinned sha256")
        payloads[key] = payload
    return payloads


def candidates(payloads: dict[str, bytes], counts: dict[str, Any]) -> list[dict[str, Any]]:
    """Every pool U and pool F question, before the time rule, the de-duplication and the screen."""
    items: list[dict[str, Any]] = []
    kuq = [
        json.loads(line) for line in payloads["kuq"].decode("utf-8").splitlines() if line.strip()
    ]
    for index, row in enumerate(kuq):
        category = str(row.get("category"))
        if not row.get("unknown") or category not in KUQ_U + KUQ_F:
            continue
        pool = "U" if category in KUQ_U else "F"
        items.append(_item(pool, "kuq", f"line{index}", category, str(row["question"])))
    for row in json.loads(payloads["selfaware"])["example"]:
        if row.get("answerable") is False:
            items.append(
                _item(
                    "U", "selfaware", str(row["question_id"]), "unanswerable", str(row["question"])
                )
            )
    counts["candidates"] = dict(
        sorted(Counter(f"{i['pool']}:{i['source_dataset']}" for i in items).items())
    )
    return items


def _item(pool: str, source: str, source_id: str, category: str, question: str) -> dict[str, Any]:
    return {
        "punans_id": item_id(pool, source, source_id),
        "pool": pool,
        "source_dataset": source,
        "source_id": source_id,
        "source_category": category,
        "user_message": question,
    }


def _disjoint_texts(root: Path) -> set[str]:
    return {
        strata.normalise(str(json.loads(line)["user_message"]))
        for path in DISJOINT_FROM
        for line in (root / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


# ── Draw ──────────────────────────────────────────────────────────────────────


def draw(root: Path, payloads: dict[str, bytes]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: dict[str, Any] = {}
    items = candidates(payloads, counts)

    dated = [i for i in items if names_past_year(i["user_message"])]
    counts["time_rule_removed"] = dict(
        sorted(Counter(f"{i['pool']}:{i['source_dataset']}" for i in dated).items())
    )
    dated_ids = {i["punans_id"] for i in dated}
    items = [i for i in items if i["punans_id"] not in dated_ids]

    # Duplicates: pool F first, then pool U; within a pool, ascending id.
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed: Counter[str] = Counter()
    for pool in ("F", "U"):
        for item in sorted((i for i in items if i["pool"] == pool), key=lambda i: i["punans_id"]):
            key = strata.normalise(item["user_message"])
            if key in seen:
                removed[pool] += 1
                continue
            seen.add(key)
            kept.append(item)
    counts["duplicates_removed"] = {pool: removed.get(pool, 0) for pool in ("U", "F")}

    others = _disjoint_texts(root)
    colliding = {i["punans_id"] for i in kept if strata.normalise(i["user_message"]) in others}
    counts["evaluation_set_collisions_removed"] = len(colliding)
    kept = [i for i in kept if i["punans_id"] not in colliding]

    shaped = [
        {"answer_id": i["punans_id"], "pool": i["pool"], "user_message": i["user_message"]}
        for i in kept
    ]
    flagged, report = strata.screen(root, shaped)
    report.pop("flagged_by_pool", None)
    report["flagged_by_pool"] = {
        pool: sum(1 for i in kept if i["pool"] == pool and i["punans_id"] in flagged)
        for pool in ("U", "F")
    }
    counts["screen"] = report
    survivors = [i for i in kept if i["punans_id"] not in flagged]
    counts["available_after_screen"] = dict(
        sorted(Counter(f"{i['pool']}:{i['source_dataset']}" for i in survivors).items())
    )

    chosen: list[dict[str, Any]] = []
    for source in ("kuq", "selfaware"):
        pool_u = sorted(
            (i for i in survivors if i["pool"] == "U" and i["source_dataset"] == source),
            key=lambda i: _rank(source, i["user_message"]),
        )
        if len(pool_u) < PER_SOURCE_U:
            counts.setdefault("pool_u_shortage", {})[source] = PER_SOURCE_U - len(pool_u)
        chosen += pool_u[:PER_SOURCE_U]
    chosen += [i for i in survivors if i["pool"] == "F"]

    tools = strata.tool_pool({"bfcl_simple_python": payloads["bfcl_simple_python"]})
    counts["tool_pool"] = {"functions": len(tools)}
    no_tool: set[str] = set()
    for item in chosen:
        item["tools"] = distractors(item["user_message"], tools)
        if not item["tools"]:
            no_tool.add(item["punans_id"])
    counts["no_eligible_tool_removed"] = len(no_tool)
    population = sorted(
        (i for i in chosen if i["punans_id"] not in no_tool),
        key=lambda i: _rank("order", i["punans_id"]),
    )
    counts["realized"] = {
        "U": sum(1 for i in population if i["pool"] == "U"),
        "F": sum(1 for i in population if i["pool"] == "F"),
        "total": len(population),
    }
    counts["realized_by_source"] = dict(
        sorted(Counter(f"{i['pool']}:{i['source_dataset']}" for i in population).items())
    )
    counts["tools_per_item"] = dict(
        sorted(Counter(str(len(i["tools"])) for i in population).items())
    )
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
        raise PUnansError(
            "inputs are not cached; run --build or --dry-run once with network access"
        )
    identity = strata.corpora_identity(root)
    if identity is None:
        raise PUnansError(
            f"the normalized corpora under {strata.CORPORA_ROOT.as_posix()} are required for the screen"
        )
    population, counts = draw(root, payloads)
    manifest = {
        "artifact_kind": "PUNANS_CANDIDATE_POPULATION",
        "schema_version": 1,
        "population_id": POPULATION_ID,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": "ADOPTED",
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "screening_year": SCREENING_YEAR,
        "per_source_pool_u": PER_SOURCE_U,
        "inputs": {
            key: {k: spec[k] for k in ("url", "revision", "sha256", "licence")}
            for key, spec in INPUTS.items()
        },
        "screen_corpora_manifest_sha256": identity,
        "disjoint_from": [p.as_posix() for p in DISJOINT_FROM],
        "counts": counts,
        "id_rule": "un1: + sha256(pool : source_dataset : source_id); KUQ source_id is the line index at the pinned revision",
        "presentation_order": "sha256(seed | order | punans_id) ascending",
        "code_sha256_lf": {module: lf_sha256(root / module) for module in CODE_MODULES},
        "gold_labels_present": False,
        "blinded_fields": ["pool", "source_dataset", "source_id", "source_category"],
        "licences": {
            "kuq": "MIT; the MIT notice travels with the items",
            "selfaware": "CC-BY-SA-4.0; share-alike applies to items built from it",
            "tools": "Apache-2.0 (BFCL)",
        },
        "statement": (
            "Candidate items for Study 002's P-UNANS, before labelling. Pool U holds questions a source marks "
            "unknowable (future events, unsolved problems, unanswerable questions); pool F holds false-premise "
            "questions. The strata are the items two non-Claude models both label UNKNOWABLE or FALSE_PREMISE "
            "(43 §9)."
        ),
    }
    return population, manifest


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    reports = (root / "reports").resolve()
    for forbidden in (
        "pdet",
        "pdet-coverage",
        "pdet-coverage-v2",
        "evaluation",
        "study-002/answer-strata-v1",
        "study-002/pconf-v1",
    ):
        blocked = reports / forbidden
        if resolved == blocked or blocked in resolved.parents:
            raise PUnansError(f"{POPULATION_ID} is never written under reports/{forbidden}/")
    return resolved


def write_population(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    if (directory / POPULATION_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise PUnansError(f"a population already exists in {directory}; it is never overwritten")
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


def write_dry_run(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **manifest,
        "artifact_kind": "PUNANS_DRY_RUN",
        "population_written": False,
        "population_records": len(population),
        "population_sha256": sha256_bytes(population_bytes(population)),
    }
    (directory / DRY_RUN_NAME).write_bytes(manifest_bytes(record))
    return record


REPRODUCED_FIELDS = (
    "counts",
    "inputs",
    "seed",
    "protocol_version",
    "per_source_pool_u",
    "screening_year",
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
    if len(items) != manifest.get("population_records"):
        errors.append("FAIL_COUNT: population_records does not match the file")
    if any(item.get("gold_policy_label") not in (None, "") for item in items):
        errors.append("FAIL_ANNOTATION: an item carries a label in the population file")
    blocked: list[str] = []
    if fetch_inputs(root, download=False) is None or strata.corpora_identity(root) is None:
        blocked.append(
            f"{BLOCKED_INPUT_MISSING}: cached inputs or normalized corpora absent; not re-derived"
        )
    else:
        rebuilt, fresh = build_population(root, download=False)
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
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.verify:
        result = verify_population(root, args.output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in (PASS, BLOCKED_INPUT_MISSING) else 1
    population, manifest = build_population(root, download=True)
    record = (write_population if args.build else write_dry_run)(
        root, args.output_dir, population, manifest
    )
    print(json.dumps(summary(record), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
