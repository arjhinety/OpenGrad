"""P-UNANS-v2: the second attempt at Study 002's unanswerable population (``study_002_prereg_v12``, doc 44).

One pool of questions no one can answer now (44 §4-§5), each paired with tools that cannot serve it (44 §6):

* KUQ ``unknowns_all.jsonl``, categories ``future unknown`` and ``unsolved problem/mistery``;
* KUQP's future questions (the unanswerable side of each pair);
* BIG-bench Known Unknowns items whose highest-scored target is ``Unknown``.

Before the draw, a question naming a year from 1900 to the screening year is dropped (the time rule), duplicates
are removed (first occurrence in source and file order), and so is every question among P-UNANS-v1's candidates.
The screen of 44 §7 then runs, tools are assigned, and one seeded order is applied: the first 100 form the trial
set ``P-UNANS-v2-TRIAL`` (44 §9), the next 800 the main set ``P-UNANS-v2``. Both are written in the same build,
before any label, so nothing the trial shows can change which questions the main set holds.

Counts and hashes only are printed: never an item's text or id.

    python -m opengrad.verification.punans_v2 --dry-run | --build | --verify
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.normalization_v3 import lf_sha256
from opengrad.hashing import sha256_bytes
from opengrad.verification import answer_strata as strata
from opengrad.verification import punans as v1
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

TRIAL_ID = "P-UNANS-v2-TRIAL"
MAIN_ID = "P-UNANS-v2"
PROTOCOL_VERSION = "punans-002-v2"
SEED = "opengrad-punans-002-v2"
PREREGISTRATION = Path("docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md")
ADOPTION_AMENDMENT = "study_002_prereg_v12"
OUTPUT_DIR = Path("reports/study-002/punans-v2")
TRIAL_NAME = "punans-v2-trial.population.jsonl"
MAIN_NAME = "punans-v2.population.jsonl"
MANIFEST_NAME = "punans-v2.manifest.json"
DRY_RUN_NAME = "punans-v2.dry-run.json"
ID_PREFIX = "un2:"
CACHE = Path(".cache/punans-v2")
TRIAL_SIZE = 100
MAIN_SIZE = 800
SCREENING_YEAR = v1.SCREENING_YEAR
KUQ_CATEGORIES = ("future unknown", "unsolved problem/mistery")
SOURCES = ("kuq", "kuqp", "bigbench-known-unknowns")
INPUTS: dict[str, dict[str, str]] = {
    "kuq": {
        "url": "https://huggingface.co/datasets/amayuelas/KUQ/resolve/"
        "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41/unknowns_all.jsonl",
        "revision": "f99b53aa226dbb0d1b086db3ec352b0da0aa8f41",
        "sha256": "8469ab010ce1142bfe3d330635fb58c1e537253568b417b4435a39d658b10855",
        "licence": "MIT",
    },
    "kuqp": {
        "url": "https://raw.githubusercontent.com/zhaoy777/kuqp-dataset/"
        "596472f31f73acfdcb95741c277413fd500f8b35/KUQP%20Dataset/future_questions.json",
        "revision": "596472f31f73acfdcb95741c277413fd500f8b35",
        "sha256": "25ee426e7881b3cf950bffb1b6d0ebc6c644eb0c1a2c5939662cce4a01ce40d1",
        "licence": "MIT",
    },
    "bigbench-known-unknowns": {
        "url": "https://raw.githubusercontent.com/google/BIG-bench/"
        "124892ccf54f85402852d68c93736a4fa57bf009/bigbench/benchmark_tasks/known_unknowns/task.json",
        "revision": "124892ccf54f85402852d68c93736a4fa57bf009",
        "sha256": "6061bdd796f2225fee6ef9c389e1b1e19f4d83da905dc7b512e0d1ca0e44c557",
        "licence": "Apache-2.0",
    },
    "bfcl_simple_python": v1.INPUTS["bfcl_simple_python"],
}
#: Tracked populations P-UNANS-v2 must not share a question with: P-UNANS-v1's candidates (44 §5) and the
#: sets of 43 §7 (44 §7).
FIRST_ATTEMPT = Path("reports/study-002/punans-v1/punans-v1.population.jsonl")
DISJOINT_FROM = (FIRST_ATTEMPT, *v1.DISJOINT_FROM)
CODE_MODULES = (
    "src/opengrad/verification/punans_v2.py",
    "src/opengrad/verification/punans.py",
    "src/opengrad/verification/answer_strata.py",
    "src/opengrad/contamination/levels.py",
)


class PUnansV2Error(ValueError):
    pass


def _rank(*parts: str) -> str:
    return sha256_bytes("|".join((SEED, *parts)).encode("utf-8"))


def item_id(source: str, source_id: str) -> str:
    return ID_PREFIX + sha256_bytes(f"{source}:{source_id}".encode())


def distractors(question: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1-3 tools that cannot serve the question: the rule of 41 §5 and 43 §6 under this document's seed."""
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
            raise PUnansV2Error(f"{key}: the file does not match its pinned sha256")
        payloads[key] = payload
    return payloads


def _item(source: str, source_id: str, category: str, author: str, question: str) -> dict[str, Any]:
    return {
        "punans_id": item_id(source, source_id),
        "source_dataset": source,
        "source_id": source_id,
        "source_category": category,
        "source_author": author,
        "user_message": question,
    }


def candidates(payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    """Every pool question, in source then file order, before any exclusion (44 §5)."""
    items: list[dict[str, Any]] = []
    for index, line in enumerate(payloads["kuq"].decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["category"] in KUQ_CATEGORIES:
            items.append(
                _item(
                    "kuq",
                    f"line{index}",
                    str(row["category"]),
                    str(row["source"]),
                    str(row["question"]),
                )
            )
    kuqp = json.loads(payloads["kuqp"])
    rows = kuqp if isinstance(kuqp, list) else next(iter(kuqp.values()))
    for index, row in enumerate(rows):
        items.append(_item("kuqp", f"item{index}", "future", "gpt", str(row["u"])))
    for index, example in enumerate(json.loads(payloads["bigbench-known-unknowns"])["examples"]):
        scores = example["target_scores"]
        if max(scores, key=scores.get) == "Unknown":
            items.append(
                _item(
                    "bigbench-known-unknowns",
                    f"example{index}",
                    "unknown",
                    "task authors",
                    str(example["input"]),
                )
            )
    return items


def _texts(root: Path, paths: tuple[Path, ...]) -> set[str]:
    return {
        strata.normalise(str(json.loads(line)["user_message"]))
        for path in paths
        for line in (root / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _by_source(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(i["source_dataset"] for i in items).items()))


# ── Draw ──────────────────────────────────────────────────────────────────────


def draw(
    root: Path, payloads: dict[str, bytes]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(trial, main, counts): 44 §5-§7."""
    counts: dict[str, Any] = {}
    items = candidates(payloads)
    counts["candidates"] = _by_source(items)

    dated = [i for i in items if v1.names_past_year(i["user_message"])]
    counts["time_rule_removed"] = _by_source(dated)
    dated_ids = {i["punans_id"] for i in dated}
    items = [i for i in items if i["punans_id"] not in dated_ids]

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = strata.normalise(item["user_message"])
        if key not in seen:
            seen.add(key)
            kept.append(item)
    counts["duplicates_removed"] = len(items) - len(kept)

    first = _texts(root, (FIRST_ATTEMPT,))
    in_first = [i for i in kept if strata.normalise(i["user_message"]) in first]
    counts["first_attempt_removed"] = _by_source(in_first)
    kept = [i for i in kept if strata.normalise(i["user_message"]) not in first]

    others = _texts(root, v1.DISJOINT_FROM)
    colliding = {i["punans_id"] for i in kept if strata.normalise(i["user_message"]) in others}
    counts["evaluation_set_collisions_removed"] = len(colliding)
    kept = [i for i in kept if i["punans_id"] not in colliding]

    shaped = [
        {"answer_id": i["punans_id"], "pool": "U", "user_message": i["user_message"]} for i in kept
    ]
    flagged, report = strata.screen(root, shaped)
    report.pop("flagged_by_pool", None)
    report["flagged_by_source"] = _by_source([i for i in kept if i["punans_id"] in flagged])
    counts["screen"] = report
    survivors = [i for i in kept if i["punans_id"] not in flagged]

    tools = strata.tool_pool({"bfcl_simple_python": payloads["bfcl_simple_python"]})
    counts["tool_pool"] = {"functions": len(tools)}
    for item in survivors:
        item["tools"] = distractors(item["user_message"], tools)
    no_tool = [i for i in survivors if not i["tools"]]
    counts["no_eligible_tool_removed"] = len(no_tool)
    survivors = [i for i in survivors if i["tools"]]
    counts["available"] = _by_source(survivors)
    if len(survivors) < TRIAL_SIZE + MAIN_SIZE:
        raise PUnansV2Error(
            f"{len(survivors)} questions survive, fewer than the {TRIAL_SIZE + MAIN_SIZE} 44 §5 draws"
        )

    ordered = sorted(survivors, key=lambda i: _rank(i["source_dataset"], i["user_message"]))
    trial = ordered[:TRIAL_SIZE]
    main = ordered[TRIAL_SIZE : TRIAL_SIZE + MAIN_SIZE]
    counts["undrawn"] = len(ordered) - len(trial) - len(main)
    for name, group in (("trial", trial), ("main", main)):
        counts[name] = {
            "total": len(group),
            "by_source": _by_source(group),
            "by_author": dict(sorted(Counter(i["source_author"] for i in group).items())),
            "tools_per_item": dict(sorted(Counter(str(len(i["tools"])) for i in group).items())),
        }

    def presented(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(group, key=lambda i: _rank("order", i["punans_id"]))

    return presented(trial), presented(main), counts


def population_bytes(population: list[dict[str, Any]]) -> bytes:
    return v1.population_bytes(population)


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return v1.manifest_bytes(manifest)


def build_populations(
    root: Path, *, download: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payloads = fetch_inputs(root, download=download)
    if payloads is None:
        raise PUnansV2Error(
            "inputs are not cached; run --build or --dry-run once with network access"
        )
    identity = strata.corpora_identity(root)
    if identity is None:
        raise PUnansV2Error(
            f"the normalized corpora under {strata.CORPORA_ROOT.as_posix()} are required for the screen"
        )
    trial, main, counts = draw(root, payloads)
    manifest = {
        "artifact_kind": "PUNANS_V2_CANDIDATE_POPULATIONS",
        "schema_version": 1,
        "population_ids": {"trial": TRIAL_ID, "main": MAIN_ID},
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": "ADOPTED",
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "screening_year": SCREENING_YEAR,
        "sizes": {"trial": TRIAL_SIZE, "main": MAIN_SIZE},
        "inputs": {
            key: {k: spec[k] for k in ("url", "revision", "sha256", "licence")}
            for key, spec in INPUTS.items()
        },
        "screen_corpora_manifest_sha256": identity,
        "disjoint_from": [p.as_posix() for p in DISJOINT_FROM],
        "counts": counts,
        "id_rule": "un2: + sha256(source_dataset : source_id); source_id is the line, item or example index at the pinned revision",
        "draw_order": "sha256(seed | source_dataset | user_message) ascending; the first 100 are the trial, the next 800 the main set",
        "presentation_order": "sha256(seed | order | punans_id) ascending, within each set",
        "code_sha256_lf": {module: lf_sha256(root / module) for module in CODE_MODULES},
        "gold_labels_present": False,
        "blinded_fields": ["source_dataset", "source_id", "source_category", "source_author"],
        "licences": {
            "kuq": "MIT; the MIT notice travels with the items",
            "kuqp": "MIT; the MIT notice travels with the items",
            "bigbench-known-unknowns": "Apache-2.0",
            "tools": "Apache-2.0 (BFCL)",
        },
        "statement": (
            "Candidate items for Study 002's second P-UNANS attempt, before labelling: questions a source marks "
            "as unknowable now (future events, unsolved problems). The trial set is labelled first and never "
            "enters P-UNANS (44 §9); P-UNANS-unknowable is the main-set items two non-Claude models both label "
            "UNKNOWABLE (44 §10)."
        ),
    }
    return trial, main, manifest


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    resolved = v1.resolve_output_dir(root, output_dir)
    blocked = (root / "reports/study-002/punans-v1").resolve()
    if resolved == blocked or blocked in resolved.parents:
        raise PUnansV2Error("P-UNANS-v2 is never written under reports/study-002/punans-v1/")
    return resolved


def _stamp(
    manifest: dict[str, Any], trial: list[dict[str, Any]], main: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        **manifest,
        "populations": {
            "trial": {
                "file": TRIAL_NAME,
                "records": len(trial),
                "sha256": sha256_bytes(population_bytes(trial)),
            },
            "main": {
                "file": MAIN_NAME,
                "records": len(main),
                "sha256": sha256_bytes(population_bytes(main)),
            },
        },
    }


def write_populations(
    root: Path,
    output_dir: Path,
    trial: list[dict[str, Any]],
    main: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    if any((directory / name).exists() for name in (TRIAL_NAME, MAIN_NAME, MANIFEST_NAME)):
        raise PUnansV2Error(f"a population already exists in {directory}; it is never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    stamped = _stamp(manifest, trial, main)
    data = manifest_bytes(stamped)
    (directory / TRIAL_NAME).write_bytes(population_bytes(trial))
    (directory / MAIN_NAME).write_bytes(population_bytes(main))
    (directory / MANIFEST_NAME).write_bytes(data)
    (directory / (MANIFEST_NAME + ".sha256")).write_bytes((sha256_bytes(data) + "\n").encode())
    return stamped


def write_dry_run(
    root: Path,
    output_dir: Path,
    trial: list[dict[str, Any]],
    main: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    directory = resolve_output_dir(root, output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **_stamp(manifest, trial, main),
        "artifact_kind": "PUNANS_V2_DRY_RUN",
        "population_written": False,
    }
    (directory / DRY_RUN_NAME).write_bytes(manifest_bytes(record))
    return record


REPRODUCED_FIELDS = (
    "counts",
    "inputs",
    "seed",
    "protocol_version",
    "sizes",
    "screening_year",
    "screen_corpora_manifest_sha256",
    "populations",
)


def verify_populations(root: Path, output_dir: Path) -> dict[str, Any]:
    directory = output_dir if output_dir.is_absolute() else root / output_dir
    paths = [directory / name for name in (TRIAL_NAME, MAIN_NAME, MANIFEST_NAME)]
    if not all(path.is_file() for path in paths):
        return {"status": FAIL, "errors": ["P-UNANS-v2 artifacts are missing"], "blocked": []}
    manifest_text = (directory / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_text)
    errors: list[str] = []
    sidecar = directory / (MANIFEST_NAME + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_bytes(
        manifest_text
    ):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")
    files: dict[str, bytes] = {}
    for name, spec in manifest["populations"].items():
        data = (directory / spec["file"]).read_bytes()
        files[name] = data
        if sha256_bytes(data) != spec["sha256"]:
            errors.append(f"FAIL_HASH: the {name} population does not match the manifest")
        items = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(items) != spec["records"]:
            errors.append(f"FAIL_COUNT: the {name} record count does not match the file")
        if any(item.get("gold_policy_label") not in (None, "") for item in items):
            errors.append(f"FAIL_ANNOTATION: a {name} item carries a label in the population file")
    trial_ids = {json.loads(line)["punans_id"] for line in files["trial"].splitlines() if line}
    main_ids = {json.loads(line)["punans_id"] for line in files["main"].splitlines() if line}
    if trial_ids & main_ids:
        errors.append("FAIL_DISJOINT: an item is in both the trial and the main set")
    blocked: list[str] = []
    if fetch_inputs(root, download=False) is None or strata.corpora_identity(root) is None:
        blocked.append(
            f"{BLOCKED_INPUT_MISSING}: cached inputs or normalized corpora absent; not re-derived"
        )
    else:
        trial, main, fresh = build_populations(root, download=False)
        rebuilt = json.loads(manifest_bytes(_stamp(fresh, trial, main)))
        if population_bytes(trial) != files["trial"] or population_bytes(main) != files["main"]:
            errors.append("FAIL_REPRODUCIBILITY: a fresh draw did not reproduce the populations")
        errors += [
            f"FAIL_PROVENANCE: manifest field {key!r} differs from a fresh draw"
            for key in REPRODUCED_FIELDS
            if manifest.get(key) != rebuilt.get(key)
        ]
    status = FAIL if errors else BLOCKED_INPUT_MISSING if blocked else PASS
    return {
        "status": status,
        "populations": {
            name: {k: spec[k] for k in ("records", "sha256")}
            for name, spec in manifest["populations"].items()
        },
        "errors": errors,
        "blocked": blocked,
    }


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """Counts and hashes only."""
    return {
        "populations": {
            name: {k: spec[k] for k in ("records", "sha256")}
            for name, spec in record["populations"].items()
        },
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
        result = verify_populations(root, args.output_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in (PASS, BLOCKED_INPUT_MISSING) else 1
    trial, main_set, manifest = build_populations(root, download=True)
    record = (write_populations if args.build else write_dry_run)(
        root, args.output_dir, trial, main_set, manifest
    )
    print(json.dumps(summary(record), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
