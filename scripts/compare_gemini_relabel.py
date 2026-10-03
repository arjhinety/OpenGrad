"""INC-0002: compare Gemini's tool-gated re-label of P-DET-COVERAGE with its original labels.

Implements ``reports/incidents/INC-0002-relabel-comparison-plan.md``, written before any comparison was computed.
For P-DET-COVERAGE-v1 (layer B), its routing layer and P-DET-COVERAGE-v2 it reports, per exposure group of the old
Gemini batch (A: read another model's answers; B: read the repository; D: browsed only outside it):

1. the change rate of Gemini's label (and, separately, of its ambiguity status), with Wilson 95% intervals;
2. the copying check, among items where gpt-5.6-sol and deepseek-v4.1-flash agree: the share where old Gemini
   matched them and new Gemini does not ("lost") and the reverse ("gained"). Copying shows as lost agreement;
3. A minus D and B minus D for the change rate and the lost-agreement share, with Newcombe 95% intervals;
4. reference changes between the frozen and the re-labelled (``gemini-r2``) references, by group;
5. the saved predictions of classifier v1's test (33 §8) and v2's test (37 §8) re-scored against the re-labelled
   references with the tests' own scoring code. The classifiers are not run: each saved prediction is looked up
   by its item's features hash. Re-scoring against the frozen references must reproduce the frozen evaluations
   exactly, or nothing is reported.

Groups come from agy's local conversation store (tool names and the location class of path arguments only). The
output is counts, rates and verdicts: no item text, item id or per-item label.

    python scripts/compare_gemini_relabel.py            # prints the comparison
    python scripts/compare_gemini_relabel.py --write    # also writes reports/incidents/INC-0002-relabel-comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import archive_external_model_labels as archive

from opengrad.hashing import sha256_bytes
from opengrad.verification import pdet_coverage_metrics as metrics
from opengrad.verification import prose_classifier_oneshot as v1
from opengrad.verification import prose_classifier_v2_oneshot as v2
from opengrad.verification.pdet_coverage_reference import ANNOTATORS, load_package

GEMINI, GPT, DEEPSEEK = ANNOTATORS
OLD_SESSION, NEW_SESSION, VARIANT = "model-gemini", "model-gemini-r2", "gemini-r2"
OUTPUT = Path("reports/incidents/INC-0002-relabel-comparison.json")
PLAN = Path("reports/incidents/INC-0002-relabel-comparison-plan.md")
REFERENCE_DIRS = {
    "pdet-coverage-v1": Path("reports/pdet-coverage/reference"),
    "pdet-coverage-v1-routing": Path("reports/pdet-coverage/reference"),
    "pdet-coverage-v2": Path("reports/pdet-coverage-v2/reference"),
}
GROUPS = ("A", "B", "D")
WEB_TOOLS = ("search_web", "read_url_content", "browser_subagent")
Z = 1.959963984540054


class ComparisonError(RuntimeError):
    pass


# -- groups ------------------------------------------------------------------------------------------------------


def exposure_group(rows: list[tuple[int, bytes | None]]) -> tuple[str, bool]:
    """(group, searched the web) of one old Gemini conversation, from tool names and path classes only."""
    answers = repo = outside = web = False
    for step_type, payload in rows:
        if step_type != archive.AGY_MODEL_STEP or not payload:
            continue
        for call in archive._field(payload, (20, 7)):
            name = (archive._field(call, (2,)) or [b"?"])[0].decode("utf-8", "replace")
            if name in WEB_TOOLS:
                web = True
                continue
            if name == "call_mcp_tool":
                outside = True
            try:
                args = json.loads((archive._field(call, (3,)) or [b"{}"])[0])
            except ValueError:
                outside = True
                continue
            for key, value in args.items() if isinstance(args, dict) else ():
                if not (
                    isinstance(value, str)
                    and re.search(r"path|dir|file|search|uri", key, re.IGNORECASE)
                ):
                    continue
                value = value.lower().replace("\\", "/")
                if "opengrad" not in value and "/appdata/local/temp/" in value:
                    continue  # the attempt's own isolated input
                outside = True
                if "opengrad" in value:
                    repo = True
                    if (
                        "/.annotation" in value
                        and "answers" in value
                        and any(session in value for session in ("model-gpt", "model-deepseek"))
                    ):
                        answers = True
    group = "A" if answers else "B" if repo else "D" if outside else "clean"
    return group, web


def item_groups(task: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Each item's group, from the old Gemini batch that recorded its label; and the per-batch tally."""
    directory = ROOT / ".annotation" / f"{task}.model-batches" / OLD_SESSION
    groups: dict[str, str] = {}
    tally: Counter[str] = Counter()
    web_batches: Counter[str] = Counter()
    for path in sorted(directory.glob("batch-*.attempt-*.run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run.get("ingested"):
            continue
        start = float(run["started_at"])
        end = start + min(60.0, float(run.get("seconds") or 0))
        files = [
            f for f in archive.AGY_CONVERSATIONS.glob("*.db") if start <= f.stat().st_ctime <= end
        ]
        if len(files) != 1:
            group, web = "unmatched", False
        else:
            connection = sqlite3.connect(f"file:{files[0]}?mode=ro", uri=True)
            try:
                rows = connection.execute("select step_type, step_payload from steps").fetchall()
            finally:
                connection.close()
            group, web = exposure_group(rows)
        batch = json.loads(
            (directory / f"batch-{run['batch_id']}.json").read_text(encoding="utf-8")
        )
        for item_id in batch["item_ids"]:
            groups[item_id] = group
        tally[group] += 1
        if web:
            web_batches[group] += 1
    return groups, {
        "batches": dict(sorted(tally.items())),
        "batches_that_searched_the_web": dict(web_batches),
    }


# -- statistics --------------------------------------------------------------------------------------------------


def wilson(k: int, n: int) -> list[float] | None:
    interval = metrics.wilson(k, n)
    return [round(interval[0], 6), round(interval[1], 6)] if interval else None


def newcombe(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any] | None:
    """Newcombe's hybrid score interval (method 10) for p1 - p2."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = metrics.wilson(k1, n1, Z) or (0.0, 1.0)
    l2, u2 = metrics.wilson(k2, n2, Z) or (0.0, 1.0)
    d = p1 - p2
    low = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {
        "difference": round(d, 6),
        "interval_95": [round(low, 6), round(high, 6)],
        "excludes_zero": low > 0 or high < 0,
    }


# -- labels ------------------------------------------------------------------------------------------------------


def labels(task: str, package: Path) -> tuple[dict[str, dict[str, tuple[str, str]]], list[str]]:
    """{annotator: {item: (label, ambiguity_status)}} from a verified package."""
    _manifest, records, item_ids = load_package(task, package, ROOT)
    out: dict[str, dict[str, tuple[str, str]]] = {a: {} for a in ANNOTATORS}
    key = "pdetcov_id"
    for record in records:
        if record.get("annotator_id") in out and record.get("status", "labeled") == "labeled":
            out[record["annotator_id"]][record[key]] = (
                str(record["gold_policy_label"]),
                str(record.get("ambiguity_status") or "NONE"),
            )
    return out, item_ids


def reference(task: str, variant: str | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    stem = f"{task}.reference" + (f".{variant}" if variant else "")
    directory = ROOT / REFERENCE_DIRS[task]
    manifest = json.loads((directory / f"{stem}.manifest.json").read_text(encoding="utf-8"))
    data = (directory / f"{stem}.jsonl").read_bytes()
    if sha256_bytes(data) != manifest["reference_sha256"]:
        raise ComparisonError(f"{stem}.jsonl does not match its manifest")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return manifest, {row["pdetcov_id"]: row for row in rows}


def compare_task(task: str) -> dict[str, Any]:
    old_manifest, old_refs = reference(task, None)
    new_manifest, new_refs = reference(task, VARIANT)
    old, items = labels(task, ROOT / old_manifest["package_manifest"])
    new, new_items = labels(task, ROOT / new_manifest["package_manifest"])
    if sorted(items) != sorted(new_items):
        raise ComparisonError(f"{task}: the two packages cover different items")
    for annotator in (GPT, DEEPSEEK):
        if old[annotator] != new[annotator]:
            raise ComparisonError(f"{task}: {annotator}'s labels differ between the packages")
    groups, tally = item_groups(task)
    if set(groups) != set(items):
        raise ComparisonError(f"{task}: not every item has an old Gemini batch")
    rows: dict[str, dict[str, Any]] = {}
    counts: dict[str, Counter[str]] = {}
    for group in sorted(set(groups.values())):
        members = [i for i in items if groups[i] == group]
        c: Counter[str] = Counter()
        for item in members:
            (old_label, old_status), (new_label, new_status) = old[GEMINI][item], new[GEMINI][item]
            c["n"] += 1
            c["label_changed"] += old_label != new_label
            c["status_changed"] += old_status != new_status
            g, d = old[GPT][item][0], old[DEEPSEEK][item][0]
            if g == d:
                c["others_agree"] += 1
                c["old_matched_others"] += old_label == g
                c["new_matched_others"] += new_label == g
                c["agreement_lost"] += old_label == g and new_label != g
                c["agreement_gained"] += old_label != g and new_label == g
            o, n = old_refs[item], new_refs[item]
            c["reference_label_changed"] += o["reference_label"] != n["reference_label"]
            c["consensus_kind_changed"] += o["consensus"] != n["consensus"]
        counts[group] = c
        rows[group] = {
            **dict(sorted(c.items())),
            "label_change_rate": round(c["label_changed"] / c["n"], 6),
            "label_change_wilson_95": wilson(c["label_changed"], c["n"]),
            "agreement_lost_rate": round(c["agreement_lost"] / c["others_agree"], 6)
            if c["others_agree"]
            else None,
            "agreement_lost_wilson_95": wilson(c["agreement_lost"], c["others_agree"]),
        }
    contrasts = {}
    base = counts.get("D")
    for group in ("A", "B"):
        if group in counts and base:
            g = counts[group]
            contrasts[f"{group}-D"] = {
                "label_change": newcombe(
                    g["label_changed"], g["n"], base["label_changed"], base["n"]
                ),
                "agreement_lost": newcombe(
                    g["agreement_lost"],
                    g["others_agree"],
                    base["agreement_lost"],
                    base["others_agree"],
                ),
            }
    return {
        "items": len(items),
        "old_batches": tally,
        "by_group": rows,
        "contrasts": contrasts,
        "reference": {
            "frozen_manifest_sha256": sha256_bytes(
                (ROOT / REFERENCE_DIRS[task] / f"{task}.reference.manifest.json").read_bytes()
            ),
            "relabelled_manifest_sha256": sha256_bytes(
                (
                    ROOT / REFERENCE_DIRS[task] / f"{task}.reference.{VARIANT}.manifest.json"
                ).read_bytes()
            ),
            "frozen_summary": old_manifest["summary"].get("reference_labels"),
            "relabelled_summary": new_manifest["summary"].get("reference_labels"),
        },
    }


# -- the classifier tests, re-scored -----------------------------------------------------------------------------


class _Saved:
    def __init__(self, label: str, step: str) -> None:
        self.label, self.step = label, step


def _lookup(
    records: list[dict[str, Any]], key: str, features: Any, predictions: dict[str, Any]
) -> Any:
    """A stand-in for the classifier: each item's saved prediction, found by its features hash."""
    table: dict[str, _Saved] = {}
    for record in records:
        saved = predictions.get(record[key])
        if saved is None:
            continue
        digest = v1.features_sha256(features(record))
        prior = table.get(digest)
        if prior is not None and (prior.label, prior.step) != (saved["prediction"], saved["step"]):
            raise ComparisonError("two items share a features hash but not a prediction")
        table[digest] = _Saved(saved["prediction"], saved["step"])
    return lambda feats: table[v1.features_sha256(feats)]


def rescore_v1(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inputs = v1.load_inputs(ROOT)
    saved_rows = v1._jsonl(
        ROOT / "reports/prose-classifier/test/prose-decision-classifier-v1.test-predictions.jsonl"
    )
    by_id = {row["item_id"]: row for row in saved_rows}
    classify_cov = _lookup(inputs["coverage_population"], "pdetcov_id", v1.coverage_features, by_id)
    classify_pdet = _lookup(inputs["pdet_population"], "pdet_id", v1.pdet_features, by_id)
    cov, _, _ = v1.coverage_items(inputs["coverage_population"], references, classify_cov)
    pdet, _, _ = v1.pdet_items(
        inputs["pdet_population"], inputs["pdet_labels"], inputs["pdet_not_in_input"], classify_pdet
    )
    return metrics.evaluate(cov + pdet, inputs["coverage_pool_strata"])


def rescore_v2(references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    coverage = v2.load_coverage_v2(ROOT)
    saved_rows = v1._jsonl(
        ROOT
        / "reports/prose-classifier/test-v2/prose-decision-classifier-v2.test-predictions.jsonl"
    )
    by_id = {row["item_id"]: row for row in saved_rows if row["population"] == v2.COVERAGE_V2}
    classify = _lookup(coverage["population"], "pdetcov_id", v1.coverage_features, by_id)
    items, _, _, _ = v2.coverage_v2_items(coverage["population"], references, classify)
    return metrics.evaluate(items, coverage["pool_strata"])


def _verdict(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "qualification": evaluation.get("qualification"),
        "c1_authorised": evaluation.get("c1_authorised"),
    }


def rescore() -> dict[str, Any]:
    frozen_v1 = json.loads(
        (
            ROOT / "reports/prose-classifier/test/prose-decision-classifier-v1.test-result.json"
        ).read_text(encoding="utf-8")
    )
    frozen_v2 = json.loads(
        (
            ROOT / "reports/prose-classifier/test-v2/prose-decision-classifier-v2.test-result.json"
        ).read_text(encoding="utf-8")
    )
    _, v1_frozen_refs = reference("pdet-coverage-v1", None)
    _, v1_new_refs = reference("pdet-coverage-v1", VARIANT)
    _, v2_frozen_refs = reference("pdet-coverage-v2", None)
    _, v2_new_refs = reference("pdet-coverage-v2", VARIANT)
    # The self-check: the lookup and the frozen references must reproduce the frozen evaluations exactly.
    if json.loads(json.dumps(rescore_v1(v1_frozen_refs))) != frozen_v1["evaluation"]:
        raise ComparisonError(
            "re-scoring classifier v1's test on the frozen reference does not reproduce it"
        )
    if json.loads(json.dumps(rescore_v2(v2_frozen_refs))) != frozen_v2["evaluation"]["gating"]:
        raise ComparisonError(
            "re-scoring classifier v2's test on the frozen reference does not reproduce it"
        )
    new_v1, new_v2 = rescore_v1(v1_new_refs), rescore_v2(v2_new_refs)
    return {
        "self_check": "re-scoring on the frozen references reproduces both frozen evaluations exactly",
        "classifier_v1_test": {
            "frozen": _verdict(frozen_v1["evaluation"]),
            "relabelled": _verdict(new_v1),
            "evaluation": new_v1,
        },
        "classifier_v2_test": {
            "frozen": _verdict(frozen_v2["evaluation"]["gating"]),
            "relabelled": _verdict(new_v2),
            "evaluation": new_v2,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = {
        "artifact_kind": "INC_0002_RELABEL_COMPARISON",
        "plan": PLAN.as_posix(),
        "plan_sha256": sha256_bytes((ROOT / PLAN).read_bytes()),
        "operationalisation": (
            "The plan's agreement drop is measured as agreement lost: among items where gpt-5.6-sol and "
            "deepseek-v4.1-flash agree, the share where old Gemini matched them and new Gemini does not. "
            "Agreement gained (the reverse) is reported beside it."
        ),
        "limit": "No batch was clean: group D also browsed (outside the repository), so it is not a pure "
        "run-to-run baseline.",
        "tasks": {task: compare_task(task) for task in REFERENCE_DIRS},
        "classifier_tests": rescore(),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        path = ROOT / OUTPUT
        if path.exists():
            raise ComparisonError(f"{OUTPUT} exists; it is written once")
        path.write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
