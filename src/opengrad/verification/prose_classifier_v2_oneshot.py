"""The one-shot test of the frozen ``prose-decision-classifier-v2`` (37 §5, 36 §3, 22 §6).

Runs the frozen v2 rules once and scores them with the preregistered rules of ``pdet-coverage-metrics-v1``,
unchanged:

* **Gating: P-DET-COVERAGE-v2** (420 first replies, contract ``prose-decision-input-v2``) against the three-model
  consensus reference of ``study_002_prereg_v6`` (36 §3.7). Every result is ``MODEL_REFERENCE`` and provisional;
  ``NO_CONSENSUS`` items are left out of every metric. It is scored alone, in the metrics module's coverage slot,
  so its challenge rows use strata R, Q, M and X (37 §5). Qualification and C1 come from this evaluation only.
* **Reported per ``unit_kind``** (single exchange, continuing conversation): population rows for each, never
  gating.
* **Reported, never gating: P-DET-v1 and P-DET-COVERAGE-v1**, loaded and verified exactly as the v1 test loads
  them, scored in a separate evaluation marked ``DEVELOPMENT_EXPOSED``: the developer read them (37 §2).

Safeguards, as for v1:

* it refuses to run unless the classifier source is byte-for-byte the frozen one (tag
  ``prose-decision-classifier-v2``);
* it refuses to run unless the population, the consensus reference and every exposed input match their pinned
  hashes or verify;
* it checks that each P-DET-COVERAGE-v2 item's features hash under contract v2 equals the one recorded at
  selection;
* it never overwrites a written result: the test runs once;
* it prints counts, rows and verdicts only, never item text or ids.

A qualification grants nothing by itself: a ``MODEL_REFERENCE`` qualification counts toward balancing permission
only provisionally (35 §1), and C1 remains the study owner's decision (37 §5).

    python -m opengrad.verification.prose_classifier_v2_oneshot --preflight
    python -m opengrad.verification.prose_classifier_v2_oneshot --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.canonical import stable_json
from opengrad.data.classifier_input import CONTRACT_V2_VERSION, ClassifierFeatures
from opengrad.verification import pdet_coverage_metrics as metrics
from opengrad.verification import prose_classifier_oneshot as v1

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier_v2.py")
FROZEN_SOURCE_SHA256_LF = "47436ca990bd4c1846113dd8d28e2c8755cff50837581acaa39654d0d2f2b382"
FROZEN_TAG = "prose-decision-classifier-v2"

COVERAGE_V2 = "P-DET-COVERAGE-v2"
COVERAGE_V2_POPULATION = Path("reports/pdet-coverage-v2/pdet-coverage-v2.population.jsonl")
COVERAGE_V2_POPULATION_SHA256 = "8fa4868c64845a93b0627cd4463887b03181e6479f2e9f48d75e8c80f626f013"
COVERAGE_V2_MANIFEST = Path("reports/pdet-coverage-v2/pdet-coverage-v2.manifest.json")
COVERAGE_V2_REFERENCE = Path("reports/pdet-coverage-v2/reference/pdet-coverage-v2.reference.jsonl")
COVERAGE_V2_REFERENCE_MANIFEST = Path("reports/pdet-coverage-v2/reference/pdet-coverage-v2.reference.manifest.json")

OUTPUT_DIR = Path("reports/prose-classifier/test-v2")
RESULT_NAME = f"{FROZEN_TAG}.test-result.json"
PREDICTIONS_NAME = f"{FROZEN_TAG}.test-predictions.jsonl"

GATING = "GATING_MODEL_REFERENCE"
DEVELOPMENT_EXPOSED = "DEVELOPMENT_EXPOSED"


class TestRunError(RuntimeError):
    """The one-shot test cannot run: an input is not the pinned one, or a result already exists."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_frozen(root: Path) -> None:
    observed = _sha256((root / CLASSIFIER_MODULE).read_bytes().replace(b"\r\n", b"\n"))
    if observed != FROZEN_SOURCE_SHA256_LF:
        raise TestRunError(f"{CLASSIFIER_MODULE.as_posix()} is not the frozen {FROZEN_TAG} (sha256 LF {observed})")


def features_sha256(features: ClassifierFeatures) -> str:
    """The contract v2 feature hash (36 §2), computed from the features alone."""
    payload = {"contract_version": CONTRACT_V2_VERSION, **features.as_dict()}
    return _sha256(stable_json(payload).encode("utf-8"))


def coverage_v2_items(
    population: Iterable[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    classify: Any,
) -> tuple[list[metrics.Item], dict[str, list[metrics.Item]], list[dict[str, Any]], dict[str, Any]]:
    """P-DET-COVERAGE-v2 items with their consensus reference and the classifier's prediction, also grouped by
    ``unit_kind``. Items sit in the metrics module's coverage slot, so its challenge strata apply unchanged."""
    items: list[metrics.Item] = []
    by_kind: dict[str, list[metrics.Item]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []
    feature_mismatch = 0
    consensus: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    for record in population:
        if record["layer"] != "B" or record.get("classifier_input_contract") != CONTRACT_V2_VERSION:
            raise TestRunError("P-DET-COVERAGE-v2 holds an item outside layer B or contract v2")
        item_id = record["pdetcov_id"]
        if item_id not in references:
            raise TestRunError("the consensus reference does not cover every P-DET-COVERAGE-v2 item")
        reference = references[item_id]
        features = v1.coverage_features(record)
        if features_sha256(features) != record["features_sha256"]:
            feature_mismatch += 1
        decision = classify(features)
        eligible = bool(reference["metric_eligible"])
        consensus[str(reference["consensus"])] += 1
        kinds[record["unit_kind"]] += 1
        item = metrics.Item(
            population=metrics.COVERAGE,
            gold=reference["reference_label"] if eligible else None,
            prediction=decision.label,
            stratum=record["stratum"],
            source=record["source_name"],
            layer="B",
            excluded=not eligible,
        )
        items.append(item)
        by_kind[record["unit_kind"]].append(item)
        predictions.append(
            {
                "population": COVERAGE_V2,
                "exposure": GATING,
                "item_id": item_id,
                "unit_kind": record["unit_kind"],
                "prediction": decision.label,
                "step": decision.step,
                "reference_label": reference["reference_label"],
                "reference": "MODEL_REFERENCE",
                "metric_eligible": eligible,
            }
        )
    if feature_mismatch:
        raise TestRunError(f"{feature_mismatch} P-DET-COVERAGE-v2 items do not reproduce their recorded features hash")
    counts = {"items": len(items), "consensus": dict(sorted(consensus.items())), "unit_kind": dict(sorted(kinds.items()))}
    return items, dict(sorted(by_kind.items())), predictions, counts


def evaluate(
    gating_items: list[metrics.Item],
    gating_pool_strata: Mapping[str, Mapping[str, int]],
    by_kind: Mapping[str, list[metrics.Item]],
    exposed_items: list[metrics.Item],
    exposed_pool_strata: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """The gating evaluation on P-DET-COVERAGE-v2 alone, and the reported-only ones beside it. Exposed items never
    enter the gating evaluation."""
    return {
        "gating": metrics.evaluate(gating_items, gating_pool_strata),
        "by_unit_kind": {kind: metrics.population_metrics(items, metrics.COVERAGE) for kind, items in by_kind.items()},
        "development_exposed": metrics.evaluate(exposed_items, exposed_pool_strata),
    }


def load_coverage_v2(root: Path) -> dict[str, Any]:
    """P-DET-COVERAGE-v2 and its consensus reference, verified. Reads no item text into the output."""
    population_bytes = (root / COVERAGE_V2_POPULATION).read_bytes()
    if _sha256(population_bytes) != COVERAGE_V2_POPULATION_SHA256:
        raise TestRunError("P-DET-COVERAGE-v2 population is not the drawn one")
    if not (root / COVERAGE_V2_REFERENCE_MANIFEST).exists():
        raise TestRunError("the P-DET-COVERAGE-v2 consensus reference does not exist yet")
    reference_manifest = json.loads((root / COVERAGE_V2_REFERENCE_MANIFEST).read_text(encoding="utf-8"))
    if reference_manifest.get("status") != "MODEL_REFERENCE_PROVISIONAL":
        raise TestRunError("the P-DET-COVERAGE-v2 reference is not the provisional three-model consensus")
    if _sha256((root / COVERAGE_V2_REFERENCE).read_bytes()) != reference_manifest["reference_sha256"]:
        raise TestRunError("the P-DET-COVERAGE-v2 reference file does not match its manifest")
    if reference_manifest.get("population_sha256") != COVERAGE_V2_POPULATION_SHA256:
        raise TestRunError("the P-DET-COVERAGE-v2 reference was built on another population")
    return {
        "population": [json.loads(line) for line in population_bytes.decode("utf-8").splitlines() if line],
        "references": {row["pdetcov_id"]: row for row in _jsonl(root / COVERAGE_V2_REFERENCE)},
        "pool_strata": json.loads((root / COVERAGE_V2_MANIFEST).read_text(encoding="utf-8"))["counts"]["pool_strata"],
        "reference_manifest_sha256": _sha256((root / COVERAGE_V2_REFERENCE_MANIFEST).read_bytes()),
    }


def run(root: Path = ROOT) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    if (out / RESULT_NAME).exists() or (out / PREDICTIONS_NAME).exists():
        raise TestRunError(f"{out / RESULT_NAME} exists: the test runs once and is never overwritten")
    check_frozen(root)
    from opengrad.data.decision_classifier_v2 import CLASSIFIER_VERSION, classify

    coverage = load_coverage_v2(root)
    exposed = v1.load_inputs(root)
    gating_items, by_kind, gating_predictions, gating_counts = coverage_v2_items(
        coverage["population"], coverage["references"], classify
    )
    cov1_items, cov1_predictions, cov1_counts = v1.coverage_items(
        exposed["coverage_population"], exposed["coverage_references"], classify
    )
    pdet_items, pdet_predictions, pdet_counts = v1.pdet_items(
        exposed["pdet_population"], exposed["pdet_labels"], exposed["pdet_not_in_input"], classify
    )
    evaluation = evaluate(
        gating_items, coverage["pool_strata"], by_kind, cov1_items + pdet_items, exposed["coverage_pool_strata"]
    )
    predictions = gating_predictions + [
        {**row, "exposure": DEVELOPMENT_EXPOSED} for row in cov1_predictions + pdet_predictions
    ]
    predictions_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in predictions
    )
    result = {
        "artifact_kind": "PROSE_CLASSIFIER_V2_ONE_SHOT_TEST",
        "classifier_version": CLASSIFIER_VERSION,
        "classifier_tag": FROZEN_TAG,
        "classifier_source_sha256_lf": FROZEN_SOURCE_SHA256_LF,
        "input_contract": CONTRACT_V2_VERSION,
        "metrics_version": metrics.METRICS_VERSION,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "gating": {
            "population": COVERAGE_V2,
            "reference": {
                "kind": "MODEL_REFERENCE (provisional)",
                "what": "two-of-three consensus of Gemini 3.8 Flash (High), gpt-5.6-sol and deepseek-v4.1-flash (36 §3.7)",
                "reference_manifest_sha256": coverage["reference_manifest_sha256"],
                "counts": gating_counts,
            },
            "metrics_slot": (
                f"scored alone in the {metrics.COVERAGE} slot of {metrics.METRICS_VERSION}, so its challenge rows use "
                "strata R, Q, M and X (37 §5); the P-DET-v1 slot is empty in this evaluation"
            ),
        },
        "development_exposed": {
            "populations": [metrics.COVERAGE, metrics.PDET_V1],
            "counts": {metrics.COVERAGE: cov1_counts, metrics.PDET_V1: pdet_counts},
            "exposure": (
                "The developer read P-DET-v1 items (model-a labels) and P-DET-COVERAGE-v1 results per row before v2 "
                "development (37 §2). Reported, never gating."
            ),
        },
        "evaluation": evaluation,
        "qualification": evaluation["gating"]["qualification"],
        "c1_authorised_by_rules": evaluation["gating"]["c1_authorised"],
        "predictions_file": PREDICTIONS_NAME,
        "predictions_sha256": _sha256(predictions_bytes),
        "statement": (
            "The single preregistered test of the frozen prose-decision-classifier-v2. Qualification comes from "
            "P-DET-COVERAGE-v2 alone, against a provisional model reference; per-unit_kind and development-exposed "
            "rows are reported and never gate. A qualification grants nothing by itself: a MODEL_REFERENCE "
            "qualification counts toward balancing permission only provisionally (35 §1), and C1 needs the study "
            "owner's decision (37 §5)."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / PREDICTIONS_NAME).write_bytes(predictions_bytes)
    (out / RESULT_NAME).write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return result


def _rows(population_metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {k: row.get(k) for k in ("status", "value", "n", "k", "threshold")}
        for name, row in population_metrics["rows"].items()
    }


def summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Counts, rows and verdicts only."""
    evaluation = result["evaluation"]
    gating = evaluation["gating"]
    return {
        "classifier": result["classifier_tag"],
        "gating_counts": result["gating"]["reference"]["counts"],
        "global_rows": gating["global_rows"],
        "qualification": gating["qualification"],
        "c1_authorised_by_rules": gating["c1_authorised"],
        "gating_rows": _rows(gating["populations"][metrics.COVERAGE]),
        "direct_poststratified_precision": {
            source: {k: value.get(k) for k in ("status", "value", "direct_predictions", "strata_without_sample")}
            for source, value in gating["direct_poststratified_precision"].items()
        },
        "by_unit_kind_rows (reported)": {kind: _rows(rows) for kind, rows in evaluation["by_unit_kind"].items()},
        "development_exposed_rows (reported)": {
            population: _rows(rows) for population, rows in evaluation["development_exposed"]["populations"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="run the one-shot test (refuses if a result exists)")
    group.add_argument("--preflight", action="store_true", help="check the freeze and every input; classify nothing")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.preflight:
        check_frozen(args.root)
        report: dict[str, Any] = {"frozen_classifier": "matches"}
        try:
            coverage = load_coverage_v2(args.root)
            report["coverage_v2_items"] = len(coverage["population"])
            report["coverage_v2_references"] = len(coverage["references"])
        except TestRunError as error:
            report["coverage_v2_reference_ready"] = str(error)
        exposed = v1.load_inputs(args.root)
        report["development_exposed_inputs"] = {
            "coverage_v1_layer_b": sum(1 for r in exposed["coverage_population"] if r["layer"] == "B"),
            "pdet_v1_items": len(exposed["pdet_population"]),
        }
        report["result_exists"] = (args.root / OUTPUT_DIR / RESULT_NAME).exists()
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(summary(run(args.root)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
