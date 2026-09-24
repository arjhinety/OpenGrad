"""The one-shot test of the frozen prose decision classifier (33 §5, 22 §6, 30 §11).

Runs ``prose-decision-classifier-v1`` once on both validation populations and scores it with the preregistered
rules of ``pdet-coverage-metrics-v1``:

* **P-DET-COVERAGE-v1**, layer B (306 items), against the three-model consensus reference of
  ``study_002_prereg_v5`` (34). Every result there is ``MODEL_REFERENCE`` and provisional. ``NO_CONSENSUS``
  items are left out of every metric.
* **P-DET-v1** (581 items) against the study owner's human labels (session ``pass-a``, one annotator, **not
  frozen gold**). Items excluded as ``EXPOSED_WORKED_EXAMPLE`` stay excluded. The 6 items that
  normalization-v3 rejected (30 §5) can never reach the classifier; they are counted, never classified.

Safeguards:

* it refuses to run unless the classifier source is byte-for-byte the frozen one (tag
  ``prose-decision-classifier-v1``);
* it refuses to run unless both populations, the consensus reference and the human-label package match their
  pinned hashes or verify;
* it checks that each coverage item's features hash equals the one recorded at selection, so the classifier
  sees exactly the contract input;
* it never overwrites a written result: the test runs once;
* it prints counts, rows and verdicts only, never item text or ids.

A qualification here grants nothing by itself. Whether a ``MODEL_REFERENCE`` qualification may grant balancing
permission is the study owner's separate decision (34 §4), and C1 needs the owner's authorisation.

    python -m opengrad.verification.prose_classifier_oneshot --run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.canonical import stable_json
from opengrad.data.classifier_input import CONTRACT_VERSION, ClassifierFeatures
from opengrad.hashing import sha256_bytes as _sha256
from opengrad.verification import pdet_coverage_metrics as metrics

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier.py")
FROZEN_SOURCE_SHA256_LF = "64293c51917a54a649bad9e96960302dd5df1b39eabd0d260d3784b3eeb6687c"
FROZEN_TAG = "prose-decision-classifier-v1"

COVERAGE_POPULATION = Path("reports/pdet-coverage/pdet-coverage-v1.population.jsonl")
COVERAGE_POPULATION_SHA256 = "755bc16e79ceb1cb9e461c0fe8b9125628c16f263ed24f9e008f55b613a7158c"
COVERAGE_MANIFEST = Path("reports/pdet-coverage/pdet-coverage-v1.manifest.json")
COVERAGE_REFERENCE = Path("reports/pdet-coverage/reference/pdet-coverage-v1.reference.jsonl")
COVERAGE_REFERENCE_MANIFEST = Path(
    "reports/pdet-coverage/reference/pdet-coverage-v1.reference.manifest.json"
)

PDET_POPULATION = Path("reports/pdet/pdet-v1.population.jsonl")
PDET_POPULATION_SHA256 = "6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b"
PDET_PACKAGE = Path("reports/pdet/annotation/wip/pdet-v1.annotation-manifest.json")
PDET_HUMAN_SESSION = "pass-a"
PDET_REPRESENTATION_AUDIT = Path("reports/normalization-v3/pdet-v1-representation-audit.json")

OUTPUT_DIR = Path("reports/prose-classifier/test")
RESULT_NAME = f"{FROZEN_TAG}.test-result.json"
PREDICTIONS_NAME = f"{FROZEN_TAG}.test-predictions.jsonl"

NOT_IN_CLASSIFIER_INPUT = "NOT_IN_NORMALIZATION_V3"


class TestRunError(RuntimeError):
    """The one-shot test cannot run: an input is not the pinned one, or a result already exists."""


def _lf_sha256(path: Path) -> str:
    return _sha256(path.read_bytes().replace(b"\r\n", b"\n"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def check_frozen(root: Path) -> None:
    observed = _lf_sha256(root / CLASSIFIER_MODULE)
    if observed != FROZEN_SOURCE_SHA256_LF:
        raise TestRunError(
            f"{CLASSIFIER_MODULE.as_posix()} is not the frozen {FROZEN_TAG} (sha256 LF {observed})"
        )


def features_sha256(features: ClassifierFeatures) -> str:
    """The contract's feature hash (32 §7), computed from the features alone."""
    payload = {"contract_version": CONTRACT_VERSION, **features.as_dict()}
    return _sha256(stable_json(payload).encode("utf-8"))


def coverage_features(record: Mapping[str, Any]) -> ClassifierFeatures:
    return ClassifierFeatures(
        user_message=record["user_message"],
        assistant_response=record["assistant_response"],
        tools=tuple(json.loads(stable_json(tool)) for tool in record["tools"]),
        structured_call_present=False,
    )


def pdet_features(record: Mapping[str, Any]) -> ClassifierFeatures:
    return ClassifierFeatures(
        user_message=record["prompt"],
        assistant_response=record["response"],
        tools=tuple(json.loads(stable_json(tool)) for tool in record["tools"]),
        structured_call_present=False,
    )


def coverage_items(
    population: Iterable[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    classify: Any,
) -> tuple[list[metrics.Item], list[dict[str, Any]], dict[str, Any]]:
    """Layer B items of P-DET-COVERAGE-v1 with their consensus reference and the classifier's prediction."""
    items: list[metrics.Item] = []
    predictions: list[dict[str, Any]] = []
    feature_mismatch = 0
    consensus: Counter[str] = Counter()
    for record in population:
        if record["layer"] != "B":
            continue
        item_id = record["pdetcov_id"]
        if item_id not in references:
            raise TestRunError("the consensus reference does not cover every layer B item")
        reference = references[item_id]
        features = coverage_features(record)
        if features_sha256(features) != record["features_sha256"]:
            feature_mismatch += 1
        decision = classify(features)
        eligible = bool(reference["metric_eligible"])
        consensus[str(reference["consensus"])] += 1
        items.append(
            metrics.Item(
                population=metrics.COVERAGE,
                gold=reference["reference_label"] if eligible else None,
                prediction=decision.label,
                stratum=record["stratum"],
                source=record["source_name"],
                layer="B",
                excluded=not eligible,
            )
        )
        predictions.append(
            {
                "population": metrics.COVERAGE,
                "item_id": item_id,
                "prediction": decision.label,
                "step": decision.step,
                "reference_label": reference["reference_label"],
                "reference": "MODEL_REFERENCE",
                "metric_eligible": eligible,
            }
        )
    if feature_mismatch:
        raise TestRunError(
            f"{feature_mismatch} coverage items do not reproduce their recorded features hash"
        )
    return (
        items,
        predictions,
        {"layer_b_items": len(items), "consensus": dict(sorted(consensus.items()))},
    )


def pdet_items(
    population: Iterable[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    not_in_input: frozenset[str],
    classify: Any,
) -> tuple[list[metrics.Item], list[dict[str, Any]], dict[str, Any]]:
    """P-DET-v1 items with the human pass-a label and the classifier's prediction."""
    items: list[metrics.Item] = []
    predictions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in population:
        item_id = record["pdet_id"]
        if item_id in not_in_input:
            counts[NOT_IN_CLASSIFIER_INPUT] += 1
            continue
        label = labels.get(item_id)
        if label is None or label.get("status") != "labeled":
            raise TestRunError("the human pass does not label every P-DET-v1 item")
        exclusions = list(label.get("metric_exclusions") or [])
        decision = classify(pdet_features(record))
        counts["classified"] += 1
        for exclusion in exclusions:
            counts[exclusion] += 1
        items.append(
            metrics.Item(
                population=metrics.PDET_V1,
                gold=label["gold_policy_label"],
                prediction=decision.label,
                challenge=record["pdet_component"] == "challenge",
                layer="B",
                excluded=bool(exclusions),
            )
        )
        predictions.append(
            {
                "population": metrics.PDET_V1,
                "item_id": item_id,
                "prediction": decision.label,
                "step": decision.step,
                "reference_label": label["gold_policy_label"],
                "reference": "HUMAN_SINGLE_ANNOTATOR_NOT_FROZEN",
                "metric_exclusions": exclusions,
            }
        )
    return items, predictions, dict(sorted(counts.items()))


def load_inputs(root: Path) -> dict[str, Any]:
    """Every pinned input, verified. Reads no item text into the output."""
    from opengrad.annotation.export import verify_package

    coverage_bytes = (root / COVERAGE_POPULATION).read_bytes()
    if _sha256(coverage_bytes) != COVERAGE_POPULATION_SHA256:
        raise TestRunError("P-DET-COVERAGE-v1 population is not the drawn one")
    reference_manifest = json.loads(
        (root / COVERAGE_REFERENCE_MANIFEST).read_text(encoding="utf-8")
    )
    reference_bytes = (root / COVERAGE_REFERENCE).read_bytes()
    if reference_manifest.get("status") != "MODEL_REFERENCE_PROVISIONAL":
        raise TestRunError("the coverage reference is not the provisional three-model consensus")
    if _sha256(reference_bytes) != reference_manifest["reference_sha256"]:
        raise TestRunError("the coverage reference file does not match its manifest")
    if reference_manifest.get("population_sha256") not in (None, COVERAGE_POPULATION_SHA256):
        raise TestRunError("the coverage reference was built on another population")

    pdet_bytes = (root / PDET_POPULATION).read_bytes()
    if _sha256(pdet_bytes) != PDET_POPULATION_SHA256:
        raise TestRunError("P-DET-v1 population is not the frozen one")
    # verify_package returns the validation result and a status summary, not the manifest itself.
    result, verification = verify_package(root / PDET_PACKAGE, root, require_source=True)
    if result.all_errors() or verification.get("status") != "PASS":
        raise TestRunError("the P-DET-v1 annotation package failed verification")
    package = json.loads((root / PDET_PACKAGE).read_text(encoding="utf-8"))
    sessions = {s["session_id"]: s for s in package.get("sessions") or []}
    if PDET_HUMAN_SESSION not in sessions:
        raise TestRunError(f"the P-DET-v1 package has no {PDET_HUMAN_SESSION} session")
    human = _jsonl((root / PDET_PACKAGE).parent / sessions[PDET_HUMAN_SESSION]["file"])
    audit = json.loads((root / PDET_REPRESENTATION_AUDIT).read_text(encoding="utf-8"))
    if audit.get("pdet_v1_population_sha256") != PDET_POPULATION_SHA256:
        raise TestRunError("the representation audit is for another P-DET-v1 population")
    return {
        "coverage_population": [
            json.loads(line) for line in coverage_bytes.decode("utf-8").splitlines() if line
        ],
        "coverage_references": {
            row["pdetcov_id"]: row for row in _jsonl(root / COVERAGE_REFERENCE)
        },
        "coverage_pool_strata": json.loads((root / COVERAGE_MANIFEST).read_text(encoding="utf-8"))[
            "counts"
        ]["pool_strata"],
        "reference_manifest": reference_manifest,
        "reference_manifest_sha256": _sha256((root / COVERAGE_REFERENCE_MANIFEST).read_bytes()),
        "pdet_population": [
            json.loads(line) for line in pdet_bytes.decode("utf-8").splitlines() if line
        ],
        "pdet_labels": {row["pdet_id"]: row for row in human},
        "pdet_package_manifest_sha256": _sha256((root / PDET_PACKAGE).read_bytes()),
        "pdet_not_in_input": frozenset(
            row["pdet_id"] for row in audit["not_structurally_equivalent"]
        ),
    }


def run(root: Path = ROOT) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    if (out / RESULT_NAME).exists() or (out / PREDICTIONS_NAME).exists():
        raise TestRunError(
            f"{out / RESULT_NAME} exists: the test runs once and is never overwritten"
        )
    check_frozen(root)
    from opengrad.data.decision_classifier import CLASSIFIER_VERSION, classify

    inputs = load_inputs(root)
    cov_items, cov_predictions, cov_counts = coverage_items(
        inputs["coverage_population"], inputs["coverage_references"], classify
    )
    pdet_items_, pdet_predictions, pdet_counts = pdet_items(
        inputs["pdet_population"], inputs["pdet_labels"], inputs["pdet_not_in_input"], classify
    )
    evaluation = metrics.evaluate(cov_items + pdet_items_, inputs["coverage_pool_strata"])
    predictions = cov_predictions + pdet_predictions
    predictions_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in predictions
    )
    result = {
        "artifact_kind": "PROSE_CLASSIFIER_ONE_SHOT_TEST",
        "classifier_version": CLASSIFIER_VERSION,
        "classifier_tag": FROZEN_TAG,
        "classifier_source_sha256_lf": FROZEN_SOURCE_SHA256_LF,
        "input_contract": CONTRACT_VERSION,
        "metrics_version": metrics.METRICS_VERSION,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "references": {
            metrics.COVERAGE: {
                "kind": "MODEL_REFERENCE (provisional)",
                "what": "two-of-three consensus of Gemini 3.8 Flash (High), gpt-5.6-sol and deepseek-v4.1-flash (34)",
                "reference_manifest_sha256": inputs["reference_manifest_sha256"],
                "counts": cov_counts,
            },
            metrics.PDET_V1: {
                "kind": "HUMAN, single annotator, not frozen gold",
                "session": PDET_HUMAN_SESSION,
                "package_manifest_sha256": inputs["pdet_package_manifest_sha256"],
                "counts": pdet_counts,
                "exposure": (
                    "The classifier's developer (Claude Opus 5) produced the model-a labels on P-DET-v1 and so read "
                    "its items before development (33 §6). It never saw the human labels or any classifier result on "
                    "them."
                ),
            },
        },
        "evaluation": evaluation,
        "predictions_file": PREDICTIONS_NAME,
        "predictions_sha256": _sha256(predictions_bytes),
        "statement": (
            "The single preregistered test of the frozen prose decision classifier. Coverage results are measured "
            "against a model reference and are provisional; P-DET-v1 results against one unfrozen human annotator. "
            "A qualification grants nothing by itself: balancing permission from a MODEL_REFERENCE qualification "
            "and C1 both need the study owner's decision (34 §4, 22 §6)."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / PREDICTIONS_NAME).write_bytes(predictions_bytes)
    (out / RESULT_NAME).write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return result


def summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Counts, rows and verdicts only."""
    evaluation = result["evaluation"]
    return {
        "classifier": result["classifier_tag"],
        "reference_counts": {name: ref["counts"] for name, ref in result["references"].items()},
        "global_rows": evaluation["global_rows"],
        "qualification": evaluation["qualification"],
        "c1_authorised_by_rules": evaluation["c1_authorised"],
        "rows": {
            population: {
                name: {k: row.get(k) for k in ("status", "value", "n", "k", "threshold")}
                for name, row in data["rows"].items()
            }
            for population, data in evaluation["populations"].items()
        },
        "direct_poststratified_precision": {
            source: {
                k: value.get(k)
                for k in ("status", "value", "direct_predictions", "strata_without_sample")
            }
            for source, value in evaluation["direct_poststratified_precision"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run", action="store_true", help="run the one-shot test (refuses if a result exists)"
    )
    group.add_argument(
        "--preflight",
        action="store_true",
        help="check the freeze and every input; classify nothing",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.preflight:
        check_frozen(args.root)
        inputs = load_inputs(args.root)
        print(
            json.dumps(
                {
                    "frozen_classifier": "matches",
                    "coverage_layer_b": sum(
                        1 for r in inputs["coverage_population"] if r["layer"] == "B"
                    ),
                    "coverage_references": len(inputs["coverage_references"]),
                    "pdet_v1_items": len(inputs["pdet_population"]),
                    "pdet_v1_human_labels": len(inputs["pdet_labels"]),
                    "pdet_v1_not_in_input": len(inputs["pdet_not_in_input"]),
                    "result_exists": (args.root / OUTPUT_DIR / RESULT_NAME).exists(),
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(summary(run(args.root)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
