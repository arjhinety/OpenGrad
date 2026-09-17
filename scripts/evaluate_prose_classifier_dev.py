"""Agreement of ``prose-decision-classifier-v1`` with the model labels on its development set (33 §5).

Reads only the development population and the exported ``model-dev`` labels. It never opens P-DET-v1 or
P-DET-COVERAGE-v1. The numbers are **agreement with model labels**, not accuracy: the model is not ground truth.

    python scripts/evaluate_prose_classifier_dev.py                 # counts and confusion matrix
    python scripts/evaluate_prose_classifier_dev.py --show 20       # also print up to 20 disagreements
    python scripts/evaluate_prose_classifier_dev.py --write         # write the agreement report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from opengrad.data.classifier_input import CONTRACT_VERSION, ClassifierFeatures
from opengrad.data.decision_classifier import ABSTAIN, CLASSIFIER_VERSION, LABELS, classify

ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = Path("reports/prose-classifier/dev")
#: set name -> (population, exported model labels, report). The check set (33 §5a) is scored once and its items
#: are never printed, so the developer never reads them.
SETS = {
    "dev": (
        DEV_DIR / "prose-classifier-dev-v1.population.jsonl",
        DEV_DIR / "annotation/wip/prose-classifier-dev-v1.annotations.model.claude-opus-5.model-dev.jsonl",
        DEV_DIR / f"{CLASSIFIER_VERSION}.dev-agreement.json",
    ),
    "devcheck": (
        DEV_DIR / "prose-classifier-devcheck-v1.population.jsonl",
        Path("reports/prose-classifier/devcheck/annotation/wip/")
        / "prose-classifier-devcheck-v1.annotations.model.claude-opus-5.model-devcheck.jsonl",
        DEV_DIR / f"{CLASSIFIER_VERSION}.devcheck-agreement.json",
    ),
}
POPULATION, LABELS_FILE, REPORT = SETS["dev"]
UNKNOWN = "UNKNOWN"
MODES = ("CALL", "DIRECT", "CLARIFY", "UNSUPPORTED")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(root: Path, population: Path = POPULATION, labels_file: Path = LABELS_FILE) -> list[dict]:
    items = {
        item["dev_id"]: item
        for item in map(json.loads, (root / population).read_text(encoding="utf-8").splitlines())
    }
    labels = [json.loads(line) for line in (root / labels_file).read_text(encoding="utf-8").splitlines() if line]
    out = []
    for record in labels:
        item = items[record["dev_id"]]
        features = ClassifierFeatures(
            user_message=item["user_message"],
            assistant_response=item["assistant_response"],
            tools=tuple(item["tools"]),
            structured_call_present=False,
        )
        decision = classify(features)
        out.append(
            {
                "dev_index": item["dev_index"],
                "model_label": record["gold_policy_label"],
                "ambiguity_status": record["ambiguity_status"],
                "prediction": decision.label,
                "step": decision.step,
                "evidence": decision.evidence,
                "item": item,
            }
        )
    return sorted(out, key=lambda row: row["dev_index"])


def summarise(results: list[dict]) -> dict:
    confusion = Counter((row["model_label"], row["prediction"]) for row in results)
    usable = [row for row in results if row["model_label"] in MODES]
    agree = sum(1 for row in usable if row["prediction"] == row["model_label"])
    per_mode = {}
    for mode in MODES:
        labelled = [row for row in usable if row["model_label"] == mode]
        predicted = [row for row in usable if row["prediction"] == mode]
        hits = sum(1 for row in labelled if row["prediction"] == mode)
        per_mode[mode] = {
            "model_labels": len(labelled),
            "predictions_on_mode_labels": len(predicted),
            "agree": hits,
            "share_of_model_labels_agreed": round(hits / len(labelled), 4) if labelled else None,
            "share_of_predictions_agreed": (
                round(sum(1 for row in predicted if row["model_label"] == mode) / len(predicted), 4) if predicted else None
            ),
        }
    return {
        "items": len(results),
        "items_with_a_mode_label": len(usable),
        "agreement_on_mode_labels": agree,
        "agreement_rate_on_mode_labels": round(agree / len(usable), 4) if usable else None,
        "abstentions_on_mode_labels": sum(1 for row in usable if row["prediction"] == ABSTAIN),
        "per_mode": per_mode,
        "predictions_on_unknown_labels": dict(
            sorted(Counter(row["prediction"] for row in results if row["model_label"] == UNKNOWN).items())
        ),
        "confusion_model_label_by_prediction": {
            label: {prediction: confusion[(label, prediction)] for prediction in LABELS if confusion[(label, prediction)]}
            for label in (*MODES, UNKNOWN)
        },
        "decisions_by_step": dict(sorted(Counter(row["step"] for row in results).items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--set", choices=sorted(SETS), default="dev")
    parser.add_argument("--show", type=int, default=0, help="print up to N disagreements (development items only)")
    parser.add_argument("--write", action="store_true", help="write the agreement report")
    args = parser.parse_args(argv)
    population, labels_file, report_path = SETS[args.set]
    if args.set == "devcheck" and args.show:
        parser.error("check-set items are never printed (33 §5a)")

    results = rows(args.root, population, labels_file)
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    shown = 0
    for row in results:
        if shown >= args.show:
            break
        if row["prediction"] == row["model_label"] or (row["model_label"] == UNKNOWN and row["prediction"] == ABSTAIN):
            continue
        shown += 1
        item = row["item"]
        names = [tool.get("name") for tool in item["tools"]]
        print(f"\n#{row['dev_index'] + 1} model={row['model_label']}/{row['ambiguity_status']} "
              f"pred={row['prediction']} step={row['step']} evidence={list(row['evidence'])} tools={names}")
        print("  A:", " ".join(item["assistant_response"].split())[:600])

    if args.write:
        report = {
            "artifact_kind": (
                "PROSE_CLASSIFIER_DEVELOPMENT_AGREEMENT"
                if args.set == "dev"
                else "PROSE_CLASSIFIER_DEVELOPMENT_CHECK_AGREEMENT"
            ),
            "statement": (
                "Agreement of the classifier with model judgments (model.claude-opus-5) on its own development "
                "set. The rules were developed against these labels, so this is in-sample agreement with a model, "
                "not accuracy and not evidence of qualification."
                if args.set == "dev"
                else "Agreement of the classifier with model judgments (model.claude-opus-5) on the held-out "
                "development check set (33 §5a), scored once before freezing; the developer never read its items. "
                "Agreement with a model, not accuracy and not evidence of qualification."
            ),
            "classifier_version": CLASSIFIER_VERSION,
            "input_contract": CONTRACT_VERSION,
            "population": {"path": population.as_posix(), "sha256": sha256(args.root / population)},
            "labels": {"path": labels_file.as_posix(), "sha256": sha256(args.root / labels_file)},
            "summary": summary,
        }
        (args.root / report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
