"""Agreement of ``prose-decision-classifier-v2`` with model labels on its development data (37 §3-§4).

Reads only development populations and their exported Claude labels: the three v1 development sets (single
exchanges, kept as development data) and the v2 development and check sets of first replies. It never opens a
validation population. The numbers are **agreement with model labels**, not accuracy. v2 sets are also summarised
per ``unit_kind``, which the classifier never reads.

    python scripts/evaluate_prose_classifier_v2_dev.py --set dev-v2             # counts and confusion matrix
    python scripts/evaluate_prose_classifier_v2_dev.py --set dev-v2 --show 20   # also print up to 20 disagreements
    python scripts/evaluate_prose_classifier_v2_dev.py --set all --round 1 --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_prose_classifier_dev import SETS as V1_SETS
from evaluate_prose_classifier_dev import UNKNOWN, sha256, summarise

from opengrad.data.classifier_input import CONTRACT_V2_VERSION, ClassifierFeatures
from opengrad.data.decision_classifier_v2 import ABSTAIN, CLASSIFIER_VERSION, classify

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path("reports/prose-classifier/dev-v2")
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier_v2.py")
#: set name -> (population, exported model labels). v1 sets keep their v1 names.
SETS: dict[str, tuple[Path, Path]] = {
    "dev-v1": V1_SETS["dev"][:2],
    "devcheck-v1": V1_SETS["devcheck"][:2],
    "devcheck-v2": V1_SETS["devcheck-v2"][:2],
    "dev-v2": (
        Path("reports/prose-classifier/dev-v2/prose-classifier-dev-v2.population.jsonl"),
        Path("reports/prose-classifier/dev-v2/annotation/wip/prose-classifier-dev-v2.annotations.model.claude-opus-5.model-dev-v2.jsonl"),
    ),
    "v2-check-1": (
        Path("reports/prose-classifier/dev-v2/prose-classifier-v2-devcheck-1.population.jsonl"),
        Path(
            "reports/prose-classifier/v2-devcheck-1/annotation/wip/"
            "prose-classifier-v2-devcheck-1.annotations.model.claude-opus-5.model-v2-devcheck-1.jsonl"
        ),
    ),
    "v2-check-2": (
        Path("reports/prose-classifier/dev-v2/prose-classifier-v2-devcheck-2.population.jsonl"),
        Path(
            "reports/prose-classifier/v2-devcheck-2/annotation/wip/"
            "prose-classifier-v2-devcheck-2.annotations.model.claude-opus-5.model-v2-devcheck-2.jsonl"
        ),
    ),
    "v2-check-3": (
        Path("reports/prose-classifier/dev-v2/prose-classifier-v2-devcheck-3.population.jsonl"),
        Path(
            "reports/prose-classifier/v2-devcheck-3/annotation/wip/"
            "prose-classifier-v2-devcheck-3.annotations.model.claude-opus-5.model-v2-devcheck-3.jsonl"
        ),
    ),
    "v2-check-4": (
        Path("reports/prose-classifier/dev-v2/prose-classifier-v2-devcheck-4.population.jsonl"),
        Path(
            "reports/prose-classifier/v2-devcheck-4/annotation/wip/"
            "prose-classifier-v2-devcheck-4.annotations.model.claude-opus-5.model-v2-devcheck-4.jsonl"
        ),
    ),
}
#: v2 check sets whose items the developer has not read (37 §4); never printed.
UNEXPOSED: frozenset[str] = frozenset()  # v2-check-1..4 exposed for rounds 2-5 (37 §7)


def rows(root: Path, population: Path, labels_file: Path) -> list[dict[str, Any]]:
    items = {item["dev_id"]: item for item in map(json.loads, (root / population).read_text(encoding="utf-8").splitlines())}
    out = []
    for record in (json.loads(line) for line in (root / labels_file).read_text(encoding="utf-8").splitlines() if line):
        item = items[record["dev_id"]]
        decision = classify(
            ClassifierFeatures(
                user_message=item["user_message"],
                assistant_response=item["assistant_response"],
                tools=tuple(item["tools"]),
                structured_call_present=False,
            )
        )
        out.append(
            {
                "dev_index": item["dev_index"],
                "model_label": record["gold_policy_label"],
                "ambiguity_status": record["ambiguity_status"],
                "prediction": decision.label,
                "step": decision.step,
                "evidence": decision.evidence,
                "unit_kind": item.get("unit_kind", "single_exchange"),
                "item": item,
            }
        )
    return sorted(out, key=lambda row: row["dev_index"])


def evaluate(root: Path, name: str) -> dict[str, Any]:
    population, labels_file = SETS[name]
    results = rows(root, population, labels_file)
    summary = summarise(results)
    by_kind = {kind: summarise([r for r in results if r["unit_kind"] == kind]) for kind in sorted({r["unit_kind"] for r in results})}
    return {
        "population": {"path": population.as_posix(), "sha256": sha256(root / population)},
        "labels": {"path": labels_file.as_posix(), "sha256": sha256(root / labels_file)},
        "summary": summary,
        "by_unit_kind": by_kind if len(by_kind) > 1 else None,
        "_rows": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--set", choices=[*sorted(SETS), "all"], default="dev-v2")
    parser.add_argument("--show", type=int, default=0, help="print up to N disagreements (development items only)")
    parser.add_argument("--round", type=int, help="development round, recorded in the report name")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    # Item text holds characters (e.g. "≈") a Windows console code page cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = sorted(SETS) if args.set == "all" else [args.set]
    if args.show and any(name in UNEXPOSED for name in names):
        parser.error("items of an unexposed check set are never printed (37 §4)")
    reports = {name: evaluate(args.root, name) for name in names}
    for name, report in reports.items():
        s = report["summary"]
        print(f"{name:12s} agree {s['agreement_on_mode_labels']}/{s['items_with_a_mode_label']} ({s['agreement_rate_on_mode_labels']})")
        if report["by_unit_kind"]:
            for kind, k in report["by_unit_kind"].items():
                print(f"  {kind:17s} agree {k['agreement_on_mode_labels']}/{k['items_with_a_mode_label']}")
        if args.set != "all":
            print(json.dumps(s, indent=2))
    shown = 0
    for name in names:
        for row in reports[name]["_rows"]:
            if shown >= args.show:
                break
            if row["prediction"] == row["model_label"] or (row["model_label"] == UNKNOWN and row["prediction"] == ABSTAIN):
                continue
            shown += 1
            item = row["item"]
            print(
                f"\n{name} #{row['dev_index'] + 1} [{row['unit_kind']}] model={row['model_label']}/{row['ambiguity_status']} "
                f"pred={row['prediction']} step={row['step']} evidence={list(row['evidence'])} tools={[t.get('name') for t in item['tools']]}"
            )
            print("  U:", " ".join(item["user_message"].split())[:300])
            print("  A:", " ".join(item["assistant_response"].split())[:600])
    if args.write:
        suffix = f".round{args.round}" if args.round is not None else ""
        source_hash = hashlib.sha256((args.root / CLASSIFIER_MODULE).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name, report in reports.items():
            body = {
                "artifact_kind": "PROSE_CLASSIFIER_V2_DEVELOPMENT_AGREEMENT",
                "statement": (
                    "Agreement of prose-decision-classifier-v2 with Claude model judgments on development data (37 §3). "
                    "In-sample where the developer has read the items; agreement with a model, not accuracy."
                ),
                "classifier_version": CLASSIFIER_VERSION,
                "classifier_source_sha256_lf": source_hash,
                "round": args.round,
                "input_contract": CONTRACT_V2_VERSION,
                **{key: value for key, value in report.items() if key != "_rows"},
            }
            path = args.root / REPORT_DIR / f"{CLASSIFIER_VERSION}{suffix}.{name}-agreement.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            print(f"wrote {path.relative_to(args.root).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
