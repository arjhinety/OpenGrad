#!/usr/bin/env python3
"""Generate the machine-checkable experiment status view at ``docs/EXPERIMENT_STATUS.md``.

Why this exists
---------------
The README, the roadmap, the reports and ``results/registry.jsonl`` can drift apart because each
is edited independently. This script derives the status table from the authoritative artifacts
only -- ``runs/<id>/experiment.json``, the per-run and central ledgers, the checkpoint registry,
the per-checkpoint evaluation metrics, and ``reports/releases/*.json`` -- so the document is a
view of state rather than a fifth place that owns it.

Nothing here is a source of truth and nothing is invented: a field with no authoritative artifact
is rendered as an em dash. Output is deterministic (no wall-clock fields), so
``tests/results/test_state_consistency.py`` can regenerate it and require a byte-identical match;
that is what turns the document into a drift detector.

Counting rules (they must match the README's prose)
---------------------------------------------------
* ``validity`` comes from the record's own metadata annotation, so a scaffold-era or mock record
  cannot be silently counted as a real result.
* An *intervention* is a distinct trained arm with real evidence: algorithm ``sft``/``dpo``,
  no ``SCAFFOLD_ONLY``/``MOCK_ONLY`` validity, and not a repeat of another identity.
* A *promoted model* must be ``PROMOTED`` and valid; the scaffold-era promotion is excluded.

Usage:
    python scripts/reporting/generate_experiment_status.py            # write the document
    python scripts/reporting/generate_experiment_status.py --check     # fail if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/EXPERIMENT_STATUS.md"

# Machine-written release records from before per-experiment identity existed carry only a
# repository name. This mapping is pinned by tests/results/test_state_consistency.py so renaming
# or republishing a repository cannot silently mis-attribute a published artifact.
REPOSITORY_EXPERIMENTS = {
    "arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final": "m0_sft_canonical_v2_final",
    "arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CorpusV2": "qwen35_2b_m0_sft_v2corpus",
    "arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute": (
        "m0_v2_final_minus_xlam_fixed_compute"
    ),
    "arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure": (
        "m0_v2_final_minus_xlam_matched_exposure"
    ),
}

# Written analyses a reader should be able to reach from the status row. The run directory link is
# emitted for every row regardless; these are the curated narrative reports.
REPORTS = {
    "tool_calling/qwen35_2b/baseline": ["reports/baselines/qwen35_2b_baseline/RESULT.md"],
    "qwen35_2b_m0_sft_full_v3": ["reports/M0_SFT_EXECUTION_REPORT.md"],
    "qwen35_2b_m1_dpo_v1": ["reports/M0_SFT_EXECUTION_REPORT.md"],
    "qwen35_2b_m0_sft_v2corpus": ["reports/M0_SFT_EXECUTION_REPORT.md"],
    "m0_sft_canonical_v2_final": [
        "reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md",
        "reports/M0_CANONICAL_V2_FINAL_EVALUATION.md",
    ],
    "m0_v2_final_minus_xlam_fixed_compute": [
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md",
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md",
    ],
    "m0_v2_final_minus_xlam_matched_exposure": [
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md",
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md",
    ],
    "m1_dpo_canonical_v2_final": ["reports/M1_DPO_EXECUTION_REPORT.md"],
    "m1_dpo_canonical_v2_final_v2": [
        "reports/M1_DPO_EXECUTION_REPORT.md",
        "reports/M1_DPO_EVALUATION.md",
    ],
    "qwen35_2b_m2_distill": ["reports/M2_DECISION.md"],
}

INVALID_VALIDITIES = {"MOCK_ONLY", "SCAFFOLD_ONLY"}
NON_INTERVENTION_VALIDITIES = {"NON_REPRODUCIBLE_REPEAT"}
# A FAILED arm stopped before it produced an evaluated model, so it is execution history rather
# than a result: an intervention must have reached a terminal evidence state.
EVIDENCE_STATUSES = {"TRAINED", "REJECTED", "PROMOTED"}


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _discover_records() -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    for path in sorted((ROOT / "runs").rglob("experiment.json")):
        data = _read_json(path)
        if data:
            records.append((str(data.get("experiment_id", path.parent.name)), data))
    return records


def _confirmatory_metrics(experiment_id: str) -> dict[str, float]:
    """The pre-registered confirmatory score, which is the number the reports quote."""
    base = ROOT / "runs" / experiment_id / "eval" / "confirmatory"
    for metrics_path in sorted(base.glob("checkpoint-*/metrics.json")):
        metrics = _read_json(metrics_path)
        comparison = (metrics.get("baseline_comparison") or {}).get("metrics") or {}
        values = {
            name: entry.get("candidate")
            for name, entry in comparison.items()
            if isinstance(entry, dict) and isinstance(entry.get("candidate"), (int, float))
        }
        if values:
            return values
    return {}


def _repo_type(entry: dict, default: str = "model") -> str:
    """The Hub repository type a release entry declares; Hub URLs differ for datasets."""
    value = str(entry.get("repo_type") or default)
    return value if value in {"model", "dataset"} else default


def _published_map() -> dict[str, tuple[str, str]]:
    """Experiment id -> (published repository, Hub repo type), from the release records.

    A record naming the experiment or the repository directly always wins. The collection item
    lists in a publication record are consulted only for experiments no direct entry covers,
    because a collection membership says that a repository is published but not what it holds.
    """
    published: dict[str, tuple[str, str]] = {}
    from_collections: dict[str, tuple[str, str]] = {}
    for path in sorted((ROOT / "reports" / "releases").glob("*.json")):
        record = _read_json(path)
        experiment_id = record.get("experiment_id")
        repository = record.get("repository") or record.get("hub_repository")
        if experiment_id and repository:
            published[str(experiment_id)] = (str(repository), _repo_type(record))
        for model in record.get("models") or []:
            repository = model.get("repository")
            mapped = REPOSITORY_EXPERIMENTS.get(str(repository))
            if mapped:
                published[mapped] = (str(repository), _repo_type(model))
        model = record.get("model")
        if isinstance(model, dict):
            mapped = REPOSITORY_EXPERIMENTS.get(str(model.get("repository")))
            if mapped:
                published[mapped] = (str(model.get("repository")), _repo_type(model))
        collections = record.get("collections")
        if isinstance(collections, dict):
            for kind, repo_type in (("models", "model"), ("datasets", "dataset")):
                collection = collections.get(kind)
                items = collection.get("items") if isinstance(collection, dict) else None
                for repository in items or []:
                    mapped = REPOSITORY_EXPERIMENTS.get(str(repository))
                    if mapped:
                        from_collections.setdefault(mapped, (str(repository), repo_type))
    for experiment_id, entry in from_collections.items():
        published.setdefault(experiment_id, entry)
    return published


def _hub_url(repository: str, repo_type: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repository}"


def _baseline_metrics() -> dict[str, float]:
    metrics = _read_json(ROOT / "reports" / "baselines" / "qwen35_2b_baseline" / "metrics.json")
    routing = metrics.get("routing") or {}
    values: dict[str, float] = {}
    for name in ("call_f1", "call_recall"):
        entry = routing.get(name)
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)):
            values[name] = float(entry["value"])
        elif isinstance(entry, (int, float)):
            values[name] = float(entry)
    return values


def _cell(value: object | None) -> str:
    return "—" if value in (None, "") else str(value)


def _link(relative: str) -> str:
    return f"[`{relative}`](../{relative})"


def build_document() -> str:
    records = _discover_records()
    published = _published_map()
    baseline = _baseline_metrics()

    rows: list[dict[str, object]] = []
    for experiment_id, record in records:
        metadata = record.get("metadata") or {}
        validity = str(metadata.get("validity", ""))
        metrics = _confirmatory_metrics(experiment_id)
        if not metrics and experiment_id == "tool_calling/qwen35_2b/baseline":
            metrics = baseline
        promotion = ""
        if str(record.get("status")) == "PROMOTED":
            promotion = "PROMOTED"
        rows.append(
            {
                "experiment_id": experiment_id,
                "algorithm": str(record.get("training_algorithm", "")),
                "status": str(record.get("status", "")),
                "validity": validity,
                "datasets": ", ".join(record.get("dataset_manifest_ids") or []) or "—",
                "metrics": metrics,
                "promotion": promotion,
                "published": published.get(experiment_id),
                "reports": REPORTS.get(experiment_id, []),
                "launched": str(record.get("launch_timestamp", "")),
            }
        )

    rows.sort(key=lambda row: (str(row["launched"]), str(row["experiment_id"])))

    def is_valid(row: dict[str, object]) -> bool:
        return str(row["validity"]) not in INVALID_VALIDITIES

    interventions = [
        row
        for row in rows
        if is_valid(row)
        and str(row["algorithm"]) in {"sft", "dpo"}
        and str(row["status"]) in EVIDENCE_STATUSES
        and str(row["validity"]) not in NON_INTERVENTION_VALIDITIES
    ]
    promoted = [row for row in rows if row["promotion"] == "PROMOTED" and is_valid(row)]
    invalid = [row for row in rows if str(row["validity"]) == "MOCK_ONLY"]
    scaffold = [row for row in rows if str(row["validity"]) == "SCAFFOLD_ONLY"]

    lines: list[str] = [
        "# Experiment status",
        "",
        "Generated view over the authoritative artifacts — `runs/<id>/experiment.json`, the per-run",
        "and central ledgers, `runs/checkpoint_registry.json`, the per-checkpoint evaluation metrics,",
        "and `reports/releases/*.json`. It is a view, not a source of truth: no field is invented, an",
        "absent value renders as an em dash, and the file is regenerated deterministically by",
        "",
        "```text",
        "python scripts/reporting/generate_experiment_status.py",
        "```",
        "",
        "A byte-identical regeneration is required by `tests/results/test_state_consistency.py`, so a",
        "stale table fails the suite instead of drifting quietly. Prose lives in the",
        "[README](../README.md#latest-results); the numeric claims there and the counting convention",
        "below are checked against this table and the registry.",
        "",
        "## Counting convention",
        "",
        (
            f"- **Executed post-training interventions with real model evidence: "
            f"{len(interventions)}.**"
            " The B0 baseline is recorded separately and is not an intervention. An arm that only"
            " failed to launch (resource, registration, or horizon failure) is execution history,"
            " not an intervention."
        ),
        f"- **Promoted models: {len(promoted)}.** A promotion only counts when the record is valid, so",
        "  the scaffold-era promotion is excluded by construction.",
        f"- **Invalid/mock records: {len(invalid)}** — preserved as evidence, never counted as trained.",
        f"- **Scaffold-era records: {len(scaffold)}** — identity and lifecycle history only; no real",
        "  model evidence.",
        "- Negative and rejected results stay in the table: the record documents failures as well as",
        "  successes.",
        "",
        "`validity` is an annotation on the authoritative experiment record (`MOCK_ONLY`,",
        "`SCAFFOLD_ONLY`, `NON_REPRODUCIBLE_REPEAT`), appended to the ledger rather than overwriting",
        "history. `status` is the lifecycle state, including when that state is historical.",
        "",
        "## Records",
        "",
        "| Experiment | Method | Datasets | Status | Validity | Promotion | Confirmatory `call_f1` / `call_recall` | Reports | Published |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = row["metrics"] if isinstance(row["metrics"], dict) else {}
        call_f1 = metrics.get("call_f1")
        call_recall = metrics.get("call_recall")
        score = (
            f"{call_f1:.4f} / {call_recall:.4f}"
            if isinstance(call_f1, (int, float)) and isinstance(call_recall, (int, float))
            else "—"
        )
        reports = " · ".join(_link(path) for path in row["reports"]) or "—"
        published = row["published"]
        published_cell = (
            f"[`{published[0]}`]({_hub_url(*published)})" if isinstance(published, tuple) else "—"
        )
        lines.append(
            "| [`{experiment_id}`](../runs/{experiment_id}/) | {algorithm} | {datasets} | "
            "{status} | {validity} | {promotion} | {score} | {reports} | {published} |".format(
                experiment_id=row["experiment_id"],
                algorithm=_cell(row["algorithm"]),
                datasets=_cell(row["datasets"]),
                status=_cell(row["status"]),
                validity=_cell(row["validity"]),
                promotion=_cell(row["promotion"]),
                score=score,
                reports=reports,
                published=published_cell,
            )
        )

    lines.extend(
        [
            "",
            "Confirmatory scores are the pre-registered partition values quoted in the reports and",
            "the model cards; the baseline row shows its full-set score, and a row with no",
            "confirmatory artifact shows an em dash rather than a convenience number. Rows are",
            "ordered by launch timestamp so the chronology is stable.",
            "",
        ]
    )
    return "\n".join(lines)


def write_document() -> int:
    path = ROOT / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_document(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


def check_document() -> int:
    path = ROOT / OUT
    expected = build_document()
    actual = path.read_text(encoding="utf-8") if path.is_file() else ""
    if expected != actual:
        print(f"STALE: {OUT} does not match the authoritative artifacts", file=sys.stderr)
        print("run: python scripts/reporting/generate_experiment_status.py", file=sys.stderr)
        return 1
    print(f"CURRENT: {OUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the document is stale")
    args = parser.parse_args()
    return check_document() if args.check else write_document()


if __name__ == "__main__":
    raise SystemExit(main())
