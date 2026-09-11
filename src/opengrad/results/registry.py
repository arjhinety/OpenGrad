"""Materialized experiment index for discovery, summaries and reporting.

What this module is
-------------------
``results/registry.jsonl`` is a **derived projection**, never a source of truth. It exists so
experiment discovery, summarisation, comparison and reporting do not have to walk the
filesystem and re-parse every artifact. It can be deleted at any time and regenerated
deterministically:

    opengrad results rebuild-registry

Authoritative state, in this order:

1. ``runs/<experiment_id>/experiment.json`` -- identity, configuration, dataset hashes,
   model/tokenizer revisions, environment, lifecycle status. Owned by ``ExperimentStore``.
2. ``runs/<experiment_id>/eval/`` -- per-checkpoint metrics, predictions and benchmark
   artifacts.
3. ``runs/central_ledger.jsonl`` -- append-only lifecycle transitions.

This module reads all three and writes none of them. It must never become a fourth place that
owns experiment state: if a value cannot be read from an authoritative artifact it is reported
as absent rather than computed, defaulted or invented.

Two deliberate refusals
-----------------------
* **No "best checkpoint".** ``docs/CHECKPOINTS.md`` states that the newest checkpoint is never
  automatically "best" and that "best" requires a formal promotion policy. Selecting a maximum
  over ``call_f1`` here would reintroduce exactly the metric-only gate the promotion policy work
  removed. The index therefore records the **promoted** checkpoint when the ledger says one was
  promoted, otherwise the **latest evaluated** checkpoint, and names which rule applied in
  ``headline_checkpoint_selection``.
* **No invented suite names.** The evaluation artifacts record the held-out *manifest* and a
  ``kind``, not a suite label, so the index reports those real identifiers (``eval_manifests``,
  ``eval_kinds``) rather than mapping them onto a suite name they never carried.

Determinism
-----------
One row per experiment, ordered by ``experiment_id``, with sorted keys and values rounded to six
decimals. Rebuilding over unchanged authoritative state produces byte-identical output. Rows are
never appended blindly: a rebuild replaces the file atomically via ``registry.jsonl.tmp`` and
``os.replace``, so an interrupted build cannot leave a half-written projection.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.experiments.ledger import ExperimentLedger

REGISTRY_RELATIVE_PATH = "results/registry.jsonl"
SCHEMA_VERSION = 1
METRICS_PRECISION = 6

# Checkpoint-selection rules, recorded on every row so the choice is never implicit.
SELECTION_PROMOTED = "PROMOTED_CHECKPOINT"
SELECTION_LATEST = "LATEST_EVALUATED_CHECKPOINT"
SELECTION_NONE = "NO_EVALUATED_CHECKPOINT"


@dataclass(frozen=True)
class RegistryFinding:
    """A single registry problem, reported rather than repaired."""

    kind: str  # "DRIFT" (registry disagrees) or "INTEGRITY" (authoritative state has a gap)
    code: str
    detail: str
    experiment_id: str | None = None
    field_name: str | None = None
    registry_value: Any = None
    authoritative_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "code": self.code,
            "experiment_id": self.experiment_id,
            "field": self.field_name,
            "registry": self.registry_value,
            "authoritative": self.authoritative_value,
            "detail": self.detail,
        }

    def render(self) -> str:
        scope = f"experiment_id: {self.experiment_id}" if self.experiment_id else "repository"
        lines = [f"{self.code} ({self.kind})", f"  {scope}"]
        if self.field_name:
            lines.append(f"  field: {self.field_name}")
        if self.registry_value is not None or self.authoritative_value is not None:
            lines.append(f"  registry:      {self.registry_value}")
            lines.append(f"  authoritative: {self.authoritative_value}")
        lines.append(f"  {self.detail}")
        return "\n".join(lines)


def _round(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, METRICS_PRECISION)
    return value


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce an artifact field to a mapping without inventing a value for it."""
    return value if isinstance(value, dict) else {}


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# --- authoritative discovery --------------------------------------------------------


def _discover_records(root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    """Every ``experiment.json`` under ``runs/``, as (id, path, raw record).

    Experiment IDs may contain slashes (the baseline namespace does), so nested record files
    are walked rather than only top-level ones.
    """
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    found: list[tuple[str, Path, dict[str, Any]]] = []
    for exp_file in sorted(runs_dir.rglob("experiment.json")):
        if not exp_file.is_file():
            continue
        raw = _read_json(exp_file)
        if raw is None:
            continue
        experiment_id = str(raw.get("experiment_id") or "")
        found.append((experiment_id, exp_file, raw))
    return sorted(found, key=lambda item: item[0])


def _ledger_events(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Central ledger events grouped by experiment id."""
    ledger = ExperimentLedger(root / "runs" / "central_ledger.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in ledger.read_events():
        grouped.setdefault(event.experiment_id, []).append(event.to_dict())
    return grouped


def _latest_timestamp(values: list[Any]) -> str | None:
    stamps = [str(value) for value in values if isinstance(value, str) and value]
    return max(stamps) if stamps else None


def _checkpoint_summaries(root: Path, experiment_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluated checkpoints read from the authoritative per-checkpoint metrics artifacts.

    Returns the summaries and the paths of any eval artifacts that could not be read. Nothing is
    recomputed: the metrics are the ``candidate`` values the evaluation wrote, and the artifact
    path is recorded so a reader can re-derive them.
    """
    eval_dir = root / "runs" / experiment_id / "eval"
    if not eval_dir.is_dir():
        return [], []
    summaries: list[dict[str, Any]] = []
    unreadable: list[str] = []
    # Measurements are namespaced by evaluation partition (`eval/dev/`, `eval/confirmatory/`) so a
    # checkpoint scored on one cannot overwrite its score on the other. The un-namespaced
    # `eval/checkpoint-N/` layout is still read, because every run recorded before the partition
    # existed uses it.
    candidates = list(eval_dir.glob("checkpoint-*/metrics.json")) + list(
        eval_dir.glob("*/checkpoint-*/metrics.json")
    )
    for metrics_path in sorted(candidates):
        payload = _read_json(metrics_path)
        if payload is None:
            unreadable.append(_relative(root, metrics_path))
            continue
        # Which partition a measurement belongs to, so a DEV-selected score is never mistaken for
        # a confirmatory one in the index. `eval/<partition>/checkpoint-N/` yields the partition
        # name; `eval/checkpoint-N/` yields the empty string.
        partition = (
            "" if metrics_path.parent.parent == eval_dir else metrics_path.parent.parent.name
        )
        lineage = _as_dict(payload.get("lineage"))
        comparison = _as_dict(payload.get("baseline_comparison"))
        metrics_block = _as_dict(comparison.get("metrics"))
        metrics: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for name, entry in metrics_block.items():
            entry_map = _as_dict(entry)
            candidate = entry_map.get("candidate")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                metrics[str(name)] = _round(candidate)
            delta = entry_map.get("delta")
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                deltas[str(name)] = _round(delta)
        step = lineage.get("checkpoint_step")
        if not isinstance(step, int):
            suffix = metrics_path.parent.name.rsplit("-", 1)[-1]
            step = int(suffix) if suffix.isdigit() else -1
        summaries.append(
            {
                "checkpoint_id": str(lineage.get("checkpoint_id") or metrics_path.parent.name),
                "checkpoint_step": step,
                "records": payload.get("records"),
                "parse_valid_rate": _round(payload.get("parse_valid_rate")),
                "metrics_artifact": _relative(root, metrics_path),
                "eval_kind": payload.get("kind"),
                "eval_partition": partition or None,
                "eval_manifest": _relative(root, root / str(payload["manifest"]))
                if payload.get("manifest")
                else None,
                "metrics": dict(sorted(metrics.items())),
                "deltas": dict(sorted(deltas.items())),
            }
        )
    return sorted(
        summaries,
        # The partition is part of the key so that two measurements of the same step order
        # deterministically instead of depending on filesystem iteration order.
        key=lambda item: (
            int(item["checkpoint_step"]),
            str(item["checkpoint_id"]),
            str(item["eval_partition"] or ""),
        ),
    ), unreadable


def _promotion_state(
    events: list[dict[str, Any]],
) -> tuple[str | None, list[str], str | None]:
    """Promoted checkpoint, rejected checkpoints, and the note, from the authoritative ledger."""
    promoted: str | None = None
    promoted_at: str | None = None
    rejected: list[str] = []
    for event in events:
        event_type = str(event.get("event_type", "")).upper()
        details = _as_dict(event.get("details"))
        checkpoint = details.get("checkpoint")
        if event_type in {"PROMOTED", "CHECKPOINT_PROMOTED"} and isinstance(checkpoint, str):
            stamp = str(event.get("timestamp", ""))
            if promoted is None or stamp >= (promoted_at or ""):
                promoted, promoted_at = checkpoint, stamp
        elif event_type in {"REJECTED", "CHECKPOINT_REJECTED"} and isinstance(checkpoint, str):
            rejected.append(checkpoint)
    return promoted, sorted(set(rejected)), promoted_at


@dataclass
class RegistryRow:
    """One normalized experiment projection."""

    experiment_id: str
    status: str
    training_algorithm: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    parent_experiment_id: str | None
    random_seed: int
    git_commit: str
    git_dirty: bool
    created_at: str | None
    updated_at: str | None
    dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    dataset_manifest_ids: list[str] = field(default_factory=list)
    training_config_fingerprint: str = ""
    evaluated_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    headline_checkpoint: str | None = None
    headline_checkpoint_selection: str = SELECTION_NONE
    headline_metrics: dict[str, float] = field(default_factory=dict)
    eval_manifests: list[str] = field(default_factory=list)
    eval_kinds: list[str] = field(default_factory=list)
    promotion_checkpoint: str | None = None
    promotion_decision: str | None = None
    rejected_checkpoints: list[str] = field(default_factory=list)
    ledger_event_count: int = 0
    run_path: str = ""
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "training_algorithm": self.training_algorithm,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "parent_experiment_id": self.parent_experiment_id,
            "random_seed": self.random_seed,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dataset_fingerprints": dict(sorted(self.dataset_fingerprints.items())),
            "dataset_manifest_ids": sorted(self.dataset_manifest_ids),
            "training_config_fingerprint": self.training_config_fingerprint,
            "evaluated_checkpoint_count": len(self.evaluated_checkpoints),
            "evaluated_checkpoints": self.evaluated_checkpoints,
            "headline_checkpoint": self.headline_checkpoint,
            "headline_checkpoint_selection": self.headline_checkpoint_selection,
            "headline_metrics": dict(sorted(self.headline_metrics.items())),
            "eval_manifests": sorted(self.eval_manifests),
            "eval_kinds": sorted(self.eval_kinds),
            "promotion_checkpoint": self.promotion_checkpoint,
            "promotion_decision": self.promotion_decision,
            "rejected_checkpoints": sorted(self.rejected_checkpoints),
            "ledger_event_count": self.ledger_event_count,
            "run_path": self.run_path,
            "provenance": dict(sorted(self.provenance.items())),
        }


def _select_headline(
    checkpoints: list[dict[str, Any]], promoted: str | None
) -> tuple[dict[str, Any] | None, str]:
    if promoted:
        for summary in checkpoints:
            if summary["checkpoint_id"] == promoted:
                return summary, SELECTION_PROMOTED
    if checkpoints:
        return checkpoints[-1], SELECTION_LATEST
    return None, SELECTION_NONE


def build_row(
    root: Path,
    experiment_id: str,
    exp_file: Path,
    record: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[RegistryRow, list[RegistryFinding]]:
    """Project one authoritative experiment record into an index row."""
    findings: list[RegistryFinding] = []
    run_path = f"runs/{experiment_id}"
    actual_path = _relative(root, exp_file.parent)
    if actual_path != run_path:
        findings.append(
            RegistryFinding(
                kind="INTEGRITY",
                code="RUN_PATH_MISMATCH",
                experiment_id=experiment_id,
                field_name="run_path",
                registry_value=run_path,
                authoritative_value=actual_path,
                detail=(
                    "the record's experiment_id does not match its directory, so "
                    "ExperimentStore.run_dir() will not resolve it"
                ),
            )
        )
        run_path = actual_path

    checkpoints, unreadable = _checkpoint_summaries(root, experiment_id)
    for path in unreadable:
        findings.append(
            RegistryFinding(
                kind="INTEGRITY",
                code="EVAL_ARTIFACT_UNREADABLE",
                experiment_id=experiment_id,
                field_name="evaluated_checkpoints",
                authoritative_value=path,
                detail="evaluation metrics artifact exists but could not be parsed",
            )
        )

    # A curve point with no metrics artifact behind it is a gap in the authoritative state,
    # not a registry defect, so it is reported as integrity rather than drift.
    curve = _read_json(root / "runs" / experiment_id / "eval" / "curve.json")
    if curve is not None:
        present = {summary["checkpoint_step"] for summary in checkpoints}
        for point in curve.get("points") or []:
            if not isinstance(point, dict):
                continue
            step = point.get("checkpoint_step")
            if isinstance(step, int) and step not in present:
                findings.append(
                    RegistryFinding(
                        kind="INTEGRITY",
                        code="CURVE_POINT_WITHOUT_METRICS",
                        experiment_id=experiment_id,
                        field_name="evaluated_checkpoints",
                        authoritative_value=step,
                        detail=(
                            f"eval/curve.json records step {step} but no "
                            f"eval/checkpoint-{step}/metrics.json exists to support it"
                        ),
                    )
                )

    promoted, rejected, _promoted_at = _promotion_state(events)
    headline, selection = _select_headline(checkpoints, promoted)

    # A record may legitimately reach a terminal evaluation status through artifacts held
    # elsewhere (the B0 baseline keeps its metrics under reports/baselines/, by contract), so a
    # missing eval directory is reported rather than treated as a registry defect.
    terminal_evaluation_status = {"EVALUATED", "PROMOTED", "REJECTED"}
    if str(record.get("status", "")) in terminal_evaluation_status and not checkpoints:
        findings.append(
            RegistryFinding(
                kind="INTEGRITY",
                code="STATUS_WITHOUT_EVAL_ARTIFACTS",
                experiment_id=experiment_id,
                field_name="evaluated_checkpoints",
                authoritative_value=str(record.get("status", "")),
                detail=(
                    f"record status is {record.get('status')} but {run_path}/eval holds no "
                    "per-checkpoint metrics; any measurement behind that status lives outside "
                    "the canonical eval location"
                ),
            )
        )

    return (
        RegistryRow(
            experiment_id=experiment_id,
            status=str(record.get("status", "")),
            training_algorithm=str(record.get("training_algorithm", "")),
            model_id=str(record.get("model_id", "")),
            model_revision=str(record.get("model_revision", "")),
            tokenizer_revision=str(record.get("tokenizer_revision", "")),
            parent_experiment_id=record.get("parent_experiment_id"),
            random_seed=int(record.get("random_seed") or 0),
            git_commit=str(record.get("git_commit", "unknown")),
            git_dirty=bool(record.get("git_dirty", False)),
            created_at=record.get("launch_timestamp"),
            updated_at=_latest_timestamp(
                [
                    record.get("launch_timestamp"),
                    record.get("completion_timestamp"),
                    *[event.get("timestamp") for event in events],
                ]
            ),
            dataset_fingerprints={
                str(key): str(value) for key, value in (record.get("dataset_hashes") or {}).items()
            },
            dataset_manifest_ids=list(record.get("dataset_manifest_ids") or []),
            training_config_fingerprint=_fingerprint(record.get("training_config") or {}),
            evaluated_checkpoints=checkpoints,
            headline_checkpoint=headline["checkpoint_id"] if headline else None,
            headline_checkpoint_selection=selection,
            headline_metrics=dict(headline["metrics"]) if headline else {},
            eval_manifests=sorted(
                {c["eval_manifest"] for c in checkpoints if c.get("eval_manifest")}
            ),
            eval_kinds=sorted({c["eval_kind"] for c in checkpoints if c.get("eval_kind")}),
            promotion_checkpoint=promoted,
            promotion_decision=record.get("promotion_decision"),
            rejected_checkpoints=rejected,
            ledger_event_count=len(events),
            run_path=run_path,
            provenance={
                "experiment": _relative(root, exp_file),
                "eval": f"{run_path}/eval",
                "ledger": "runs/central_ledger.jsonl",
            },
        ),
        findings,
    )


def build_registry(root: Path) -> tuple[list[RegistryRow], list[RegistryFinding]]:
    """Read authoritative state and project it. Writes nothing."""
    events_by_experiment = _ledger_events(root)
    rows: list[RegistryRow] = []
    findings: list[RegistryFinding] = []
    for experiment_id, exp_file, record in _discover_records(root):
        row, row_findings = build_row(
            root, experiment_id, exp_file, record, events_by_experiment.get(experiment_id, [])
        )
        rows.append(row)
        findings.extend(row_findings)
    # Canonical ordering; duplicate ids cannot survive because discovery is keyed by record.
    rows.sort(key=lambda row: row.experiment_id)
    return rows, findings


# --- materialization ---------------------------------------------------------------


def registry_path(root: Path) -> Path:
    return root / REGISTRY_RELATIVE_PATH


def write_registry(root: Path, rows: list[RegistryRow]) -> Path:
    """Atomically replace the materialized index. Never appends."""
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    payload = "".join(
        json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    return path


def rebuild_registry(root: Path) -> dict[str, Any]:
    """Deterministically regenerate the index from authoritative state."""
    rows, findings = build_registry(root)
    path = write_registry(root, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "registry": _relative(root, path),
        "experiments": len(rows),
        "rows_written": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "with_eval_artifacts": sum(1 for row in rows if row.evaluated_checkpoints),
        "integrity_findings": [finding.to_dict() for finding in findings],
    }


def registry_exists(root: Path) -> bool:
    return registry_path(root).is_file()


def load_registry(root: Path) -> list[dict[str, Any]]:
    """Read the materialized index. Malformed lines are skipped, not repaired."""
    path = registry_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


# --- drift detection ---------------------------------------------------------------


def _compare_rows(
    expected: dict[str, Any], actual: dict[str, Any], experiment_id: str
) -> list[RegistryFinding]:
    findings: list[RegistryFinding] = []
    for key in sorted(set(expected) | set(actual)):
        want, got = expected.get(key), actual.get(key)
        if want != got:
            findings.append(
                RegistryFinding(
                    kind="DRIFT",
                    code="FIELD_MISMATCH",
                    experiment_id=experiment_id,
                    field_name=key,
                    registry_value=_short(got),
                    authoritative_value=_short(want),
                    detail="registry disagrees with the projection of authoritative state",
                )
            )
    return findings


def _short(value: Any, limit: int = 220) -> Any:
    if isinstance(value, (list, dict)) and len(json.dumps(value, default=str)) > limit:
        return json.dumps(value, sort_keys=True, default=str)[:limit] + "…"
    return value


def validate_registry(root: Path) -> list[RegistryFinding]:
    """Compare the materialized index against authoritative state. Repairs nothing."""
    findings: list[RegistryFinding] = []
    rows, integrity = build_registry(root)
    findings.extend(integrity)
    expected = {row.experiment_id: row.to_dict() for row in rows}

    path = registry_path(root)
    if not path.is_file():
        findings.append(
            RegistryFinding(
                kind="DRIFT",
                code="REGISTRY_MISSING",
                field_name="registry",
                authoritative_value=REGISTRY_RELATIVE_PATH,
                detail=(
                    f"{len(rows)} experiment(s) exist but {REGISTRY_RELATIVE_PATH} does not; "
                    "run `opengrad results rebuild-registry`"
                ),
            )
        )
        return findings

    raw = path.read_text(encoding="utf-8")
    actual: dict[str, dict[str, Any]] = {}
    malformed = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(payload, dict):
            malformed += 1
            continue
        experiment_id = str(payload.get("experiment_id") or "")
        if not experiment_id:
            malformed += 1
            continue
        if experiment_id in actual:
            findings.append(
                RegistryFinding(
                    kind="DRIFT",
                    code="DUPLICATE_EXPERIMENT_ID",
                    experiment_id=experiment_id,
                    detail="the materialized registry contains more than one row for this id",
                )
            )
            continue
        actual[experiment_id] = payload

    if malformed:
        findings.append(
            RegistryFinding(
                kind="DRIFT",
                code="MALFORMED_REGISTRY_ROW",
                field_name="registry",
                registry_value=malformed,
                detail="registry lines that are not readable experiment objects",
            )
        )

    # The bootstrap placeholder is zero bytes; that is only acceptable when it is also correct.
    if not actual and expected and not raw.strip():
        findings.append(
            RegistryFinding(
                kind="DRIFT",
                code="REGISTRY_NOT_BUILT",
                field_name="registry",
                registry_value=0,
                authoritative_value=len(expected),
                detail=(
                    f"{REGISTRY_RELATIVE_PATH} is empty while {len(expected)} experiment(s) "
                    "exist; run `opengrad results rebuild-registry`"
                ),
            )
        )

    for experiment_id in sorted(set(expected) - set(actual)):
        findings.append(
            RegistryFinding(
                kind="DRIFT",
                code="EXPERIMENT_MISSING_FROM_REGISTRY",
                experiment_id=experiment_id,
                detail="authoritative experiment has no row in the materialized registry",
            )
        )
    for experiment_id in sorted(set(actual) - set(expected)):
        findings.append(
            RegistryFinding(
                kind="DRIFT",
                code="REGISTRY_ROW_WITHOUT_EXPERIMENT",
                experiment_id=experiment_id,
                detail="registry row has no authoritative experiment record",
            )
        )
    for experiment_id in sorted(set(actual) & set(expected)):
        findings.extend(
            _compare_rows(expected[experiment_id], actual[experiment_id], experiment_id)
        )

    # Provenance paths must resolve from the repository root.
    for experiment_id, payload in sorted(actual.items()):
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            continue
        for name, value in sorted(provenance.items()):
            if isinstance(value, str) and not (root / value).exists():
                findings.append(
                    RegistryFinding(
                        kind="DRIFT",
                        code="PROVENANCE_PATH_UNRESOLVED",
                        experiment_id=experiment_id,
                        field_name=f"provenance.{name}",
                        registry_value=value,
                        detail="registry provenance points at a path that does not exist",
                    )
                )
    return findings


def render_findings(findings: list[RegistryFinding]) -> str:
    if not findings:
        return "registry validation: OK (matches authoritative state)"
    drift = [f for f in findings if f.kind == "DRIFT"]
    integrity = [f for f in findings if f.kind == "INTEGRITY"]
    lines = [f"registry validation: {len(findings)} finding(s)"]
    if drift:
        lines.append("")
        lines.append(f"DRIFT — registry disagrees with authoritative state ({len(drift)}):")
        lines.extend(finding.render() for finding in drift)
    if integrity:
        lines.append("")
        lines.append(
            f"INTEGRITY — authoritative state has gaps, registry faithfully mirrors them "
            f"({len(integrity)}):"
        )
        lines.extend(finding.render() for finding in integrity)
    return "\n".join(lines)


def render_registry_table(rows: list[dict[str, Any]]) -> str:
    header = f"{'experiment_id':<34}{'status':<10}{'algo':<8}{'evals':>6}{'calls':>7}  headline"
    lines = [header, "-" * len(header)]
    for row in rows:
        headline = row.get("headline_checkpoint") or "—"
        if row.get("headline_checkpoint_selection") == SELECTION_PROMOTED:
            headline = f"{headline} (promoted)"
        lines.append(
            f"{row.get('experiment_id')!s:<34}{row.get('status')!s:<10}"
            f"{row.get('training_algorithm')!s:<8}"
            f"{int(row.get('evaluated_checkpoint_count') or 0):>6}"
            f"{int(row.get('ledger_event_count') or 0):>7}  {headline}"
        )
    if not rows:
        lines.append("(no experiments recorded)")
    return "\n".join(lines)
