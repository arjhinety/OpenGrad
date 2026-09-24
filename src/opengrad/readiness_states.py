"""Readiness gate states: one ``_<name>_state`` function per gate that needs more than a lookup.

Each returns ``(ok, detail, code)``; `opengrad.readiness.readiness` turns it into a gate. A new gate's
state function belongs here (the `opengrad-experiments-readiness` skill). Split out of `readiness.py` on
2026-09-24 with no change in behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opengrad.contamination.audit import (
    audit_path_for,
    benchmark_fingerprint,
    evaluate_audit,
    load_audit,
    load_quarantine,
    quarantine_path_for,
    training_corpus_fingerprint,
)
from opengrad.contamination.heldout import output_path_for as contamination_report_path
from opengrad.readiness_contracts import (
    DEFAULT_TRAINING_RELEASE_DIR,
    SOURCE_LABEL_ALIASES,
    _as_dict,
    _read_json,
    _resolve_project_path,
    _safe_item_count,
)
from opengrad.training.determinism import UNDECLARED, resolve_determinism
from opengrad.training.model_components import (
    MODEL_COMPONENT_POLICY_VERSION,
    ModelComponentError,
    resolve_component_settings,
)


def _materialized_split_state(root: Path, split: dict[str, Any]) -> tuple[bool, str, set[str]]:
    source = _resolve_project_path(root, split.get("source"), root / "__missing__")
    manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
    data = _read_json(manifest_path)
    if (
        not data
        or data.get("finalized") is not True
        or not isinstance(data.get("shards"), list)
        or not data["shards"]
    ):
        return (
            False,
            f"split {split.get('id')} materialization manifest is missing, unfinished, or empty",
            set(),
        )
    counts = _as_dict(data.get("counts"))
    written = _safe_item_count({"items": counts.get("written")})
    expected = _safe_item_count(split)
    if written != expected:
        return (
            False,
            f"split {split.get('id')} materialized count {written} does not match frozen count {expected}",
            set(),
        )
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    declared_hash = data.get("content_hash") or data.get("manifest_sha256")
    # A split content hash is a data-contract hash, not necessarily the JSON
    # manifest file hash. If the materializer declares one, it must match. When
    # it does not declare one, the frozen content hash cannot be verified.
    if declared_hash != split.get("content_hash"):
        return False, f"split {split.get('id')} content hash disagrees with materialization", set()
    shards = set()
    actual_rows = 0
    seen_ids: set[str] = set()
    content_digest = hashlib.sha256()
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        for name in data["shards"]:
            if not isinstance(name, str) or not name or name in shards:
                return False, f"split {split.get('id')} has invalid shard names", set()
            shard = source.parent / name if source.name == "manifest.json" else source / name
            if not shard.is_file():
                return False, f"split {split.get('id')} shard is missing: {name}", set()
            shards.add(name)
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    example_id = row.get("example_id") if isinstance(row, dict) else None
                    if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
                        return (
                            False,
                            f"split {split.get('id')} contains missing or duplicate example_id",
                            set(),
                        )
                    seen_ids.add(example_id)
                    content_digest.update(
                        json.dumps(
                            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                        + b"\n"
                    )
                    actual_rows += 1
    except Exception as exc:  # noqa: BLE001 - unreadable shards must fail the gate, never pass it
        return False, f"split {split.get('id')} shard content cannot be verified: {exc}", set()
    if actual_rows != written:
        return (
            False,
            f"split {split.get('id')} shard rows {actual_rows} do not match manifest count {written}",
            set(),
        )
    if content_digest.hexdigest() != split.get("content_hash"):
        return False, f"split {split.get('id')} content hash does not match shard rows", set()
    return (
        True,
        f"split {split.get('id')} is materialized ({written} records; manifest {actual_hash[:12]})",
        shards,
    )


def _contamination_report(root: Path, release_dir: Path | None = None) -> dict[str, Any] | None:
    """The scan report for this corpus.

    Corpus-scoped: a scan measures one benchmark against one training corpus, so reports for two
    corpora must not share a file. Reading the wrong one would evaluate a held-out set against a
    corpus the experiment is not training on.
    """
    return _read_json(contamination_report_path(root, release_dir))


def _corpus_source_labels(root: Path, release_dir: Path | None = None) -> set[str] | None:
    """The source labels a corpus actually contains, read from its release manifest.

    Derived rather than hardcoded. A fixed six-source set cannot describe a corpus built from a
    different number of sources, and the check that matters is not "were all six scanned" but
    "was every source this corpus contains scanned, and nothing else". ``None`` means the corpus
    could not be read, so the caller must not treat the check as satisfied.
    """
    base = (root / (release_dir or Path(DEFAULT_TRAINING_RELEASE_DIR))).resolve()
    manifest = _read_json(base / "release-manifest.json")
    if manifest is None:
        return None
    labels: set[str] = set()
    for source in manifest.get("sources") or []:
        if isinstance(source, dict) and source.get("source"):
            label = str(source["source"])
            labels.add(SOURCE_LABEL_ALIASES.get(label, label))
    return labels or None


def _contamination_state(
    root: Path,
    report: dict[str, Any] | None,
    release_dir: Path | None = None,
) -> tuple[bool, str]:
    """Combine machine-measured levels 1-4 with the human Level-5 adjudication artifact.

    The generated report is never trusted for Level 5: its verdict is recomputed from the
    durable audit artifact and the quarantine list, so a hand-edited report cannot pass the
    gate and a stale judgment cannot survive a change in the evidence.

    ``release_dir`` is the corpus the configuration will actually train on. It matters: the
    Level-5 verdicts bind to a training-corpus fingerprint, so evaluating them against a
    different corpus than the experiment uses would either pass a judgment that no longer
    applies or fail one that does. Before this parameter existed the comparison was hardcoded to
    the v1 release, which is exactly that mistake once a newer corpus exists.
    """
    if not report or report.get("manifest_id") != "behavioral-heldout-v2":
        return True, "contamination report is missing or bound to the wrong evaluation manifest"
    levels = _as_dict(report.get("levels"))
    required = {
        "1_exact_canonical_conversation_hash",
        "2_normalized_prompt_hash",
        "3_near_duplicate_ngram_minhash",
        "4_semantic_similarity",
        "5_manual_audit",
    }
    machine_levels = required - {"5_manual_audit"}
    level_ok = {"MEASURED", "CLEAN", "PASSED", "COMPLETE"}
    missing = sorted(required - set(levels))
    machine_pending = [
        name for name in sorted(machine_levels) if str(levels.get(name, "")).upper() not in level_ok
    ]
    sources = (
        {str(value) for value in report.get("training_sources_checked", [])}
        if isinstance(report.get("training_sources_checked"), list)
        else set()
    )
    expected_sources = _corpus_source_labels(root, release_dir)
    if expected_sources is None:
        source_ok = False
        source_detail = (
            "the corpus being evaluated could not be read, so the scan coverage cannot be checked"
        )
    else:
        source_ok = sources == expected_sources
        source_detail = f"scanned {sorted(sources)}; corpus contains {sorted(expected_sources)}" + (
            "" if source_ok else " -> MISMATCH"
        )

    audit = load_audit(audit_path_for(root, release_dir))
    quarantine = load_quarantine(quarantine_path_for(root, release_dir))
    evaluation = evaluate_audit(
        report,
        audit,
        quarantine,
        benchmark_fp=benchmark_fingerprint(root),
        training_fp=training_corpus_fingerprint(root, release_dir),
    )

    machine_ok = not missing and not machine_pending and source_ok
    blocked = not machine_ok or not evaluation.complete
    # The effective status comes from the audit evaluation, which owns the CLEAN vs
    # SEMANTIC_REVIEW_COMPLETE distinction: quarantining a contaminated example is a
    # completed review, not evidence of a clean corpus.
    status = "REVIEW_REQUIRED_LEVEL_5_PENDING" if blocked else evaluation.effective_status

    detail = (
        f"status={status}; missing_levels={missing or 'none'}; "
        f"pending_levels={machine_pending or 'none'}; sources_ok={source_ok}; "
        f"scan_coverage[{source_detail}]; "
        f"level_5={evaluation.level_5}; audit[{evaluation.detail()}]"
    )
    return blocked, detail


def _renderability_state(root: Path, raw: dict[str, Any]) -> tuple[bool, str, str | None]:
    """Evaluate a configuration's declared per-source trainability report.

    Canonical validity does not imply trainability: Canonical-v1 released 213,951 canonical
    records whose training boundary yielded 9 tool-call targets, and every canonical-count gate
    passed. A source that is supposed to teach function calling and yields none has collapsed,
    and that must block rather than contribute silently.

    The gate is dormant until a build declares `datasets.yield_report`, so it never fails a
    configuration that predates the measurement -- but once measured, a collapse blocks.
    """
    declared = (raw.get("datasets") or {}).get("yield_report")
    if not declared:
        return (
            True,
            "No yield report declared: per-source trainability is not measured for this config",
            None,
        )
    path = root / str(declared)
    if not path.is_file():
        return False, f"Declared yield report is missing: {declared}", "YIELD_REPORT_MISSING"
    payload = _read_json(path)
    if payload is None:
        return False, f"Declared yield report is unreadable: {declared}", "YIELD_REPORT_UNREADABLE"
    sources = payload.get("sources") or []
    collapsed = [str(item.get("source")) for item in sources if item.get("status") == "COLLAPSE"]
    anomalies = [str(item.get("source")) for item in sources if item.get("status") == "ANOMALY"]
    detail = (
        f"{len(sources)} sources measured from {declared}; "
        f"collapse={collapsed or 'none'}; anomaly={anomalies or 'none'}"
    )
    if collapsed:
        return False, detail, "DATASET_TRAINABILITY_COLLAPSE"
    return True, detail, None


#: The pre-training gate of `full-model-components-v1` (docs/MODEL_COMPONENT_POLICY.md §10).
MODEL_COMPONENTS_VALIDATION = Path("reports/training/model-components-validation.json")


REQUIRED_MODEL_COMPONENT_CHECKS = ("gpu_smoke_test", "gguf_export")


#: Study 001's training configs. They predate the determinism declaration Study 002 requires
#: (`14-HETEROGENEITY-POLICY.md`), are frozen and are not run again, so an undeclared mode is
#: recorded for them rather than blocking. Every other training config must declare one.
STUDY_001_TRAINING_CONFIGS = frozenset(
    f"configs/experiments/{name}.yaml"
    for name in (
        "m0_sft",
        "m0_sft_canonical_v2_final",
        "m0_v2_final_minus_xlam_fixed_compute",
        "m0_v2_final_minus_xlam_matched_exposure",
        "m0_v2_final_supervision_call_prediction_only",
        "m0_v2_final_supervision_complete_trajectory_only",
        "m1_dpo",
        "m1_dpo_canonical_v2_final",
        "m1_dpo_canonical_v2_final_v2",
        "qwen35_2b_m0_sft_full_v3",
        "qwen35_2b_m0_sft_micro",
        "qwen35_2b_m0_sft_v2corpus",
        "qwen35_2b_m1_dpo_v1",
        "qwen35_2b_m1_dpo_v1_restore",
    )
)


def _determinism_state(
    root: Path, config_path: Path, raw: dict[str, Any]
) -> tuple[bool, str, str | None]:
    """Whether the config declares its kernel-determinism mode (docs 05 and 14)."""
    try:
        mode = resolve_determinism(raw)
    except ValueError as exc:
        return False, str(exc), "DETERMINISM_INVALID"
    if mode != UNDECLARED:
        return True, f"reproducibility.determinism: {mode}", None
    try:
        relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = config_path.as_posix()
    if relative in STUDY_001_TRAINING_CONFIGS:
        return True, "Study 001 config, frozen: determinism recorded as UNDECLARED", None
    detail = (
        "reproducibility.determinism is not declared; set DECLARED_DETERMINISTIC or "
        "NON_DETERMINISTIC_KERNEL (14-HETEROGENEITY-POLICY.md)"
    )
    return False, detail, "DETERMINISM_UNDECLARED"


def _model_components_state(root: Path, raw: dict[str, Any]) -> tuple[bool, str, str | None]:
    """Block a real run that carries vision or MTP until the component path is validated.

    Vision and MTP training was implemented and tested on a tiny CPU model only. Before any real
    run depends on it, a GPU smoke test on the pinned model and a GGUF export that keeps the
    components must both be recorded as PASS, each pointing at evidence that exists. A text-only
    run (both components excluded, as every Study 002 arm is) never exercises that path and is
    exempt.
    """
    trainer = raw.get("trainer") or {}
    algorithm = str(trainer.get("type", "")).lower()
    try:
        settings = resolve_component_settings(trainer, algorithm=algorithm)
    except ModelComponentError as exc:
        return False, str(exc), "CONFIG_INVALID"
    if settings.vision == "exclude" and settings.mtp == "exclude":
        return (
            True,
            "Text-only run (vision and MTP excluded); component validation is not required",
            None,
        )
    carried = [name for name in ("vision", "mtp") if getattr(settings, name) == "include"]
    path = root / MODEL_COMPONENTS_VALIDATION
    payload = _read_json(path)
    if payload is None:
        return (
            False,
            f"Run carries {carried} but {MODEL_COMPONENTS_VALIDATION} is missing or unreadable",
            "MODEL_COMPONENTS_UNVALIDATED",
        )
    if payload.get("policy_version") != MODEL_COMPONENT_POLICY_VERSION:
        return (
            False,
            (
                f"{MODEL_COMPONENTS_VALIDATION} validates {payload.get('policy_version')!r}, "
                f"not the current {MODEL_COMPONENT_POLICY_VERSION!r}"
            ),
            "MODEL_COMPONENTS_UNVALIDATED",
        )
    checks = payload.get("checks") or {}
    pending: list[str] = []
    for name in REQUIRED_MODEL_COMPONENT_CHECKS:
        check = checks.get(name) or {}
        evidence = check.get("evidence")
        if check.get("status") != "PASS" or not evidence or not (root / str(evidence)).exists():
            pending.append(name)
    if pending:
        return (
            False,
            (
                f"Run carries {carried}; pre-training checks not passed with evidence: {pending} "
                f"({MODEL_COMPONENTS_VALIDATION}). Exclude both components for a text-only run."
            ),
            "MODEL_COMPONENTS_UNVALIDATED",
        )
    return True, f"Component path validated: {list(REQUIRED_MODEL_COMPONENT_CHECKS)}", None


def _dpo_contract_state(root: Path, raw: dict[str, Any]) -> tuple[bool, str, str | None]:
    """Validate the DPO-specific identity before a real M1 launch."""
    datasets = _as_dict(raw.get("datasets"))
    trainer = _as_dict(raw.get("trainer"))
    evaluation = _as_dict(raw.get("evaluation"))
    errors: list[str] = []

    preference_value = datasets.get("preference_path")
    preference_path = _resolve_project_path(root, preference_value, root / "__missing__")
    pair_count = 0
    if not preference_path.is_file():
        errors.append("preference_path is missing")
    else:
        digest = hashlib.sha256(preference_path.read_bytes()).hexdigest()
        expected = _as_dict(datasets.get("hashes")).get("preference")
        if expected != digest:
            errors.append(f"preference hash mismatch: config={expected} on_disk={digest}")
        try:
            pair_count = sum(
                1
                for line in preference_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeError):
            pair_count = 0
        if pair_count < 100:
            errors.append(f"preference dataset has only {pair_count} rows; need at least 100")

    report_paths = datasets.get("manifest_paths")
    if not isinstance(report_paths, list) or len(report_paths) != 1:
        errors.append("DPO must declare exactly one preference evidence report")
    else:
        report_path = _resolve_project_path(root, report_paths[0], root / "__missing__")
        report = _read_json(report_path)
        if report is None:
            errors.append("preference evidence report is missing or unreadable")
        else:
            if report.get("artifact_kind") != "M1_CALIBRATION_PREFERENCE_PAIRS":
                errors.append("preference evidence report has the wrong artifact kind")
            if report.get("sha256") != _as_dict(datasets.get("hashes")).get("preference"):
                errors.append("preference evidence report hash disagrees with the config")
            if report.get("pairs_out") != pair_count:
                errors.append("preference evidence report count disagrees with the preference file")

    initial_value = trainer.get("initial_checkpoint")
    initial = _resolve_project_path(root, initial_value, root / "__missing__")
    if not initial.is_dir() or not (initial / "model.safetensors").is_file():
        errors.append("initial_checkpoint is not a loadable local checkpoint")
        initial_hash = None
    else:
        initial_hash = hashlib.sha256((initial / "model.safetensors").read_bytes()).hexdigest()
        if trainer.get("initial_checkpoint_sha256") != initial_hash:
            errors.append("initial_checkpoint_sha256 does not match model.safetensors")

    reference = _resolve_project_path(
        root, trainer.get("reference_checkpoint"), root / "__missing__"
    )
    if not reference.is_dir() or not (reference / "model.safetensors").is_file():
        errors.append("reference_checkpoint is not a loadable local checkpoint")
    elif initial_hash is not None:
        reference_hash = hashlib.sha256((reference / "model.safetensors").read_bytes()).hexdigest()
        if reference_hash != initial_hash:
            errors.append("reference checkpoint hash differs from the initial M1 checkpoint")

    parent_id = trainer.get("parent_checkpoint_id")
    if not isinstance(parent_id, str) or not parent_id:
        errors.append("parent_checkpoint_id is missing")
    else:
        registry = _read_json(root / "runs/checkpoint_registry.json") or {}
        matches = [
            item
            for item in registry.get("checkpoints", [])
            if isinstance(item, dict) and item.get("checkpoint_id") == parent_id
        ]
        if len(matches) != 1:
            errors.append(f"parent checkpoint is not uniquely registered: {parent_id}")
        elif initial_value and Path(str(matches[0].get("path"))).resolve() != initial.resolve():
            errors.append("parent checkpoint registry path differs from initial_checkpoint")

    manifest = evaluation.get("checkpoint_selection_manifest")
    manifest_path = _resolve_project_path(root, manifest, root / "__missing__")
    if not manifest_path.is_file():
        errors.append("frozen checkpoint-selection manifest is missing")

    if trainer.get("reference") != "explicit_checkpoint":
        errors.append("M1 must use the explicit selected-M0 reference checkpoint")
    if raw.get("parent_experiment_id") != "m0_sft_canonical_v2_final":
        errors.append("M1 parent experiment must be m0_sft_canonical_v2_final")
    if _as_dict(raw.get("promotion")).get("policy_version") != "tool_use_promotion_v4":
        errors.append("M1 must pin prospective promotion policy tool_use_promotion_v4")

    if errors:
        return False, "; ".join(errors), "DPO_CONTRACT_INVALID"
    return (
        True,
        f"M1 parent, reference, preference hash/count, frozen evaluation manifest, and policy are pinned ({pair_count} pairs)",
        None,
    )


def _supervision_composition_state(root: Path, raw: dict[str, Any]) -> tuple[bool, str, str | None]:
    """Report the training mixture by supervision kind, and honour the config's selection.

    A single aggregate trainable count cannot distinguish a corpus of complete trajectories from
    a corpus of call-prediction examples, and those teach different things. The mixture is
    therefore stated per kind before training, and a config that selected kinds absent from the
    corpus fails rather than silently training on something else.
    """
    declared = (raw.get("datasets") or {}).get("yield_report")
    if not declared:
        return (
            True,
            "No yield report declared: supervision composition is not measured for this config",
            None,
        )
    payload = _read_json(root / str(declared))
    if payload is None:
        return False, f"Declared yield report is unreadable: {declared}", "YIELD_REPORT_UNREADABLE"

    selected = [str(item) for item in ((raw.get("supervision") or {}).get("include") or [])]
    per_kind: dict[str, dict[str, int]] = {}
    total_trainable = 0
    invalid_counts: list[str] = []
    for source in payload.get("sources") or []:
        kinds = source.get("supervision_kinds_trainable") or {}
        for kind, raw_count in kinds.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError, OverflowError):
                invalid_counts.append(f"{source.get('source', '<unknown>')}:{kind}={raw_count!r}")
                continue
            if count < 0:
                invalid_counts.append(f"{source.get('source', '<unknown>')}:{kind}={count}")
                continue
            if count == 0:
                continue
            entry = per_kind.setdefault(str(kind), {"trainable": 0, "sources": 0})
            entry["trainable"] += count
            entry["sources"] += 1
            total_trainable += count
    if invalid_counts:
        return (
            False,
            f"Yield report has invalid supervision counts: {invalid_counts}",
            "SUPERVISION_COMPOSITION_INVALID",
        )
    if not per_kind:
        if selected:
            return (
                False,
                (
                    "Yield report has no positive trainable supervision kinds but config selects "
                    f"{selected}: {declared}"
                ),
                "SUPERVISION_SELECTION_MISMATCH",
            )
        return (
            False,
            f"Yield report declares no positive trainable supervision kinds: {declared}",
            "SUPERVISION_COMPOSITION_MISSING",
        )

    composition = {
        kind: {
            **entry,
            "share": round(entry["trainable"] / total_trainable, 6),
        }
        for kind, entry in sorted(per_kind.items())
    }
    unclassified = per_kind.get("UNCLASSIFIED", {}).get("trainable", 0)
    absent = sorted(kind for kind in selected if kind not in per_kind)
    detail = (
        f"trainable by supervision kind: "
        f"{ {k: v['trainable'] for k, v in composition.items()} }; "
        f"shares: { {k: v['share'] for k, v in composition.items()} }"
    )
    if selected:
        detail += f"; config selects {selected}"
    if absent:
        return (
            False,
            f"{detail}; config selects kinds the corpus does not contain: {absent}",
            "SUPERVISION_SELECTION_MISMATCH",
        )
    if unclassified:
        # An unclassified trainable record means a record reached training without a declared
        # contract. It must not be silently averaged into the mixture.
        return (
            False,
            f"{detail}; {unclassified} trainable records carry no supervision kind",
            "SUPERVISION_UNCLASSIFIED",
        )
    return True, detail, None
