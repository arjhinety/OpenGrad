"""Authoritative repository status, readiness gates, and bounded GPU boundary smoke.

This module is deliberately conservative: CPU mocks and frozen placeholders are
reported as evidence of plumbing only, never as real baseline or SFT readiness.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import re
import subprocess
from pathlib import Path
from typing import Any

from opengrad.env_capture import capture
from opengrad.experiments.preflight import run_experiment_preflight
from opengrad.experiments.schema import ExperimentConfig
from opengrad.experiments.store import ExperimentStore
from opengrad.hardware.probe import probe_hardware
from opengrad.registry.validate import validate as validate_registry

BASELINE_CONFIG = Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml")
BASELINE_MANIFEST = Path("reports/evaluation/behavioral-heldout-v2.manifest.json")
BASELINE_METRICS = Path("reports/baselines/qwen35_2b_baseline/metrics.json")
BASELINE_PREDICTIONS = Path("reports/baselines/qwen35_2b_baseline/predictions.jsonl")
BASELINE_RESIDUALS = Path("reports/failures/qwen35_2b_baseline/residual-profile.json")
BASELINE_ENVIRONMENT = Path("reports/baselines/qwen35_2b_baseline/environment.json")
PINNED_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
PINNED_TEMPLATE_HASH = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
PINNED_EVALUATOR_REVISION = "2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc"
CANONICAL_MODEL_ID = "Qwen/Qwen3.5-2B"
TRAINING_SOURCE_IDS = {
    "xlam-function-calling-60k", "toolace", "looptool-23k",
    "glaive-function-calling-v2", "button", "when2call-sft",
}
TRAINING_SOURCE_MANIFESTS = {
    "data/processed/normalization-v1/xlam",
    "data/processed/normalization-v1/toolace",
    "data/processed/normalization-v1/looptool",
    "data/processed/normalization-v1/glaive",
    "data/processed/normalization-v1/button",
    "data/processed/normalization-v1/when2call-sft",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _baseline_state(root: Path) -> dict[str, Any]:
    metrics_path = root / BASELINE_METRICS
    predictions_path = root / BASELINE_PREDICTIONS
    residual_path = root / BASELINE_RESIDUALS
    env_path = root / BASELINE_ENVIRONMENT
    metrics = _read_json(metrics_path)
    residuals = _read_json(residual_path)
    environment = _read_json(env_path)
    artifacts = {
        "metrics": metrics_path.exists(),
        "predictions": predictions_path.exists(),
        "residual_profile": residual_path.exists(),
        "environment": env_path.exists(),
    }
    prediction_rows: list[dict[str, Any]] = []
    predictions_valid = False
    prediction_ids: set[str] = set()
    if predictions_path.exists():
        try:
            raw_rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            ids = [row.get("example_id") for row in raw_rows if isinstance(row, dict)]
            decisions = {"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"}
            predictions_valid = bool(raw_rows) and all(isinstance(item, str) and item for item in ids) and len(ids) == len(set(ids)) == len(raw_rows) and all(
                isinstance(row, dict)
                and isinstance(row.get("raw_output"), str)
                and row["raw_output"].strip()
                and row.get("parser", {}).get("status") == "RAW_VALID"
                and row.get("prediction", {}).get("decision") in decisions
                for row in raw_rows
            )
            if predictions_valid:
                prediction_rows = raw_rows
                prediction_ids = set(ids)
        except (OSError, json.JSONDecodeError, TypeError):
            prediction_rows = []
    manifest = _read_json(root / BASELINE_MANIFEST)
    expected_records = sum(_safe_item_count(split) or 0 for split in (manifest or {}).get("splits", []) if isinstance(split, dict))
    expected_ids = _materialized_evaluation_ids(root, manifest)
    config = _read_yaml(root / BASELINE_CONFIG)
    manifest_file = root / BASELINE_MANIFEST
    manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest() if manifest_file.exists() else None
    current_commit = _git_state(root).get("commit")
    experiment_records = _experiments(root)
    baseline_experiment = next((record for record in experiment_records if record.get("experiment_id") in {"tool_calling/qwen35_2b/baseline", "qwen35_2b_baseline"}), None)
    residual_values = residuals.get("residuals", {}) if isinstance(residuals, dict) else {}
    residual_contract_ok = bool(
        isinstance(residuals, dict)
        and residuals.get("schema_version") == 1
        and residuals.get("baseline_experiment") == "tool_calling/qwen35_2b/baseline"
        and residuals.get("model_id") == CANONICAL_MODEL_ID
        and residuals.get("model_revision") == PINNED_MODEL_REVISION
        and residuals.get("manifest_sha256") == manifest_sha256
        and isinstance(residual_values, dict)
        and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in residual_values.values())
    )
    artifact_contract_ok = bool(
        metrics and metrics.get("status") == "EXECUTED" and metrics.get("backend") == "transformers"
        and metrics.get("model_id") == config.get("model_id") == CANONICAL_MODEL_ID
        and metrics.get("model_revision") == config.get("model_revision") == PINNED_MODEL_REVISION
        and metrics.get("tokenizer_revision") == config.get("tokenizer_revision") == PINNED_MODEL_REVISION
        and metrics.get("manifest") == config.get("evaluations", {}).get("behavioral_manifest")
        and metrics.get("manifest_sha256") == manifest_sha256
        and metrics.get("template_hash") == config.get("template_hash") == PINNED_TEMPLATE_HASH
        and metrics.get("renderer") == "qwen3_5_2b_v1"
        and metrics.get("run_id") == "tool_calling/qwen35_2b/baseline"
        and metrics.get("records", 0) == len(prediction_rows) > 0
        and predictions_valid and expected_ids is not None and prediction_ids == expected_ids
        and metrics.get("records") == expected_records
        and residual_contract_ok and residuals.get("sample_count") == metrics.get("records")
        and environment and environment.get("backend") == metrics.get("backend")
        and environment.get("dry_run") is False
        and environment.get("run_id") == metrics.get("run_id")
        and environment.get("model_id") == CANONICAL_MODEL_ID
        and environment.get("model_revision") == PINNED_MODEL_REVISION
        and environment.get("manifest_sha256") == manifest_sha256
        and isinstance(metrics.get("git_commit"), str) and metrics.get("git_commit") == current_commit
        and baseline_experiment is not None
        and baseline_experiment.get("status") in {"EVALUATED", "REVIEW", "PROMOTED"}
        and baseline_experiment.get("model_id") == CANONICAL_MODEL_ID
        and baseline_experiment.get("model_revision") == PINNED_MODEL_REVISION
    )
    real_status = bool(all(artifacts.values()) and artifact_contract_ok)
    dry_run_complete = bool(metrics and metrics.get("status") == "DRY_RUN" and metrics.get("backend") == "deterministic-mock" and metrics.get("records") == len(prediction_rows) > 0)
    return {
        "id": "qwen35_2b_baseline",
        "status": "REAL_COMPLETE" if real_status else ("CPU_DRY_RUN" if dry_run_complete else "PLANNED"),
        "real": real_status,
        "metrics": metrics,
        "residual_profile": residuals,
        "artifacts": artifacts,
        "artifact_contract_ok": artifact_contract_ok,
        "prediction_records": len(prediction_rows),
        "expected_records": expected_records,
        "artifact_paths": {key: str(path) for key, path in {
            "metrics": metrics_path, "predictions": predictions_path,
            "residual_profile": residual_path, "environment": env_path,
        }.items()},
    }


def _experiments(root: Path) -> list[dict[str, Any]]:
    store = ExperimentStore(root)
    return [record.to_dict() for record in store.list_experiments()]


def _check_revision(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", value))


def _safe_item_count(split: Any) -> int | None:
    if not isinstance(split, dict):
        return None
    try:
        value = int(split.get("items", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _materialized_evaluation_ids(root: Path, manifest: dict[str, Any] | None) -> set[str] | None:
    if not isinstance(manifest, dict):
        return None
    try:
        import pyarrow.parquet as pq
        result: set[str] = set()
        for split in manifest.get("splits", []):
            source = _resolve_project_path(root, split.get("source"), root / "__missing__")
            manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
            data = _read_json(manifest_path)
            if not data or data.get("finalized") is not True or not isinstance(data.get("shards"), list) or not data["shards"]:
                return None
            base = manifest_path.parent
            for name in data["shards"]:
                shard = base / str(name)
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["example_id"]):
                    for row in batch.to_pylist():
                        value = row.get("example_id")
                        if not isinstance(value, str) or not value or value in result:
                            return None
                        result.add(value)
        return result or None
    except Exception:
        # Readiness is a fail-closed projection. Corrupt/missing parquet must
        # produce an unavailable ID set, never an optimistic empty set.
        return None


def _read_dataset_registry(root: Path) -> dict[str, dict[str, Any]]:
    try:
        import yaml
        raw = yaml.safe_load((root / "registry/datasets.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, TypeError, ValueError):
        return {}
    rows = raw.get("datasets", []) if isinstance(raw, dict) else []
    return {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}


def _training_data_contract(root: Path, raw: dict[str, Any]) -> tuple[bool, str]:
    datasets = raw.get("datasets") if isinstance(raw.get("datasets"), dict) else {}
    ids = datasets.get("manifest_ids")
    hashes = datasets.get("hashes")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item.strip() for item in ids):
        return False, "SFT dataset manifest IDs are missing or invalid"
    if not isinstance(hashes, dict) or set(hashes) != set(ids) or not all(_check_revision(value) for value in hashes.values()):
        return False, "SFT dataset hashes must be one pinned SHA revision per manifest ID"
    registry = _read_dataset_registry(root)
    unknown = [item for item in ids if item not in registry]
    if unknown:
        return False, f"SFT dataset IDs are absent from registry: {unknown}"
    unsafe = []
    for item in ids:
        row = registry[item]
        intended = {str(value).lower() for value in row.get("intended_stages", [])} if isinstance(row.get("intended_stages"), list) else set()
        forbidden = {str(value).lower() for value in row.get("forbidden_splits", [])} if isinstance(row.get("forbidden_splits"), list) else set()
        if "evaluation" in intended or "preference" in intended or forbidden & {"evaluation", "heldout", "mcq_test", "llm_judge_test"}:
            unsafe.append(item)
        source_revision = row.get("source_revision", {}).get("value") if isinstance(row.get("source_revision"), dict) else None
        if not _check_revision(source_revision):
            return False, f"SFT dataset source revision is not pinned for {item}"
        processed = row.get("processed_dataset_hash") if isinstance(row.get("processed_dataset_hash"), dict) else {}
        processed_value = processed.get("value")
        if not isinstance(processed_value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", processed_value):
            return False, f"SFT processed dataset hash is not materialized for {item}"
        if hashes.get(item) != processed_value:
            return False, f"SFT dataset hash does not match registry for {item}"
    if unsafe:
        return False, f"Evaluation/preference datasets cannot enter SFT: {unsafe}"
    return True, "SFT dataset IDs, registry eligibility, source revisions, and hashes are pinned"


def repository_status(root: Path) -> dict[str, Any]:
    env = capture(root)
    git = _git_state(root)
    baseline = _baseline_state(root)
    experiments = _experiments(root)
    checkpoints = _read_json(root / "runs/checkpoint_registry.json") or {"checkpoints": []}
    # The baseline config is YAML, so use the typed parser instead of inventing
    # a parallel schema for this read-only projection.
    try:
        import yaml

        baseline_config = yaml.safe_load((root / BASELINE_CONFIG).read_text(encoding="utf-8")) or {}
    except Exception:
        baseline_config = {}
    active = experiments[-1] if experiments else None
    if baseline["real"]:
        phase = "BASELINE"
    elif active and active.get("status") in {"TRAINING", "PREFLIGHT"}:
        phase = "SFT_RUNNING" if active.get("training_algorithm") == "sft" else "TRAINING"
    elif active and active.get("status") in {"TRAINED", "EVALUATING", "EVALUATED", "REVIEW"}:
        phase = "POST_SFT_EVALUATION" if active.get("training_algorithm") == "sft" else "REVIEW"
    else:
        phase = "PRE_BASELINE"
    errors = validate_registry(root)
    return {
        "schema_version": 1,
        "state": phase,
        "git": git,
        "environment": env,
        "model": {
            "id": baseline_config.get("model_id", baseline_config.get("model", {}).get("model_id")),
            "revision": baseline_config.get("model_revision", baseline_config.get("model", {}).get("model_revision")),
        },
        "dataset": {
            "manifest": baseline_config.get("dataset_manifest", baseline_config.get("datasets", {}).get("manifest_ids")),
            "revision": baseline_config.get("dataset_hash", baseline_config.get("datasets", {}).get("hashes")),
        },
        "baseline": baseline,
        "active_experiment": active,
        "experiments": experiments,
        "checkpoints": checkpoints.get("checkpoints", []),
        "validation": {"status": "PASS" if not errors else "FAIL", "errors": errors},
        "gpu": {"status": "NOT_RUN", "evidence": "Use opengrad gpu-smoke --json; placeholder config is not evidence."},
    }


def _gate(name: str, status: str, details: str, code: str | None = None) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details, "error_code": code}


def _resolve_project_path(root: Path, value: Any, default: Path) -> Path:
    candidate = Path(str(value)) if value else default
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return root / "__invalid_path__"
    return resolved if resolved.is_relative_to(root) else root / "__outside_project__"


def _baseline_config_contract(raw: dict[str, Any], root: Path | None = None) -> tuple[bool, str]:
    required = {
        "schema_version", "status", "model_id", "model_revision", "tokenizer_revision",
        "renderer", "template_hash", "seed", "generation", "evaluations", "runtime",
        "outputs", "provenance",
    }
    missing = sorted(required - set(raw))
    if missing:
        return False, f"baseline config is missing required field(s): {', '.join(missing)}"
    if raw.get("schema_version") != 1 or raw.get("status") != "FROZEN_PRE_GPU":
        return False, "baseline config must use schema 1 and FROZEN_PRE_GPU status"
    if raw.get("model_id") != CANONICAL_MODEL_ID or raw.get("model_revision") != PINNED_MODEL_REVISION or raw.get("tokenizer_revision") != PINNED_MODEL_REVISION:
        return False, "baseline config model and tokenizer revisions must match the pinned canonical model"
    if raw.get("renderer") != "qwen3_5_2b_v1" or raw.get("template_hash") != PINNED_TEMPLATE_HASH or raw.get("seed") != 0:
        return False, "baseline renderer, template hash, or seed is not pinned"
    generation = raw.get("generation")
    if not isinstance(generation, dict) or generation.get("do_sample") is not False or generation.get("temperature") != 0.0 or generation.get("top_p") != 1.0 or not isinstance(generation.get("max_new_tokens"), int) or generation["max_new_tokens"] < 1:
        return False, "baseline generation must be deterministic and bounded"
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("backend") != "transformers" or runtime.get("precision") != "bfloat16" or runtime.get("device_policy") != "accelerator_required" or not isinstance(runtime.get("context_length"), int) or runtime["context_length"] < 1:
        return False, "baseline runtime must require the pinned transformers/BF16 accelerator path"
    evaluations = raw.get("evaluations")
    provenance = raw.get("provenance")
    outputs = raw.get("outputs")
    if not isinstance(evaluations, dict) or not isinstance(provenance, dict) or not isinstance(outputs, dict):
        return False, "baseline evaluations, outputs, and provenance must be objects"
    if not isinstance(evaluations.get("behavioral_manifest"), str) or provenance.get("evaluator_revision") != PINNED_EVALUATOR_REVISION or provenance.get("manifest_status") != "FROZEN_PRE_GPU":
        return False, "baseline evaluation provenance is not pinned"
    if set(outputs) != {"predictions", "metrics", "residual_profile", "environment"} or not all(isinstance(value, str) and value for value in outputs.values()):
        return False, "baseline outputs must name all four artifact paths"
    return True, "frozen baseline model, renderer, generation, runtime, and provenance contract match"


def _is_baseline_config(raw: dict[str, Any]) -> bool:
    return _baseline_config_contract(raw)[0]


def _is_experiment_config(raw: dict[str, Any]) -> bool:
    required = {"experiment_id", "hypothesis", "model", "datasets", "trainer", "evaluation", "checkpointing", "promotion", "reproducibility"}
    return required.issubset(raw)


def _manifest_contract_ok(root: Path, manifest: dict[str, Any] | None, expected_revision: Any, expected_template: Any) -> tuple[bool, str]:
    if not manifest:
        return False, "evaluation manifest is missing or invalid JSON"
    if manifest.get("schema_version") != 1 or manifest.get("frozen") is not True or manifest.get("status") not in {"FROZEN_PRE_GPU", "MATERIALIZED", "EXECUTED"}:
        return False, "manifest must use schema 1 and be frozen with a supported status"
    if manifest.get("manifest_id") != "behavioral-heldout-v2":
        return False, "manifest ID does not match the frozen behavioral evaluation contract"
    if manifest.get("freeze_revision") != PINNED_EVALUATOR_REVISION:
        return False, "manifest freeze revision does not match the pinned evaluator revision"
    contract = manifest.get("model_renderer_contract")
    if not isinstance(contract, dict):
        return False, "manifest model_renderer_contract is missing"
    if contract.get("model_revision") != PINNED_MODEL_REVISION or expected_revision != PINNED_MODEL_REVISION:
        return False, "model revision does not match the pinned baseline revision"
    if contract.get("renderer") != "qwen3_5_2b_v1":
        return False, "manifest renderer does not match the pinned native renderer"
    if contract.get("template_hash") != PINNED_TEMPLATE_HASH or expected_template != PINNED_TEMPLATE_HASH:
        return False, "template hash does not match the pinned baseline template"
    splits = manifest.get("splits")
    if not isinstance(splits, list) or not splits:
        return False, "manifest has no valid split contract"
    ids = set()
    for split in splits:
        if not isinstance(split, dict) or not isinstance(split.get("id"), str) or not split["id"] or split["id"] in ids or _safe_item_count(split) is None:
            return False, "manifest has an invalid or duplicate split item contract"
        ids.add(split["id"])
        if not isinstance(split.get("content_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", split["content_hash"]):
            return False, f"split {split.get('id')} has no pinned content hash"
        if not isinstance(split.get("source"), str) or not split["source"]:
            return False, f"split {split.get('id')} has no materialization source"
    return True, "frozen model, renderer, template, split, and content-hash contract match"


def _materialized_split_state(root: Path, split: dict[str, Any]) -> tuple[bool, str, set[str]]:
    source = _resolve_project_path(root, split.get("source"), root / "__missing__")
    manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
    data = _read_json(manifest_path)
    if not data or data.get("finalized") is not True or not isinstance(data.get("shards"), list) or not data["shards"]:
        return False, f"split {split.get('id')} materialization manifest is missing, unfinished, or empty", set()
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    written = _safe_item_count({"items": counts.get("written")})
    expected = _safe_item_count(split)
    if written != expected:
        return False, f"split {split.get('id')} materialized count {written} does not match frozen count {expected}", set()
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
        import pyarrow.parquet as pq
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
                        return False, f"split {split.get('id')} contains missing or duplicate example_id", set()
                    seen_ids.add(example_id)
                    content_digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
                    actual_rows += 1
    except Exception as exc:
        return False, f"split {split.get('id')} shard content cannot be verified: {exc}", set()
    if actual_rows != written:
        return False, f"split {split.get('id')} shard rows {actual_rows} do not match manifest count {written}", set()
    if content_digest.hexdigest() != split.get("content_hash"):
        return False, f"split {split.get('id')} content hash does not match shard rows", set()
    return True, f"split {split.get('id')} is materialized ({written} records; manifest {actual_hash[:12]})", shards


def _contamination_state(report: dict[str, Any] | None) -> tuple[bool, str]:
    if not report or report.get("manifest_id") != "behavioral-heldout-v2":
        return True, "contamination report is missing or bound to the wrong evaluation manifest"
    levels = report.get("levels") if isinstance(report.get("levels"), dict) else {}
    required = {"1_exact_canonical_conversation_hash", "2_normalized_prompt_hash", "3_near_duplicate_ngram_minhash", "4_semantic_similarity", "5_manual_audit"}
    missing = sorted(required - set(levels))
    pending = [str(name) for name in sorted(required) if str(levels.get(name, "")).upper() not in {"MEASURED", "CLEAN", "PASSED", "COMPLETE"}]
    sources = {str(value) for value in report.get("training_sources_checked", [])} if isinstance(report.get("training_sources_checked"), list) else set()
    source_ok = sources == {item.rsplit("/", 1)[-1] for item in TRAINING_SOURCE_MANIFESTS} or sources == TRAINING_SOURCE_IDS
    blocked = str(report.get("status", "UNKNOWN")).upper() not in {"CLEAN", "SEMANTIC_REVIEW_COMPLETE"} or bool(missing or pending) or not source_ok
    detail = f"status={report.get('status', 'UNKNOWN')}; missing_levels={missing or 'none'}; pending_levels={pending or 'none'}; sources_ok={source_ok}"
    return blocked, detail


def readiness(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config_path = _resolve_project_path(root, config_path, root / BASELINE_CONFIG).resolve()
    gates: list[dict[str, Any]] = []
    gate_map: dict[str, dict[str, Any]] = {}

    def add(name: str, status: str, details: str, code: str | None = None) -> None:
        gate = _gate(name, status, details, code)
        gates.append(gate)
        gate_map[name] = gate

    errors = validate_registry(root)
    add("repository_validation", "PASS" if not errors else "FAIL", "All registries validate" if not errors else "; ".join(errors), "CONFIG_INVALID" if errors else None)
    raw = _read_yaml(config_path)
    if not raw:
        add("config_validation", "FAIL", f"Config missing or invalid: {config_path}", "CONFIG_INVALID")
    elif _is_baseline_config(raw):
        add("config_validation", "PASS", f"Valid frozen baseline config: {config_path.relative_to(root) if config_path.is_relative_to(root) else config_path}")
    elif _is_experiment_config(raw):
        try:
            ExperimentConfig.from_file(config_path)
            add("config_validation", "PASS", f"Valid experiment config: {config_path.relative_to(root) if config_path.is_relative_to(root) else config_path}")
        except (OSError, ValueError, TypeError) as exc:
            add("config_validation", "FAIL", str(exc), "CONFIG_INVALID")
    else:
        add("config_validation", "FAIL", "Config is neither a frozen baseline evaluation contract nor a complete experiment config", "CONFIG_INVALID")

    revision = raw.get("model_revision", raw.get("model", {}).get("model_revision"))
    revision_ok = revision == PINNED_MODEL_REVISION
    add("model_revision", "PASS" if revision_ok else "FAIL", f"Pinned model revision: {revision}", "TOKENIZER_MISMATCH" if not revision_ok else None)
    model_id = raw.get("model_id", raw.get("model", {}).get("model_id"))
    model_id_ok = model_id == CANONICAL_MODEL_ID
    add("model_identity", "PASS" if model_id_ok else "FAIL", f"Canonical model ID: {model_id}", "MODEL_INVALID" if not model_id_ok else None)
    tokenizer_revision = raw.get("tokenizer_revision", raw.get("model", {}).get("tokenizer_revision"))
    tokenizer_ok = _check_revision(tokenizer_revision) and tokenizer_revision == PINNED_MODEL_REVISION
    add("tokenizer_revision", "PASS" if tokenizer_ok else "FAIL", f"Pinned tokenizer revision: {tokenizer_revision}", "TOKENIZER_MISMATCH" if not tokenizer_ok else None)
    config_is_baseline = _is_baseline_config(raw)
    model_contract = raw.get("model_renderer_contract") if isinstance(raw.get("model_renderer_contract"), dict) else {}
    template_hash = raw.get("template_hash", model_contract.get("template_hash"))
    if config_is_baseline:
        template_ok = template_hash == PINNED_TEMPLATE_HASH
        template_detail = f"template_hash={template_hash}"
    else:
        # Experiment configs do not own the evaluation renderer contract; the
        # frozen held-out manifest remains the authority for evaluation.
        template_ok = True
        template_detail = "Experiment config delegates renderer contract to the frozen evaluation manifest"
    add("chat_template_contract", "PASS" if template_ok else "FAIL", template_detail, "TOKENIZER_MISMATCH" if not template_ok else None)

    manifest_rel = raw.get("dataset_manifest", raw.get("evaluations", {}).get("behavioral_manifest", BASELINE_MANIFEST.as_posix()))
    manifest_path = _resolve_project_path(root, manifest_rel, root / BASELINE_MANIFEST)
    manifest = _read_json(manifest_path)
    expected_template = template_hash if config_is_baseline else PINNED_TEMPLATE_HASH
    manifest_ok, manifest_detail = _manifest_contract_ok(root, manifest, revision, expected_template)
    add("evaluation_manifest", "PASS" if manifest_ok else "FAIL", manifest_detail, "CHECKSUM_MISMATCH" if not manifest_ok else None)
    materialization_errors = []
    if manifest:
        for split in manifest.get("splits", []):
            valid_split, detail, _ = _materialized_split_state(root, split)
            if not valid_split:
                materialization_errors.append(detail)
    else:
        materialization_errors.append("evaluation manifest is unavailable")
    add("evaluation_materialization", "PASS" if not materialization_errors else "FAIL", "All frozen evaluation split manifests are materialized" if not materialization_errors else "; ".join(materialization_errors), "DATASET_NOT_FOUND" if materialization_errors else None)
    if _is_experiment_config(raw):
        training_ok, training_detail = _training_data_contract(root, raw)
        add("dataset_revision", "PASS" if training_ok else "FAIL", training_detail, "NO_FLOATING_DATASET_REVISION" if not training_ok else None)
        add("dataset_snapshot", "PASS" if training_ok else "FAIL", training_detail, "DATASET_NOT_FOUND" if not training_ok else None)
    else:
        add("dataset_revision", "PASS", "Baseline dataset contract is represented by the frozen evaluation manifest")
        add("dataset_snapshot", "PASS", "Baseline uses the frozen evaluation manifest")
    contamination = _read_json(root / "reports/data/behavioral-heldout-v2-contamination.json")
    contamination_blocked, contamination_detail = _contamination_state(contamination)
    add("contamination_gate", "FAIL" if contamination_blocked else "PASS", contamination_detail, "CONTAMINATION_FAILURE" if contamination_blocked else None)
    policy = manifest.get("contamination_policy", {}) if isinstance(manifest, dict) else {}
    excluded = policy.get("training_manifests_excluded")
    heldout_source_ok = bool(
        policy.get("derived_prompts_excluded") is True
        and isinstance(excluded, list)
        and {str(value).rstrip("/") for value in excluded} == {value.rstrip("/") for value in TRAINING_SOURCE_MANIFESTS}
    )
    add("evaluation_leakage", "PASS" if heldout_source_ok else "FAIL", "Evaluation-only manifest excludes exactly the training manifests" if heldout_source_ok else "Manifest leakage policy is missing or does not cover the training manifests", "CONTAMINATION_FAILURE" if not heldout_source_ok else None)

    disk = shutil.disk_usage(root)
    free_gib = disk.free / 2**30
    add("disk_capacity", "PASS" if free_gib >= 2 else ("WARN" if free_gib >= 1 else "FAIL"), f"{free_gib:.2f} GiB free", "PATH_NOT_WRITABLE" if free_gib < 1 else None)
    runs = root / "runs"
    storage_error = ""
    try:
        runs.mkdir(parents=True, exist_ok=True)
        probe = runs / ".opengrad-readiness-probe"
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
        storage_status = "PASS"
    except OSError as exc:
        storage_status, storage_error = "FAIL", str(exc)
    add("artifact_storage", storage_status, "runs/ is writable" if storage_status == "PASS" else storage_error, "PATH_NOT_WRITABLE" if storage_status != "PASS" else None)
    parser_exists = (root / "src/opengrad/formatting/parser.py").exists()
    add("native_parser", "PASS" if parser_exists else "FAIL", "Qwen native parser is present" if parser_exists else "Qwen native parser is missing", "TOKENIZER_MISMATCH" if not parser_exists else None)

    hw = probe_hardware()
    gpu_status = "PASS" if hw.gpu_available and hw.bf16_supported else "FAIL"
    add("gpu_probe", gpu_status, f"{hw.gpu_name or 'no accelerator detected'}; VRAM={hw.total_vram_gb:.2f} GiB; BF16={hw.bf16_supported}", "GPU_UNAVAILABLE" if not hw.gpu_available else None)
    smoke_path = root / "reports/hardware/qwen_gpu_smoke.json"
    smoke = _read_json(smoke_path)
    smoke_checks = smoke.get("checks", []) if smoke else []
    required_smoke_checks = {"model_access", "native_template", "model_load", "one_generation", "native_parser", "vram", "cleanup"}
    smoke_names = {check.get("name") for check in smoke_checks if isinstance(check, dict) and check.get("status") == "PASS"}
    smoke_parser_ok = any(check.get("name") == "native_parser" and check.get("parser_status") == "RAW_VALID" for check in smoke_checks if isinstance(check, dict) and check.get("status") == "PASS")
    boundary_ok = bool(
        smoke and smoke.get("status") == "PASS" and smoke.get("kind") == "GPU_BOUNDARY_VERIFIED"
        and smoke.get("model_id") == "Qwen/Qwen3.5-2B" and smoke.get("model_revision") == PINNED_MODEL_REVISION
        and smoke.get("hardware", {}).get("gpu_available") is True
        and required_smoke_checks.issubset(smoke_names) and smoke_parser_ok
    )
    add("gpu_boundary", "PASS" if boundary_ok else "FAIL", "Bounded Qwen smoke receipt has complete valid evidence" if boundary_ok else "GPU boundary is not verified with complete RAW_VALID smoke evidence", "GPU_SMOKE_FAILED" if not boundary_ok else None)

    baseline = _baseline_state(root)
    required_artifacts = all(baseline["artifacts"].values()) and baseline.get("artifact_contract_ok") is True
    is_sft_config = _is_experiment_config(raw) and str(raw.get("trainer", {}).get("type", "")).lower() == "sft"
    if is_sft_config:
        try:
            preflight = run_experiment_preflight(config_path, root=root)
            preflight_ok = preflight.overall_status == "PASS"
            preflight_detail = f"OpenGrad experiment preflight: {preflight.overall_status}"
        except (OSError, TypeError, ValueError) as exc:
            preflight_ok = False
            preflight_detail = f"OpenGrad experiment preflight failed: {exc}"
        add("experiment_preflight", "PASS" if preflight_ok else "FAIL", preflight_detail, "PREFLIGHT_FAILED" if not preflight_ok else None)
    else:
        add("experiment_preflight", "PASS", "Baseline contract does not require SFT experiment preflight")
    if is_sft_config:
        dataset_ids = raw.get("datasets", {}).get("manifest_ids", [])
        eval_terms = {"heldout", "evaluation", "mcq", "judge", "preference", "bfcl", "tau2"}
        unsafe_ids = [str(item) for item in dataset_ids if any(term in str(item).lower() for term in eval_terms)]
        add("training_data_policy", "FAIL" if unsafe_ids else "PASS", f"Evaluation-only dataset IDs are excluded: {unsafe_ids}" if unsafe_ids else "SFT dataset manifest IDs contain no evaluation-only names", "NO_TRAINING_ON_EVAL_DATA" if unsafe_ids else None)
    else:
        add("training_data_policy", "PASS", "Not an SFT configuration; training data policy deferred")
    add("real_b0", "PASS" if baseline["real"] else "FAIL", baseline["status"], "BASELINE_NOT_FOUND" if not baseline["real"] else None)
    add("baseline_artifacts", "PASS" if required_artifacts else "FAIL", str(baseline["artifact_paths"]), "BASELINE_NOT_FOUND" if not required_artifacts else None)

    # Baseline execution is the operation that establishes real_b0. It may
    # require the hardware probe and repository/data contracts, but must not
    # require its own post-run artifacts or a prior GPU receipt. The receipt is
    # produced by the immediately preceding smoke stage in B0_WORKFLOW.
    baseline_prerequisites = {
        "repository_validation", "config_validation", "model_revision", "tokenizer_revision",
        "chat_template_contract", "evaluation_manifest", "evaluation_materialization",
        "contamination_gate", "evaluation_leakage", "disk_capacity", "artifact_storage",
        "native_parser", "gpu_probe",
    }
    ready_for_baseline = all(gate_map[name]["status"] == "PASS" for name in baseline_prerequisites)
    sft_names = baseline_prerequisites | {"gpu_boundary", "real_b0", "baseline_artifacts", "training_data_policy", "dataset_revision", "dataset_snapshot", "experiment_preflight"}
    ready_for_sft = all(gate_map[name]["status"] == "PASS" for name in sft_names)
    blocking = [gate["name"] for gate in gates if gate["status"] == "FAIL"]
    warnings = [gate["name"] for gate in gates if gate["status"] == "WARN"]
    overall = "FAIL" if blocking else ("WARN" if warnings else "PASS")
    return {
        "schema_version": 1,
        "status": overall,
        "ready_for_baseline": ready_for_baseline,
        "ready_for_sft": ready_for_sft,
        "blocking_gates": blocking,
        "warnings": warnings,
        "gates": gates,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": raw.get("dataset_hash", raw.get("datasets", {}).get("hashes")),
        "git_commit": _git_state(root)["commit"],
        "baseline": baseline,
        "gpu": hw.to_dict(),
        "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
    }


def gpu_smoke(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    hw = probe_hardware()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED" if not hw.gpu_available else "INCOMPLETE",
        "kind": "GPU_BOUNDARY_VERIFIED",
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": PINNED_MODEL_REVISION,
        "hardware": hw.to_dict(),
        "checks": [],
        "limitations": [],
    }
    if not hw.gpu_available:
        receipt["checks"] = [{"name": "hardware", "status": "FAIL", "code": "GPU_UNAVAILABLE", "details": "No CUDA accelerator detected."}]
        receipt["limitations"].append("No model load or generation attempted.")
        return _write_gpu_receipt(root, receipt)
    checks: list[dict[str, Any]] = []
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        receipt["checks"] = [{"name": "dependencies", "status": "FAIL", "code": "GPU_SMOKE_FAILED", "details": str(exc)}]
        receipt["limitations"].append("Install the pinned gpu-evaluation extra before retrying.")
        return _write_gpu_receipt(root, receipt)

    cfg = config_path or root / "configs/evaluation/tool_calling/qwen35_2b_baseline.yaml"
    try:
        import yaml

        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        model_id = str(raw.get("model_id", "Qwen/Qwen3.5-2B"))
        model_id = {"qwen3.5-2b": "Qwen/Qwen3.5-2B"}.get(model_id.lower(), model_id)
        revision = str(raw.get("model_revision", ""))
        template_hash = raw.get("template_hash")
        if model_id != "Qwen/Qwen3.5-2B" or revision != PINNED_MODEL_REVISION or template_hash != PINNED_TEMPLATE_HASH:
            raise ValueError("GPU smoke requires the pinned Qwen model revision and template hash")
        kwargs = {"revision": revision, "trust_remote_code": False}
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **kwargs)
        checks.append({"name": "model_access", "status": "PASS", "details": model_id})
        tools = [{"type": "function", "function": {"name": "lookup", "description": "Lookup a value", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Look up worker 12."}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        checks.append({"name": "native_template", "status": "PASS", "characters": len(str(rendered))})
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", **kwargs)
        model.eval()
        device = next(model.parameters()).device
        if getattr(device, "type", None) != "cuda":
            raise RuntimeError(f"model loaded on {device}, not CUDA")
        checks.append({"name": "model_load", "status": "PASS", "device": str(device)})
        encoded = tokenizer(str(rendered), return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=8, do_sample=False)
        generated = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False)
        checks.append({"name": "one_generation", "status": "PASS", "output_chars": len(generated)})
        from opengrad.formatting.parser import parse_qwen_native_output

        parsed = parse_qwen_native_output(generated)
        parser_ok = parsed.status == "RAW_VALID" and parsed.decision in {"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"}
        checks.append({"name": "native_parser", "status": "PASS" if parser_ok else "FAIL", "parser_status": parsed.status, "decision": parsed.decision, "errors": parsed.errors or []})
        if not parser_ok:
            raise RuntimeError(f"native parser rejected smoke output: {parsed.status}: {parsed.errors or []}")
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        checks.append({"name": "vram", "status": "PASS", "allocated_gib": round(allocated, 3), "peak_allocated_gib": round(peak, 3)})
        del model, tokenizer
        torch.cuda.empty_cache()
        checks.append({"name": "cleanup", "status": "PASS"})
        receipt["status"] = "PASS"
    except Exception as exc:  # bounded smoke must emit an auditable failure, never hide it
        checks.append({"name": "runtime", "status": "FAIL", "code": "GPU_SMOKE_FAILED", "details": f"{type(exc).__name__}: {exc}"})
        receipt["limitations"].append("Real baseline and SFT are blocked until this boundary passes.")
    receipt["checks"] = checks
    return _write_gpu_receipt(root, receipt)


def _write_gpu_receipt(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    path = root / "reports/hardware/qwen_gpu_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["artifact"] = str(path.relative_to(root))
    return receipt
