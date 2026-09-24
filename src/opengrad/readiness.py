"""Authoritative repository status, readiness gates, and bounded GPU boundary smoke.

This module is deliberately conservative: CPU mocks and frozen placeholders are
reported as evidence of plumbing only, never as real baseline or SFT readiness.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

from opengrad.env_capture import capture
from opengrad.experiments.preflight import run_experiment_preflight
from opengrad.experiments.schema import ExperimentConfig
from opengrad.hardware.probe import HardwareProbeResult, probe_hardware
from opengrad.readiness_contracts import (
    BASELINE_CONFIG,
    BASELINE_MANIFEST,
    BASELINE_METRICS,
    BASELINE_PREDICTIONS,
    CANONICAL_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
    SMOKE_MAX_NEW_TOKENS,
    SMOKE_MIN_NEW_TOKENS,
    TRAINING_SOURCE_MANIFESTS,
    _as_dict,
    _baseline_state,
    _check_revision,
    _commit_is_ancestor,
    _configured_release_dir,
    _experiments,
    _gate,
    _git_state,
    _is_baseline_config,
    _is_experiment_config,
    _manifest_contract_ok,
    _read_json,
    _read_yaml,
    _resolve_project_path,
    _training_data_contract,
)
from opengrad.readiness_states import (
    MODEL_COMPONENTS_VALIDATION,
    STUDY_001_TRAINING_CONFIGS,
    _contamination_report,
    _contamination_state,
    _determinism_state,
    _dpo_contract_state,
    _materialized_split_state,
    _model_components_state,
    _renderability_state,
    _supervision_composition_state,
)
from opengrad.registry.validate import validate as validate_registry

# Re-exported: tests and callers import these from `opengrad.readiness`.
__all__ = [
    "BASELINE_MANIFEST",
    "BASELINE_METRICS",
    "BASELINE_PREDICTIONS",
    "MODEL_COMPONENTS_VALIDATION",
    "SMOKE_MAX_NEW_TOKENS",
    "SMOKE_MIN_NEW_TOKENS",
    "STUDY_001_TRAINING_CONFIGS",
    "_baseline_state",
    "_commit_is_ancestor",
    "_determinism_state",
    "_git_state",
    "_materialized_split_state",
    "_model_components_state",
    "_renderability_state",
    "_supervision_composition_state",
    "_training_data_contract",
    "gpu_smoke",
    "readiness",
    "repository_status",
]


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
    except Exception:  # noqa: BLE001 - status must degrade to an empty projection, not crash
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
            "revision": baseline_config.get(
                "model_revision", baseline_config.get("model", {}).get("model_revision")
            ),
        },
        "dataset": {
            "manifest": baseline_config.get(
                "dataset_manifest", baseline_config.get("datasets", {}).get("manifest_ids")
            ),
            "revision": baseline_config.get(
                "dataset_hash", baseline_config.get("datasets", {}).get("hashes")
            ),
        },
        "baseline": baseline,
        "active_experiment": active,
        "experiments": experiments,
        "checkpoints": checkpoints.get("checkpoints", []),
        "validation": {"status": "PASS" if not errors else "FAIL", "errors": errors},
        "gpu": {
            "status": "NOT_RUN",
            "evidence": "Use opengrad gpu-smoke --json; placeholder config is not evidence.",
        },
    }


class _AddGate(Protocol):
    """Records one gate in `readiness()`'s ordered list; the phase functions below call it."""

    def __call__(self, name: str, status: str, details: str, code: str | None = None) -> None: ...


def _config_gates(add: _AddGate, config_path: Path, root: Path) -> dict[str, Any]:
    """Registries validate and the config is a frozen baseline contract or a complete experiment."""
    errors = validate_registry(root)
    add(
        "repository_validation",
        "PASS" if not errors else "FAIL",
        "All registries validate" if not errors else "; ".join(errors),
        "CONFIG_INVALID" if errors else None,
    )
    raw = _read_yaml(config_path)
    if not raw:
        add(
            "config_validation",
            "FAIL",
            f"Config missing or invalid: {config_path}",
            "CONFIG_INVALID",
        )
    elif _is_baseline_config(raw):
        add(
            "config_validation",
            "PASS",
            f"Valid frozen baseline config: {config_path.relative_to(root) if config_path.is_relative_to(root) else config_path}",
        )
    elif _is_experiment_config(raw):
        try:
            ExperimentConfig.from_file(config_path)
            add(
                "config_validation",
                "PASS",
                f"Valid experiment config: {config_path.relative_to(root) if config_path.is_relative_to(root) else config_path}",
            )
        except (OSError, ValueError, TypeError) as exc:
            add("config_validation", "FAIL", str(exc), "CONFIG_INVALID")
    else:
        add(
            "config_validation",
            "FAIL",
            "Config is neither a frozen baseline evaluation contract nor a complete experiment config",
            "CONFIG_INVALID",
        )
    return raw


def _model_identity_gates(add: _AddGate, raw: dict[str, Any]) -> tuple[bool, Any, Any, Any]:
    """Pinned model and tokenizer revisions, the canonical model id and the chat-template contract."""
    revision = raw.get("model_revision", raw.get("model", {}).get("model_revision"))
    revision_ok = revision == PINNED_MODEL_REVISION
    add(
        "model_revision",
        "PASS" if revision_ok else "FAIL",
        f"Pinned model revision: {revision}",
        "TOKENIZER_MISMATCH" if not revision_ok else None,
    )
    model_id = raw.get("model_id", raw.get("model", {}).get("model_id"))
    model_id_ok = model_id == CANONICAL_MODEL_ID
    add(
        "model_identity",
        "PASS" if model_id_ok else "FAIL",
        f"Canonical model ID: {model_id}",
        "MODEL_INVALID" if not model_id_ok else None,
    )
    tokenizer_revision = raw.get(
        "tokenizer_revision", raw.get("model", {}).get("tokenizer_revision")
    )
    tokenizer_ok = (
        _check_revision(tokenizer_revision) and tokenizer_revision == PINNED_MODEL_REVISION
    )
    add(
        "tokenizer_revision",
        "PASS" if tokenizer_ok else "FAIL",
        f"Pinned tokenizer revision: {tokenizer_revision}",
        "TOKENIZER_MISMATCH" if not tokenizer_ok else None,
    )
    config_is_baseline = _is_baseline_config(raw)
    model_contract = _as_dict(raw.get("model_renderer_contract"))
    template_hash = raw.get("template_hash", model_contract.get("template_hash"))
    if config_is_baseline:
        template_ok = template_hash == PINNED_TEMPLATE_HASH
        template_detail = f"template_hash={template_hash}"
    else:
        # Experiment configs do not own the evaluation renderer contract; the
        # frozen held-out manifest remains the authority for evaluation.
        template_ok = True
        template_detail = (
            "Experiment config delegates renderer contract to the frozen evaluation manifest"
        )
    add(
        "chat_template_contract",
        "PASS" if template_ok else "FAIL",
        template_detail,
        "TOKENIZER_MISMATCH" if not template_ok else None,
    )
    return config_is_baseline, revision, template_hash, tokenizer_revision


def _evaluation_data_gates(
    add: _AddGate,
    config_is_baseline: bool,
    raw: dict[str, Any],
    revision: Any,
    root: Path,
    template_hash: Any,
) -> dict[str, Any] | None:
    """The frozen evaluation manifest, its materialized splits, and the training data's pinned revision."""
    manifest_rel = raw.get(
        "dataset_manifest",
        raw.get("evaluations", {}).get("behavioral_manifest", BASELINE_MANIFEST.as_posix()),
    )
    manifest_path = _resolve_project_path(root, manifest_rel, root / BASELINE_MANIFEST)
    manifest = _read_json(manifest_path)
    expected_template = template_hash if config_is_baseline else PINNED_TEMPLATE_HASH
    manifest_ok, manifest_detail = _manifest_contract_ok(
        root, manifest, revision, expected_template
    )
    add(
        "evaluation_manifest",
        "PASS" if manifest_ok else "FAIL",
        manifest_detail,
        "CHECKSUM_MISMATCH" if not manifest_ok else None,
    )
    materialization_errors = []
    if manifest:
        for split in manifest.get("splits", []):
            valid_split, detail, _ = _materialized_split_state(root, split)
            if not valid_split:
                materialization_errors.append(detail)
    else:
        materialization_errors.append("evaluation manifest is unavailable")
    add(
        "evaluation_materialization",
        "PASS" if not materialization_errors else "FAIL",
        "All frozen evaluation split manifests are materialized"
        if not materialization_errors
        else "; ".join(materialization_errors),
        "DATASET_NOT_FOUND" if materialization_errors else None,
    )
    if _is_experiment_config(raw):
        training_ok, training_detail = _training_data_contract(root, raw)
        add(
            "dataset_revision",
            "PASS" if training_ok else "FAIL",
            training_detail,
            "NO_FLOATING_DATASET_REVISION" if not training_ok else None,
        )
        add(
            "dataset_snapshot",
            "PASS" if training_ok else "FAIL",
            training_detail,
            "DATASET_NOT_FOUND" if not training_ok else None,
        )
    else:
        add(
            "dataset_revision",
            "PASS",
            "Baseline dataset contract is represented by the frozen evaluation manifest",
        )
        add("dataset_snapshot", "PASS", "Baseline uses the frozen evaluation manifest")
    return manifest


def _contamination_gates(
    add: _AddGate, manifest: dict[str, Any] | None, raw: dict[str, Any], root: Path
) -> None:
    """The contamination report for the config's corpus, and the manifest's leakage policy."""
    configuration_release = _configured_release_dir(root, raw)
    contamination = _contamination_report(root, configuration_release)
    contamination_blocked, contamination_detail = _contamination_state(
        root, contamination, configuration_release
    )
    add(
        "contamination_gate",
        "FAIL" if contamination_blocked else "PASS",
        contamination_detail,
        "CONTAMINATION_FAILURE" if contamination_blocked else None,
    )
    policy = manifest.get("contamination_policy", {}) if isinstance(manifest, dict) else {}
    excluded = policy.get("training_manifests_excluded")
    heldout_source_ok = bool(
        policy.get("derived_prompts_excluded") is True
        and isinstance(excluded, list)
        and {str(value).rstrip("/") for value in excluded}
        == {value.rstrip("/") for value in TRAINING_SOURCE_MANIFESTS}
    )
    add(
        "evaluation_leakage",
        "PASS" if heldout_source_ok else "FAIL",
        "Evaluation-only manifest excludes exactly the training manifests"
        if heldout_source_ok
        else "Manifest leakage policy is missing or does not cover the training manifests",
        "CONTAMINATION_FAILURE" if not heldout_source_ok else None,
    )


def _storage_gates(add: _AddGate, root: Path) -> None:
    """Disk capacity, a writable runs/ directory, and the native parser."""
    disk = shutil.disk_usage(root)
    free_gib = disk.free / 2**30
    add(
        "disk_capacity",
        "PASS" if free_gib >= 2 else ("WARN" if free_gib >= 1 else "FAIL"),
        f"{free_gib:.2f} GiB free",
        "PATH_NOT_WRITABLE" if free_gib < 1 else None,
    )
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
    add(
        "artifact_storage",
        storage_status,
        "runs/ is writable" if storage_status == "PASS" else storage_error,
        "PATH_NOT_WRITABLE" if storage_status != "PASS" else None,
    )
    parser_exists = (root / "src/opengrad/formatting/parser.py").exists()
    add(
        "native_parser",
        "PASS" if parser_exists else "FAIL",
        "Qwen native parser is present" if parser_exists else "Qwen native parser is missing",
        "TOKENIZER_MISMATCH" if not parser_exists else None,
    )


def _gpu_gates(add: _AddGate, root: Path) -> HardwareProbeResult:
    """The hardware probe and the bounded GPU smoke receipt."""
    hw = probe_hardware()
    gpu_status = "PASS" if hw.gpu_available and hw.bf16_supported else "FAIL"
    add(
        "gpu_probe",
        gpu_status,
        f"{hw.gpu_name or 'no accelerator detected'}; VRAM={hw.total_vram_gb:.2f} GiB; BF16={hw.bf16_supported}",
        "GPU_UNAVAILABLE" if not hw.gpu_available else None,
    )
    smoke_path = root / "reports/hardware/qwen_gpu_smoke.json"
    smoke = _read_json(smoke_path)
    smoke_checks = smoke.get("checks", []) if smoke else []
    required_smoke_checks = {
        "model_access",
        "native_template",
        "model_load",
        "one_generation",
        "native_parser",
        "vram",
        "cleanup",
    }
    smoke_names = {
        check.get("name")
        for check in smoke_checks
        if isinstance(check, dict) and check.get("status") == "PASS"
    }
    smoke_parser_ok = any(
        check.get("name") == "native_parser" and check.get("parser_status") == "RAW_VALID"
        for check in smoke_checks
        if isinstance(check, dict) and check.get("status") == "PASS"
    )
    boundary_ok = bool(
        smoke
        and smoke.get("status") == "PASS"
        and smoke.get("kind") == "GPU_BOUNDARY_VERIFIED"
        and smoke.get("model_id") == "Qwen/Qwen3.5-2B"
        and smoke.get("model_revision") == PINNED_MODEL_REVISION
        and smoke.get("hardware", {}).get("gpu_available") is True
        and required_smoke_checks.issubset(smoke_names)
        and smoke_parser_ok
    )
    add(
        "gpu_boundary",
        "PASS" if boundary_ok else "FAIL",
        "Bounded Qwen smoke receipt has complete valid evidence"
        if boundary_ok
        else "GPU boundary is not verified with complete RAW_VALID smoke evidence",
        "GPU_SMOKE_FAILED" if not boundary_ok else None,
    )
    return hw


def _training_gates(
    add: _AddGate, config_path: Path, raw: dict[str, Any], root: Path
) -> tuple[dict[str, Any], bool]:
    """Gates that apply to SFT and DPO configs: preflight, data policy and composition, B0, DPO, components, determinism."""
    baseline = _baseline_state(root)
    required_artifacts = (
        all(baseline["artifacts"].values()) and baseline.get("artifact_contract_ok") is True
    )
    trainer_type = str(raw.get("trainer", {}).get("type", "")).lower()
    is_sft_config = _is_experiment_config(raw) and trainer_type == "sft"
    is_dpo_config = _is_experiment_config(raw) and trainer_type == "dpo"
    if is_sft_config or is_dpo_config:
        try:
            preflight = run_experiment_preflight(config_path, root=root)
            preflight_ok = preflight.overall_status == "PASS"
            preflight_detail = f"OpenGrad experiment preflight: {preflight.overall_status}"
        except (OSError, TypeError, ValueError) as exc:
            preflight_ok = False
            preflight_detail = f"OpenGrad experiment preflight failed: {exc}"
        add(
            "experiment_preflight",
            "PASS" if preflight_ok else "FAIL",
            preflight_detail,
            "PREFLIGHT_FAILED" if not preflight_ok else None,
        )
    else:
        add(
            "experiment_preflight",
            "PASS",
            "Configuration does not require a training experiment preflight",
        )
    if is_sft_config:
        dataset_ids = raw.get("datasets", {}).get("manifest_ids", [])
        eval_terms = {"heldout", "evaluation", "mcq", "judge", "preference", "bfcl", "tau2"}
        unsafe_ids = [
            str(item)
            for item in dataset_ids
            if any(term in str(item).lower() for term in eval_terms)
        ]
        add(
            "training_data_policy",
            "FAIL" if unsafe_ids else "PASS",
            f"Evaluation-only dataset IDs are excluded: {unsafe_ids}"
            if unsafe_ids
            else "SFT dataset manifest IDs contain no evaluation-only names",
            "NO_TRAINING_ON_EVAL_DATA" if unsafe_ids else None,
        )
    else:
        add(
            "training_data_policy",
            "PASS",
            "Not an SFT configuration; training data policy deferred",
        )
    if is_sft_config:
        yield_ok, yield_detail, yield_code = _renderability_state(root, raw)
        add(
            "renderability_yield",
            "PASS" if yield_ok else "FAIL",
            yield_detail,
            yield_code,
        )
    else:
        add(
            "renderability_yield",
            "PASS",
            "Not an SFT configuration; per-source trainability deferred",
        )
    if is_sft_config:
        composition_ok, composition_detail, composition_code = _supervision_composition_state(
            root, raw
        )
        add(
            "supervision_composition",
            "PASS" if composition_ok else "FAIL",
            composition_detail,
            composition_code,
        )
    else:
        add(
            "supervision_composition",
            "PASS",
            "Not an SFT configuration; supervision composition deferred",
        )
    add(
        "real_b0",
        "PASS" if baseline["real"] else "FAIL",
        baseline["status"],
        "BASELINE_NOT_FOUND" if not baseline["real"] else None,
    )
    add(
        "baseline_artifacts",
        "PASS" if required_artifacts else "FAIL",
        (
            f"{baseline['artifact_paths']}; "
            f"raw_valid_rate={baseline.get('parse_valid_rate')} "
            f"(min {baseline.get('min_parse_valid_rate')})"
        ),
        "BASELINE_NOT_FOUND" if not required_artifacts else None,
    )
    if is_dpo_config:
        dpo_ok, dpo_detail, dpo_code = _dpo_contract_state(root, raw)
        add("dpo_contract", "PASS" if dpo_ok else "FAIL", dpo_detail, dpo_code)
    else:
        add("dpo_contract", "PASS", "Not a DPO configuration; DPO contract deferred")
    if is_sft_config or is_dpo_config:
        components_ok, components_detail, components_code = _model_components_state(root, raw)
        add(
            "model_components_validation",
            "PASS" if components_ok else "FAIL",
            components_detail,
            components_code,
        )
    else:
        add(
            "model_components_validation",
            "PASS",
            "Not a training configuration; component validation deferred",
        )
    if is_sft_config or is_dpo_config:
        determinism_ok, determinism_detail, determinism_code = _determinism_state(
            root, config_path, raw
        )
        add(
            "determinism_declared",
            "PASS" if determinism_ok else "FAIL",
            determinism_detail,
            determinism_code,
        )
    else:
        add(
            "determinism_declared",
            "PASS",
            "Not a training configuration; determinism declaration deferred",
        )
    return baseline, is_dpo_config


def readiness(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config_path = _resolve_project_path(root, config_path, root / BASELINE_CONFIG).resolve()
    gates: list[dict[str, Any]] = []
    gate_map: dict[str, dict[str, Any]] = {}

    def add(name: str, status: str, details: str, code: str | None = None) -> None:
        gate = _gate(name, status, details, code)
        gates.append(gate)
        gate_map[name] = gate

    raw = _config_gates(add, config_path, root)

    config_is_baseline, revision, template_hash, tokenizer_revision = _model_identity_gates(
        add, raw
    )

    manifest = _evaluation_data_gates(add, config_is_baseline, raw, revision, root, template_hash)
    _contamination_gates(add, manifest, raw, root)

    _storage_gates(add, root)

    hw = _gpu_gates(add, root)

    baseline, is_dpo_config = _training_gates(add, config_path, raw, root)

    # Baseline execution is the operation that establishes real_b0. It may
    # require the hardware probe and repository/data contracts, but must not
    # require its own post-run artifacts or a prior GPU receipt. The receipt is
    # produced by the immediately preceding smoke stage in B0_WORKFLOW.
    baseline_prerequisites = {
        "repository_validation",
        "config_validation",
        "model_revision",
        "tokenizer_revision",
        "chat_template_contract",
        "evaluation_manifest",
        "evaluation_materialization",
        "contamination_gate",
        "evaluation_leakage",
        "disk_capacity",
        "artifact_storage",
        "native_parser",
        "gpu_probe",
    }
    ready_for_baseline = all(gate_map[name]["status"] == "PASS" for name in baseline_prerequisites)
    sft_names = baseline_prerequisites | {
        "gpu_boundary",
        "real_b0",
        "baseline_artifacts",
        "training_data_policy",
        "dataset_revision",
        "dataset_snapshot",
        "experiment_preflight",
        "renderability_yield",
        "supervision_composition",
        "model_components_validation",
        "determinism_declared",
    }
    ready_for_sft = all(gate_map[name]["status"] == "PASS" for name in sft_names)
    dpo_names = baseline_prerequisites | {
        "gpu_boundary",
        "real_b0",
        "baseline_artifacts",
        "experiment_preflight",
        "dpo_contract",
        "model_components_validation",
        "determinism_declared",
    }
    ready_for_dpo = is_dpo_config and all(gate_map[name]["status"] == "PASS" for name in dpo_names)
    blocking = [gate["name"] for gate in gates if gate["status"] == "FAIL"]
    warnings = [gate["name"] for gate in gates if gate["status"] == "WARN"]
    overall = "FAIL" if blocking else ("WARN" if warnings else "PASS")
    return {
        "schema_version": 1,
        "status": overall,
        "ready_for_baseline": ready_for_baseline,
        "ready_for_sft": ready_for_sft,
        "ready_for_dpo": ready_for_dpo,
        "blocking_gates": blocking,
        "warnings": warnings,
        "gates": gates,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": raw.get("dataset_hash", raw.get("datasets", {}).get("hashes")),
        "git_commit": _git_state(root)["commit"],
        "baseline": baseline,
        "gpu": hw.to_dict(),
        "config": str(config_path.relative_to(root))
        if config_path.is_relative_to(root)
        else str(config_path),
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
        receipt["checks"] = [
            {
                "name": "hardware",
                "status": "FAIL",
                "code": "GPU_UNAVAILABLE",
                "details": "No CUDA accelerator detected.",
            }
        ]
        receipt["limitations"].append("No model load or generation attempted.")
        return _write_gpu_receipt(root, receipt)
    checks: list[dict[str, Any]] = []
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        receipt["checks"] = [
            {
                "name": "dependencies",
                "status": "FAIL",
                "code": "GPU_SMOKE_FAILED",
                "details": str(exc),
            }
        ]
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
        if (
            model_id != "Qwen/Qwen3.5-2B"
            or revision != PINNED_MODEL_REVISION
            or template_hash != PINNED_TEMPLATE_HASH
        ):
            raise ValueError("GPU smoke requires the pinned Qwen model revision and template hash")
        kwargs = {"revision": revision, "trust_remote_code": False}
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **kwargs)
        checks.append({"name": "model_access", "status": "PASS", "details": model_id})
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Look up worker 12."}],
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        checks.append(
            {"name": "native_template", "status": "PASS", "characters": len(str(rendered))}
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", **kwargs
        )
        model.eval()
        device = next(model.parameters()).device
        if getattr(device, "type", None) != "cuda":
            raise RuntimeError(f"model loaded on {device}, not CUDA")
        checks.append({"name": "model_load", "status": "PASS", "device": str(device)})
        encoded = tokenizer(str(rendered), return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=SMOKE_MAX_NEW_TOKENS, do_sample=False)
        generated = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=False
        )
        checks.append({"name": "one_generation", "status": "PASS", "output_chars": len(generated)})
        from opengrad.formatting.parser import parse_qwen_native_output

        parsed = parse_qwen_native_output(generated)
        parser_ok = parsed.status == "RAW_VALID" and parsed.decision in {
            "CALL",
            "ANSWER",
            "CLARIFY",
            "UNSUPPORTED",
        }
        checks.append(
            {
                "name": "native_parser",
                "status": "PASS" if parser_ok else "FAIL",
                "parser_status": parsed.status,
                "decision": parsed.decision,
                "errors": parsed.errors or [],
            }
        )
        if not parser_ok:
            raise RuntimeError(
                f"native parser rejected smoke output: {parsed.status}: {parsed.errors or []}"
            )
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        checks.append(
            {
                "name": "vram",
                "status": "PASS",
                "allocated_gib": round(allocated, 3),
                "peak_allocated_gib": round(peak, 3),
            }
        )
        del model, tokenizer
        torch.cuda.empty_cache()
        checks.append({"name": "cleanup", "status": "PASS"})
        receipt["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - bounded smoke must emit an auditable failure, never hide it
        checks.append(
            {
                "name": "runtime",
                "status": "FAIL",
                "code": "GPU_SMOKE_FAILED",
                "details": f"{type(exc).__name__}: {exc}",
            }
        )
        receipt["limitations"].append(
            "Real baseline and SFT are blocked until this boundary passes."
        )
    receipt["checks"] = checks
    return _write_gpu_receipt(root, receipt)


def _write_gpu_receipt(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    path = root / "reports/hardware/qwen_gpu_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["artifact"] = str(path.relative_to(root))
    return receipt
