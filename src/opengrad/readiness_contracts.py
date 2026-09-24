"""Readiness building blocks: the pinned constants, file and git helpers, and the baseline, training-data,
config and manifest contracts that `opengrad.readiness` and `opengrad.readiness_states` read.

Split out of `readiness.py` on 2026-09-24 with no change in behaviour; `readiness` re-exports every name
tests and other modules import from it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from opengrad.contamination.audit import (
    QUARANTINE_PATH as CONTAMINATION_QUARANTINE_PATH,
)
from opengrad.contamination.audit import (
    load_quarantine,
)
from opengrad.evaluation.runner import SUPPORTED_ENGINES
from opengrad.experiments.store import ExperimentStore

BASELINE_CONFIG = Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml")


BASELINE_MANIFEST = Path("reports/evaluation/behavioral-heldout-v2.manifest.json")


BASELINE_METRICS = Path("reports/baselines/qwen35_2b_baseline/metrics.json")


BASELINE_PREDICTIONS = Path("reports/baselines/qwen35_2b_baseline/predictions.jsonl")


BASELINE_RESIDUALS = Path("reports/failures/qwen35_2b_baseline/residual-profile.json")


BASELINE_ENVIRONMENT = Path("reports/baselines/qwen35_2b_baseline/environment.json")


PINNED_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


# Generation budget for the bounded smoke. The smoke prompt elicits a tool call
# ("Look up worker 12." with a `lookup` tool), and a complete native call
# (`<tool_call>{"name": "lookup", "arguments": {"q": "worker 12"}}</tool_call>`) needs
# roughly 20 tokens. A smaller budget truncates mid-call, which the parser correctly
# rejects as UNCLOSED_TOOL_CALL, failing the boundary for a harness reason rather than a
# model or parser defect. Keep this comfortably above one complete call.
SMOKE_MIN_NEW_TOKENS = 64


SMOKE_MAX_NEW_TOKENS = 128


PINNED_TEMPLATE_HASH = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"


PINNED_EVALUATOR_REVISION = "2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc"


CANONICAL_MODEL_ID = "Qwen/Qwen3.5-2B"


TRAINING_SOURCE_IDS = {
    "xlam-function-calling-60k",
    "toolace",
    "looptool-23k",
    "glaive-function-calling-v2",
    "button",
    "when2call-sft",
}


TRAINING_SOURCE_MANIFESTS = {
    "data/processed/normalization-v1/xlam",
    "data/processed/normalization-v1/toolace",
    "data/processed/normalization-v1/looptool",
    "data/processed/normalization-v1/glaive",
    "data/processed/normalization-v1/button",
    "data/processed/normalization-v1/when2call-sft",
}


# The corpus a configuration trains on when it does not name one. Kept as the v1 default because
# B0 and every result recorded before Canonical-v2 is pinned to it.
DEFAULT_TRAINING_RELEASE_DIR = ".release/hf/toolpolicy-canonical-v1"


# The scanner labels the When2Call SFT split "when2call-sft" while release manifests call it
# "when2call"; the alias keeps the coverage comparison from reporting a spurious mismatch.
SOURCE_LABEL_ALIASES = {"when2call": "when2call-sft"}


def _configured_release_dir(root: Path, raw: dict[str, Any]) -> Path | None:
    """The release directory a config trains on, or None for the default corpus."""
    declared = (raw.get("datasets") or {}).get("release_dir")
    if not declared:
        return None
    path = Path(str(declared))
    return path if path.is_absolute() else root / path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - an unreadable config must project as absent, not crash status
        return {}
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow an arbitrary artifact field to a mapping without inventing a value for it.

    The ``x.get(k) if isinstance(x.get(k), dict) else {}`` idiom reads correctly but types as a
    union that still includes ``None``, so every later ``.get`` on it needs its own guard. One
    helper keeps the narrowing honest and the call sites readable.
    """
    return value if isinstance(value, dict) else {}


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _commit_is_ancestor(root: Path, recorded: Any, head: Any) -> bool:
    """Whether ``recorded`` is ``head`` or predates it in this history.

    Evidence must come from a real commit in this repository, but it must not be invalidated
    by the next commit: results generate documentation, so requiring equality with the
    current HEAD would expire every baseline the moment anything else was committed. Ancestry
    plus a clean-tree requirement says what is actually meant -- "produced from a clean
    checkout of a commit that is in this history" -- and still fails closed on an unknown,
    rewritten, or fabricated commit.
    """
    if not isinstance(recorded, str) or not isinstance(head, str) or not recorded or not head:
        return False
    if recorded == head:
        return True
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", recorded, head],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    # 0 = ancestor, 1 = not an ancestor, other = unknown ref/repo -> fail closed.
    return result.returncode == 0


def _min_parse_valid_rate(root: Path) -> float:
    """The pinned native-parse quality bound for the frozen baseline.

    A real model on a finite completion budget will occasionally truncate mid-tool-call, so
    requiring every row to parse is unsatisfiable rather than strict. The bound lives in the
    frozen config so it is reviewable and changing it is a contract change.
    """
    config = _read_yaml(root / BASELINE_CONFIG)
    quality = config.get("evaluations", {}).get("quality", {})
    value = quality.get("min_parse_valid_rate") if isinstance(quality, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.99


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
    parse_valid_rate = 0.0
    if predictions_path.exists():
        try:
            raw_rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [row.get("example_id") for row in raw_rows if isinstance(row, dict)]
            decisions = {"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"}
            # Two separate requirements, because conflating them made the gate unsatisfiable:
            #
            # 1. Structural integrity must hold for *every* row. This is about the artifact
            #    being a faithful record: one row per example, unique ids, recorded output,
            #    and a decision drawn from the canonical set (a malformed parse still yields
            #    a decision, and those rows are kept and reported rather than dropped).
            # 2. Native-parse quality is bounded, not required to be perfect. A real model
            #    given a finite completion budget will occasionally run past it mid-call; the
            #    parser is right to call that RAW_VALID's absence. Demanding 100% meant no
            #    real run could ever produce passing evidence, so the bound is pinned in the
            #    frozen config instead, and the measured rate is reported.
            structurally_valid = (
                bool(raw_rows)
                and all(isinstance(item, str) and item for item in ids)
                and len(ids) == len(set(ids)) == len(raw_rows)
                and all(
                    isinstance(row, dict)
                    and isinstance(row.get("raw_output"), str)
                    and row["raw_output"].strip()
                    and row.get("prediction", {}).get("decision") in decisions
                    for row in raw_rows
                )
            )
            raw_valid = sum(
                1 for row in raw_rows if row.get("parser", {}).get("status") == "RAW_VALID"
            )
            parse_valid_rate = raw_valid / len(raw_rows) if raw_rows else 0.0
            min_parse_valid_rate = _min_parse_valid_rate(root)
            predictions_valid = structurally_valid and parse_valid_rate >= min_parse_valid_rate
            if structurally_valid:
                prediction_rows = raw_rows
                # `structurally_valid` already requires every id to be a non-empty string, so
                # this keeps the values that check admitted and drops nothing real.
                prediction_ids = {item for item in ids if isinstance(item, str)}
        except (OSError, json.JSONDecodeError, TypeError):
            prediction_rows = []
    manifest = _read_json(root / BASELINE_MANIFEST)
    expected_records = sum(
        _safe_item_count(split) or 0
        for split in (manifest or {}).get("splits", [])
        if isinstance(split, dict)
    )
    expected_ids = _materialized_evaluation_ids(root, manifest)
    # Quarantined examples are excluded from evaluation, so the expected record count
    # must match what the runner actually loads rather than the raw materialized total.
    quarantined_ids = {
        example_id
        for ids in load_quarantine(root / CONTAMINATION_QUARANTINE_PATH).by_split().values()
        for example_id in ids
    }
    if expected_ids is not None:
        expected_ids = expected_ids - quarantined_ids
        expected_records = len(expected_ids)
    config = _read_yaml(root / BASELINE_CONFIG)
    manifest_file = root / BASELINE_MANIFEST
    manifest_sha256 = (
        hashlib.sha256(manifest_file.read_bytes()).hexdigest() if manifest_file.exists() else None
    )
    current_commit = _git_state(root).get("commit")
    experiment_records = _experiments(root)
    baseline_experiment = next(
        (
            record
            for record in experiment_records
            if record.get("experiment_id")
            in {"tool_calling/qwen35_2b/baseline", "qwen35_2b_baseline"}
        ),
        None,
    )
    residual_values = residuals.get("residuals", {}) if isinstance(residuals, dict) else {}
    residual_contract_ok = bool(
        isinstance(residuals, dict)
        and residuals.get("schema_version") == 1
        and residuals.get("baseline_experiment") == "tool_calling/qwen35_2b/baseline"
        and residuals.get("model_id") == CANONICAL_MODEL_ID
        and residuals.get("model_revision") == PINNED_MODEL_REVISION
        and residuals.get("manifest_sha256") == manifest_sha256
        and isinstance(residual_values, dict)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            for value in residual_values.values()
        )
    )
    # The engine that produced the baseline is part of the measurement, so it must be the
    # engine the frozen config declares, and its version must be recorded. A baseline from
    # an unnamed or unrecorded engine is not comparable to anything.
    declared_engine = (config.get("runtime") or {}).get("backend")
    recorded_engine = metrics.get("engine") if isinstance(metrics, dict) else None
    engine_contract_ok = bool(
        isinstance(declared_engine, str)
        and metrics
        and metrics.get("backend") == declared_engine
        and isinstance(recorded_engine, dict)
        and recorded_engine.get("name") == declared_engine
        and isinstance(recorded_engine.get("version"), str)
        and recorded_engine.get("version")
    )
    artifact_contract_ok = bool(
        metrics
        and metrics.get("status") == "EXECUTED"
        and engine_contract_ok
        and metrics.get("model_id") == config.get("model_id") == CANONICAL_MODEL_ID
        and metrics.get("model_revision") == config.get("model_revision") == PINNED_MODEL_REVISION
        and metrics.get("tokenizer_revision")
        == config.get("tokenizer_revision")
        == PINNED_MODEL_REVISION
        and metrics.get("manifest") == config.get("evaluations", {}).get("behavioral_manifest")
        and metrics.get("manifest_sha256") == manifest_sha256
        and metrics.get("template_hash") == config.get("template_hash") == PINNED_TEMPLATE_HASH
        and metrics.get("renderer") == "qwen3_5_2b_v1"
        and metrics.get("run_id") == "tool_calling/qwen35_2b/baseline"
        and metrics.get("records", 0) == len(prediction_rows) > 0
        and predictions_valid
        and expected_ids is not None
        and prediction_ids == expected_ids
        and metrics.get("records") == expected_records
        and residual_contract_ok
        and isinstance(residuals, dict)
        and residuals.get("sample_count") == metrics.get("records")
        and environment
        and environment.get("backend") == metrics.get("backend")
        and environment.get("engine") == metrics.get("engine")
        and environment.get("dry_run") is False
        and environment.get("run_id") == metrics.get("run_id")
        and environment.get("model_id") == CANONICAL_MODEL_ID
        and environment.get("model_revision") == PINNED_MODEL_REVISION
        and environment.get("manifest_sha256") == manifest_sha256
        and isinstance(metrics.get("git_commit"), str)
        and _commit_is_ancestor(root, metrics.get("git_commit"), current_commit)
        # Real evidence must come from a clean checkout: a run from a dirty tree cannot be
        # traced to a code state, which is the point of recording the commit at all.
        and metrics.get("git_dirty") is False
        and baseline_experiment is not None
        and baseline_experiment.get("status") in {"EVALUATED", "REVIEW", "PROMOTED"}
        and baseline_experiment.get("model_id") == CANONICAL_MODEL_ID
        and baseline_experiment.get("model_revision") == PINNED_MODEL_REVISION
    )
    real_status = bool(all(artifacts.values()) and artifact_contract_ok)
    dry_run_complete = bool(
        metrics
        and metrics.get("status") == "DRY_RUN"
        and metrics.get("backend") == "deterministic-mock"
        and metrics.get("records") == len(prediction_rows) > 0
    )
    return {
        "id": "qwen35_2b_baseline",
        "status": "REAL_COMPLETE"
        if real_status
        else ("CPU_DRY_RUN" if dry_run_complete else "PLANNED"),
        "real": real_status,
        "metrics": metrics,
        "residual_profile": residuals,
        "artifacts": artifacts,
        "artifact_contract_ok": artifact_contract_ok,
        "prediction_records": len(prediction_rows),
        "parse_valid_rate": round(parse_valid_rate, 6),
        "min_parse_valid_rate": _min_parse_valid_rate(root),
        "expected_records": expected_records,
        "artifact_paths": {
            key: str(path)
            for key, path in {
                "metrics": metrics_path,
                "predictions": predictions_path,
                "residual_profile": residual_path,
                "environment": env_path,
            }.items()
        },
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
    """Distinct held-out example ids, or None when they cannot be established.

    Duplicates *within* one split mean corruption and fail closed. Overlap *across* splits is
    a real property of the frozen benchmark -- the upstream llm-judge file repeats 300 rows
    from the mcq file byte for byte -- so the evaluation identity is the distinct union;
    otherwise those rows would be counted twice in every aggregate.
    """
    if not isinstance(manifest, dict):
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        result: set[str] = set()
        for split in manifest.get("splits", []):
            source = _resolve_project_path(root, split.get("source"), root / "__missing__")
            manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
            data = _read_json(manifest_path)
            if (
                not data
                or data.get("finalized") is not True
                or not isinstance(data.get("shards"), list)
                or not data["shards"]
            ):
                return None
            base = manifest_path.parent
            seen_in_split: set[str] = set()
            for name in data["shards"]:
                shard = base / str(name)
                for batch in pq.ParquetFile(shard).iter_batches(
                    batch_size=256, columns=["example_id"]
                ):
                    for row in batch.to_pylist():
                        value = row.get("example_id")
                        if not isinstance(value, str) or not value or value in seen_in_split:
                            return None
                        seen_in_split.add(value)
            result |= seen_in_split
        return result or None
    except Exception:  # noqa: BLE001 - readiness is a fail-closed projection
        # Corrupt/missing parquet must produce an unavailable ID set, never an
        # optimistic empty set.
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
    datasets = _as_dict(raw.get("datasets"))
    ids = datasets.get("manifest_ids")
    hashes = datasets.get("hashes")
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(item, str) and item.strip() for item in ids)
    ):
        return False, "SFT dataset manifest IDs are missing or invalid"
    extra_hash_aliases = (
        {"preference"} if str(raw.get("trainer", {}).get("type", "")).lower() == "dpo" else set()
    )
    if (
        not isinstance(hashes, dict)
        or set(hashes) - set(ids) - extra_hash_aliases
        or not set(ids).issubset(hashes)
        or not all(_check_revision(value) for value in hashes.values())
    ):
        return False, "Training dataset hashes must pin every manifest ID with no unknown aliases"
    registry = _read_dataset_registry(root)
    unknown = [item for item in ids if item not in registry]
    if unknown:
        return False, f"SFT dataset IDs are absent from registry: {unknown}"
    unsafe = []
    for item in ids:
        row = registry[item]
        # Registry stage names are spelled with a `future_` prefix (`future_sft`,
        # `future_preference`), so an exact match on "preference" missed
        # `future_preference` entirely and let a preference corpus into SFT. Normalise the
        # prefix away before testing the stage.
        intended = (
            {str(value).lower().removeprefix("future_") for value in row.get("intended_stages", [])}
            if isinstance(row.get("intended_stages"), list)
            else set()
        )
        allowed = (
            {str(value).lower() for value in row.get("allowed_splits", [])}
            if isinstance(row.get("allowed_splits"), list)
            else set()
        )
        # A dataset is unsafe for SFT if it is *intended* for evaluation or preference, or if
        # it *permits* evaluation/held-out splits. The previous test inspected
        # forbidden_splits, which is inverted: declaring a split forbidden is exactly what
        # makes a corpus safe to train on, so it refused the datasets that had done the right
        # thing. A corpus that forbids the held-out tests is fine; one that allows them is not.
        if (
            "evaluation" in intended
            or "preference" in intended
            or allowed & {"evaluation", "heldout", "mcq_test", "llm_judge_test"}
        ):
            unsafe.append(item)
        source_revision = (
            row.get("source_revision", {}).get("value")
            if isinstance(row.get("source_revision"), dict)
            else None
        )
        if not _check_revision(source_revision):
            return False, f"SFT dataset source revision is not pinned for {item}"
        processed = _as_dict(row.get("processed_dataset_hash"))
        processed_value = processed.get("value")
        if not isinstance(processed_value, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", processed_value
        ):
            return False, f"SFT processed dataset hash is not materialized for {item}"
        if hashes.get(item) != processed_value:
            return False, f"SFT dataset hash does not match registry for {item}"
    if unsafe:
        return False, f"Evaluation/preference datasets cannot enter SFT: {unsafe}"
    return True, "SFT dataset IDs, registry eligibility, source revisions, and hashes are pinned"


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
        "schema_version",
        "status",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "renderer",
        "template_hash",
        "seed",
        "generation",
        "evaluations",
        "runtime",
        "outputs",
        "provenance",
    }
    missing = sorted(required - set(raw))
    if missing:
        return False, f"baseline config is missing required field(s): {', '.join(missing)}"
    if raw.get("schema_version") != 1 or raw.get("status") != "FROZEN_PRE_GPU":
        return False, "baseline config must use schema 1 and FROZEN_PRE_GPU status"
    if (
        raw.get("model_id") != CANONICAL_MODEL_ID
        or raw.get("model_revision") != PINNED_MODEL_REVISION
        or raw.get("tokenizer_revision") != PINNED_MODEL_REVISION
    ):
        return (
            False,
            "baseline config model and tokenizer revisions must match the pinned canonical model",
        )
    if (
        raw.get("renderer") != "qwen3_5_2b_v1"
        or raw.get("template_hash") != PINNED_TEMPLATE_HASH
        or raw.get("seed") != 0
    ):
        return False, "baseline renderer, template hash, or seed is not pinned"
    generation = raw.get("generation")
    if (
        not isinstance(generation, dict)
        or generation.get("do_sample") is not False
        or generation.get("temperature") != 0.0
        or generation.get("top_p") != 1.0
        or not isinstance(generation.get("max_new_tokens"), int)
        or generation["max_new_tokens"] < 1
    ):
        return False, "baseline generation must be deterministic and bounded"
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        return False, "baseline runtime must be an object"
    engine = runtime.get("backend")
    if engine not in SUPPORTED_ENGINES:
        return False, (
            f"baseline runtime must declare a supported engine "
            f"({', '.join(SUPPORTED_ENGINES)}); got {engine!r}"
        )
    if engine == "vllm" and not isinstance(runtime.get("vllm_version"), str):
        return False, "vllm runtime must pin vllm_version"
    if engine == "transformers" and not isinstance(runtime.get("transformers_version"), str):
        return False, "transformers runtime must pin transformers_version"
    if engine == "vllm":
        window = runtime.get("max_model_len")
        if not isinstance(window, int):
            return False, "vllm runtime must pin max_model_len"
        if window < int(runtime.get("context_length", 0)) + int(
            (raw.get("generation") or {}).get("max_new_tokens", 0)
        ):
            return False, (
                "vllm max_model_len must cover context_length plus the completion budget, "
                "because overflow prompts are bucketed rather than truncated"
            )
    if (
        runtime.get("precision") != "bfloat16"
        or runtime.get("device_policy") != "accelerator_required"
        or not isinstance(runtime.get("context_length"), int)
        or runtime["context_length"] < 1
    ):
        return (
            False,
            "baseline runtime must pin BF16 precision, an accelerator requirement, and a context length",
        )
    evaluations = raw.get("evaluations")
    provenance = raw.get("provenance")
    outputs = raw.get("outputs")
    if (
        not isinstance(evaluations, dict)
        or not isinstance(provenance, dict)
        or not isinstance(outputs, dict)
    ):
        return False, "baseline evaluations, outputs, and provenance must be objects"
    if (
        not isinstance(evaluations.get("behavioral_manifest"), str)
        or provenance.get("evaluator_revision") != PINNED_EVALUATOR_REVISION
        or provenance.get("manifest_status") != "FROZEN_PRE_GPU"
    ):
        return False, "baseline evaluation provenance is not pinned"
    quality = evaluations.get("quality")
    if not isinstance(quality, dict):
        return False, "baseline evaluation config must define a quality bound"
    rate = quality.get("min_parse_valid_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 < float(rate) <= 1:
        return False, "min_parse_valid_rate must be a number in (0, 1]"
    if set(outputs) != {"predictions", "metrics", "residual_profile", "environment"} or not all(
        isinstance(value, str) and value for value in outputs.values()
    ):
        return False, "baseline outputs must name all four artifact paths"
    return (
        True,
        "frozen baseline model, renderer, generation, runtime, and provenance contract match",
    )


def _is_baseline_config(raw: dict[str, Any]) -> bool:
    return _baseline_config_contract(raw)[0]


def _is_experiment_config(raw: dict[str, Any]) -> bool:
    required = {
        "experiment_id",
        "hypothesis",
        "model",
        "datasets",
        "trainer",
        "evaluation",
        "checkpointing",
        "promotion",
        "reproducibility",
    }
    return required.issubset(raw)


def _manifest_contract_ok(
    root: Path, manifest: dict[str, Any] | None, expected_revision: Any, expected_template: Any
) -> tuple[bool, str]:
    if not manifest:
        return False, "evaluation manifest is missing or invalid JSON"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("frozen") is not True
        or manifest.get("status") not in {"FROZEN_PRE_GPU", "MATERIALIZED", "EXECUTED"}
    ):
        return False, "manifest must use schema 1 and be frozen with a supported status"
    if manifest.get("manifest_id") != "behavioral-heldout-v2":
        return False, "manifest ID does not match the frozen behavioral evaluation contract"
    if manifest.get("freeze_revision") != PINNED_EVALUATOR_REVISION:
        return False, "manifest freeze revision does not match the pinned evaluator revision"
    contract = manifest.get("model_renderer_contract")
    if not isinstance(contract, dict):
        return False, "manifest model_renderer_contract is missing"
    if (
        contract.get("model_revision") != PINNED_MODEL_REVISION
        or expected_revision != PINNED_MODEL_REVISION
    ):
        return False, "model revision does not match the pinned baseline revision"
    if contract.get("renderer") != "qwen3_5_2b_v1":
        return False, "manifest renderer does not match the pinned native renderer"
    if (
        contract.get("template_hash") != PINNED_TEMPLATE_HASH
        or expected_template != PINNED_TEMPLATE_HASH
    ):
        return False, "template hash does not match the pinned baseline template"
    splits = manifest.get("splits")
    if not isinstance(splits, list) or not splits:
        return False, "manifest has no valid split contract"
    ids = set()
    for split in splits:
        if (
            not isinstance(split, dict)
            or not isinstance(split.get("id"), str)
            or not split["id"]
            or split["id"] in ids
            or _safe_item_count(split) is None
        ):
            return False, "manifest has an invalid or duplicate split item contract"
        ids.add(split["id"])
        if not isinstance(split.get("content_hash"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", split["content_hash"]
        ):
            return False, f"split {split.get('id')} has no pinned content hash"
        if not isinstance(split.get("source"), str) or not split["source"]:
            return False, f"split {split.get('id')} has no materialization source"
    return True, "frozen model, renderer, template, split, and content-hash contract match"
