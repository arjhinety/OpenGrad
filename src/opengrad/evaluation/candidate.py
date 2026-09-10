"""Candidate evaluation against the frozen behavioral held-out.

A trained checkpoint cannot be measured by ``run_baseline``: that path refuses any config that
is not the pinned canonical model, which is exactly the guard that keeps B0 immutable. But the
comparison the experiment needs is a candidate measured *the same way* as B0, so this module
adds the missing path rather than relaxing that guard.

What is deliberately shared with the baseline:

* the frozen behavioral manifest,
* the model-family renderer and the pinned template hash,
* the generation settings and the context bucketing,
* the engine and its version,
* the native parser and the routing metrics,

all through :func:`opengrad.evaluation.runner.measure_predictions`, so the two runs cannot
drift into two different measurements.

What is deliberately different: the model is a checkpoint directory rather than a Hub
revision, outputs go to a candidate namespace and never to the baseline evidence paths, and the
result records the lineage (base model, checkpoint, parent experiment) that makes the delta
attributable. Baseline artifacts are never written or overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.env_capture import capture
from opengrad.evaluation.routing import routing_metrics
from opengrad.evaluation.runner import (
    PINNED_EVALUATOR_REVISION,
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
    _project_path,
    _residuals,
    _write_json,
    build_backend,
    load_evaluation_examples,
    measure_predictions,
)
from opengrad.formatting.parser import parse_qwen_native_output  # noqa: F401  (contract marker)

CANDIDATE_STATUS = "CANDIDATE_EVALUATION"
BASELINE_CONFIG = Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml")
BASELINE_METRICS = Path("reports/baselines/qwen35_2b_baseline/metrics.json")

REQUIRED_FIELDS = {
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

# Fields that define the measurement. A candidate may change the model and nothing else, or the
# delta is not a measurement of the model.
INVARIANT_PATHS = (
    ("renderer",),
    ("template_hash",),
    ("seed",),
    ("generation",),
    ("runtime",),
    ("evaluations", "behavioral_manifest"),
)


def _dig(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_candidate_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Refuse anything that would make the comparison meaningless."""
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        raise ValueError(f"candidate config is missing required field(s): {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("candidate config must be schema version 1")
    if config["status"] != CANDIDATE_STATUS:
        raise ValueError(f"candidate config status must be {CANDIDATE_STATUS}")

    checkpoint = Path(str(config["model_id"]))
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError(
            f"candidate model_id must be an existing checkpoint directory: {checkpoint}"
        )
    for required in ("config.json", "tokenizer_config.json"):
        if not (checkpoint / required).is_file():
            raise ValueError(f"checkpoint is not loadable, missing {required}: {checkpoint}")

    if str(config["tokenizer_revision"]) != PINNED_MODEL_REVISION:
        raise ValueError("candidate must use the pinned tokenizer revision")

    baseline = yaml.safe_load((root / BASELINE_CONFIG).read_text(encoding="utf-8"))
    for path in INVARIANT_PATHS:
        expected, actual = _dig(baseline, path), _dig(config, path)
        if actual != expected:
            raise ValueError(
                f"candidate changes the measurement contract at {'.'.join(path)}: "
                f"baseline={expected!r} candidate={actual!r}"
            )

    provenance = config["provenance"]
    for field in ("parent_experiment_id", "checkpoint_id", "checkpoint_step", "base_model_id"):
        if not provenance.get(field):
            raise ValueError(f"candidate provenance must record {field}")
    if provenance.get("evaluator_revision") != PINNED_EVALUATOR_REVISION:
        raise ValueError("candidate provenance must pin the evaluator revision")
    if str(config["template_hash"]) != PINNED_TEMPLATE_HASH:
        raise ValueError("candidate template hash does not match the frozen template")
    return config


def run_candidate_evaluation(
    config_path: Path,
    *,
    root: Path | None = None,
    backend: Any = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure one checkpoint on the frozen held-out set and diff it against B0."""
    import time

    config_path = Path(config_path).resolve()
    root = (root or config_path.parents[3]).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("candidate config must be a YAML object")
    validate_candidate_config(config, root)

    manifest_path = _project_path(
        root, config["evaluations"]["behavioral_manifest"], "behavioral manifest"
    )
    examples = load_evaluation_examples(root, manifest_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        examples = examples[:limit]

    engine_name = str(config["runtime"].get("backend", "vllm"))
    model = backend
    if model is None:
        if dry_run:
            from opengrad.benchmarks.backends.mock import DeterministicFakeBackend

            model = DeterministicFakeBackend(mode="ar")
        else:
            model = build_backend(engine_name, config)
    renderer = Qwen35_2BRenderer(
        revision=str(config["tokenizer_revision"]),
        enable_thinking=bool(config.get("thinking", False)),
    )

    started = time.monotonic()
    predictions = measure_predictions(config, examples, model, renderer)

    output_config = config.get("outputs")
    if not isinstance(output_config, dict):
        raise TypeError("candidate config must define an outputs object")
    outputs = {
        name: _project_path(root, value, f"{name} output") for name, value in output_config.items()
    }
    run_dir = root / "runs" / str(config["provenance"]["parent_experiment_id"])
    for path in outputs.values():
        if not path.is_relative_to(run_dir):
            raise ValueError(f"candidate outputs must live under {run_dir}: {path}")

    actual = [str(row["expected_decision"]) for row in predictions]
    predicted = [str(row["prediction"]["decision"]) for row in predictions]
    metrics = routing_metrics(actual, predicted) if predictions else {"records": 0}

    from collections import Counter

    parse_counts = Counter(str(row["parser"]["status"]) for row in predictions)
    context_counts = Counter(str(row["context_bucket"]) for row in predictions)
    engine = None
    engine_metadata = getattr(model, "engine_metadata", None)
    if callable(engine_metadata):
        engine = engine_metadata()

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": f"{config['provenance']['parent_experiment_id']}/eval/{config['provenance']['checkpoint_id']}",
        "status": "DRY_RUN" if dry_run else "EXECUTED",
        "kind": CANDIDATE_STATUS,
        "model_id": str(config["model_id"]),
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "backend": model.name,
        "engine": engine,
        "manifest": str(config["evaluations"]["behavioral_manifest"]),
        "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "renderer": "qwen3_5_2b_v1",
        "template_hash": PINNED_TEMPLATE_HASH,
        "lineage": dict(config["provenance"]),
        "records": len(predictions),
        "parse_valid_rate": (
            round(
                sum(1 for row in predictions if row["parser"]["status"] == "RAW_VALID")
                / len(predictions),
                6,
            )
            if predictions
            else 0.0
        ),
        "generation": config["generation"],
        "routing": metrics,
        "parser_status": dict(sorted(parse_counts.items())),
        "context_buckets": dict(sorted(context_counts.items())),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "artifacts": {
            name: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for name, path in outputs.items()
        },
    }

    baseline_metrics_path = root / BASELINE_METRICS
    if baseline_metrics_path.is_file():
        baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        result["baseline_comparison"] = compare_routing(
            baseline.get("routing", {}), metrics, baseline.get("run_id")
        )

    residual = _residuals(
        predictions,
        str(result["run_id"]),
        model_id=str(result["model_id"]),
        model_revision=str(result["model_revision"]),
        manifest_sha256=str(result["manifest_sha256"]),
    )
    environment = capture(root)
    environment.update(
        {
            "run_id": result["run_id"],
            "backend": model.name,
            "engine": engine,
            "dry_run": dry_run,
            "model_id": result["model_id"],
            "manifest_sha256": result["manifest_sha256"],
        }
    )

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs.setdefault("predictions", run_dir / "eval" / "predictions.jsonl")
    # The generator writes concurrently; stream predictions before the aggregate artifacts so a
    # crash mid-run still leaves the raw generations for re-scoring.
    with outputs["predictions"].open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(outputs["metrics"], result)
    _write_json(outputs["residual_profile"], residual)
    _write_json(outputs["environment"], environment)
    return result


def compare_routing(
    baseline: dict[str, Any], candidate: dict[str, Any], baseline_id: str | None = None
) -> dict[str, Any]:
    """Delta of the routing metrics, plus an explicit better/worse verdict per metric.

    Only metrics present in both are compared, so adding a metric to a future evaluator cannot
    silently turn into a zero-valued regression here.
    """
    keys = [
        "call_precision",
        "call_recall",
        "call_f1",
        "over_call_rate",
        "under_call_rate",
        "clarification_accuracy",
        "unsupported_accuracy",
        "parse_valid_rate",
    ]
    deltas: dict[str, Any] = {}
    for key in keys:
        if key in baseline and key in candidate:
            before, after = float(baseline[key]), float(candidate[key])
            # For the error rates, down is better; for the rest, up is better.
            lower_is_better = key.endswith("_rate") and key not in {
                "call_recall",
                "parse_valid_rate",
            }
            improved = after < before if lower_is_better else after > before
            deltas[key] = {
                "baseline": round(before, 6),
                "candidate": round(after, 6),
                "delta": round(after - before, 6),
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "verdict": "IMPROVED"
                if improved
                else ("UNCHANGED" if after == before else "REGRESSED"),
            }
    return {
        "baseline_run_id": baseline_id,
        "candidate_run_id": candidate.get("run_id"),
        "metrics": deltas,
        "improved": sorted(k for k, v in deltas.items() if v["verdict"] == "IMPROVED"),
        "regressed": sorted(k for k, v in deltas.items() if v["verdict"] == "REGRESSED"),
    }


def write_candidate_config(
    root: Path,
    *,
    checkpoint: Path,
    checkpoint_id: str,
    checkpoint_step: int,
    parent_experiment_id: str,
    out_path: Path,
) -> Path:
    """Produce a candidate config by copying the frozen measurement and swapping the model.

    Generated rather than hand-written so the invariant block is literally the baseline's,
    which is what :func:`validate_candidate_config` then re-checks.
    """
    baseline = yaml.safe_load((root / BASELINE_CONFIG).read_text(encoding="utf-8"))
    candidate = dict(baseline)
    candidate["status"] = CANDIDATE_STATUS
    # Prefer a repo-relative path: an absolute one bakes this machine's layout into a committed
    # config, so the same file would not measure the same checkpoint anywhere else.
    try:
        candidate["model_id"] = str(checkpoint.resolve().relative_to(root.resolve()))
    except ValueError:
        candidate["model_id"] = str(checkpoint)
    candidate["model_revision"] = None
    outputs = {
        name: str(
            Path("runs") / parent_experiment_id / "eval" / checkpoint_id / Path(str(value)).name
        )
        for name, value in dict(baseline["outputs"]).items()
    }
    candidate["outputs"] = outputs
    candidate["provenance"] = dict(baseline["provenance"])
    candidate["provenance"].update(
        {
            "parent_experiment_id": parent_experiment_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "base_model_id": baseline["model_id"],
            "base_model_revision": baseline["model_revision"],
        }
    )
    candidate.pop("hardware", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# GENERATED by opengrad.evaluation.candidate.write_candidate_config.\n"
        "# The measurement block is the frozen baseline's, unmodified; only the model and the\n"
        "# output namespace differ, so the delta is attributable to the checkpoint.\n"
        + yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )
    return out_path
