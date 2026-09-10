"""The executable baseline boundary.

The runner is intentionally backend-agnostic.  Everything around
``InferenceBackend.generate`` is exercised by the deterministic backend, so a
GPU run only swaps the model-generation implementation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import yaml

from opengrad.contamination.audit import QUARANTINE_PATH, load_quarantine
from opengrad.data.canonical import CanonicalEvaluationExample
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.env_capture import capture
from opengrad.evaluation.routing import routing_metrics
from opengrad.experiments.schema import ExperimentRecord, ExperimentStatus
from opengrad.experiments.store import ExperimentStore
from opengrad.formatting.parser import ParsedNativeOutput, parse_qwen_native_output

PINNED_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
PINNED_TEMPLATE_HASH = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
PINNED_EVALUATOR_REVISION = "2d97c7d5a8de0b16a2e58e4376e231fe06ab16dc"
CANONICAL_MODEL_ID = "Qwen/Qwen3.5-2B"
BASELINE_EXPERIMENT_ID = "tool_calling/qwen35_2b/baseline"


class InferenceBackend(Protocol):
    """Stable generation interface shared by fake and model-backed runners."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str: ...


def _decision(value: str) -> str:
    return {
        "tool_call": "CALL",
        "call": "CALL",
        "must_call": "CALL",
        "direct": "ANSWER",
        "can_answer": "ANSWER",
        "answer": "ANSWER",
        "request_for_info": "CLARIFY",
        "clarify": "CLARIFY",
        "cannot_answer": "UNSUPPORTED",
        "unsupported": "UNSUPPORTED",
    }.get(value.casefold(), "UNKNOWN")


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON column") from exc
    return value


def _qwen_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Accept legacy evaluation aliases while keeping training schemas strict."""
    result = dict(tool)
    parameters = result.get("parameters")

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "object"
                    if key == "type" and item == "dict"
                    else "array"
                    if key == "type" and item == "list"
                    else visit(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    if isinstance(parameters, dict):
        result["parameters"] = visit(parameters)
    return result


def _resolve_inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    return result


def _materialized_rows(root: Path, split: dict[str, Any]) -> list[dict[str, Any]]:
    source = _resolve_inside(root, split.get("source"), f"split {split.get('id')} source")
    manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation materialization manifest is missing: {manifest_path}")
    shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if shard_manifest.get("finalized") is not True or not isinstance(shard_manifest.get("shards"), list) or not shard_manifest["shards"]:
        raise ValueError(f"evaluation materialization is not finalized: {manifest_path}")
    expected_count = split.get("items")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError(f"split {split.get('id')} has an invalid item count")
    rows: list[dict[str, Any]] = []
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    seen_shards: set[str] = set()
    seen_ids: set[str] = set()
    for shard_name in shard_manifest["shards"]:
        if not isinstance(shard_name, str) or not shard_name or shard_name in seen_shards:
            raise ValueError(f"split {split.get('id')} has invalid shard names")
        seen_shards.add(shard_name)
        shard_path = _resolve_inside(manifest_path.parent, shard_name, f"split {split.get('id')} shard")
        if not shard_path.is_file():
            raise FileNotFoundError(f"evaluation shard is missing: {shard_path}")
        for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                example_id = row.get("example_id") if isinstance(row, dict) else None
                if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
                    raise ValueError(f"split {split.get('id')} contains missing or duplicate example_id")
                seen_ids.add(example_id)
                rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"split {split.get('id')} has {len(rows)} rows; expected {expected_count}")
    digest = hashlib.sha256(b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)).hexdigest()
    if shard_manifest.get("content_hash") != split.get("content_hash") or shard_manifest.get("content_hash") != digest:
        raise ValueError(f"split {split.get('id')} content hash does not match materialized rows")
    return rows


def load_evaluation_examples(root: Path, manifest_path: Path) -> list[CanonicalEvaluationExample]:
    """Load exactly the finalized, hash-bound materialized splits.

    Examples quarantined after a CONTAMINATED Level-5 contamination verdict are excluded:
    a quarantined benchmark item must not be measured by B0 or any later held-out run.
    """
    root = root.resolve()
    if manifest_path.is_absolute():
        if not manifest_path.resolve().is_relative_to(root):
            raise ValueError("evaluation manifest must remain inside the repository")
        manifest_value = str(manifest_path.resolve().relative_to(root))
    else:
        manifest_value = str(manifest_path)
    manifest_path = _resolve_inside(root, manifest_value, "evaluation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("manifest_id") != "behavioral-heldout-v2" or manifest.get("frozen") is not True or manifest.get("status") not in {"FROZEN_PRE_GPU", "MATERIALIZED", "EXECUTED"}:
        raise ValueError("evaluation manifest is not the pinned frozen contract")
    contract = manifest.get("model_renderer_contract", {})
    if contract.get("model_revision") != PINNED_MODEL_REVISION or contract.get("renderer") != "qwen3_5_2b_v1" or contract.get("template_hash") != PINNED_TEMPLATE_HASH:
        raise ValueError("evaluation manifest renderer contract is not pinned")
    quarantined = load_quarantine(root / QUARANTINE_PATH).by_split()
    examples: list[CanonicalEvaluationExample] = []
    for split in manifest.get("splits", []):
        excluded = quarantined.get(str(split.get("id")), set())
        for raw in _materialized_rows(root, split):
            if excluded and str(raw.get("example_id")) in excluded:
                continue
            row = dict(raw)
            for key in ("source", "tools", "candidates", "metadata"):
                row[key] = _json(row[key])
            example = CanonicalEvaluationExample(
                str(row["example_id"]), row["source"], str(row["question"]),
                [_qwen_tool(tool) for tool in row["tools"]], str(row["expected_decision"]),
                row["candidates"], row["metadata"],
            )
            example.validate()
            examples.append(example)
    return examples


class FakeDeterministicBackend:
    """CPU backend that emits native-looking output for every routing class."""

    name = "deterministic-mock"

    def __init__(
        self,
        *,
        adversarial: bool = False,
        scenarios: dict[str, str] | None = None,
    ) -> None:
        self.adversarial = adversarial
        self.scenarios = scenarios or {}
        self.last_truncated = False

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str:
        self.last_truncated = False
        explicit = self.scenarios.get(example.example_id)
        if explicit is not None:
            if explicit == "cutoff":
                self.last_truncated = True
                return '<tool_call>{"name":"cutoff","arguments":'
            if explicit == "observation":
                return (
                    '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\nObservation: value x'
                )
            if explicit == "multiple":
                return '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>'
            if explicit == "malformed":
                return '<tool_call>{"name":"broken","arguments":</tool_call>'
            return explicit
        expected = _decision(example.expected_decision)
        if self.adversarial:
            index = int(hashlib.sha256(example.example_id.encode()).hexdigest()[:2], 16) % 7
            if index == 0:
                return '<tool_call>{"name":"broken","arguments":</tool_call>'
            if index == 1:
                return (
                    '<tool_call>{"name":"a","arguments":{}}</tool_call>'
                    '<tool_call>{"name":"b","arguments":{}}</tool_call>'
                )
            if index == 2:
                return "Could you clarify which account you mean?"
            if index == 3:
                return "I cannot help with that unsupported request."
            if index == 4:
                return (
                    '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\nObservation: value x'
                )
            if index == 5:
                self.last_truncated = True
                return '<tool_call>{"name":"cutoff","arguments":'
        if expected == "CALL":
            name = str(example.tools[0].get("name", "tool")) if example.tools else "tool"
            return f"<tool_call>{json.dumps({'name': name, 'arguments': {}}, sort_keys=True)}</tool_call>"
        if expected == "CLARIFY":
            return "Could you clarify the missing information?"
        if expected == "UNSUPPORTED":
            return "I cannot help with that unsupported request."
        return "Here is the requested information."


class TransformersInferenceBackend:
    """Lazy Transformers backend; importing/loading it is never needed for dry-run."""

    name = "transformers"

    def __init__(self, *, model_id: str, revision: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self._tokenizer: Any = None
        self._model: Any = None
        self.last_truncated = False

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("GPU inference requires the evaluation extra") from exc
        kwargs: dict[str, Any] = {"revision": self.revision, "trust_remote_code": False}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto", **kwargs
        )
        self._model.eval()
        return self._tokenizer, self._model

    def generate(
        self,
        prompt: str,
        *,
        example: CanonicalEvaluationExample,
        generation_config: dict[str, Any],
    ) -> str:
        tokenizer, model = self._load()
        torch = importlib.import_module("torch")

        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        kwargs = {
            key: value
            for key, value in generation_config.items()
            if key in {"temperature", "top_p", "max_new_tokens", "do_sample"}
        }
        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)
        generated = output[0, encoded["input_ids"].shape[1] :]
        self.last_truncated = len(generated) >= int(generation_config.get("max_new_tokens", 0))
        return str(tokenizer.decode(generated, skip_special_tokens=False))


def _prediction(
    example: CanonicalEvaluationExample,
    prompt: str,
    input_tokens: int,
    context_limit: int,
    raw: str,
    parsed: ParsedNativeOutput,
) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "source": example.source.get("dataset_id", "unknown"),
        "expected_decision": _decision(example.expected_decision),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_tokens": input_tokens,
        "context_bucket": "overflow" if input_tokens > context_limit else "base",
        "prediction": {
            "decision": parsed.decision,
            "calls": [
                {"name": call.name, "arguments": call.arguments, "id": call.call_id}
                for call in parsed.calls
            ],
            "content": parsed.content,
        },
        "raw_output": raw,
        "parser": {
            "status": parsed.status,
            "errors": parsed.errors or [],
            "truncated": parsed.truncated,
        },
    }


def _residuals(
    predictions: Iterable[dict[str, Any]],
    baseline_experiment: str,
    *,
    model_id: str,
    model_revision: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    for row in predictions:
        total += 1
        expected = row["expected_decision"]
        predicted = row["prediction"]["decision"]
        if row["parser"]["status"] != "RAW_VALID":
            counts["FORMAT_ERROR"] += 1
        elif expected == "CALL" and predicted != "CALL":
            counts["UNDER_CALL"] += 1
        elif expected != "CALL" and predicted == "CALL":
            counts["OVER_CALL"] += 1
        elif expected != predicted:
            counts["PREMATURE_STOP"] += 1
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "manifest_sha256": manifest_sha256,
        "baseline_experiment": baseline_experiment,
        "sample_count": total,
        "residuals": {key: value / total for key, value in sorted(counts.items())} if total else {},
        "failure_counts": dict(sorted(counts.items())),
    }


def _project_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must remain inside the repository")
    return path


def _validate_baseline_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "status", "model_id", "model_revision", "tokenizer_revision", "renderer", "template_hash", "seed", "generation", "evaluations", "runtime", "outputs", "provenance"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"baseline config is missing required field(s): {', '.join(missing)}")
    if config["schema_version"] != 1 or config["status"] != "FROZEN_PRE_GPU":
        raise ValueError("baseline config must be schema 1 and FROZEN_PRE_GPU")
    if config["model_id"] != CANONICAL_MODEL_ID or config["model_revision"] != PINNED_MODEL_REVISION or config["tokenizer_revision"] != PINNED_MODEL_REVISION:
        raise ValueError("baseline config must pin the canonical model, model revision, and tokenizer revision")
    if config["renderer"] != "qwen3_5_2b_v1" or config["template_hash"] != PINNED_TEMPLATE_HASH or config["seed"] != 0:
        raise ValueError("baseline config renderer, template, or seed is not the frozen contract")
    if config.get("evaluations", {}).get("behavioral_manifest") != "reports/evaluation/behavioral-heldout-v2.manifest.json":
        raise ValueError("baseline config must use the pinned behavioral-heldout-v2 manifest")
    provenance = config["provenance"]
    if provenance.get("evaluator_revision") != PINNED_EVALUATOR_REVISION or provenance.get("manifest_status") != "FROZEN_PRE_GPU":
        raise ValueError("baseline provenance must pin the evaluator revision and frozen manifest status")
    generation = config["generation"]
    if not isinstance(generation, dict) or generation.get("max_new_tokens", 0) < 1 or generation.get("do_sample") is not False or generation.get("temperature") != 0.0 or generation.get("top_p") != 1.0:
        raise ValueError("baseline generation config is not deterministic and bounded")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or runtime.get("backend") != "transformers" or runtime.get("precision") != "bfloat16" or runtime.get("device_policy") != "accelerator_required" or not isinstance(runtime.get("context_length"), int) or runtime["context_length"] < 1:
        raise ValueError("baseline runtime config is not the pinned transformers/BF16 contract")
    evaluations = config["evaluations"]
    if not isinstance(evaluations, dict) or not isinstance(evaluations.get("behavioral_manifest"), str) or evaluations.get("evaluator_revision") not in {None, PINNED_MODEL_REVISION}:
        raise ValueError("baseline evaluation config is malformed")
    if not isinstance(config["provenance"], dict):
        raise TypeError("baseline provenance must be an object")
    outputs = config["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"predictions", "metrics", "residual_profile", "environment"} or not all(isinstance(value, str) and value for value in outputs.values()):
        raise ValueError("baseline must define exactly predictions, metrics, residual_profile, and environment outputs")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_baseline(
    config_path: Path,
    *,
    root: Path | None = None,
    backend: InferenceBackend | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the frozen baseline from manifest through artifacts.

    Real runs are immutable lifecycle events: they cannot overwrite an existing
    baseline artifact set or ledger record.  A dry-run is explicitly plumbing
    only and is redirected away from the canonical evidence paths when the
    default config is used.
    """
    root = (root or config_path.parents[3]).resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("baseline config must be a YAML object")
    _validate_baseline_config(config)
    manifest_value = config.get("evaluations", {}).get("behavioral_manifest")
    manifest_path = _project_path(root, manifest_value, "behavioral manifest")
    examples = load_evaluation_examples(root, manifest_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    model = backend or (
        FakeDeterministicBackend()
        if dry_run
        else TransformersInferenceBackend(
            model_id="Qwen/Qwen3.5-2B", revision=str(config["model_revision"])
        )
    )
    renderer = Qwen35_2BRenderer(
        revision=str(config["model_revision"]), enable_thinking=bool(config.get("thinking", False))
    )
    predictions: list[dict[str, Any]] = []
    context_limit = int(config["runtime"]["context_length"])
    started = time.monotonic()
    for example in examples:
        rendered = renderer.render_evaluation(example)
        input_tokens = renderer.text_token_length(rendered.text)
        raw = model.generate(
            rendered.text, example=example, generation_config=dict(config["generation"])
        )
        parsed = parse_qwen_native_output(
            raw, truncated=bool(getattr(model, "last_truncated", False))
        )
        predictions.append(
            _prediction(example, rendered.text, input_tokens, context_limit, raw, parsed)
        )
    output_config = config.get("outputs")
    if not isinstance(output_config, dict) or set(output_config) != {"predictions", "metrics", "residual_profile", "environment"}:
        raise ValueError("baseline config must define predictions, metrics, residual_profile, and environment outputs")
    output_paths = {}
    for name in output_config:
        value = output_config[name]
        if not dry_run:
            output_paths[name] = _project_path(root, value, f"{name} output")
        elif isinstance(value, str) and Path(value).is_absolute():
            # An explicit absolute destination (tests, scratch experiments) is honored.
            output_paths[name] = Path(value).resolve()
        else:
            # A dry run must never write to the canonical evidence namespace. Doing so
            # would both pollute the evidence paths with deterministic-mock output and
            # permanently block the real B0 behind the overwrite guard.
            output_paths[name] = _project_path(
                root,
                str(Path("runs/.dry-run/qwen35_2b_baseline") / Path(str(value)).name),
                f"{name} output",
            )
    canonical_outputs = {
        "predictions": "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
        "metrics": "reports/baselines/qwen35_2b_baseline/metrics.json",
        "residual_profile": "reports/failures/qwen35_2b_baseline/residual-profile.json",
        "environment": "reports/baselines/qwen35_2b_baseline/environment.json",
    }
    if not dry_run and {name: str(path.relative_to(root)) for name, path in output_paths.items()} != canonical_outputs:
        raise ValueError("real baseline must write the canonical evidence paths")
    if not dry_run:
        if any(path.exists() for path in output_paths.values()):
            raise FileExistsError("real baseline evidence already exists; use a new immutable run identity")
        try:
            ExperimentStore(root).get_experiment(BASELINE_EXPERIMENT_ID)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("real baseline experiment record already exists; refusing to overwrite")
    prediction_path = output_paths["predictions"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_temporary = prediction_path.with_name(prediction_path.name + ".tmp")
    try:
        with prediction_temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        prediction_temporary.replace(prediction_path)
    finally:
        prediction_temporary.unlink(missing_ok=True)
    actual = [str(row["expected_decision"]) for row in predictions]
    predicted = [str(row["prediction"]["decision"]) for row in predictions]
    metrics = routing_metrics(actual, predicted) if predictions else {"records": 0}
    parse_counts = Counter(str(row["parser"]["status"]) for row in predictions)
    context_counts = Counter(str(row["context_bucket"]) for row in predictions)
    result = {
        "schema_version": 1,
        "run_id": "tool_calling/qwen35_2b/baseline",
        "status": "DRY_RUN" if dry_run else "EXECUTED",
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "backend": model.name,
        "manifest": str(config["evaluations"]["behavioral_manifest"]),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "renderer": "qwen3_5_2b_v1",
        "template_hash": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        "records": len(predictions),
        "generation": config["generation"],
        "routing": metrics,
        "parser_status": dict(sorted(parse_counts.items())),
        "context_buckets": dict(sorted(context_counts.items())),
        "benchmarks": {
            "when2call": {"status": "EXECUTED_LOCAL_BEHAVIORAL", "metrics": metrics},
            "bfcl": {"status": "FROZEN_NOT_EXECUTED"},
            "tau2": {"status": "FROZEN_NOT_EXECUTED"},
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "artifacts": {
            name: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for name, path in output_paths.items()
        },
    }
    residual = _residuals(
        predictions,
        str(result["run_id"]),
        model_id=str(result["model_id"]),
        model_revision=str(result["model_revision"]),
        manifest_sha256=str(result["manifest_sha256"]),
    )
    environment = capture(root)
    environment.update({
        "run_id": result["run_id"],
        "backend": model.name,
        "dry_run": dry_run,
        "model_id": result["model_id"],
        "model_revision": result["model_revision"],
        "manifest_sha256": result["manifest_sha256"],
    })
    result["git_commit"] = environment.get("git_sha")
    result["git_dirty"] = environment.get("git_dirty")
    _write_json(output_paths["metrics"], result)
    _write_json(output_paths["residual_profile"], residual)
    _write_json(output_paths["environment"], environment)
    if not dry_run:
        store = ExperimentStore(root)
        record = ExperimentRecord(
            experiment_id=BASELINE_EXPERIMENT_ID,
            hypothesis="Frozen Qwen3.5-2B behavioral baseline inference",
            model_id=str(result["model_id"]),
            model_revision=str(result["model_revision"]),
            tokenizer_revision=str(config["tokenizer_revision"]),
            training_algorithm="evaluation",
            training_config={},
            dataset_manifest_ids=[str(config["evaluations"]["behavioral_manifest"])],
            dataset_hashes={str(config["evaluations"]["behavioral_manifest"]): str(result["manifest_sha256"])},
            git_commit=str(result.get("git_commit") or "unknown"),
            git_dirty=bool(result.get("git_dirty")),
            environment=environment,
            random_seed=int(config["seed"]),
            hardware_info=dict(environment.get("gpu", {}) or {}),
            status=ExperimentStatus.EVALUATED.value,
            metadata={
                "kind": "REAL_BASELINE",
                "manifest": result["manifest"],
                "manifest_sha256": result["manifest_sha256"],
                "metrics": str(output_paths["metrics"].relative_to(root)) if output_paths["metrics"].is_relative_to(root) else str(output_paths["metrics"]),
                "predictions": str(output_paths["predictions"].relative_to(root)) if output_paths["predictions"].is_relative_to(root) else str(output_paths["predictions"]),
                "residual_profile": str(output_paths["residual_profile"].relative_to(root)) if output_paths["residual_profile"].is_relative_to(root) else str(output_paths["residual_profile"]),
            },
        )
        try:
            store.register_record(record)
        except FileExistsError as exc:
            raise FileExistsError("real baseline experiment record already exists; refusing to overwrite") from exc
    return result
