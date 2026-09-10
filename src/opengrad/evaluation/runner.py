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

from opengrad.data.canonical import CanonicalEvaluationExample
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.env_capture import capture
from opengrad.evaluation.routing import routing_metrics
from opengrad.formatting.parser import ParsedNativeOutput, parse_qwen_native_output


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


def load_evaluation_examples(root: Path, manifest_path: Path) -> list[CanonicalEvaluationExample]:
    """Load exactly the materialized splits named by a frozen manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen") is not True or manifest.get("status") not in {
        "FROZEN_PRE_GPU",
        "MATERIALIZED",
        "EXECUTED",
    }:
        raise ValueError("evaluation manifest is not frozen")
    examples: list[CanonicalEvaluationExample] = []
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    for split in manifest.get("splits", []):
        source = root / str(split["source"])
        if source.name == "manifest.json":
            source = source.parent
        if not source.exists():
            raise FileNotFoundError(f"evaluation split is missing: {source}")
        shard_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        for shard_name in shard_manifest.get("shards", []):
            for batch in pq.ParquetFile(source / shard_name).iter_batches(batch_size=128):
                for raw in batch.to_pylist():
                    row = dict(raw)
                    for key in ("source", "tools", "candidates", "metadata"):
                        row[key] = _json(row[key])
                    example = CanonicalEvaluationExample(
                        str(row["example_id"]),
                        row["source"],
                        str(row["question"]),
                        [_qwen_tool(tool) for tool in row["tools"]],
                        str(row["expected_decision"]),
                        row["candidates"],
                        row["metadata"],
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


def _residuals(predictions: Iterable[dict[str, Any]], baseline_experiment: str) -> dict[str, Any]:
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
        "model": "qwen3.5-2b",
        "baseline_experiment": baseline_experiment,
        "sample_count": total,
        "residuals": {key: value / total for key, value in sorted(counts.items())} if total else {},
        "failure_counts": dict(sorted(counts.items())),
    }


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
    """Run the frozen baseline from manifest through artifacts."""
    root = root or config_path.parents[3]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("status") != "FROZEN_PRE_GPU":
        raise ValueError("baseline config must be FROZEN_PRE_GPU")
    manifest_path = root / str(config["evaluations"]["behavioral_manifest"])
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
    output_config = config.get("outputs", {})
    prediction_path = root / str(output_config["predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    actual = [str(row["expected_decision"]) for row in predictions]
    predicted = [str(row["prediction"]["decision"]) for row in predictions]
    metrics = routing_metrics(actual, predicted) if predictions else {"records": 0}
    parse_counts = Counter(str(row["parser"]["status"]) for row in predictions)
    context_counts = Counter(str(row["context_bucket"]) for row in predictions)
    result = {
        "schema_version": 1,
        "run_id": "tool_calling/qwen3.5-2b/baseline",
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
            "predictions": str(output_config["predictions"]),
            "metrics": str(output_config["metrics"]),
            "residual_profile": str(output_config["residual_profile"]),
            "environment": str(output_config["environment"]),
        },
    }
    residual = _residuals(predictions, str(result["run_id"]))
    environment = capture(root)
    environment.update({"run_id": result["run_id"], "backend": model.name, "dry_run": dry_run})
    _write_json(root / str(output_config["metrics"]), result)
    _write_json(root / str(output_config["residual_profile"]), residual)
    _write_json(root / str(output_config["environment"]), environment)
    return result
