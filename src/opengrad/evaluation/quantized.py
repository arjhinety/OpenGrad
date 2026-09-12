"""Score generations from a deployment runtime against the frozen behavioral contract.

A quantized artifact cannot be measured by :func:`opengrad.evaluation.candidate.run_candidate_evaluation`.
That function's ``INVARIANT_PATHS`` pins the entire ``runtime`` block to the vLLM baseline, which is
exactly the guard that keeps a checkpoint comparison attributable to the checkpoint. A GGUF or
ExecuTorch artifact changes the engine by definition, so that path refuses it — correctly. This
module adds the missing measurement rather than relaxing that guard.

What is deliberately shared with the baseline: the native parser, the routing metrics, the decision
vocabulary, and the context bucketing. What is deliberately different: generation happens somewhere
else entirely, and arrives here as text.

The accounting is strict on purpose. A runtime that times out, refuses a prompt, or silently returns
fewer completions than it was given would otherwise shrink the denominator and report a *better*
score for having answered less. Every condition that could do that raises instead.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from opengrad.evaluation.routing import routing_metrics
from opengrad.evaluation.runner import _decision
from opengrad.formatting.parser import parse_qwen_native_output

PROMPT_SCHEMA_VERSION = 1

# The baseline's own prediction row keys. Kept as a constant so a drift in either direction is a
# test failure rather than a downstream surprise in tooling that reads predictions.jsonl.
PREDICTION_KEYS = frozenset(
    {
        "example_id",
        "source",
        "expected_decision",
        "prompt_sha256",
        "input_tokens",
        "context_bucket",
        "prediction",
        "raw_output",
        "parser",
    }
)


class RuntimeAccountingError(RuntimeError):
    """A runtime returned a generation set that does not reconcile with what it was given."""


def _report(kind: str, ids: Iterable[str]) -> str:
    offenders = sorted(ids)
    shown = ", ".join(offenders[:10])
    suffix = ", …" if len(offenders) > 10 else ""
    return f"{kind}: {len(offenders)} example(s) [{shown}{suffix}]"


def canonical_decision(value: str) -> str:
    """Map a raw dataset label to the canonical decision vocabulary.

    Delegates to the runner's table rather than restating it: two copies of this mapping would be
    two measurements the moment one of them gained a label.
    """
    return _decision(value)


def load_frozen_prompts(path: Path, *, partition: str | None = None) -> list[dict[str, Any]]:
    """Load pre-rendered prompts, optionally restricted to one frozen partition."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if partition is not None and str(row.get("partition")) != partition:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no prompts loaded from {path} (partition={partition!r})")
    rows.sort(key=lambda row: str(row["example_id"]))
    return rows


def build_predictions(
    prompts: Sequence[Mapping[str, Any]],
    generations: Iterable[Mapping[str, Any]],
    *,
    context_length: int = 4096,
) -> list[dict[str, Any]]:
    """Pair every prompt with exactly one generation and parse it.

    Raises rather than dropping: an unmatched prompt or an extra generation means the run and the
    frozen population disagree, and any metric computed over that disagreement is not the metric
    it claims to be.
    """
    by_id = {str(prompt["example_id"]): prompt for prompt in prompts}

    collected: dict[str, Mapping[str, Any]] = {}
    unknown: list[str] = []
    duplicate: list[str] = []
    for generation in generations:
        example_id = str(generation["example_id"])
        if example_id not in by_id:
            unknown.append(example_id)
            continue
        if example_id in collected:
            duplicate.append(example_id)
            continue
        collected[example_id] = generation
    if unknown:
        raise RuntimeAccountingError(_report("generations for unknown example_id", unknown))
    if duplicate:
        raise RuntimeAccountingError(_report("duplicate generations", duplicate))

    missing = set(by_id) - set(collected)
    if missing:
        raise RuntimeAccountingError(_report("prompts with no generation", missing))

    failed = [
        example_id
        for example_id, generation in collected.items()
        if generation.get("error") not in (None, "")
    ]
    if failed:
        raise RuntimeAccountingError(_report("generations reporting a runtime error", failed))

    predictions: list[dict[str, Any]] = []
    for example_id in sorted(by_id):
        prompt = by_id[example_id]
        generation = collected[example_id]
        raw = generation.get("raw")
        if not isinstance(raw, str):
            raise RuntimeAccountingError(
                f"generation for {example_id} has non-string raw output: {type(raw).__name__}"
            )
        parsed = parse_qwen_native_output(raw, truncated=bool(generation.get("truncated", False)))
        input_tokens = int(prompt["input_tokens"])
        predictions.append(
            {
                "example_id": example_id,
                "source": str(prompt["source"]),
                "expected_decision": str(prompt["expected_decision"]),
                "prompt_sha256": str(prompt["prompt_sha256"]),
                "input_tokens": input_tokens,
                "context_bucket": "overflow" if input_tokens > context_length else "base",
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
        )
    return predictions


def score_runtime_generations(
    prompts: Sequence[Mapping[str, Any]],
    generations: Iterable[Mapping[str, Any]],
    *,
    context_length: int = 4096,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Metrics for one runtime artifact, in the same schema the baseline reports."""
    predictions = build_predictions(prompts, generations, context_length=context_length)
    if len(predictions) != len(prompts):
        # Unreachable given build_predictions' checks; kept because a silent denominator change is
        # the specific failure this module exists to make impossible.
        raise RuntimeAccountingError(
            f"scored {len(predictions)} rows for {len(prompts)} submitted prompts"
        )

    # Both sides of this ratio are counted in prediction rows. Counting parser errors in the
    # numerator and rows in the denominator would mix units and silently change the metric.
    valid_rows = sum(1 for row in predictions if row["parser"]["status"] == "RAW_VALID")
    total_rows = len(predictions)

    metrics = routing_metrics(
        [str(row["expected_decision"]) for row in predictions],
        [str(row["prediction"]["decision"]) for row in predictions],
    )
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "records": total_rows,
        "submitted": len(prompts),
        "parse_valid_rate": round(valid_rows / total_rows, 6),
        "routing": metrics,
        "parser_status": dict(sorted(Counter(r["parser"]["status"] for r in predictions).items())),
        "context_buckets": dict(sorted(Counter(r["context_bucket"] for r in predictions).items())),
        "decision_counts": dict(
            sorted(Counter(r["prediction"]["decision"] for r in predictions).items())
        ),
        "runtime": dict(runtime) if runtime else None,
        "predictions": predictions,
    }
