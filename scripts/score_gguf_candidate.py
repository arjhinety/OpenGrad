#!/usr/bin/env python3
"""Score one llama.cpp GGUF artifact and attribute its deltas to the right cause.

The study has two different comparisons and they answer different questions. Collapsing them is
the specific mistake this script is built to prevent:

* **Engine parity** — vLLM/HF BF16 vs llama.cpp BF16. Changes the engine *and* the tokenizer at
  once. Only meaningful for the BF16 GGUF, and it carries the known `\\p{M}` pre-tokenizer
  divergence on 6 of the 1277 confirmatory prompts.
* **Quantization loss** — llama.cpp BF16 vs llama.cpp QX. Engine and tokenizer are identical on
  both sides, so the tokenizer divergence is held constant and cancels. This is the comparison
  that can be attributed to quantization.

The frozen `quantization_preservation_v1` gate is computed from the vLLM BF16 reference and is
applied unchanged — it is never recomputed against the llama.cpp baseline, because that would be
moving the gate after seeing results. The llama.cpp-relative deltas are reported *alongside* it
for attribution, not in place of it.

No example is ever dropped. `score_runtime_generations` raises on any accounting mismatch, so a
run that answered fewer prompts fails loudly instead of scoring better for having answered less.

Usage:
    python scripts/score_gguf_candidate.py --artifact m1-v2-bf16 --generations <path.jsonl>
    python scripts/score_gguf_candidate.py --artifact m1-v2-Q4_K_M --generations <path.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.quantized import load_frozen_prompts, score_runtime_generations
from opengrad.promotion.quantization import (
    PRESERVATION_POLICY_VERSION,
    evaluate_quantization_preservation,
)

PROMPTS = ROOT / "results/quantization/frozen_prompts_v1.jsonl"
REFERENCE = ROOT / "results/quantization/m1_v2_reference.json"
GATE = ROOT / "results/quantization/quantization_preservation_v1.json"
OUT_DIR = ROOT / "results/quantization/gguf"
BASELINE_STEM = "m1-v2-bf16"

# The preservation policy requires BOTH its own behavioural verdict and the already-executed M1
# tool-policy verdict, and treats a missing tool-policy record as a failed check rather than
# silently omitting it. That is correct behaviour, but it means the verdict must actually be
# supplied — otherwise every rung reports a failure for a bookkeeping reason that has nothing to
# do with its metrics. The path is the one named by the frozen reference itself.
TOOL_POLICY = ROOT / "reports/data/m1-dpo-canonical-v2-final-v2-promotion.json"

# The llama.cpp BF16 GGUF is the attribution baseline for every quantized rung. It is NOT the gate
# reference; the gate reference stays the frozen vLLM BF16 measurement.
BASELINE_METRICS = OUT_DIR / f"score_{BASELINE_STEM}_confirmatory.json"


def flat_metrics(scored: dict) -> dict[str, float]:
    """The seven gate metrics in one flat mapping, from the scorer's nested output."""
    routing = scored["routing"]
    out = {
        name: float(routing[name])
        for name in (
            "call_f1",
            "call_precision",
            "call_recall",
            "clarification_accuracy",
            "unsupported_accuracy",
            "over_call_rate",
        )
    }
    out["parse_valid_rate"] = float(scored["parse_valid_rate"])
    return out


def agreement(candidate_predictions: list[dict], baseline_predictions: list[dict]) -> dict:
    """Per-example decision agreement against the llama.cpp BF16 baseline.

    Two artifacts can post identical aggregate metrics while disagreeing on many examples — the
    errors just land in different places. Aggregate retention alone would call that "preserved".
    """
    base = {row["example_id"]: row for row in baseline_predictions}
    cand = {row["example_id"]: row for row in candidate_predictions}
    if set(base) != set(cand):
        raise SystemExit(
            f"candidate and baseline cover different example sets: "
            f"{len(set(base) ^ set(cand))} differ"
        )
    flips = []
    identical_text = 0
    for example_id in sorted(base):
        b, c = base[example_id], cand[example_id]
        if b["raw_output"] == c["raw_output"]:
            identical_text += 1
        if b["prediction"]["decision"] != c["prediction"]["decision"]:
            flips.append(
                {
                    "example_id": example_id,
                    "expected": b["expected_decision"],
                    "baseline_decision": b["prediction"]["decision"],
                    "candidate_decision": c["prediction"]["decision"],
                    # Which way the flip went relative to the label.
                    "effect": (
                        "fixed" if c["prediction"]["decision"] == b["expected_decision"]
                        else "broke" if b["prediction"]["decision"] == b["expected_decision"]
                        else "lateral"
                    ),
                }
            )
    total = len(base)
    return {
        "examples": total,
        "decision_agreement": round((total - len(flips)) / total, 6),
        "decision_flips": len(flips),
        "byte_identical_outputs": identical_text,
        "byte_identical_rate": round(identical_text / total, 6),
        "flips": flips,
        "flip_effects": {
            effect: sum(1 for f in flips if f["effect"] == effect)
            for effect in ("fixed", "broke", "lateral")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="stem, e.g. m1-v2-Q4_K_M")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--partition", default="confirmatory")
    parser.add_argument("--quantization", default=None, help="e.g. Q4_K_M; omit for the BF16 base")
    parser.add_argument("--artifact-bytes", type=int, default=None)
    parser.add_argument("--artifact-sha256", default=None)
    args = parser.parse_args()

    prompts = load_frozen_prompts(PROMPTS, partition=args.partition)
    generations = [
        json.loads(line)
        for line in Path(args.generations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_doc = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference = {k: float(v) for k, v in reference_doc["metrics"].items()}

    scored = score_runtime_generations(
        prompts,
        generations,
        context_length=4096,
        runtime={
            "engine": "llama.cpp",
            "artifact": args.artifact,
            "quantization": args.quantization or "BF16",
            "partition": args.partition,
        },
    )
    metrics = flat_metrics(scored)

    # The mandated gate, unchanged, against the frozen vLLM BF16 reference.
    if not TOOL_POLICY.is_file():
        raise SystemExit(
            f"missing the executed M1 tool-policy verdict at {TOOL_POLICY.relative_to(ROOT)}; "
            "the preservation policy requires it and refusing here is better than scoring every "
            "rung as failed for a missing input"
        )
    tool_policy = json.loads(TOOL_POLICY.read_text(encoding="utf-8"))
    expected_parent = reference_doc["experiment_id"]
    if tool_policy.get("experiment_id") != expected_parent:
        raise SystemExit(
            f"tool-policy verdict is for {tool_policy.get('experiment_id')!r}, but the frozen "
            f"reference is {expected_parent!r}; refusing to attach another model's promotion"
        )
    verdict = evaluate_quantization_preservation(
        metrics, reference, existing_tool_policy=tool_policy
    )

    result = {
        "artifact": args.artifact,
        "quantization": args.quantization or "BF16",
        "partition": args.partition,
        "policy_version": PRESERVATION_POLICY_VERSION,
        "submitted": scored["submitted"],
        "records": scored["records"],
        "metrics": metrics,
        "parser_status": scored["parser_status"],
        "decision_counts": scored["decision_counts"],
        "context_buckets": scored["context_buckets"],
        "gate_vs_frozen_vllm_reference": verdict,
        "artifact_bytes": args.artifact_bytes,
        "artifact_sha256": args.artifact_sha256,
    }

    if args.artifact == BASELINE_STEM:
        # Engine parity: this is the only comparison where the engine changed, and it is confounded
        # with the tokenizer divergence by construction. Labelled so, never called "quantization".
        result["engine_parity_vs_vllm_bf16"] = {
            "note": (
                "vLLM BF16 -> llama.cpp BF16 changes engine AND tokenizer together; the 6-prompt "
                "\\p{M} pre-tokenizer divergence is inside this delta and is not a quantization "
                "effect"
            ),
            "reference_metrics": reference,
            "delta": {k: round(metrics[k] - reference[k], 6) for k in metrics},
            "relative_retention": {
                k: round(metrics[k] / reference[k], 6) if reference[k] else None
                for k in metrics
            },
        }
    else:
        if not BASELINE_METRICS.is_file():
            raise SystemExit(
                f"missing llama.cpp BF16 baseline at {BASELINE_METRICS.relative_to(ROOT)}; "
                "score the BF16 GGUF before any quantized rung"
            )
        baseline_doc = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
        baseline_metrics = baseline_doc["metrics"]
        baseline_predictions = json.loads(
            (OUT_DIR / f"predictions_{BASELINE_STEM}_{args.partition}.json").read_text(
                encoding="utf-8"
            )
        )
        result["quantization_loss_vs_llamacpp_bf16"] = {
            "note": (
                "engine and tokenizer identical on both sides, so this delta is attributable to "
                "quantization"
            ),
            "baseline_metrics": baseline_metrics,
            "delta": {k: round(metrics[k] - baseline_metrics[k], 6) for k in metrics},
            "relative_retention": {
                k: round(metrics[k] / baseline_metrics[k], 6) if baseline_metrics[k] else None
                for k in metrics
            },
            "output_agreement": agreement(scored["predictions"], baseline_predictions),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUT_DIR / f"predictions_{args.artifact}_{args.partition}.json"
    predictions_path.write_text(
        json.dumps(scored["predictions"], indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_path = OUT_DIR / f"score_{args.artifact}_{args.partition}.json"
    score_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {score_path.relative_to(ROOT)}")
    print(f"wrote {predictions_path.relative_to(ROOT)}  ({len(scored['predictions'])} rows)")
    print(f"\n{args.artifact}  ({result['quantization']})  records={result['records']}")
    for name, value in metrics.items():
        print(f"  {name:24} {value:.6f}")
    print(f"\ngate vs frozen vLLM BF16 reference: {verdict['decision']}")
    for check in verdict["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['dimension']:28} {check['observed']}  {check['requirement']}")
    if "quantization_loss_vs_llamacpp_bf16" in result:
        qa = result["quantization_loss_vs_llamacpp_bf16"]["output_agreement"]
        print(
            f"\nvs llama.cpp BF16: decision agreement {qa['decision_agreement']:.4f} "
            f"({qa['decision_flips']} flips: {qa['flip_effects']}), "
            f"byte-identical {qa['byte_identical_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
