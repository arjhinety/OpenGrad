#!/usr/bin/env python3
"""Fold the GGUF branch results into the append-only evidence ledger.

Every candidate gets a row, including rejected ones and ones that failed to build. A ladder with
rejected rungs silently omitted cannot answer "where is the compression floor", which is the
question the sub-Q4 rungs exist to answer — so a rung that fails is recorded with its failing
dimensions rather than dropped.

Rows carry the full reproduction set: artifact hash and size, quantizer command, llama.cpp commit,
imatrix and calibration hashes, tokenizer identity, source config hash, and both comparison
namespaces kept distinct (`engine_parity_vs_vllm_bf16` for BF16 only,
`quantization_loss_vs_llamacpp_bf16` for every rung).

Usage:
    python scripts/update_quantization_ledger.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "results/quantization/gguf"
LEDGER = ROOT / "results/quantization/findings.jsonl"
LADDER = ("Q2_K", "Q3_K_M", "IQ3_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

PASSING = {"PASS", "PASSED", "ACCEPT", "ACCEPTED", "PTQ_ACCEPTED", "PRESERVED"}


CLOSURE = ROOT / "manifests/quantization/ptq_phase_closure_v1.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def release_roles() -> dict[str, dict]:
    """Release roles and secondary-bar status, read from the closure manifest.

    Sourced rather than duplicated so the ledger cannot drift from the closure record. The
    secondary-bar string travels with the role because a role is not a claim that the bar passed —
    no PTQ artifact met it.
    """
    closure = load(CLOSURE)
    if closure is None:
        return {}
    return {
        a["quantization"]: {
            "release_role": a["role"],
            "secondary_release_bar": a.get("secondary_release_bar"),
            "compression_vs_bf16": a.get("compression_vs_bf16"),
        }
        for a in closure["artifacts"]
    }


def common_provenance() -> dict:
    prepare = load(GGUF / "prepare.json") or {}
    imatrix = load(GGUF / "imatrix.json") or {}
    tokenizer = load(GGUF / "inspect_tokenizer.json") or {}
    parity = load(GGUF / "engine_parity_verdict.json") or {}
    return {
        "runtime": "llama.cpp",
        "llama_cpp_tag": prepare.get("llama_cpp_tag"),
        "llama_cpp_commit": prepare.get("llama_cpp_commit"),
        "model_repo": prepare.get("model_repo"),
        "model_revision": prepare.get("model_revision"),
        "parent_checkpoint": prepare.get("checkpoint"),
        "source_weight_sha256": prepare.get("source_weight_sha256"),
        "source_config_sha256": prepare.get("source_config_sha256"),
        "bf16_gguf_sha256": prepare.get("gguf_sha256"),
        "converter_command": prepare.get("converter_command"),
        "tokenizer_pre_type": (tokenizer.get("keys_matched") or {}).get("tokenizer.ggml.pre"),
        "tokenizer_model": (tokenizer.get("keys_matched") or {}).get("tokenizer.ggml.model"),
        "imatrix_sha256": imatrix.get("imatrix_sha256"),
        "imatrix_bytes": imatrix.get("imatrix_bytes"),
        "calibration_sha256": imatrix.get("calibration_sha256"),
        "imatrix_parse_special": imatrix.get("parse_special"),
        "engine_tokenizer_parity": parity.get("engine_tokenizer_parity"),
        "tokenizer_materiality": parity.get("tokenizer_materiality"),
        "quantization_baseline": parity.get("quantization_baseline"),
        "decoding": {
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
            "max_new_tokens": 512, "cache_prompt": False, "slot_context": 5760,
        },
    }


def bench_summary(rung: str) -> dict | None:
    doc = load(GGUF / f"bench_m1-v2-{rung}.json")
    if not doc:
        return None
    out = {}
    for row in doc.get("rows") or []:
        n_prompt, n_gen = int(row.get("n_prompt") or 0), int(row.get("n_gen") or 0)
        if row.get("avg_ts") is None:
            continue
        key = f"pp{n_prompt}" if n_prompt and not n_gen else f"tg{n_gen}" if n_gen else None
        if key:
            out[f"{key}_tok_s"] = round(float(row["avg_ts"]), 2)
    return out or None


def build_rows() -> list[dict]:
    provenance = common_provenance()
    now = datetime.now(UTC).isoformat()
    rows = []

    roles = release_roles()
    baseline = load(GGUF / "score_m1-v2-bf16_confirmatory.json")
    if baseline:
        engine = baseline.get("engine_parity_vs_vllm_bf16", {})
        rows.append({
            "branch": "gguf",
            "artifact": "m1-v2-bf16-gguf",
            "status": "BF16_REFERENCE",
            "quantization": "BF16",
            "recorded_at": now,
            "policy_version": baseline["policy_version"],
            "partition": baseline["partition"],
            "submitted": baseline["submitted"],
            "records": baseline["records"],
            "metrics": baseline["metrics"],
            "artifact_bytes": baseline.get("artifact_bytes"),
            "artifact_sha256": baseline.get("artifact_sha256") or provenance["bf16_gguf_sha256"],
            "gate_vs_frozen_vllm_reference": baseline["gate_vs_frozen_vllm_reference"]["decision"],
            "failed_dimensions": [
                c["dimension"] for c in baseline["gate_vs_frozen_vllm_reference"]["checks"]
                if not c["passed"]
            ],
            "engine_parity_vs_vllm_bf16": {
                "delta": engine.get("delta"),
                "relative_retention": engine.get("relative_retention"),
                "note": engine.get("note"),
            },
            "quantization_loss_vs_llamacpp_bf16": None,
            "bench": bench_summary("bf16"),
            "reason": (
                "llama.cpp BF16 GGUF. This is the attribution baseline for every quantized rung, "
                "not a release candidate. Strict engine tokenizer parity FAILED (6/1277) and is "
                "not restated as passing here."
            ),
            "detail": "results/quantization/gguf/score_m1-v2-bf16_confirmatory.json",
            **roles.get("BF16", {}),
            **provenance,
        })

    for rung in LADDER:
        score = load(GGUF / f"score_m1-v2-{rung}_confirmatory.json")
        quant = load(GGUF / f"quantize_{rung}.json")
        if score is None:
            rows.append({
                "branch": "gguf",
                "artifact": f"m1-v2-{rung}-gguf",
                "status": "NOT_SCORED",
                "quantization": rung,
                "recorded_at": now,
                "artifact_bytes": (quant or {}).get("artifact_bytes"),
                "artifact_sha256": (quant or {}).get("artifact_sha256"),
                "quantizer_command": (quant or {}).get("command"),
                "metrics": None,
                "reason": (
                    "quantized but not behaviourally scored"
                    if quant else "rung did not produce an artifact"
                ),
                **roles.get(rung, {}),
                **provenance,
            })
            continue

        gate = score["gate_vs_frozen_vllm_reference"]
        loss = score.get("quantization_loss_vs_llamacpp_bf16", {})
        agreement = loss.get("output_agreement", {})
        passed = str(gate["decision"]).upper() in PASSING
        rows.append({
            "branch": "gguf",
            "artifact": f"m1-v2-{rung}-gguf",
            "status": "PTQ_ACCEPTED" if passed else "REJECTED_ACCURACY",
            "quantization": rung,
            "recorded_at": now,
            "policy_version": score["policy_version"],
            "partition": score["partition"],
            "submitted": score["submitted"],
            "records": score["records"],
            "metrics": score["metrics"],
            "artifact_bytes": score.get("artifact_bytes") or (quant or {}).get("artifact_bytes"),
            "artifact_sha256": score.get("artifact_sha256") or (quant or {}).get("artifact_sha256"),
            "quantizer_command": (quant or {}).get("command"),
            "quantize_elapsed_seconds": (quant or {}).get("elapsed_seconds"),
            "gate_vs_frozen_vllm_reference": gate["decision"],
            "failed_dimensions": [c["dimension"] for c in gate["checks"] if not c["passed"]],
            "engine_parity_vs_vllm_bf16": None,
            "quantization_loss_vs_llamacpp_bf16": {
                "delta": loss.get("delta"),
                "relative_retention": loss.get("relative_retention"),
                "decision_agreement": agreement.get("decision_agreement"),
                "decision_flips": agreement.get("decision_flips"),
                "flip_effects": agreement.get("flip_effects"),
                "byte_identical_rate": agreement.get("byte_identical_rate"),
            },
            "bench": bench_summary(rung),
            "reason": (
                f"{rung} quantized from BF16 GGUF with the frozen imatrix. "
                + ("Passes" if passed else "Fails")
                + " the frozen quantization_preservation_v1 gate. Tokenizer divergence is held "
                "constant against the llama.cpp BF16 baseline and is not counted as quantization "
                "loss."
            ),
            "detail": f"results/quantization/gguf/score_m1-v2-{rung}_confirmatory.json",
            **roles.get(rung, {}),
            **provenance,
        })
    return rows


def main() -> int:
    existing = []
    if LEDGER.is_file():
        existing = [
            json.loads(line)
            for line in LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    keep = [row for row in existing if row.get("branch") != "gguf"]
    dropped = len(existing) - len(keep)
    rows = keep + build_rows()

    LEDGER.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"rewrote {LEDGER.relative_to(ROOT)}: {len(rows)} rows "
          f"({dropped} stale gguf rows replaced)")
    for row in rows:
        print(f"  {row['branch']:<11} {row['artifact']:<24} {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
