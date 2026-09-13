#!/usr/bin/env python3
"""Decide, from the artifacts themselves, what the 8da4w export actually quantized.

Phase 3B forbids inferring this from the file size. A 3.09x shrink is consistent with "every
weight is int4" and equally consistent with "two thirds of the weights are int4 and the rest were
silently skipped", and those are very different models. This script settles it by reconciling the
fp32 and 8da4w named-data stores against each other and against `config.json`.

The reconciliation works on the *size multiset* rather than on names, because ExecuTorch keys
named data by content hash, not by module path. That turns out to be a stronger check than a
name lookup: every fp32 weight class of S bytes must reappear in the 8da4w store as an int4
payload of S/2 **and** a group-scale buffer of S/(4*group)*2, at identical multiplicity. If a
class reappears at its original size it was skipped. If anything is left over at the end, the
model is not what either reading claims and the audit fails rather than rounding the difference
away.

Inputs are the two JSON records written by `scripts/modal/executorch_export.py::audit_pte`.

Usage:
    python scripts/audit_executorch_quantization.py
    python scripts/audit_executorch_quantization.py --json   # machine-readable, for the ledger
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results/quantization/executorch"
CONFIG = ROOT / ".workspace/quantization/source/m1-v2/dpo-checkpoint-30/config.json"

FP32_ITEMSIZE = 4
GROUP_SIZE = 32  # asserted here, then verified against every payload/scale ratio below
SCALE_ITEMSIZE = 2


def expected_weight_classes(config: dict) -> dict[int, str]:
    """Map each distinct weight element-count to what the architecture says it is.

    Every entry is derived from config.json rather than from the artifact, so a mismatch between
    this map and the measured store is a real disagreement and not a tautology.
    """
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    inter = config["intermediate_size"]
    k_heads, k_dim = config["linear_num_key_heads"], config["linear_key_head_dim"]
    v_heads, v_dim = config["linear_num_value_heads"], config["linear_value_head_dim"]
    conv_k = config["linear_conv_kernel_dim"]
    vocab = config["vocab_size"]

    # attn_output_gate doubles the q projection's output width.
    q_out = heads * head_dim * (2 if config.get("attn_output_gate") else 1)
    qkv = k_heads * k_dim + v_heads * v_dim + v_heads * v_dim  # fused q,k,v for Gated DeltaNet

    return {
        vocab * hidden: "lm_head (tied to the embedding matrix)",
        inter * hidden: "MLP gate/up/down, and the fused DeltaNet in_proj_qkv",
        q_out * hidden: "full-attention q_proj (output-gated, 2x width)",
        heads * head_dim * hidden: "full-attention o_proj, DeltaNet in_proj_z and out_proj",
        kv_heads * head_dim * hidden: "full-attention k_proj / v_proj",
        v_heads * hidden: "DeltaNet a_proj / b_proj (per-head gate scalars)",
        qkv * conv_k: f"DeltaNet causal depthwise conv1d ({qkv} channels x kernel {conv_k})",
    }


def load_sizes(name: str) -> tuple[dict, Counter]:
    path = AUDIT_DIR / f"audit_{name}.json"
    if not path.is_file():
        raise SystemExit(f"missing audit record: {path.relative_to(ROOT)}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = doc.get("named_data_store", {}).get("entries") or []
    if not entries:
        raise SystemExit(f"{path.name} has no named-data entries; re-run audit_pte")
    return doc, Counter(entry["bytes"] for entry in entries)


def reconcile(fp32: Counter, quant: Counter, labels: dict[int, str]) -> dict:
    remaining = Counter(quant)
    classes = []
    quantized_params = skipped_params = 0

    # Largest first. Two distinct classes can collide on a byte size (a 1,048,576-element weight's
    # int4 payload is the same 524,288 bytes as an 8,388,608-element weight's scale buffer), so a
    # deterministic order plus a hard "nothing left over" check is what keeps the match honest.
    for size, count in sorted(fp32.items(), reverse=True):
        elements = size // FP32_ITEMSIZE
        payload = elements // 2
        scale = (elements // GROUP_SIZE) * SCALE_ITEMSIZE
        label = labels.get(elements, "UNIDENTIFIED — not predicted by config.json")

        if remaining.get(payload, 0) >= count and remaining.get(scale, 0) >= count:
            remaining[payload] -= count
            remaining[scale] -= count
            verdict, params = "QUANTIZED_INT4", elements * count
            quantized_params += params
        elif remaining.get(size, 0) >= count:
            remaining[size] -= count
            verdict, params = "SKIPPED_FP32", elements * count
            skipped_params += params
        else:
            verdict, params = "UNMATCHED", elements * count

        classes.append(
            {
                "elements_each": elements,
                "count": count,
                "parameters": elements * count,
                "verdict": verdict,
                "int4_payload_bytes": payload if verdict == "QUANTIZED_INT4" else None,
                "scale_bytes": scale if verdict == "QUANTIZED_INT4" else None,
                "identity": label,
            }
        )

    leftover = {size: n for size, n in remaining.items() if n}
    return {
        "classes": classes,
        "unconsumed_buffers": leftover,
        "quantized_parameters": quantized_params,
        "skipped_parameters": skipped_params,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", default="qwen3_5_2b_fp32")
    parser.add_argument("--quantized", default="qwen3_5_2b_8da4w")
    parser.add_argument("--json", action="store_true", help="emit the ledger record only")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    labels = expected_weight_classes(config)
    fp32_doc, fp32 = load_sizes(args.fp32)
    quant_doc, quant = load_sizes(args.quantized)
    result = reconcile(fp32, quant, labels)

    # The embedding lookup table lives in the *program* as a constant, not in the named-data store,
    # because `aten::embedding.out` stays outside the XNNPACK partition. It is therefore invisible
    # to the reconciliation above and has to be accounted for separately — and it is the single
    # largest thing in the artifact, so leaving it out would flatter the result badly.
    const_bytes = sum(quant_doc["constant_bytes_by_dtype"].values())
    const_elements = sum(quant_doc["constant_elements_by_dtype"].values())
    fp32_const_bytes = sum(fp32_doc["constant_bytes_by_dtype"].values())
    embedding = config["vocab_size"] * config["hidden_size"]

    named_total = result["quantized_parameters"] + result["skipped_parameters"]
    stored_total = named_total + const_elements

    record = {
        "artifact": quant_doc["artifact"],
        "reference_artifact": fp32_doc["artifact"],
        "group_size": GROUP_SIZE,
        "scale_itemsize_bytes": SCALE_ITEMSIZE,
        "bytes": {
            "fp32": fp32_doc["bytes"],
            "quantized": quant_doc["bytes"],
            "ratio": round(fp32_doc["bytes"] / quant_doc["bytes"], 4),
            "weight_region_fp32": sum(s * c for s, c in fp32.items()),
            "weight_region_quantized": sum(s * c for s, c in quant.items()),
            "program_constants_fp32": fp32_const_bytes,
            "program_constants_quantized": const_bytes,
        },
        "named_data": result,
        "program_constants": {
            "elements": const_elements,
            "bytes": const_bytes,
            "unchanged_between_artifacts": const_bytes == fp32_const_bytes,
            "embedding_parameters": embedding,
            "embedding_bytes": embedding * FP32_ITEMSIZE,
            "embedding_share_of_quantized_artifact": round(
                embedding * FP32_ITEMSIZE / quant_doc["bytes"], 4
            ),
            "norm_and_other_parameters": const_elements - embedding,
        },
        "coverage": {
            "named_data_quantized_fraction": round(
                result["quantized_parameters"] / named_total, 6
            ),
            "stored_parameters": stored_total,
            "overall_quantized_fraction": round(
                result["quantized_parameters"] / stored_total, 6
            ),
        },
        "reconciled": not result["unconsumed_buffers"]
        and all(c["verdict"] != "UNMATCHED" for c in result["classes"]),
    }

    if args.json:
        print(json.dumps(record, indent=2))
        return 0 if record["reconciled"] else 1

    print(f"fp32      {fp32_doc['artifact']}  {fp32_doc['bytes']:,} bytes")
    print(f"quantized {quant_doc['artifact']}  {quant_doc['bytes']:,} bytes  "
          f"({record['bytes']['ratio']}x smaller)\n")
    header = f"{'parameters':>15} {'n':>4}  {'verdict':<15}  identity"
    print(header)
    print("-" * (len(header) + 30))
    for item in result["classes"]:
        print(f"{item['parameters']:>15,} {item['count']:>4}  {item['verdict']:<15}  "
              f"{item['identity']}")

    print(f"\nunconsumed buffers: {result['unconsumed_buffers'] or 'none'}")
    print(f"named-data weights quantized: {result['quantized_parameters']:,} / {named_total:,} "
          f"= {record['coverage']['named_data_quantized_fraction'] * 100:.2f}%")
    print(f"named-data weights skipped:   {result['skipped_parameters']:,}")
    print(f"\nprogram constants (fp32 in BOTH artifacts): {const_elements:,} params, "
          f"{const_bytes:,} bytes")
    print(f"  of which the embedding table: {embedding:,} params, "
          f"{embedding * FP32_ITEMSIZE:,} bytes — "
          f"{record['program_constants']['embedding_share_of_quantized_artifact'] * 100:.1f}% "
          "of the quantized artifact")
    print(f"  remainder (normalization weights): "
          f"{record['program_constants']['norm_and_other_parameters']:,} params")
    print(f"\noverall quantized share of stored parameters: "
          f"{record['coverage']['overall_quantized_fraction'] * 100:.2f}%")
    print(f"reconciled: {record['reconciled']}")
    return 0 if record["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
