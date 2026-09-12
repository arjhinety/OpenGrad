#!/usr/bin/env python3
"""Pre-render the frozen evaluation prompts once, so every runtime scores identical bytes.

The measurement contract lives in the prompt, not in the engine. vLLM produced the BF16 reference
from text rendered by OpenGrad's pinned Qwen renderer; llama.cpp and ExecuTorch must be handed that
same text rather than re-applying a chat template of their own. Letting each runtime render would
put the frozen prompt contract in the hands of three components the frozen config does not name,
and any behavioural delta would then be uninterpretable.

Rendering once and shipping the result also makes the ExecuTorch handoff honest: whoever runs the
artifact elsewhere scores the same prompt bytes, verifiable by sha256, without needing this
repository or its tokenizer.

Usage:
    python scripts/render_frozen_prompts.py                          # both partitions
    python scripts/render_frozen_prompts.py --partition confirmatory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.data.canonical import CanonicalEvaluationExample  # noqa: E402
from opengrad.data.renderers import Qwen35_2BRenderer  # noqa: E402
from opengrad.evaluation.quantized import canonical_decision  # noqa: E402
from opengrad.evaluation.runner import (  # noqa: E402
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
)

SOURCE = Path("results/quantization/frozen_behavioral_eval_v1.jsonl")
OUT_JSONL = Path("results/quantization/frozen_prompts_v1.jsonl")
OUT_MANIFEST = Path("results/quantization/frozen_prompts_v1.json")

# The frozen measurement, restated here only so the rendered manifest is self-describing for a
# consumer who does not have the baseline config.
CONTEXT_LENGTH = 4096
ENGINE_WINDOW = 5760
GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "do_sample": False}

# Reproduction check. These are not guesses: they are the context_buckets recorded by the
# authoritative reference run in
# runs/m1_dpo_canonical_v2_final_v2/eval/confirmatory/checkpoint-30/metrics.json
# ("context_buckets": {"base": 1275, "overflow": 2}, "records": 1277). If this script renders a
# different distribution, the tokenizer or the template has moved and nothing downstream is
# comparable to the BF16 reference.
CONFIRMATORY_RECORDS = 1277
CONFIRMATORY_BASE = 1275
CONFIRMATORY_OVERFLOW = 2


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_stats(values: list[int]) -> dict[str, Any]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 3),
        "max": max(values),
        "over_context_length": sum(1 for value in values if value > CONTEXT_LENGTH),
        "over_engine_window": sum(1 for value in values if value > ENGINE_WINDOW),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="all", choices=["all", "dev", "confirmatory"])
    args = parser.parse_args()

    source_path = ROOT / SOURCE
    if not source_path.is_file():
        raise SystemExit(
            f"missing {SOURCE}; run scripts/prepare_quantization_inputs.py first"
        )

    renderer = Qwen35_2BRenderer(revision=PINNED_MODEL_REVISION, enable_thinking=False)
    rows: list[dict[str, Any]] = []
    verified_template = False

    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if args.partition != "all" and raw["partition"] != args.partition:
                continue
            example = CanonicalEvaluationExample(
                str(raw["example_id"]),
                raw["source"],
                str(raw["question"]),
                raw["tools"],
                str(raw["expected_decision"]),
                raw["candidates"],
                raw["metadata"],
            )
            rendered = renderer.render_evaluation(example)
            if not verified_template:
                if rendered.chat_template_hash != PINNED_TEMPLATE_HASH:
                    raise SystemExit(
                        "chat template does not match the frozen contract:\n"
                        f"  rendered: {rendered.chat_template_hash}\n"
                        f"  pinned  : {PINNED_TEMPLATE_HASH}"
                    )
                verified_template = True
            prompt = rendered.text
            rows.append(
                {
                    "example_id": str(raw["example_id"]),
                    "partition": str(raw["partition"]),
                    "source": str((raw["source"] or {}).get("dataset_id", "unknown")),
                    "expected_decision": canonical_decision(str(raw["expected_decision"])),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "input_tokens": renderer.text_token_length(prompt),
                }
            )

    if not rows:
        raise SystemExit(f"no rows selected for partition {args.partition!r}")
    rows.sort(key=lambda row: row["example_id"])

    confirmatory = [row for row in rows if row["partition"] == "confirmatory"]
    if confirmatory:
        overflow = sum(1 for row in confirmatory if row["input_tokens"] > CONTEXT_LENGTH)
        base = len(confirmatory) - overflow
        if (len(confirmatory), base, overflow) != (
            CONFIRMATORY_RECORDS,
            CONFIRMATORY_BASE,
            CONFIRMATORY_OVERFLOW,
        ):
            raise SystemExit(
                "confirmatory partition does not reproduce the reference context buckets:\n"
                f"  observed: records={len(confirmatory)} base={base} overflow={overflow}\n"
                f"  expected: records={CONFIRMATORY_RECORDS} base={CONFIRMATORY_BASE} "
                f"overflow={CONFIRMATORY_OVERFLOW}"
            )

    out_jsonl = ROOT / OUT_JSONL
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    partitions = sorted({row["partition"] for row in rows})
    manifest = {
        "schema_version": 1,
        "source_jsonl": str(SOURCE).replace("\\", "/"),
        "source_sha256": sha256_file(source_path),
        "output_jsonl": str(OUT_JSONL).replace("\\", "/"),
        "output_sha256": sha256_file(out_jsonl),
        "renderer": "qwen3_5_2b_v1",
        "tokenizer_revision": PINNED_MODEL_REVISION,
        "template_hash": PINNED_TEMPLATE_HASH,
        "enable_thinking": False,
        "generation": GENERATION,
        "context_length": CONTEXT_LENGTH,
        "engine_window": ENGINE_WINDOW,
        "records": len(rows),
        "partitions": {
            name: sum(1 for row in rows if row["partition"] == name) for name in partitions
        },
        "token_stats": {
            name: token_stats([r["input_tokens"] for r in rows if r["partition"] == name])
            for name in partitions
        },
    }
    (ROOT / OUT_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"rendered {len(rows)} prompts -> {OUT_JSONL}")
    for name in partitions:
        stats = manifest["token_stats"][name]
        print(
            f"  {name:<13} n={manifest['partitions'][name]:<5} "
            f"tokens min={stats['min']} median={stats['median']} max={stats['max']} "
            f"over_{CONTEXT_LENGTH}={stats['over_context_length']} "
            f"over_{ENGINE_WINDOW}={stats['over_engine_window']}"
        )
    print(f"  template_hash {PINNED_TEMPLATE_HASH}")
    print(f"  output_sha256 {manifest['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
