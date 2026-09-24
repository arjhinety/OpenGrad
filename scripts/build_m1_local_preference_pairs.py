#!/usr/bin/env python3
"""Build a local, provenance-rich M1 preference set from base/M0 disagreements.

The source corpus is training data, never the frozen DEV/confirmatory population. For each selected
prompt, the immutable base and selected M0 policy generate deterministic responses. A pair is kept
only when exactly one parsed decision matches the canonical gold decision. That makes the preference
label an observed behavioral correction, not a fabricated teacher judgment.

The output is balanced by gold decision, hashes both model artifacts, and records the generation
contract in every row. No external API is used.

Usage:
    python scripts/build_m1_local_preference_pairs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_file as sha256

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
M0_CHECKPOINT = ROOT / "runs/m0_sft_canonical_v2_final/checkpoints/checkpoint-1800"
CORPUS = ROOT / ".release/hf/toolpolicy-canonical-v2-final"
PARTITION = ROOT / "reports/evaluation/behavioral-heldout-v2-partition.json"
OUTPUT = ROOT / "data/processed/m1_local_calibration_pairs_v1.jsonl"
REPORT = ROOT / "reports/data/m1-local-calibration-pairs-v1.json"
SEED = 42
TARGETS = {"CALL": 300, "CLARIFY": 200, "UNSUPPORTED": 200, "ANSWER": 100}
MAX_CANDIDATES_PER_CLASS = 1600
MAX_NEW_TOKENS = 128
BATCH_SIZE = 32


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def prompt_from_row(tokenizer: Any, row: dict[str, Any]) -> str | None:
    messages = parse_json(row.get("messages"))
    tools = parse_json(row.get("tools")) or []
    if not isinstance(messages, list) or not messages:
        return None
    last_user = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        default=-1,
    )
    if last_user < 0:
        return None
    prefix = [message for message in messages[: last_user + 1] if isinstance(message, dict)]
    if not prefix:
        return None
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if isinstance(tools, list) and tools:
        kwargs["tools"] = tools
    try:
        return str(tokenizer.apply_chat_template(prefix, **kwargs))
    except Exception:  # noqa: BLE001 - an unrenderable source row is skipped deterministically
        return None


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("canonical_hash") or row.get("id") or "")


def load_candidates() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    heldout: set[str] = set()
    if PARTITION.is_file():
        partition = json.loads(PARTITION.read_text(encoding="utf-8"))
        heldout = {
            str(item) for ids in (partition.get("example_ids") or {}).values() for item in ids
        }
    rows: list[dict[str, Any]] = []
    for shard in sorted(CORPUS.glob("*.parquet")):
        table = pq.read_table(
            shard,
            columns=[
                "canonical_hash",
                "source_dataset",
                "behavior_decision",
                "messages",
                "tools",
            ],
        )
        for row in table.to_pylist():
            identifier = row_id(row)
            category = str(row.get("behavior_decision") or "")
            if identifier in heldout or category not in TARGETS:
                continue
            if not prompt_from_row(_TOKENIZER, row):
                continue
            rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["behavior_decision"])].append(row)
    selected: list[dict[str, Any]] = []
    for category in TARGETS:
        ordered = sorted(
            grouped[category],
            key=lambda row: hashlib.sha256(row_id(row).encode()).hexdigest(),
        )
        selected.extend(ordered[:MAX_CANDIDATES_PER_CLASS])
    return selected


def generate_outputs(
    model_id: str, revision_or_path: str, prompts: list[str], tokenizer: Any
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        revision_or_path,
        revision=None if Path(revision_or_path).is_dir() else BASE_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    results: list[str] = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=5760,
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        for index, row in enumerate(output):
            results.append(tokenizer.decode(row[input_width:], skip_special_tokens=False))
    del model
    torch.cuda.empty_cache()
    return results


def main() -> int:
    global _TOKENIZER
    from transformers import AutoTokenizer

    from opengrad.formatting.parser import parse_qwen_native_output

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    _TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False
    )
    _TOKENIZER.padding_side = "left"
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    partition = json.loads(PARTITION.read_text(encoding="utf-8")) if PARTITION.is_file() else {}
    heldout = {str(item) for ids in (partition.get("example_ids") or {}).values() for item in ids}
    rows = load_candidates()
    prompts = [prompt_from_row(_TOKENIZER, row) for row in rows]
    prompts = [prompt for prompt in prompts if prompt is not None]
    if len(prompts) != len(rows):
        raise RuntimeError("candidate prompt count changed during preparation")

    m0_hash = sha256(M0_CHECKPOINT / "model.safetensors")
    print(f"candidates: {len(rows)}; generating base outputs", flush=True)
    base_outputs = generate_outputs(BASE_MODEL, BASE_MODEL, prompts, _TOKENIZER)
    print("generating selected M0 outputs", flush=True)
    m0_outputs = generate_outputs(BASE_MODEL, str(M0_CHECKPOINT), prompts, _TOKENIZER)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    for row, prompt, base_output, m0_output in zip(
        rows, prompts, base_outputs, m0_outputs, strict=True
    ):
        expected = str(row["behavior_decision"])
        base_decision = parse_qwen_native_output(base_output).decision
        m0_decision = parse_qwen_native_output(m0_output).decision
        base_ok = base_decision == expected
        m0_ok = m0_decision == expected
        if base_ok == m0_ok:
            rejected["both_or_neither_correct"] += 1
            continue
        chosen, rejected_output, chosen_decision, rejected_decision = (
            (base_output, m0_output, base_decision, m0_decision)
            if base_ok
            else (m0_output, base_output, m0_decision, base_decision)
        )
        by_category[expected].append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected_output,
                "preference_source": "local_base_m0_gold_decision_v1",
                "canonical_id": row_id(row),
                "source_dataset": str(row.get("source_dataset") or ""),
                "expected_decision": expected,
                "chosen_decision": chosen_decision,
                "rejected_decision": rejected_decision,
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "m0_checkpoint": str(M0_CHECKPOINT.relative_to(ROOT)),
                "m0_checkpoint_sha256": m0_hash,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": SEED,
                },
            }
        )

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for category, target in TARGETS.items():
        candidates = sorted(
            by_category[category],
            key=lambda row: hashlib.sha256(row["canonical_id"].encode()).hexdigest(),
        )
        rng.shuffle(candidates)
        selected.extend(candidates[:target])
    selected.sort(key=lambda row: hashlib.sha256(row["canonical_id"].encode()).hexdigest())

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    output_hash = sha256(output)
    report = {
        "schema_version": 1,
        "artifact_kind": "M1_LOCAL_CALIBRATION_PREFERENCE_PAIRS",
        "output": str(output.relative_to(ROOT)),
        "sha256": output_hash,
        "records_in": len(rows),
        "pairs_usable_before_balancing": sum(len(v) for v in by_category.values()),
        "pairs_out": len(selected),
        "selected_by_expected_decision": dict(
            Counter(row["expected_decision"] for row in selected)
        ),
        "available_by_expected_decision": {
            key: len(value) for key, value in sorted(by_category.items())
        },
        "rejected": dict(rejected),
        "heldout_excluded": len(heldout),
        "base_model": {"id": BASE_MODEL, "revision": BASE_REVISION},
        "m0_checkpoint": str(M0_CHECKPOINT.relative_to(ROOT)),
        "m0_checkpoint_sha256": m0_hash,
        "selection": "deterministic hash order with fixed per-decision caps; no API-generated labels",
        "note": (
            "Each pair compares actual deterministic outputs from the immutable base and selected "
            "M0 policy on Canonical-v2 training prompts. A pair is retained only when exactly one "
            "decision matches the canonical behavior label. Frozen DEV and confirmatory IDs are "
            "excluded before generation; no evaluation prompt enters this preference set."
        ),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


_TOKENIZER: Any = None

if __name__ == "__main__":
    raise SystemExit(main())
