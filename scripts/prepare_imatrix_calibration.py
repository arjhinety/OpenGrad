#!/usr/bin/env python3
"""Freeze the training-side calibration corpus used to build the GGUF importance matrix.

An importance matrix decides which weights a k-quant spends its bits on, so the text it is computed
from is part of the quantized artifact. Three constraints follow.

**Contamination.** Calibrating on DEV or confirmatory examples would tune the quantizer against the
set that later judges it, and the resulting score would be unfalsifiable. Every frozen held-out id
is excluded by construction and the overlap is recorded, not assumed.

**Labelling.** Behaviour is derived with `parse_qwen_native_output` — the same function that scores
the model — rather than read from `metadata.behavior.decision`. That field is coarse: the adapters
derive "no tool call -> ANSWER" and never distinguish the sub-types, so `when2call-sft` declares
14,829 ANSWER records when by content roughly half are clarifications and half are refusals. A
corpus balanced on the declared field would be balanced on a fiction. (An earlier v1 of this corpus
was built on that misreading; it drew only from the preference pairs and is superseded by this
file. Nothing consumed it.)

**Distribution.** The frozen evaluation is 35.5% CALL, 29.0% CLARIFY and 35.5% UNSUPPORTED with no
ANSWER examples at all, so the calibration set targets a roughly even spread across the four
behaviours rather than mirroring raw corpus frequencies, where a source like `xlam` (100% CALL by
content, 59,370 records) would otherwise dominate and starve the two metrics already weakest in the
BF16 reference (`unsupported_accuracy` 0.5386, `clarification_accuracy` 0.7655).

Each record is deployment-shaped text: a rendered prompt followed by the assistant continuation, so
activations cover both the prefill region (tool schemas, user turn) and the generation region (the
tool-call XML, the clarification, or the refusal). Preference records use `chosen` and never
`rejected` — calibrating on behaviour the model was trained away from would weight the quantizer
toward it.

IMPORTANT — the consumer must pass `--parse-special`:

    llama-imatrix -m model.gguf -f m1_v2_imatrix_calibration_v2.txt --parse-special ...

`llama-imatrix` defaults `parse_special` to false, which would tokenize `<|im_start|>` as literal
characters and compute the matrix over a token distribution that never occurs at inference.

Usage:
    python scripts/prepare_imatrix_calibration.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.contamination.audit import QUARANTINE_PATH, load_quarantine  # noqa: E402
from opengrad.data.materialize import iter_materialized_rows  # noqa: E402
from opengrad.data.real_analysis import _restore_row  # noqa: E402
from opengrad.data.renderers import Qwen35_2BRenderer, _qwen_messages  # noqa: E402
from opengrad.data.schema import normalize_tool  # noqa: E402
from opengrad.evaluation.runner import (  # noqa: E402
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
    _qwen_tool,
)
from opengrad.formatting.parser import parse_qwen_native_output  # noqa: E402

CORPUS_ID = "m1_v2_imatrix_calibration_v2"
SUPERSEDES = "m1_v2_imatrix_calibration_v1"

PREFERENCE = Path("data/processed/m1_calibration_preference_pairs_v1.jsonl")
PARTITION = Path("reports/evaluation/behavioral-heldout-v2-partition.json")
MATERIALIZED = Path("data/processed/normalization-v1")

# The canonical-v2 corpus, exactly as configs/experiments/m0_sft_canonical_v2_final.yaml declares
# it. BUTTON and LoopTool are excluded there as upstream-unavailable and are excluded here too, so
# calibration text stays inside the distribution the model was actually trained on.
SFT_SOURCES = ("when2call-sft", "toolace", "glaive", "xlam")

OUT_JSONL = Path(f"manifests/quantization/{CORPUS_ID}.jsonl")
OUT_MANIFEST = Path(f"manifests/quantization/{CORPUS_ID}.json")
OUT_TEXT = Path(f"manifests/quantization/{CORPUS_ID}.txt")

BEHAVIORS = ("ANSWER", "CALL", "CLARIFY", "UNSUPPORTED")
RECORD_SEPARATOR = "\n\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_key(identity: str) -> str:
    """Stable shuffle that does not depend on an RNG seed, filesystem, or insertion order."""
    return hashlib.sha256(identity.encode()).hexdigest()


def held_out_ids() -> tuple[set[str], dict[str, Any]]:
    partition = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    dev = set(partition["example_ids"]["dev"])
    confirmatory = set(partition["example_ids"]["confirmatory"])
    quarantined: set[str] = set()
    for ids in load_quarantine(ROOT / QUARANTINE_PATH).by_split().values():
        quarantined |= set(ids)
    return dev | confirmatory | quarantined, {
        "dev_examples": len(dev),
        "dev_fingerprint": partition["dev"]["fingerprint"],
        "confirmatory_examples": len(confirmatory),
        "confirmatory_fingerprint": partition["confirmatory"]["fingerprint"],
        "quarantined_examples": len(quarantined),
    }


def behaviour_of(messages: list[dict[str, Any]]) -> str | None:
    """Label the final assistant turn with the evaluator's own parser.

    A turn carrying structured tool calls is CALL without consulting the parser: the parser reads
    Qwen's *emitted* text, and a materialized training turn holds the calls as data rather than as
    rendered XML.
    """
    final = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if final is None:
        return None
    if final.get("tool_calls"):
        return "CALL"
    content = str(final.get("content") or "").strip()
    if not content:
        return None
    parsed = parse_qwen_native_output(content)
    return parsed.decision if parsed.decision in BEHAVIORS else None


class _Messages:
    """Duck type for the renderer's private message mapper, which only reads `.messages`."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages


def render_calibration_text(renderer: Qwen35_2BRenderer, row: dict[str, Any]) -> str:
    """Render a training row into the text the model actually sees, without the training gate.

    `render_sft` is two things: `validate_training_semantics()` followed by the chat template. The
    strictness lives entirely in the first half — it rejects schema keywords the training contract
    disallows, orphaned tool results, and non-FIFO call ordering, which caused 14,379 of 24,000
    scanned rows to fail and left CALL badly under-filled. None of that matters for an importance
    matrix: the quantizer needs the activation distribution of real deployment text, and at
    deployment the lenient evaluation contract is what applies.

    So this calls the same pinned template on the same messages, with the evaluator's own
    `_qwen_tool` leniency applied to the tool schemas. It deliberately does not construct a
    `CanonicalEvaluationExample`, which would require relabelling a training row as
    `evaluation_only` — a lie about eligibility to satisfy a validator.
    """
    tokenizer = renderer._load()
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": renderer.enable_thinking,
    }
    if row["tools"]:
        kwargs["tools"] = [normalize_tool(_qwen_tool(tool)) for tool in row["tools"]]
    return str(tokenizer.apply_chat_template(_qwen_messages(_Messages(row["messages"])), **kwargs))


def load_preference(excluded: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overlap: list[str] = []
    with (ROOT / PREFERENCE).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            identity = str(raw.get("canonical_id") or raw.get("example_id") or f"pref-{index:06d}")
            if identity in excluded:
                overlap.append(identity)
                continue
            behaviour = str(raw["expected_decision"])
            if behaviour not in BEHAVIORS:
                raise SystemExit(f"record {identity} has unknown behaviour {behaviour!r}")
            text = str(raw["prompt"]) + str(raw["chosen"])
            rows.append(
                {
                    "calibration_id": identity,
                    "behavior": behaviour,
                    "behavior_source": "preference_expected_decision",
                    "source_dataset": str(raw.get("source_dataset", "unknown")),
                    "origin": "m1_calibration_preference_pairs_v1",
                    "text": text,
                    "text_sha256": sha256_text(text),
                    "characters": len(text),
                }
            )
    if overlap:
        raise SystemExit(
            f"preference corpus overlaps {len(overlap)} frozen held-out id(s): {sorted(overlap)[:10]}"
        )
    return rows


def collect_sft(
    excluded: set[str],
    *,
    scan_limit: int,
    per_source_cap: int,
    targets: dict[str, int],
    have: Counter[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically top up each behaviour from the canonical-v2 SFT sources."""
    renderer = Qwen35_2BRenderer(revision=PINNED_MODEL_REVISION, enable_thinking=False)
    candidates: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    scanned: dict[str, int] = {}
    unlabelled: Counter[str] = Counter()
    overlap = 0

    for source in SFT_SOURCES:
        directory = ROOT / MATERIALIZED / source
        if not (directory / "manifest.json").is_file():
            raise SystemExit(f"missing materialization for {source}: {directory}")
        seen = 0
        for raw in iter_materialized_rows(directory):
            if seen >= scan_limit:
                break
            seen += 1
            row = _restore_row(raw)
            identity = str(row["id"])
            if identity in excluded:
                overlap += 1
                continue
            behaviour = behaviour_of(row["messages"])
            if behaviour is None:
                unlabelled[source] += 1
                continue
            candidates[behaviour].append((order_key(f"{source}:{identity}"), source, row))
        scanned[source] = seen

    if overlap:
        raise SystemExit(f"SFT scan hit {overlap} frozen held-out id(s); refusing to calibrate")

    selected: list[dict[str, Any]] = []
    per_source: Counter[str] = Counter()
    render_failures: Counter[str] = Counter()
    for behaviour in BEHAVIORS:
        needed = max(0, targets[behaviour] - have[behaviour])
        if not needed:
            continue
        for _, source, row in sorted(candidates[behaviour], key=lambda item: item[0]):
            if needed <= 0:
                break
            if per_source[source] >= per_source_cap:
                continue
            try:
                text = render_calibration_text(renderer, row)
            except Exception as exc:  # quarantine rather than crash the corpus build
                # Bucket by error code, not by message: these carry the offending field names, so
                # keying on the message produces thousands of near-unique entries and a manifest
                # that is mostly noise.
                code = str(exc).split(":", 1)[0].strip() or type(exc).__name__
                render_failures[f"{type(exc).__name__}/{code}"] += 1
                continue
            per_source[source] += 1
            needed -= 1
            selected.append(
                {
                    "calibration_id": str(row["id"]),
                    "behavior": behaviour,
                    "behavior_source": "parse_qwen_native_output",
                    "source_dataset": source,
                    "origin": "canonical_v2_sft",
                    "text": text,
                    "text_sha256": sha256_text(text),
                    "characters": len(text),
                }
            )

    return selected, {
        "scanned_per_source": scanned,
        "scan_limit": scan_limit,
        "per_source_cap": per_source_cap,
        "unlabelled_rows": dict(unlabelled),
        "render_failures": dict(render_failures),
        "available_per_behavior": {b: len(candidates[b]) for b in BEHAVIORS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-per-behavior", type=int, default=300)
    parser.add_argument("--per-source-cap", type=int, default=250)
    parser.add_argument("--scan-limit", type=int, default=8000)
    args = parser.parse_args()

    excluded, exclusion = held_out_ids()
    rows = load_preference(excluded)
    have = Counter(row["behavior"] for row in rows)
    targets = {name: args.target_per_behavior for name in BEHAVIORS}

    supplement, scan_report = collect_sft(
        excluded,
        scan_limit=args.scan_limit,
        per_source_cap=args.per_source_cap,
        targets=targets,
        have=have,
    )
    rows.extend(supplement)
    rows.sort(key=lambda row: order_key(row["calibration_id"]))

    out_jsonl = ROOT / OUT_JSONL
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    text_path = ROOT / OUT_TEXT
    text_path.write_text(
        RECORD_SEPARATOR.join(row["text"] for row in rows) + "\n", encoding="utf-8", newline="\n"
    )

    characters = [row["characters"] for row in rows]
    behaviour_counts = Counter(row["behavior"] for row in rows)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "IMATRIX_CALIBRATION_CORPUS",
        "corpus_id": CORPUS_ID,
        "supersedes": SUPERSEDES,
        "supersession_reason": (
            "v1 balanced on metadata.behavior.decision, which labels every non-tool-call turn "
            "ANSWER and therefore hid that when2call-sft is roughly half clarification and half "
            "refusal by content. v2 labels with parse_qwen_native_output, the same function that "
            "scores the model. v1 was never consumed by any artifact."
        ),
        "purpose": "GGUF k-quant importance matrix for M1-v2 deployment descendants",
        "records": len(rows),
        "behavior_counts": {name: behaviour_counts.get(name, 0) for name in BEHAVIORS},
        "behavior_label_source": {
            "preference_expected_decision": sum(
                1 for r in rows if r["behavior_source"] == "preference_expected_decision"
            ),
            "parse_qwen_native_output": sum(
                1 for r in rows if r["behavior_source"] == "parse_qwen_native_output"
            ),
        },
        "origin_counts": dict(sorted(Counter(r["origin"] for r in rows).items())),
        "source_counts": dict(sorted(Counter(r["source_dataset"] for r in rows).items())),
        "selection": {
            "target_per_behavior": args.target_per_behavior,
            **scan_report,
        },
        "character_stats": {
            "min": min(characters),
            "median": statistics.median(characters),
            "mean": round(statistics.fmean(characters), 3),
            "max": max(characters),
            "total": sum(characters),
        },
        "outputs": {
            "jsonl": str(OUT_JSONL).replace("\\", "/"),
            "jsonl_sha256": sha256_file(out_jsonl),
            "text": str(OUT_TEXT).replace("\\", "/"),
            "text_sha256": sha256_file(text_path),
            "text_bytes": text_path.stat().st_size,
        },
        "contamination": {**exclusion, "excluded_ids_considered": len(excluded), "overlap_count": 0},
        "render_contract": {
            "renderer": "qwen3_5_2b_v1",
            "tokenizer_revision": PINNED_MODEL_REVISION,
            "template_hash": PINNED_TEMPLATE_HASH,
            "preference_text": "prompt + chosen (rejected completions are never used)",
            "sft_text": "render_sft of the full canonical conversation",
        },
        "consumer_requirements": {
            "tool": "llama-imatrix",
            "required_flags": ["--parse-special"],
            "reason": (
                "llama-imatrix defaults parse_special to false; without the flag the chat special "
                "tokens are tokenized as literal text and the matrix is computed over a "
                "distribution that never occurs at inference"
            ),
        },
    }
    (ROOT / OUT_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"calibration records : {len(rows)}")
    print(f"behaviour           : {manifest['behavior_counts']}")
    print(f"label source        : {manifest['behavior_label_source']}")
    print(f"origins             : {manifest['origin_counts']}")
    print(f"sources             : {manifest['source_counts']}")
    print(f"characters          : {manifest['character_stats']['total']:,}")
    print(f"overlap with frozen : {manifest['contamination']['overlap_count']}")
    print(f"text sha256         : {manifest['outputs']['text_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
