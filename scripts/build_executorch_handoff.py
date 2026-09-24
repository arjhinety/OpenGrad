#!/usr/bin/env python3
"""Assemble the OpenWeights handoff package for an exported ExecuTorch artifact.

This study exports the `.pte` files but does not score them, so the handoff has to carry enough
for someone else's run to be comparable to the BF16 reference rather than merely similar. That
means four things, and the reason for each is worth stating because skipping any one of them
silently produces an unfalsifiable number:

* **The prompts, pre-rendered and pre-tokenized.** Rendering happens once, here, with the pinned
  Qwen renderer. Shipping token ids alongside the text removes the BOS question entirely: a runner
  fed ids cannot prepend anything.
* **A scorer that vendors OpenGrad's real parser.** `opengrad/formatting/parser.py` depends only on
  the standard library, so it is copied byte-for-byte rather than reimplemented, and a repository
  test asserts the copy never drifts. The parser is already byte-compatible with OpenWeights'
  `ToolCallParser.parseTaggedXml`, which is what makes the two measurements the same measurement.
* **The frozen gate.** So the verdict comes back in the same schema and against thresholds that
  were fixed before any candidate existed.
* **Parity notes.** The BOS collision in particular: the stock ExecuTorch metadata declares
  `get_bos_id: 248045`, which is `<|im_start|>`, and every rendered prompt already begins with that
  token.

Usage:
    python scripts/build_executorch_handoff.py --target cpu --pte <path-or-none>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.runner import (
    PINNED_EVALUATOR_REVISION,
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
)
from opengrad.hashing import sha256_bytes

PROMPTS = ROOT / "results/quantization/frozen_prompts_v1.jsonl"
REFERENCE = ROOT / "results/quantization/m1_v2_reference.json"
GATE = ROOT / "results/quantization/quantization_preservation_v1.json"
PARSER_SOURCE = ROOT / "src/opengrad/formatting/parser.py"
POLICY_SOURCE = ROOT / "src/opengrad/promotion/quantization.py"

BOS_TOKEN_ID = 248045  # <|im_start|>
EOS_TOKEN_IDS = (248046, 248044)  # <|im_end|>, <|endoftext|>
MAX_NEW_TOKENS = 512
CONTEXT_LENGTH = 4096
ENGINE_WINDOW = 5760

SCORER = '''#!/usr/bin/env python3
"""Score ExecuTorch generations against OpenGrad's frozen preservation gate.

Standalone: needs only Python 3.10+ and the two vendored OpenGrad modules beside it. Run it on the
raw generations and it produces the same metric schema and the same verdict the GGUF branch uses.

    python score_generations.py --generations my_run.jsonl --artifact qwen3_5_2b_fp32

`--generations` is JSONL with one object per example:

    {"example_id": "...", "raw": "<the model's output text>", "truncated": false}

Every example_id in frozen_prompts_confirmatory_v1.jsonl must appear exactly once. Missing,
duplicate, unknown, or errored generations are refused rather than dropped: a shrinking denominator
would report a better score for having answered less.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from opengrad_min.parser import parse_qwen_native_output
from opengrad_min.policy import evaluate_quantization_preservation
from opengrad_min.routing import routing_metrics

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--prompts", default=str(HERE / "frozen_prompts_confirmatory_v1.jsonl"))
    parser.add_argument("--reference", default=str(HERE / "m1_v2_reference.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prompts = {row["example_id"]: row for row in read_jsonl(args.prompts)}
    generations = read_jsonl(args.generations)

    seen = {}
    unknown, duplicate, failed = [], [], []
    for row in generations:
        identity = str(row["example_id"])
        if identity not in prompts:
            unknown.append(identity)
        elif identity in seen:
            duplicate.append(identity)
        else:
            seen[identity] = row
            if row.get("error"):
                failed.append(identity)
    missing = sorted(set(prompts) - set(seen))
    for label, offenders in (
        ("unknown example_id", unknown),
        ("duplicate example_id", duplicate),
        ("missing generation", missing),
        ("runtime error", failed),
    ):
        if offenders:
            raise SystemExit(
                "%s: %d example(s) %s" % (label, len(offenders), sorted(offenders)[:10])
            )

    predictions = []
    for identity in sorted(prompts):
        row = seen[identity]
        parsed = parse_qwen_native_output(
            str(row["raw"]), truncated=bool(row.get("truncated", False))
        )
        predictions.append(
            {
                "example_id": identity,
                "expected_decision": prompts[identity]["expected_decision"],
                "decision": parsed.decision,
                "status": parsed.status,
                "input_tokens": prompts[identity]["input_tokens"],
            }
        )

    metrics = routing_metrics(
        [row["expected_decision"] for row in predictions],
        [row["decision"] for row in predictions],
    )
    valid = sum(1 for row in predictions if row["status"] == "RAW_VALID")
    candidate = {
        "call_f1": metrics["call_f1"],
        "call_precision": metrics["call_precision"],
        "call_recall": metrics["call_recall"],
        "over_call_rate": metrics["over_call_rate"],
        "clarification_accuracy": metrics["clarification_accuracy"],
        "unsupported_accuracy": metrics["unsupported_accuracy"],
        "parse_valid_rate": valid / len(predictions),
    }
    with open(args.reference, encoding="utf-8") as handle:
        reference = {k: float(v) for k, v in json.load(handle)["metrics"].items()}

    verdict = evaluate_quantization_preservation(
        candidate,
        reference,
        existing_tool_policy={"decision": "PROMOTE", "policy_version": "tool_use_promotion_v4"},
    )
    result = {
        "artifact": args.artifact,
        "records": len(predictions),
        "submitted": len(prompts),
        "metrics": candidate,
        "reference": reference,
        "retention": {
            name: candidate[name] / reference[name]
            for name in (
                "call_f1",
                "call_precision",
                "call_recall",
                "clarification_accuracy",
                "unsupported_accuracy",
            )
        },
        "over_call_rate_delta": candidate["over_call_rate"] - reference["over_call_rate"],
        "verdict": verdict,
        "parser_status": dict(sorted(Counter(r["status"] for r in predictions).items())),
    }
    destination = args.out or ("%s.score.json" % args.artifact)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print("%s  ->  %s" % (args.artifact, verdict["decision"]))
    for name, kept in sorted(result["retention"].items()):
        flag = "" if kept >= 0.99 else "   <-- below 99%"
        print("  %-24s %.6f  retention %.4f%s" % (name, candidate[name], kept, flag))
    print("  %-24s %.6f  delta %+.6f" % (
        "over_call_rate", candidate["over_call_rate"], result["over_call_rate_delta"]))
    print("  %-24s %.6f" % ("parse_valid_rate", candidate["parse_valid_rate"]))
    if verdict["failed_dimensions"]:
        print("  FAILED: %s" % ", ".join(verdict["failed_dimensions"]))
    print("wrote %s" % destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

ROUTING = '''"""Vendored from opengrad/evaluation/routing.py.

The only change is that DECISIONS is defined here instead of imported from
opengrad.data.behavior, which pulls in a YAML taxonomy this package does not need. The frozenset
is identical.
"""

from __future__ import annotations

from collections.abc import Iterable

DECISIONS = frozenset({"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"})


def routing_metrics(actual: Iterable[str], predicted: Iterable[str]) -> dict[str, object]:
    actual_list, predicted_list = list(actual), list(predicted)
    if len(actual_list) != len(predicted_list) or not actual_list:
        raise ValueError("actual and predicted must have equal non-zero length")
    matrix = {a: {p: 0 for p in DECISIONS} for a in DECISIONS}
    for truth, guess in zip(actual_list, predicted_list):
        if truth not in DECISIONS or guess not in DECISIONS:
            raise ValueError("routing labels must use the canonical decisions")
        matrix[truth][guess] += 1
    call_tp = matrix["CALL"]["CALL"]
    call_pred = sum(matrix[a]["CALL"] for a in DECISIONS)
    call_actual = sum(matrix["CALL"].values())
    precision = call_tp / call_pred if call_pred else 0.0
    recall = call_tp / call_actual if call_actual else 0.0
    return {
        "confusion_matrix": matrix,
        "call_precision": precision,
        "call_recall": recall,
        "call_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "must_call_accuracy": recall,
        "no_call_accuracy": matrix["ANSWER"]["ANSWER"] / sum(matrix["ANSWER"].values())
        if sum(matrix["ANSWER"].values())
        else 0.0,
        "clarification_accuracy": matrix["CLARIFY"]["CLARIFY"] / sum(matrix["CLARIFY"].values())
        if sum(matrix["CLARIFY"].values())
        else 0.0,
        "unsupported_accuracy": matrix["UNSUPPORTED"]["UNSUPPORTED"]
        / sum(matrix["UNSUPPORTED"].values())
        if sum(matrix["UNSUPPORTED"].values())
        else 0.0,
        "under_call_rate": matrix["CALL"]["ANSWER"] / call_actual if call_actual else 0.0,
        "over_call_rate": sum(matrix[a]["CALL"] for a in ("ANSWER", "CLARIFY", "UNSUPPORTED"))
        / (len(actual_list) - call_actual)
        if len(actual_list) > call_actual
        else 0.0,
    }
'''


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def token_ids(prompts: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Tokenize with the pinned tokenizer so the handoff cannot inherit a BOS ambiguity."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-2B", revision=PINNED_MODEL_REVISION, trust_remote_code=False
    )
    if getattr(tokenizer, "add_bos_token", False):
        raise SystemExit("pinned tokenizer unexpectedly reports add_bos_token=True")
    return {
        row["example_id"]: tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        for row in prompts
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["cpu", "qnn", "mediatek"])
    parser.add_argument("--pte", default=None, help="path to the exported .pte, if one exists")
    parser.add_argument(
        "--export-json",
        action="append",
        default=None,
        help="export result json; repeat for a target with several artifacts (fp32 and 8da4w)",
    )
    parser.add_argument("--partition", default="confirmatory")
    args = parser.parse_args()

    out = ROOT / "release/executorch" / args.target
    (out / "opengrad_min").mkdir(parents=True, exist_ok=True)
    # Running the scorer locally leaves __pycache__ behind, and a published handoff should contain
    # only the files someone is meant to read or run.
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    prompts = [
        row
        for row in (
            json.loads(line)
            for line in PROMPTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["partition"] == args.partition
    ]
    prompts.sort(key=lambda row: row["example_id"])
    ids = token_ids(prompts)

    prompt_file = out / f"frozen_prompts_{args.partition}_v1.jsonl"
    with prompt_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in prompts:
            handle.write(
                json.dumps(
                    {
                        "example_id": row["example_id"],
                        "expected_decision": row["expected_decision"],
                        "prompt": row["prompt"],
                        "prompt_sha256": row["prompt_sha256"],
                        "input_tokens": row["input_tokens"],
                        "token_ids": ids[row["example_id"]],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    # Vendored verbatim: parser.py imports only json/re/dataclasses/typing, so the copy is the
    # source. tests/evaluation/test_executorch_handoff.py asserts it stays byte-identical.
    shutil.copyfile(PARSER_SOURCE, out / "opengrad_min/parser.py")
    shutil.copyfile(POLICY_SOURCE, out / "opengrad_min/policy.py")
    (out / "opengrad_min/routing.py").write_text(ROUTING, encoding="utf-8", newline="\n")
    (out / "opengrad_min/__init__.py").write_text(
        '"""Minimal vendored OpenGrad measurement code for the ExecuTorch handoff."""\n',
        encoding="utf-8",
        newline="\n",
    )
    (out / "score_generations.py").write_text(SCORER, encoding="utf-8", newline="\n")
    shutil.copyfile(REFERENCE, out / "m1_v2_reference.json")
    shutil.copyfile(GATE, out / "quantization_preservation_v1.json")

    # Artifact metadata comes from the export records, not from the file being present locally: the
    # .pte files are multi-gigabyte and are pushed to the Hub straight from the Modal volume, so
    # they never touch this machine. The export record already carries the authoritative size and
    # hash, and the upload verifies that hash on the way out.
    exports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.export_json or [])
    ]
    artifacts = []
    for export in exports:
        if not export.get("exported"):
            continue
        artifacts.append(
            {
                "file": Path(str(export["artifact"])).name,
                "bytes": export["artifact_bytes"],
                "sha256": export["artifact_sha256"],
                "quantization": export.get("quantization") or export.get("pt2e_quantize"),
                "group_size": export.get("group_size"),
                "status": export["status"],
                "toolchain_patch": export.get("toolchain_patch"),
            }
        )
    export = exports[0] if exports else None

    if args.pte and Path(args.pte).is_file():
        destination = out / Path(args.pte).name
        shutil.copyfile(args.pte, destination)
        artifacts.append(
            {
                "file": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "status": "EXPORTED_PENDING_EVALUATION",
            }
        )

    manifest = {
        "schema_version": 1,
        "target": args.target,
        "built_at": datetime.now(UTC).isoformat(),
        "parent_experiment_id": "m1_dpo_canonical_v2_final_v2",
        "parent_checkpoint": "m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30",
        "parent_model_revision": "f33d20308982f37deb459076f489e794d5521ee3",
        "status": (export or {}).get("status", "EXPORTED_PENDING_EVALUATION"),
        "artifacts": artifacts,
        "export_records": exports,
        "measurement_contract": {
            "renderer": "qwen3_5_2b_v1",
            "tokenizer_revision": PINNED_MODEL_REVISION,
            "template_hash": PINNED_TEMPLATE_HASH,
            "evaluator_revision": PINNED_EVALUATOR_REVISION,
            "partition": args.partition,
            "examples": len(prompts),
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "context_length": CONTEXT_LENGTH,
            "engine_window": ENGINE_WINDOW,
            "add_bos_token": False,
            "bos_token_id_declared_by_pte_metadata": BOS_TOKEN_ID,
            "eos_token_ids": list(EOS_TOKEN_IDS),
        },
        "files": {
            "prompts": prompt_file.name,
            "prompts_sha256": sha256_file(prompt_file),
            "scorer": "score_generations.py",
            "vendored_parser_sha256": sha256_file(out / "opengrad_min/parser.py"),
            "reference": "m1_v2_reference.json",
            "gate": "quantization_preservation_v1.json",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "PARITY_NOTES.md").write_text(parity_notes(args.target, len(prompts)), encoding="utf-8")

    print(f"handoff -> {out.relative_to(ROOT)}")
    print(f"  prompts        {len(prompts)} ({prompt_file.name})")
    if artifacts:
        for item in artifacts:
            print(
                f"  artifact       {item['file']}  {item['bytes']:,} bytes  "
                f"({item.get('quantization') or 'fp32'})"
            )
    else:
        print("  artifact       NONE (no export produced a .pte)")
    print(f"  status         {manifest['status']}")
    return 0


def parity_notes(target: str, examples: int) -> str:
    return f"""# Runtime parity notes — ExecuTorch {target}

Read this before running. Each item below is a way to produce numbers that look valid and are not
comparable to the BF16 reference.

## 1. Do not prepend BOS

The pinned Qwen3.5 tokenizer sets `add_bos_token=False` and `bos_token=None`. The stock ExecuTorch
config nevertheless declares `get_bos_id: {BOS_TOKEN_ID}` in its metadata, and {BOS_TOKEN_ID} is
`<|im_start|>` — the token every rendered prompt *already starts with*. A runner that honours that
metadata emits `<|im_start|><|im_start|>system…` and measures a different model.

`frozen_prompts_{{partition}}_v1.jsonl` ships `token_ids` for exactly this reason. Feeding ids
directly is the safest path; if you feed text, disable BOS insertion and verify the first token id
is {BOS_TOKEN_ID} exactly once.

## 2. Use the shipped prompt bytes verbatim

Do not re-apply a chat template. The prompts were rendered once with the pinned renderer
(`qwen3_5_2b_v1`, template hash `{PINNED_TEMPLATE_HASH}`). `prompt_sha256` is in the file; check a
few before a long run.

## 3. Decoding must be greedy

`temperature 0.0`, `top_p 1.0`, `do_sample false`, `max_new_tokens {MAX_NEW_TOKENS}`. The reference
was produced greedily; any sampling makes the comparison noise-limited.

## 4. Stop tokens

Stop on `<|im_end|>` ({EOS_TOKEN_IDS[0]}) or `<|endoftext|>` ({EOS_TOKEN_IDS[1]}). If generation
stops because it hit the {MAX_NEW_TOKENS}-token budget, set `"truncated": true` on that row — the
parser treats a mid-tool-call cutoff as a format error rather than a silent wrong answer.

## 5. Context window

The longest prompt in this partition is 5,235 tokens and the completion budget is
{MAX_NEW_TOKENS}, so the runtime needs a {ENGINE_WINDOW}-token window. The stock Qwen3.5 ExecuTorch
config ships `max_seq_length: 2048`, which would truncate prompts into a different measurement;
these exports raise it to {ENGINE_WINDOW}. Two of the {examples} examples exceed 4,096 tokens and
are bucketed as `overflow` rather than dropped.

## 6. Static shape and sequential prefill

The Qwen3.5 ExecuTorch bring-up is fp32 with `enable_dynamic_shape=False`, and `runner.native`
falls back to sequential token prefill for multi-token prompts. This is a throughput property, not
a correctness one, but it makes a full {examples}-example run slow on a single stream — shard it.

## 7. Do not let a failed example disappear

`score_generations.py` refuses a run whose generations do not reconcile exactly with the submitted
prompts: missing, duplicate, unknown, or errored rows are errors, not omissions. A shrinking
denominator reports a better score for having answered less.

## 8. Tool-call format

The model emits Qwen3.5's XML form,
`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`. llama.cpp's built-in
tool parser does not recognise it; OpenWeights' `ToolCallParser.parseTaggedXml` does, and the
vendored `opengrad_min/parser.py` is byte-compatible with it. Score with the vendored parser, not
with a runtime's own tool extraction.

## What a result is worth

These artifacts carry `EXPORTED_PENDING_EVALUATION`. They have no preservation verdict until this
scorer produces one. Passing `quantization_preservation_v1` is what makes an artifact releasable;
a successful export on its own is not evidence of preserved behaviour.
"""


if __name__ == "__main__":
    raise SystemExit(main())
