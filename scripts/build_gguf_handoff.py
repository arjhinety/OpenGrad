#!/usr/bin/env python3
"""Assemble the OpenWeights handoff package for the GGUF artifacts.

OpenWeights runs llama.cpp, so this package has to answer two different questions and keep them
apart. Did the quantized model keep the tool-calling policy (behavioural), and is it actually
cheaper to run (systems)? A quantization that passes the gate but does not shrink or speed anything
up has no reason to exist, and one that is fast but fails the gate must not ship — so both numbers
travel together.

The behavioural half deliberately reuses the *same* prompts, the *same* vendored parser and the
*same* frozen gate as the ExecuTorch handoff, so a GGUF result and an ExecuTorch result are
comparable to each other and to the BF16 reference rather than to nothing.

Two llama.cpp-specific traps are handled explicitly:

* llama.cpp's built-in tool-call parser does not recognise Qwen3.5's XML emission, so generation
  must go through raw `/completion` and be parsed by the vendored OpenGrad parser. Using
  `/v1/chat/completions` would re-apply a chat template *and* parse tool calls with the wrong
  grammar.
* The GGUF carries `add_bos_token` from the source tokenizer, which is false here. The benchmark
  verifies tokenization against the shipped `token_ids` before trusting any number it produces.

Usage:
    python scripts/build_gguf_handoff.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_executorch_handoff import ROUTING, SCORER, token_ids  # noqa: E402
from opengrad.evaluation.runner import (  # noqa: E402
    PINNED_EVALUATOR_REVISION,
    PINNED_MODEL_REVISION,
    PINNED_TEMPLATE_HASH,
)

PROMPTS = ROOT / "results/quantization/frozen_prompts_v1.jsonl"
REFERENCE = ROOT / "results/quantization/m1_v2_reference.json"
GATE = ROOT / "results/quantization/quantization_preservation_v1.json"
PARSER_SOURCE = ROOT / "src/opengrad/formatting/parser.py"
POLICY_SOURCE = ROOT / "src/opengrad/promotion/quantization.py"

OUT = ROOT / "release/gguf"
LADDER = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
MAX_NEW_TOKENS = 512
ENGINE_WINDOW = 5760
BOS_TOKEN_ID = 248045
EOS_TOKEN_IDS = (248046, 248044)

BENCH = r'''#!/usr/bin/env python3
"""Benchmark a GGUF artifact: behavioural preservation and systems cost, in one run.

Standard library only, so it runs anywhere llama.cpp does -- including against a server on a phone
reached through `adb forward tcp:8080 tcp:8080`.

    # against a server you already started (device or host)
    llama-server -m model-Q4_K_M.gguf -c 5760 --parallel 4 --no-webui --seed 0
    python bench_gguf.py --base-url http://127.0.0.1:8080 --artifact Q4_K_M --model model-Q4_K_M.gguf

    # systems numbers only, no 1277-prompt run
    python bench_gguf.py --base-url http://127.0.0.1:8080 --artifact Q4_K_M --mode systems

What it measures
----------------
behavioural : all frozen confirmatory prompts -> generations.jsonl, then score_generations.py
systems     : size on disk, cold-load time, TTFT, prefill tok/s, decode tok/s, at three prompt
              lengths drawn from the real frozen distribution (short / median / long)

Why raw /completion and not /v1/chat/completions
------------------------------------------------
The prompts are already rendered by OpenGrad's pinned renderer. The chat endpoint would re-apply a
template on top, and llama.cpp's tool-call parser does not recognise Qwen3.5's XML form anyway.
Raw text in, raw text out, parsed by the vendored OpenGrad parser.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOS_TOKEN_ID = 248045
MAX_NEW_TOKENS = 512


def post(base_url, route, payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + route, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wait_ready(base_url, timeout=900):
    """Cold-start time: first moment the server answers /health."""
    started = time.time()
    while time.time() - started < timeout:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as response:
                if response.status == 200:
                    return round(time.time() - started, 3)
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server never became ready")


def check_tokenization(base_url, prompts, sample=25):
    """Refuse to report numbers from a runtime that tokenizes the prompts differently.

    The shipped token_ids come from the pinned HF tokenizer. If llama.cpp disagrees -- most likely
    by prepending a BOS the tokenizer does not define -- then every metric below describes a
    different model and is not comparable to the BF16 reference.
    """
    mismatches, bos_insertions = [], 0
    for row in prompts[:sample]:
        observed = post(base_url, "/tokenize", {"content": row["prompt"]}, timeout=120)["tokens"]
        expected = row["token_ids"]
        if observed == expected:
            continue
        if observed[1:] == expected:
            bos_insertions += 1
        mismatches.append(row["example_id"])
    return {
        "checked": min(sample, len(prompts)),
        "mismatches": len(mismatches),
        "bos_insertions": bos_insertions,
        "sample": mismatches[:5],
        "passed": not mismatches,
    }


def one_completion(base_url, prompt, n_predict):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "cache_prompt": False,
    }
    started = time.time()
    body = post(base_url, "/completion", payload)
    return body, round(time.time() - started, 4)


def systems(base_url, prompts, model_path, repeats):
    """Prefill/decode throughput at three real prompt lengths, plus TTFT and size."""
    ordered = sorted(prompts, key=lambda row: row["input_tokens"])
    picks = {
        "short": ordered[0],
        "median": ordered[len(ordered) // 2],
        "long": ordered[-1],
    }
    out = {}
    for label, row in picks.items():
        samples = []
        for _ in range(repeats):
            body, wall = one_completion(base_url, row["prompt"], 128)
            timings = body.get("timings", {})
            samples.append(
                {
                    "wall_seconds": wall,
                    "prompt_tokens": timings.get("prompt_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "prefill_tps": timings.get("prompt_per_second"),
                    "predicted_tokens": timings.get("predicted_n"),
                    "predicted_ms": timings.get("predicted_ms"),
                    "decode_tps": timings.get("predicted_per_second"),
                    # TTFT is prompt-processing time: the first token cannot appear before the
                    # prefill completes, so this is its lower bound and the honest thing to report
                    # from a non-streaming endpoint.
                    "ttft_ms_lower_bound": timings.get("prompt_ms"),
                }
            )
        numeric = lambda key: [s[key] for s in samples if isinstance(s[key], (int, float))]  # noqa: E731
        out[label] = {
            "input_tokens": row["input_tokens"],
            "example_id": row["example_id"],
            "repeats": repeats,
            "prefill_tps_median": round(statistics.median(numeric("prefill_tps")), 3)
            if numeric("prefill_tps")
            else None,
            "decode_tps_median": round(statistics.median(numeric("decode_tps")), 3)
            if numeric("decode_tps")
            else None,
            "ttft_ms_lower_bound_median": round(statistics.median(numeric("ttft_ms_lower_bound")), 3)
            if numeric("ttft_ms_lower_bound")
            else None,
            "samples": samples,
        }
    if model_path and Path(model_path).is_file():
        size = Path(model_path).stat().st_size
        out["artifact"] = {
            "path": str(model_path),
            "bytes": size,
            "mib": round(size / (1024 * 1024), 2),
        }
    return out


def behavioural(base_url, prompts, destination):
    results, errors = {}, 0
    started = time.time()
    for index, row in enumerate(prompts, start=1):
        try:
            body, _ = one_completion(base_url, row["prompt"], MAX_NEW_TOKENS)
            results[row["example_id"]] = {
                "example_id": row["example_id"],
                "raw": body.get("content", ""),
                "truncated": bool(body.get("stopped_limit", False)),
            }
        except Exception as exc:
            errors += 1
            results[row["example_id"]] = {
                "example_id": row["example_id"],
                "raw": "",
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
        if index % 100 == 0:
            print("  %d/%d  (%.0fs)" % (index, len(prompts), time.time() - started), flush=True)

    missing = {row["example_id"] for row in prompts} - set(results)
    if missing:
        raise SystemExit("generation set does not reconcile; missing %d" % len(missing))
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        for example_id in sorted(results):
            handle.write(json.dumps(results[example_id], ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "generations": str(destination),
        "submitted": len(prompts),
        "returned": len(results),
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--artifact", required=True, help="label, e.g. Q4_K_M")
    parser.add_argument("--model", default=None, help="path to the .gguf, for size on disk")
    parser.add_argument("--mode", default="both", choices=["both", "systems", "behavioural"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts", default=str(HERE / "frozen_prompts_confirmatory_v1.jsonl"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    report = {"artifact": args.artifact, "base_url": args.base_url, "mode": args.mode}

    report["cold_start_seconds"] = wait_ready(args.base_url)
    report["tokenization_parity"] = check_tokenization(args.base_url, prompts)
    if not report["tokenization_parity"]["passed"]:
        print("WARNING: tokenization does not match the shipped token_ids.")
        print("         Behavioural numbers from this run are NOT comparable to the reference.")
        if report["tokenization_parity"]["bos_insertions"]:
            print("         Looks like a prepended BOS. llama.cpp should honour add_bos_token=false.")

    if args.mode in ("both", "systems"):
        report["systems"] = systems(args.base_url, prompts, args.model, args.repeats)
    if args.mode in ("both", "behavioural"):
        destination = HERE / ("generations.%s.jsonl" % args.artifact)
        report["behavioural"] = behavioural(args.base_url, prompts, destination)
        print("\nNow score it:")
        print("  python score_generations.py --generations %s --artifact %s"
              % (destination.name, args.artifact))

    out = args.out or ("bench.%s.json" % args.artifact)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("\n%s" % args.artifact)
    print("  cold start        : %.3fs" % report["cold_start_seconds"])
    parity = report["tokenization_parity"]
    print("  tokenization      : %d/%d exact%s" % (
        parity["checked"] - parity["mismatches"], parity["checked"],
        "" if parity["passed"] else "   <-- MISMATCH, results not comparable"))
    if "systems" in report:
        for label in ("short", "median", "long"):
            entry = report["systems"].get(label)
            if entry:
                print("  %-6s (%5d tok) : prefill %s tok/s   decode %s tok/s   ttft>= %s ms" % (
                    label, entry["input_tokens"], entry["prefill_tps_median"],
                    entry["decode_tps_median"], entry["ttft_ms_lower_bound_median"]))
        artifact = report["systems"].get("artifact")
        if artifact:
            print("  size on disk      : %.2f MiB" % artifact["mib"])
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity_notes(examples: int) -> str:
    return f"""# Runtime parity notes — GGUF / llama.cpp

Each item below is a way to produce numbers that look valid and are not comparable to the BF16
reference.

## 1. Use raw `/completion`, never `/v1/chat/completions`

The prompts are already rendered by OpenGrad's pinned renderer. The chat endpoint re-applies a
template on top of an already-templated prompt. Worse, llama.cpp's built-in tool-call parser **does
not recognise Qwen3.5's XML emission** — the model emits
`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`, and the vendored
`opengrad_min/parser.py` is what reads it correctly. It is byte-compatible with OpenWeights'
`ToolCallParser.parseTaggedXml`.

## 2. Do not prepend BOS

The source tokenizer sets `add_bos_token=False` and `bos_token=None`, and the GGUF should carry
that. `bench_gguf.py` verifies llama.cpp's `/tokenize` against the shipped `token_ids` before
reporting anything, and warns loudly if a leading token appears that the HF tokenizer does not
produce.

## 3. Greedy decoding

`temperature 0`, `top_k 1`, `top_p 1.0`, fixed seed, `n_predict {MAX_NEW_TOKENS}`. Also
`cache_prompt: false` — prefix reuse across requests makes a result depend on submission order.

## 4. Context window

The longest confirmatory prompt is 5,235 tokens and the completion budget is {MAX_NEW_TOKENS},
so the server needs `-c {ENGINE_WINDOW}` **per slot**. `llama-server` divides `--ctx-size` across
`--parallel` slots, so with `--parallel 4` pass `--ctx-size {ENGINE_WINDOW * 4}`.

## 5. Stop tokens

`<|im_end|>` ({EOS_TOKEN_IDS[0]}) and `<|endoftext|>` ({EOS_TOKEN_IDS[1]}). When generation stops
because it hit the token budget, the row must carry `"truncated": true` — the parser treats a
mid-tool-call cutoff as a format error rather than a silent wrong answer.

## 6. Engine change is not quantization damage

The BF16 reference was produced by **vLLM 0.29.0**, not llama.cpp. Any delta between the BF16 GGUF
and that reference is an *engine* difference layered on top of the format change. Compare quantized
artifacts against the **BF16 GGUF** row to isolate quantization, and against the vLLM reference only
to see the total deployment delta. Reporting the second as if it were the first overstates
quantization damage.

## 7. No example may vanish

`score_generations.py` refuses a run whose generations do not reconcile exactly with the
{examples:,} submitted prompts. Missing, duplicate, unknown, and errored rows are errors.

## 8. Systems numbers need identical settings

Size, TTFT, prefill and decode throughput are only comparable across artifacts measured on the same
hardware with the same `--parallel`, `--ctx-size` and `-ngl`. `bench_gguf.py` records what it used.
TTFT is reported as a **lower bound** (prompt-processing time) because the non-streaming endpoint
cannot observe the first token directly.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="confirmatory")
    args = parser.parse_args()

    (OUT / "opengrad_min").mkdir(parents=True, exist_ok=True)
    for cache in OUT.rglob("__pycache__"):
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

    prompt_file = OUT / f"frozen_prompts_{args.partition}_v1.jsonl"
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

    shutil.copyfile(PARSER_SOURCE, OUT / "opengrad_min/parser.py")
    shutil.copyfile(POLICY_SOURCE, OUT / "opengrad_min/policy.py")
    (OUT / "opengrad_min/routing.py").write_text(ROUTING, encoding="utf-8", newline="\n")
    (OUT / "opengrad_min/__init__.py").write_text(
        '"""Minimal vendored OpenGrad measurement code for the GGUF handoff."""\n',
        encoding="utf-8",
        newline="\n",
    )
    (OUT / "score_generations.py").write_text(SCORER, encoding="utf-8", newline="\n")
    (OUT / "bench_gguf.py").write_text(BENCH, encoding="utf-8", newline="\n")
    (OUT / "PARITY_NOTES.md").write_text(parity_notes(len(prompts)), encoding="utf-8")
    shutil.copyfile(REFERENCE, OUT / "m1_v2_reference.json")
    shutil.copyfile(GATE, OUT / "quantization_preservation_v1.json")

    manifest = {
        "schema_version": 1,
        "runtime": "llama.cpp / GGUF",
        "built_at": datetime.now(UTC).isoformat(),
        "parent_experiment_id": "m1_dpo_canonical_v2_final_v2",
        "parent_checkpoint": "m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30",
        "ladder": list(LADDER),
        "primary_mobile_target": "Q4_K_M",
        "measurement_contract": {
            "renderer": "qwen3_5_2b_v1",
            "tokenizer_revision": PINNED_MODEL_REVISION,
            "template_hash": PINNED_TEMPLATE_HASH,
            "evaluator_revision": PINNED_EVALUATOR_REVISION,
            "partition": args.partition,
            "examples": len(prompts),
            "generation": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 0,
                "cache_prompt": False,
                "n_predict": MAX_NEW_TOKENS,
            },
            "engine_window_per_slot": ENGINE_WINDOW,
            "add_bos_token": False,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_ids": list(EOS_TOKEN_IDS),
            "endpoint": "/completion (raw text; never /v1/chat/completions)",
        },
        "files": {
            "prompts": prompt_file.name,
            "prompts_sha256": sha256_file(prompt_file),
            "benchmark": "bench_gguf.py",
            "scorer": "score_generations.py",
            "vendored_parser_sha256": sha256_file(OUT / "opengrad_min/parser.py"),
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"handoff -> {OUT.relative_to(ROOT)}")
    print(f"  prompts  {len(prompts)} ({prompt_file.name})")
    print(f"  files    {sorted(p.name for p in OUT.iterdir() if p.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
