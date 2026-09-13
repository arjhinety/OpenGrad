#!/usr/bin/env python3
"""Freeze the six tokenizer-divergent examples as permanent regression fixtures.

These fixtures characterise a **known, measured disagreement**. They are deliberately NOT a
requirement that the two tokenizers agree — they already do not, that failure is recorded, and
forcing agreement here would quietly delete the finding. What they protect is the *characterisation*:
if either tokenizer's behaviour drifts from what was measured, or if a divergence that was
behaviourally inert becomes material (or vice versa), the suite fails and someone has to look.

Each fixture records, for one example:

* the exact token ids from the pinned HF tokenizer
* the exact token ids from stock llama.cpp's `qwen35` pre-tokenizer
* the decoded prompt text and its sha256
* where the streams first diverge and where they reconverge
* the special-token census on both sides
* the policy decision each tokenization produced
* the expected class from the frozen evaluation set
* whether the divergence is known, and whether it changes behaviour

Usage:
    python scripts/build_tokenizer_regression_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "results/quantization/gguf/tokenizer_divergence_probe.json"
VERDICT = ROOT / "results/quantization/gguf/engine_parity_verdict.json"
OUT = ROOT / "tests/fixtures/tokenizer_divergence_v1.json"


def main() -> int:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    verdict = json.loads(VERDICT.read_text(encoding="utf-8"))
    by_id = {r["example_id"]: r for r in verdict["per_prompt"]}
    prov = probe["provenance"]

    fixtures = []
    for item in probe["examples"]:
        example_id = item["example_id"]
        row = by_id[example_id]
        text = item["prompt_text"]
        fixtures.append({
            "example_id": example_id,
            "expected_class": item["expected_decision"],
            "source": item["source"],
            "script": "Thai",
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "prompt_chars": len(text),
            "prompt_text": text,
            "hf": {
                "token_ids": item["hf_token_ids"],
                "token_count": item["hf_token_count"],
                "special_census": item["hf_special_census"],
                "decision": row["decision_llamacpp_hf_token_ids"],
            },
            "llamacpp": {
                "token_ids": item["llamacpp_token_ids"],
                "token_count": item["llamacpp_token_count"],
                "special_census": item["llamacpp_special_census"],
                "decision": row["decision_llamacpp_own_tokenization"],
            },
            "divergence": {
                "known": True,
                "cause": "llama.cpp QWEN35 folds \\p{M} combining marks into the letter run; "
                         "the pinned tokenizer's \\p{L}+ does not",
                "token_count_delta": item["token_count_delta"],
                "first_divergence_index": row["first_divergence"],
                "reconverged_suffix_tokens": row["reconverged_suffix"],
                "divergent_span_hf": row["divergent_span_hf"],
                "divergent_span_llamacpp": row["divergent_span_llamacpp"],
                "special_tokens_identical": row["special_token_census_matches"],
                "changes_behaviour": not row["decisions_agree_across_tokenizations"],
                "outputs_byte_identical": row["output_byte_identical_across_tokenizations"],
            },
        })

    fixtures.sort(key=lambda f: f["example_id"])
    payload = {
        "fixture_version": "tokenizer_divergence_v1",
        "purpose": (
            "Characterisation lock, not an agreement requirement. The two tokenizers are known to "
            "disagree on these six prompts; this file pins exactly how, so drift is detectable."
        ),
        "provenance": {
            "tokenizer_source_sha256": prov["source_file_sha256"],
            "special_tokens": prov["special_tokens"],
            "llama_cpp_tag": prov["llama_cpp_tag"],
            "llama_cpp_commit": prov["llama_cpp_commit"],
            "gguf_sha256": prov["gguf_sha256"],
            "transformers": prov["transformers"],
            "generation": prov["generation"],
            "direct_token_path_validation": prov["direct_token_path_validation"],
        },
        "summary": {
            "examples": len(fixtures),
            "behaviour_changing": sum(1 for f in fixtures if f["divergence"]["changes_behaviour"]),
            "behaviour_preserving": sum(
                1 for f in fixtures if not f["divergence"]["changes_behaviour"]
            ),
            "materiality_verdict": verdict["tokenizer_materiality"],
            "parity_verdict": verdict["engine_tokenizer_parity"],
            "of_confirmatory_examples": verdict["checked"],
        },
        "fixtures": fixtures,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(fixtures)} fixtures, "
          f"{payload['summary']['behaviour_changing']} behaviour-changing)")
    for f in fixtures:
        d = f["divergence"]
        print(f"  {f['example_id'][:8]}  {f['hf']['token_count']:>4} -> "
              f"{f['llamacpp']['token_count']:>4} ({d['token_count_delta']:+d})  "
              f"HF={f['hf']['decision']:<12} LC={f['llamacpp']['decision']:<12} "
              f"{'CHANGES BEHAVIOUR' if d['changes_behaviour'] else 'inert'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
