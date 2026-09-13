---
pretty_name: OpenGrad ToolPolicy Canonical v2 — minus xLAM
language:
- en
license: other
task_categories:
- text-generation
configs:
- config_name: canonical
  data_files:
  - split: train
    path: "*.parquet"
---

> This is the byte-identical training view for a **joint xLAM-plus-`CALL_PREDICTION` removal**
> experiment. xLAM is currently the corpus's only source of that supervision contract, so this is
> not a pure source-content ablation. It carries no result of its own and is not a recommended
> mixture. It is part of [OpenGrad Study 001](https://opengrad.arjhinety.com/studies/001).

## What this is

[`OpenGrad-ToolPolicy-Canonical-v2`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2)
with **one source removed**: xLAM/APIGen. Three sources remain, 115,895 canonical records, 118
shards.

It is the exact input of OpenGrad's paired fixed-compute and matched-exposure runs, published
because an experiment is only checkable if its input is available. Both runs have been executed and
published as negative results:
[`M0-ABL-MinusXLAM-FixedCompute`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-FixedCompute)
and
[`M0-ABL-MinusXLAM-MatchedExposure`](https://huggingface.co/arrochi112/OpenGrad-Qwen3.5-2B-M0-ABL-MinusXLAM-MatchedExposure).
The matched-exposure arm saw 1.19× the reference's supervised tokens, so its exposure was not in
fact matched (see the OpenGrad `reports/ERRATA.md`). Removing xLAM also removes all
`CALL_PREDICTION` supervision, so the runs estimate that joint intervention rather than an
xLAM-content-only effect.

## It is a selection, not a rebuild

The parent corpus writes its shards per source, so removing a source is the removal of that
source's shards and nothing else. Every retained shard here is a **byte-for-byte copy** of the
parent's: no record was parsed, rewritten, re-encoded, or re-sharded.

That is verifiable rather than asserted. Every shard's SHA-256 is recorded in
`release-manifest.json` under `output_shards`, and each one equals the parent's hash for the same
file. The manifest also records, under `derived_view`, the parent's fingerprint, its Hub revision,
and the names and hashes of the 58 shards that were excluded — so the exclusion can be checked
against the parent dataset as well as against this one.

```text
parent fingerprint : 8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161
current fingerprint: 5fc739040b1a09fa7b8ac20494609bd4cbe89efc6ed54e4dbc9c2df12268b7f3
prior fingerprint  : f8ba687e16d8ab740b78a503adb41f4be28bd775f925291d6da7ec31be1ac5ac
                     superseded only because its interpretation metadata called this a source
                     ablation without stating the inseparable supervision-channel removal;
                     retained parquet bytes and counts are unchanged
```

Consequences worth stating: supervision contracts, renderer behaviour, loss masks, contamination
decisions, source provenance, and trainability decisions are **identical to the parent** for every
retained record. Mechanically, the row-selection difference is that xLAM is absent. Experimentally,
that same removal eliminates the entire `CALL_PREDICTION` channel because no other source currently
provides it.

## What was removed, and what it costs

| Source | Canonical records | Trainable | Shards |
|---|---:|---:|---:|
| glaive-function-calling-v2 | 98,339 | 97,112 | 99 |
| toolace | 11,051 | 2,259 | 12 |
| when2call | 6,505 | 6,505 | 7 |
| **retained total** | **115,895** | **105,876** | **118** |
| *xLAM/APIGen (removed)* | *57,342* | *56,090* | *58* |

xLAM is **34.6% of the parent's trainable records but only 11.7% of its loss-bearing tokens.**
Those two numbers differ because xLAM records are short: under its `CALL_PREDICTION` contract the
supervised target is the tool call itself and no tool result or final response follows it, so a
record ends where a `COMPLETE_TRAJECTORY` record would continue. Measured across the parent's
rendered samples:

| | full corpus | minus xLAM | ratio |
|---|---:|---:|---:|
| trainable records | 161,966 | 105,876 | 0.654 |
| rendered tokens | 99,717,548 | 64,146,939 | 0.643 |
| supervised (loss-bearing) tokens | 33,565,721 | 29,630,369 | 0.883 |

That asymmetry is the reason the ablation needed two arms and two different step counts rather
than one. It is also the most transferable thing this view demonstrates: in a heterogeneous corpus,
a source's share of records is a poor proxy for its share of training signal, and an ablation that
matched on record counts would have confounded two variables.

## Refusal-shaped targets

The parent corpus carries 18,114 records (10.5% of 173,237) whose target is a refusal but whose
decision label is ANSWER, found by a heuristic detector whose precision has not yet been measured.
None of them come from xLAM (Glaive 14,066, When2Call 4,038, ToolACE 10), so **all 18,114 are in
this view: 15.6% of its 115,895 records**, a larger share than in the parent. OpenGrad Study 001
found a general-capability regression associated with tool-policy post-training on
When2Call-derived data, including refusal of every zero-shot GSM8K question; this supervision is a
candidate explanation, not a demonstrated cause. Source:
`results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json` in the
OpenGrad repository.

## Supervision contracts

xLAM was the only source carrying `CALL_PREDICTION`, so this view is entirely `COMPLETE_TRAJECTORY`
— 105,876 trainable records. The configs explicitly declare that complete post-filter contract set;
readiness rejects any retained empty or absent selection. Although the implementation filters by
source, the experimental treatment necessarily removes both xLAM and the supervision channel.

| Contract | Trainable here | In the parent |
|---|---:|---:|
| `COMPLETE_TRAJECTORY` | 105,876 | 105,876 |
| `CALL_PREDICTION` | 0 | 56,090 |

## Intended use

As the input of the paired joint xLAM-plus-`CALL_PREDICTION` removal experiment, and as a comparison
artifact for anyone examining the parent corpus. Training on it is not a recommendation: the three
remaining sources are the parent's, unbalanced and unmodified, and the parent was never a
recommended mixture either. No xLAM-specific causal attribution is valid unless another evidence-
backed `CALL_PREDICTION` source separates source identity from supervision type.

Not suitable as a benchmark. A subset of a training corpus is still training data.

## Attribution

The parent's `CITATIONS.bib` and `source-licenses.md` are inherited verbatim, so they still list
xLAM alongside the three retained sources. That is deliberate: attribution files should
over-include rather than risk dropping a licence obligation. xLAM's entry documents the source
this view removes, and the parent dataset is where those terms continue to apply.
