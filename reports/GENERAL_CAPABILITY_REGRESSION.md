# General-capability regression — diagnosis across the OpenGrad checkpoint ladder

**Question.** The OpenWeights sentinel showed the promoted checkpoint declining two tasks it should
be able to do — `100 − 7×12` and a three-element JSON array — while both tool cases passed. The
frozen tool-policy partition contains **no ANSWER examples**, so nothing in the primary evaluation
could have detected it. Is that lost capability, learned refusal, or noise from seven cases?

**Answer label: `GENERAL_DEGRADATION`**

> accuracy_given_answer falls by 21.3pp on at least one measured edge -- the ability itself is worse, not merely unused

The 21.3pp is MMLU-Pro on `BASE->M0_SFT`, and it excludes 2,136 truncated Base items from Base's
denominator. On the 9,637 items both stages attempted the drop is 17.7pp (60.0% → 42.4%). No
interval is given for either figure; the label holds on both.

This is an ordering statement about measured stages, not a causal one. A label says the pattern first becomes observable after a stage; it does not establish that the stage caused it, because no stage was re-run with a controlled intervention.

## Evidence labels used in this report

| label | meaning |
|---|---|
| CONFIRMED | reproduced from per-example artifacts on a set large enough to support the claim |
| MEASURED | a real number from a real run, stated without a causal or generalising claim |
| SUGGESTIVE | too small or too indirect to conclude from — the 7-case sentinel lives here |
| UNKNOWN | not measured |
| BLOCKED | could not be measured; reason stated |

The 5/7 sentinel result was, and remains, **SUGGESTIVE**. It is not evidence of general-capability
regression on its own and is never promoted by this report.

## The checkpoint ladder

| stage | repo | revision | location | status |
|---|---|---|---|---|
| BASE | `Qwen/Qwen3.5-2B` | `15852e8c1636` | repo root | available |
| M0_SFT | `arrochi112/OpenGrad-Qwen3.5-2B-M0-SFT-CanonicalV2-Final` | `acffaf6eb068` | repo root | available |
| M1_DPO_CURRENT | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` | `f33d20308982` | `dpo-checkpoint-30` | available |
| M1_DPO_HISTORICAL | `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` | `3a1250c7b8e8` | `checkpoint-300` | available |

Missing checkpoints: None — every stage on the primary path resolved.

The decisive path is **BASE → M0_SFT → M1_DPO_CURRENT**. Every stage on it shares one chat template
and one tokenizer *file lineage*. `M1_DPO_HISTORICAL` (M1-v1) is DPO applied directly to the base
model — `parent_experiment_id: null`, `reference: initial_policy`, preference data
`when2call_pref_v1` — with no SFT stage. It is supplementary because it is a different lineage, but
it matters: it refuses 70.7% of zero-shot GSM8K, so the regression is not specific to SFT.

## Transition matrix

| stage | tool call_f1 | IFEval strict | GSM8K 0-shot | GSM8K acc\|answer | GSM8K 8-shot | MMLU-Pro | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (Qwen3.5-2B, no SFT, no DPO) | — | 67.8 | 67.4 | 67.4 | 70.4 | 49.0 | 100.0 | 0.0 | 5/7 |
| M0 — SFT | — | 45.1 | 0.0 | — | 56.3 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v2 — DPO (promoted) | 0.751 | 45.8 | 0.0 | — | 55.5 | 37.0 | 0.0 | 100.0 | 5/7 |
| M1-v1 — DPO (other lineage) | — | 43.4 | 17.3 | 59.8 | 70.4 | — | 28.9 | 70.7 | 3/7 |

All figures are percentages except `call_f1`. `answer rate` and `refusal rate` are GSM8K 0-shot.

## Stage-to-stage deltas (percentage points)

| edge | IFEval strict | GSM8K acc | GSM8K acc\|answer | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|
| `BASE->M0_SFT` | -22.7 | -67.4 | — | -100.0 | +100.0 | +0 |
| `M0_SFT->M1_DPO_CURRENT` | +0.7 | +0.0 | — | +0.0 | +0.0 | +0 |

## Where each signal first becomes observable

| signal | first observable on |
|---|---|
| answer rate falls | `BASE->M0_SFT` (-100.0pp) |
| conditional instruction accuracy falls | `BASE->M0_SFT` (-15.6pp) |
| conditional math accuracy falls | not measurable at 0-shot (M0 attempted none); 8-shot: `BASE->M0_SFT` (-14.1pp) |
| fewshot math accuracy falls | `BASE->M0_SFT` (-14.1pp) |
| instruction following falls | `BASE->M0_SFT` (-22.7pp) |
| math accuracy falls | `BASE->M0_SFT` (-67.4pp) |
| refusal rises | `BASE->M0_SFT` (+100.0pp) |

These are **ordering statements**. A signal appearing after a stage does not establish that the
stage caused it: no stage was re-run with a controlled intervention, so chronology is all that is
available.

## Answer rate versus conditional accuracy

This is the measurement that separates the two hypotheses, and it is reported un-collapsed:

- **`accuracy`** falls if the model gets fewer right, for any reason.
- **`accuracy_given_answer`** falls only if the model is worse *when it actually attempts*.

A large drop in `answer_rate` with `accuracy_given_answer` broadly held means the ability is intact
and unused — over-alignment. A drop in `accuracy_given_answer` means the ability itself degraded.
The threshold for "material" is **5 percentage points**, declared
before the numbers existed.

Worst observed on any measured edge: answer rate **-100.0pp**,
conditional accuracy **-21.3pp** (MMLU-Pro; -17.7pp on items both stages attempted).

### The pre-registered 8-shot control

GSM8K runs two arms. `zeroshot` measures deployed behaviour. `fewshot8` supplies eight worked
exemplars, which demonstrate the answer format without changing the arithmetic. It was registered
in the adapter **before any generation**, precisely so that "the model can do it but declines to"
is falsifiable rather than asserted: a checkpoint that recovers under exemplars has the capability.

## IFEval failure composition

Which failures are refusals, and which are genuine instruction-following errors:

| failure class | Base (Qwen3.5-2B, no SFT, no DPO) | M0 | M1-v2 | M1-v1 |
|---|---:|---:|---:|---:|
| EXTRA_FORBIDDEN_CONTENT | 34 | 43 | 45 | 33 |
| FORMAT_VIOLATION | 47 | 56 | 51 | 52 |
| LENGTH_VIOLATION | 28 | 42 | 44 | 29 |
| MISSING_REQUIRED_ELEMENT | 62 | 64 | 60 | 70 |
| REFUSAL | 1 | 88 | 89 | 116 |
| WRONG_CONTENT | 2 | 4 | 4 | 6 |

`REFUSAL` is a **HEURISTIC** regex classification and is labelled as such everywhere. Every other
class is **DETERMINISTIC**, derived from which upstream instruction checker failed. The official
IFEval strict/loose numbers are computed by the vendored upstream checkers and are **not** adjusted
by this classification — a refusal counts as a failure there, exactly as upstream scores it.

### MMLU-Pro generation budget — a confound found, corrected, and residually present

The first MMLU-Pro pass used a 768-token generation budget. That was too small for 5-shot CoT over
ten options, and it did not bite evenly:

| stage | truncated at 768 | truncated at 2048 |
|---|---:|---:|
| Base | **4,527 / 12,032 (37.6%)** | 2,627 (21.8%) |
| M0 — SFT | 869 (7.2%) | 759 (6.3%) |
| M1-v2 — DPO | 903 (7.5%) | 747 (6.2%) |
| M1-v1 (supplementary) | 5,978 (49.7%) | not measured — run terminated |

At 768, **99.6% of Base's unattempted examples (4,292 of 4,309) were truncations**, not refusals
or malformed answers. The benchmark was measuring verbosity, not accuracy, and the Base-vs-post-trained
comparison was invalid. The whole pass was discarded and re-run at 2048; its cost is counted in
full in the ledger.

**A residual remains, and it is quantified rather than asserted.** Base still stops at the cap
~3.5× more often than the post-trained stages. An earlier draft called the observed drop a
"conservative lower bound"; that was wrong, and it is corrected here. A truncated generation could
in principle have resolved either way, so the observed gap is a **point estimate inside an
interval**, not a floor.

Defining an example as *unresolved* when it is incorrect **and** hit the token cap — the cases a
larger budget could plausibly have turned correct — each stage's true accuracy lies in
`[correct/n, (correct + unresolved)/n]`:

| | Base | M0 — SFT |
|---|---|---|
| accuracy interval | **[49.0%, 70.1%]** | **[37.0%, 42.6%]** |
| unresolved (incorrect ∧ truncated) | 2,537 | 676 |

| Base − M0 gap | value |
|---|---:|
| worst case (Base floor − M0 ceiling) | **+6.4pp** |
| observed | **+12.0pp** |
| best case (Base ceiling − M0 floor) | +33.1pp |

**The gap stays positive under the maximally adversarial resolution of every truncated example**,
so the direction of the finding does not depend on the generation budget. This is a deterministic
range, not a confidence interval, and it assumes no distribution. The same derivation applied to
GSM8K 8-shot gives worst **+13.9pp** / observed **+14.1pp** / best +16.3pp.

## MMLU-Pro per category

A localised regression is scientifically different from a global one, so the aggregate is never
reported alone:

| category | Base (Qwen3.5-2B, no SFT, no DPO) | M0 | M1-v2 |
|---|---:|---:|---:|
| biology | 69.6 | 64.7 | 64.0 |
| business | 53.2 | 41.2 | 40.8 |
| chemistry | 56.3 | 36.7 | 36.8 |
| computer science | 48.8 | 37.1 | 36.1 |
| economics | 59.1 | 45.5 | 43.7 |
| engineering | 23.6 | 24.0 | 25.3 |
| health | 49.4 | 40.2 | 39.7 |
| history | 33.9 | 26.8 | 27.3 |
| law | 24.8 | 15.7 | 16.2 |
| math | 62.7 | 40.4 | 40.9 |
| other | 42.0 | 34.4 | 34.1 |
| philosophy | 38.5 | 31.5 | 32.9 |
| physics | 55.4 | 36.4 | 37.0 |
| psychology | 58.0 | 47.9 | 47.4 |

## Benchmarks, datasets and scorers

| benchmark | source | revision | split | n | scorer |
|---|---|---|---|---:|---|
| IFEval | `google/IFEval` | `966cd89545d6` | train | 541 | google-research checkers, vendored unmodified |
| GSM8K | `openai/gsm8k` (main) | `740312add88f` | test | 1319 ×2 arms | deterministic numeric extraction |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` | test | 12032 | upstream answer regexes |

All three are real upstream datasets at pinned revisions. No synthetic records, no placeholder
tasks, no approximated subsets presented as official scores. The request sets are frozen and
hashed; the validation gate in `tests/evaluation/test_capability_benchmarks.py` must pass before
any result is labelled COMPLETE.

**No LLM judge is used anywhere.** Every metric comes from a deterministic checker or a regex.

## Inference conditions

Identical across every stage:

| | |
|---|---|
| engine | vLLM 0.29.0 |
| dtype | bfloat16 |
| decoding | greedy — `temperature 0.0`, `top_p 1.0`, `top_k 1`, seed 0 |
| prefix caching | disabled — would make MMLU-Pro depend on submission order |
| max model len | 5760 |
| GPU | NVIDIA H200 |
| torch / CUDA | 2.13.0+cu130 / 13.0 |

Each checkpoint is served with **its own** tokenizer and chat template, because those are part of
the artifact. Substituting one checkpoint's tokenizer into another would evaluate a model that does
not exist.

## Caveats that constrain these conclusions

**Tokenizer.** **TOKENIZER_DIVERGENCE_INERT_FOR_CAPABILITY_SUITE** — 0 of 15211 benchmark prompts tokenize differently between the base and the post-trained tokenizers.

The base repo and the post-trained checkpoints do not ship byte-identical tokenizers: the vocab and
merges are identical, but the base carries the Qwen3.5 pre-tokenizer regex aware of Unicode
combining marks (`\p{M}`) while the post-trained stages carry the older Qwen2 form, and the
post-trained stages register 7 extra audio/TTS tokens. This is a property of the published
artifacts, recorded in the ladder, and is not patched.

**Architecture.** `Qwen/Qwen3.5-2B` publishes `Qwen3_5ForConditionalGeneration` with a vision tower;
the post-trained stages publish text-only `Qwen3_5ForCausalLM`. The BASE rung is evaluated as
published, text-only, with no images. The language stack was **not** extracted and re-saved as a
causal LM — a reconstructed artifact is not the original. This is a caveat on BASE deltas only; the
M0→M1 edge has the same architecture on both sides.

**Engine.** Engine effects are held entirely separate. The recorded vLLM↔llama.cpp per-example
agreement of **0.9836 (21 flips)** is a different example set and a different metric; it is not
folded into any stage delta here. Every stage above ran on the same engine and version, so engine
choice cannot explain a stage difference. The conservative form of that earlier finding stands:
**runtime choice produces measurable non-zero per-example behavioural differences** — runtime, not
engine alone, because that comparison also changed hardware (H200 vs A100).

**Statistical power.** IFEval is 541 prompts and GSM8K 1319; a one-example change is 0.18pp and
0.08pp respectively. The sentinel is 7 cases, where one case is 14pp — which is why it cannot carry
a conclusion.

**Not an intervention.** This run is diagnostic. No checkpoint was trained, tuned, merged or
re-prompted against these results, and no benchmark failure was fed back into any dataset.

## Cost

| category | USD |
|---|---:|
| retained runs (includes $0.757 for a terminated run that reported no results) | $5.01 |
| superseded runs (counted in full) | $3.89 |
| INFRASTRUCTURE_WASTE | $0.00 |
| **this continuation total** | **$8.90** |
| prior run (preserved) | $2.49 |
| **campaign total** | **$11.39** |

Spend that produced no reported result (superseded runs plus the terminated M1-v1 MMLU-Pro run) is
$4.64, 52% of the continuation.

`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE` — the Modal API exposes consumption, not a remaining
balance. Only the configured envelope and observed spend are known.

## Reproduction

```bash
python scripts/preserve_h200_state.py --verify        # completed run is untouched
python scripts/build_checkpoint_ladder.py --verify    # ladder identities still resolve
python scripts/prepare_capability_benchmarks.py       # rebuild request sets (byte-identical)
python -m pytest tests/evaluation/test_capability_benchmarks.py
modal run scripts/modal/h200_capability.py --stage <STAGE> --jobs ifeval,gsm8k,sentinel
python scripts/fetch_and_score_capability.py --stage <STAGE>
python scripts/build_regression_analysis.py
python scripts/build_capability_report.py
```

Per-example evidence for every row above lives in
`results/benchmarks/h200/capability_v1/<STAGE>/*_per_example.jsonl`, including raw model output, so
every aggregate here is recomputable without re-running the GPU — in a working tree that has them.
Those files are bulk and not committed; see `FINAL_CAMPAIGN_AUDIT.md` §16.
