# Final campaign audit — tool-policy post-training and the general-capability regression

**Audit posture: adversarial.** Every figure below was recomputed from per-example artifacts by
[`scripts/audit_campaign_final.py`](../scripts/audit_campaign_final.py), not copied from prose. Where
a recomputation contradicted an existing report, the artifact won and the contradiction is recorded
in [§9](#9-claims-corrected-during-this-audit). No GPU work was performed for this audit.

Machine-readable counterpart: [`results/final_campaign_verdict.json`](../results/final_campaign_verdict.json).
Raw recomputation: `results/benchmarks/h200/capability_v1/final_campaign_audit.json`.

---

## 1. Executive conclusion

Tool-policy post-training on When2Call-derived data is associated with **two separable
regressions**, and the M0 → M1-v2 preference stage changed neither. The regression appears after
SFT (M0) and also after DPO applied directly to the base (M1-v1, 70.7% GSM8K zero-shot refusal), so
it is not specific to SFT. Causation is not established: one lineage, one seed, no replicate.

1. **A prompt-regime-conditioned refusal policy.** On bare zero-shot GSM8K the model went from
   answering every question to declining every question. The same model answers the *same*
   questions when eight worked exemplars are present, and never refuses 5-shot MMLU-Pro.
2. **Genuine capability degradation**, measurable in regimes where refusal is absent and therefore
   not explainable by refusal.

The promoted checkpoint improved the metric it was optimized for relative to B0 (its gain over M0,
+0.0078 call_f1 or 7 of 453 calls on one seed, is within noise). The regression was invisible to
the promotion gate because the frozen confirmatory partition contains **no ANSWER examples**. The
successful artifact of this campaign is the evaluation methodology and the resulting diagnosis —
not the checkpoint.

---

## 2. Primary lineage — verified, not assumed

| stage | repo | revision | weights sha256 | arch |
|---|---|---|---|---|
| Base | `Qwen/Qwen3.5-2B` | `15852e8c1636` | `aa33250c4fc6` | `Qwen3_5ForConditionalGeneration` |
| M0 — SFT | `arrochi112/…-M0-SFT-CanonicalV2-Final` | `acffaf6eb068` | `7144579aeece` | `Qwen3_5ForCausalLM` |
| M1-v2 — DPO | `arrochi112/…-M1-DPO-CanonicalV2-Final-v2` | `f33d20308982` | `903f9b115d41` | `Qwen3_5ForCausalLM` |

**M0 is the exact parent of M1-v2 — independently re-verified.** The frozen reference records
`parent_weight_sha256 = 7144579aeece…`. That value equals both (a) the Hub LFS digest of the
published M0 shard and (b) a sha256 recomputed over the downloaded file inside the H200 container.
Two independent paths, same digest. The `M0 → M1-v2` edge therefore isolates the preference stage.

All four stages carry **distinct weight digests**; no checkpoint mix-up. All stages ran on an
**identical engine configuration** (vLLM 0.29.0, bfloat16, `max_model_len` 5760, greedy
`temperature 0.0 / top_p 1.0 / top_k 1`, seed 0, prefix caching off), so no stage delta is
attributable to the runtime.

**M1-v1 is NOT on the primary path.** It is DPO applied directly to the base model:
`runs/qwen35_2b_m1_dpo_v1/experiment.json` records `parent_experiment_id: null`,
`reference: initial_policy` and preference data `when2call_pref_v1`. It has no SFT parent (an
earlier ladder record placed it on a CorpusV2 SFT parent; that was wrong). No persisted metadata
places it on `Base → M0 → M1-v2`. It is reported as an alternate lineage only — see
[§7](#7-alternate-lineage--m1-v1).

**Two architecture/tokenizer caveats, both measured:**

- Base publishes a multimodal `Qwen3_5ForConditionalGeneration`; the post-trained stages are
  text-only. Base was evaluated **as published**, text-only, with no images. Its language stack was
  not extracted and re-saved — a reconstructed artifact is not the original. This caveats Base
  deltas only; the `M0 → M1-v2` edge is identical on both sides.
- Base and the post-trained stages ship different `tokenizer.json` bytes (identical vocab and
  merges; the base carries the Qwen3.5 pre-tokenizer regex aware of `\p{M}` combining marks, the
  post-trained stages the older Qwen2 form). Measured impact: **0 of 15,211 benchmark prompts
  tokenize differently** (`tokenizer_divergence_census.json`). The difference is behaviourally
  inert for this suite; it is not claimed to be inert in general.

---

## 3. Canonical benchmark results

All recomputed from per-example rows. Bucket reconciliation passes for every GSM8K cell
(correct + wrong + refused + unparseable = n).

### GSM8K — 1,319 questions, both arms over the *same* questions

| stage | arm | accuracy | answer rate | refusal | acc given answer |
|---|---|---:|---:|---:|---:|
| Base | zero-shot | **67.4%** | 100.0% | 0.0% | 67.4% |
| Base | 8-shot | **70.4%** | 100.0% | 0.0% | 70.4% |
| M0 — SFT | zero-shot | **0.0%** | 0.0% | **100.0%** | n/a |
| M0 — SFT | 8-shot | **56.3%** | 100.0% | 0.0% | 56.3% |
| M1-v2 | zero-shot | **0.0%** | 0.0% | **100.0%** | n/a |
| M1-v2 | 8-shot | **55.5%** | 100.0% | 0.0% | 55.5% |

`acc given answer` is `n/a`, not `0.0`, where nothing was attempted — a model that never tried has
no conditional accuracy, and `0.0` would read as "tried everything and failed".

### IFEval — 541 prompts, upstream checkers, **seeded**

| stage | prompt strict | prompt loose | answer rate | refusal | acc given answer |
|---|---:|---:|---:|---:|---:|
| Base | **67.8%** | 72.6% | 99.6% | 0.4% | 67.9% |
| M0 — SFT | **45.1%** | 48.2% | 81.0% | 19.0% | 52.3% |
| M1-v2 | **45.8%** | 48.8% | 80.2% | 19.8% | 53.0% |
| M1-v1 | 43.4% | 49.9% | 70.6% | 29.4% | 50.3% |

### MMLU-Pro — 12,032 items, 5-shot CoT, **2048-token budget (canonical)**

| stage | accuracy | answer rate | refusal | invalid option | acc given answer |
|---|---:|---:|---:|---:|---:|
| Base | **49.0%** | 82.1% | 0.0% | 0.2% | 59.7% |
| M0 — SFT | **37.0%** | 96.3% | 0.0% | 0.3% | 38.4% |
| M1-v2 | **37.0%** | 96.3% | 0.0% | 0.4% | 38.5% |

`accuracy_given_answer = correct / attempted`, where *attempted* means the response yielded a
letter **within that item's own option range** (MMLU-Pro items have 3–10 options, so an out-of-range
letter is an invalid option, not a wrong answer). Truncated generations that produced no in-range
letter are therefore excluded from the denominator, which is why answer rate and truncation rate
move together for Base.

### OpenWeights sentinel — 7 cases, **SENTINEL / REGRESSION SMOKE TEST**

| stage | pass | tool | general | refusal failures | wrong-content failures |
|---|---:|---|---|---:|---:|
| Base | 5/7 | 2/2 | 3/5 | **0** | **2** |
| M0 — SFT | 5/7 | 2/2 | 3/5 | **2** | 0 |
| M1-v2 | 5/7 | 2/2 | 3/5 | **2** | 0 |
| M1-v1 | 3/7 | 1/2 | 2/5 | 1 | 3 |

**Base and the promoted checkpoint score identically and fail completely differently.** Base
attempts `trap-arithmetic` and `multi-step-change` and gets them *wrong* (answers "1" and "12");
M0/M1-v2 pass `trap-arithmetic` but *refuse* `multi-step-change` and `format-constraint`, which Base
passes (`sentinel_scores.json`). This is the clearest demonstration in the campaign that an aggregate score
can hide a total change in behaviour — and the reason 7 cases are never used as evidence for or
against general capability.

---

## 4. Superseded results — preserved, with one gap

### MMLU-Pro @768 — `SUPERSEDED_PROTOCOL_INVALID`

Retained at `results/benchmarks/h200/capability_v1/<STAGE>/superseded_mmlu_768/` for Base, M0 and
M1-v2. M1-v1's `superseded_mmlu_768/` directory is empty: its 49.7% truncation figure survives only
in the cost ledger (`gpu_runs.jsonl`, `cost_ledger.json`).

| stage | accuracy @768 | truncation @768 | accuracy @2048 | truncation @2048 |
|---|---:|---:|---:|---:|
| Base | 38.0% | **37.6%** | **49.0%** | 21.8% |
| M0 — SFT | 36.8% | 7.2% | **37.0%** | 6.3% |
| M1-v2 | 36.9% | 7.5% | **37.0%** | 6.2% |
| M1-v1 | — | 49.7% | **UNMEASURED** | — |

**This is the most important methodological finding of the campaign.** At 768 tokens the three
stages looked flat — 38.0 / 36.8 / 36.9 — which would have been reported as *"MMLU-Pro shows no
degradation."* That conclusion was an artifact: the budget bit Base five times harder than the
post-trained stages, and **99.6% of Base's unattempted examples at 768 (4,292 of 4,309) were
truncations**, not refusals or malformed answers. The benchmark was measuring verbosity, not accuracy.

Corrected at 2048, the same comparison shows a **12.0pp** Base advantage. **The corrected protocol
strengthened the conclusion that the flawed protocol had hidden.** A uniform-looking truncation
rate would have been far more dangerous, because it would have been believed.

### IFEval @1280 — `SUPERSEDED_PROTOCOL_INVALID`

Retained at `M1_DPO_CURRENT/ifeval_scores_budget1280_superseded*`. 65/541 responses hit the cap.

The 2560-token re-run did **not** remove IFEval truncation: 47 (Base), 67 (M0), 63 (M1-v2) and 17
(M1-v1) of 541 responses still hit the cap (`finish_reason: length` in the generations). This applies
to every IFEval row in §3, which do not restate it; no interval has been computed for them.

---

## 5. Refusal characterization — prompt-regime conditioned

| stage | GSM8K 0-shot | GSM8K 8-shot | MMLU-Pro 5-shot | IFEval 0-shot |
|---|---:|---:|---:|---:|
| Base | 0.0% | 0.0% | 0.0% | 0.4% |
| M0 — SFT | **100.0%** | **0.0%** | **0.0%** | 19.0% |
| M1-v2 | **100.0%** | **0.0%** | **0.0%** | 19.8% |
| M1-v1 | 70.7% | 2.0% | UNMEASURED | 29.4% |

**The decisive contrast holds content constant.** Both GSM8K arms cover the identical 1,319
questions; only the presence of eight worked exemplars differs. Refusal moves 100% → 0%. The
zero-shot arm had **0 truncated generations** for M0 and M1-v2, so this measurement carries no
generation-budget confound whatsoever.

The supported statement is:

> Tool-policy post-training on When2Call-derived data is associated with a learned conditional
> refusal behaviour that is strongly activated by the bare zero-shot request regime and is
> suppressed by in-context demonstrations. It appears after SFT (M0) and also after DPO applied
> directly to the base (M1-v1), so it is not specific to SFT. Causation is not established: one
> lineage, one seed, no replicate.

Refusal surface forms are **stereotyped**: M1-v2 emitted 1,426 refusals across 185 distinct
12-token openings, with the top 10 covering 82.3%. Base emitted 2 refusals total. Stereotypy is
consistent with a learned behaviour, but it is **not** evidence about the internal mechanism — see
[§10](#10-evidence-hierarchy).

A CPU audit of the supervision found refusal-shaped supervised targets labelled `ANSWER`. The
first audit (`sft_refusal_supervision_audit.json`, 21,749 of 217,903 = 10.0%; `when2call-sft`
7,490 / 14,829) measured the normalization-v1 sources, not M0's corpus. On the published
Canonical-v2 final corpus that M0 trained on (`sft_refusal_supervision_audit_canonical_v2.json`):
**18,114 of 173,237 single-exchange records (10.5%)** have a refusal target labelled `ANSWER` —
When2Call **4,038 of 6,505 (62.1%)**, Glaive 14,066 of 98,339 (14.3%), ToolACE 10, xLAM 0 — and
**zero** are labelled `CANNOT_ANSWER`. This is an **association**, not a demonstrated cause. The ablation that would establish causation is
specified in `ROADMAP.md` step 16 and has not been run.

---

## 6. Capability loss — separable from refusal

Refusal explains the zero-shot answer-rate collapse. It does **not** explain everything.

| evidence | regime | refusal present? | Base | M0 | delta |
|---|---|---|---:|---:|---:|
| GSM8K 8-shot accuracy | exemplars | **0%** | 70.4% | 56.3% | **−14.1pp** |
| MMLU-Pro accuracy | 5-shot | **0%** | 49.0% | 37.0% | **−12.0pp** |
| MMLU-Pro acc given answer | 5-shot | **0%** | 59.7% | 38.4% | **−21.3pp** |
| IFEval acc given answer | 0-shot | 19% | 67.9% | 52.3% | −15.6pp |

The first three rows are measured in regimes with **essentially no refusal**, so refusal cannot
account for them. The conditional MMLU-Pro row excludes 2,136 truncated Base items from Base's
denominator; on the 9,637 items both stages attempted the drop is **−17.7pp** (60.0% → 42.4%). No
interval is computed for either conditional figure.

### Truncation-adversarial intervals — derived, not asserted

Truncation rates differ by stage, so each difference is bounded rather than stated as exact. An
example is **unresolved** if it is incorrect *and* hit the token cap — the cases a larger budget
could plausibly have turned correct. True accuracy then lies in
`[correct/n, (correct + unresolved)/n]`.

| comparison | worst case | observed | best case | survives adversarial resolution |
|---|---:|---:|---:|---|
| MMLU-Pro, Base − M0 | **+6.4pp** | **+12.0pp** | +33.1pp | **yes** |
| GSM8K 8-shot, Base − M0 | **+13.9pp** | **+14.1pp** | +16.3pp | **yes** |
| GSM8K 0-shot, Base − M0 | +67.4pp | +67.4pp | +77.4pp | yes |

These are **deterministic ranges under adversarial resolution, not confidence intervals**, and they
assume no distribution. The direction of the finding does not depend on the generation budget.

---

## 7. Alternate lineage — M1-v1

DPO applied directly to the base model (`parent_experiment_id: null`, `reference: initial_policy`,
preference data `when2call_pref_v1`); no SFT stage. **Not on the primary causal chain.** Informative
as a natural ablation: it shows the refusal behaviour also appears without any SFT stage, and is not
all-or-nothing across lineages.

- GSM8K zero-shot refusal **70.7%** (933 / 1,319), accuracy 17.3%, acc-given-answer 59.8%
- GSM8K 8-shot refusal **2.0%**, accuracy 70.4%
- IFEval refusal **29.4%**, strict 43.4%
- MMLU-Pro at the corrected budget: **UNMEASURED** — the run was terminated mid-flight on budget

**No interpolation is offered for the missing MMLU-Pro value.** It is unmeasured and stays that way.
M1-v1 also truncated heavily on GSM8K (137 zero-shot, 119 few-shot of 1,319), so its figures carry a
larger budget caveat than the primary path.

---

## 8. Preference stage — behaviourally flat

| measure | M0 | M1-v2 | delta |
|---|---:|---:|---:|
| GSM8K 0-shot refusal | 100.0% | 100.0% | 0.0pp |
| GSM8K 8-shot accuracy | 56.3% | 55.5% | −0.8pp |
| IFEval prompt strict | 45.1% | 45.8% | +0.7pp |
| MMLU-Pro accuracy | 37.0% | 37.0% | ~0.0pp |
| Sentinel | 5/7 | 5/7 | 0 |

Corroborated independently: **81.96% of GSM8K generations are byte-identical** between M0 and
M1-v2, against ~0% for every pair involving Base.

The supported statement is:

> Preference training produced no material behavioural recovery or degradation on the measured
> evaluation suite.

This is a statement about **these evaluations only**. It is *not* a claim that DPO made small weight
changes, that the optimizer was ineffective, or that preference optimization could not repair this
behaviour if explicitly targeted. None of those were measured.

---

## 9. Claims corrected during this audit

| # | prior claim | status | correction |
|---|---|---|---|
| 1 | "only 1 of 2,638 GSM8K generations truncated" | **FALSE** | True of M1-v2 alone. Actual: Base **178**, M0 2, M1-v2 1, M1-v1 **256**. A single stage's number was generalized to the benchmark. |
| 2 | "the −12pp MMLU-Pro drop is a conservative lower bound" | **INVALID** | No bound had been computed. Replaced with a derived interval: worst **+6.4pp**, observed **+12.0pp**, best +33.1pp. |
| 3 | IFEval scores | **NON-REPRODUCIBLE** | Scorer was non-deterministic — one fixed file scored 0.4510 / 0.4492 / 0.4492. Root cause: `langdetect.detect()` and `random.*` in `build_description`. Both seeded; re-verified identical across three runs. All stages re-scored. |
| 4 | MMLU-Pro interval definition | **INCONSISTENT** | Two scripts used different "unresolved" definitions, yielding +8.3pp and +6.4pp. Reconciled to `incorrect ∧ truncated`; both now agree at **+6.4pp**. |
| 5 | "engine change is nearly as disruptive as quantization" | **QUALIFIED** | The controlled comparison does exist (same 1,277 examples, same metric): engine 21 flips vs quantization 25. But the engine arm also changed hardware (H200 vs A100), so it is a *runtime-stack* comparison, not engine-only. |
| 6 | "$22.70 remaining / envelope" | **CORRECTED** | Never a balance. Recorded as a planning envelope with `envelope_is_not_a_balance: true`; actual remaining credit was far lower. |
| 7 | Base "5/7 same as promoted" | **RE-FRAMED** | Identical score, opposite failure mode (wrong answers vs refusals). Now used as evidence *against* aggregate-score reasoning. |

---

## 10. Evidence hierarchy

**CONFIRMED** — reproduced from per-example artifacts on sets large enough to support the claim:

- Base → M0 introduces a 0% → 100% bare-prompt GSM8K refusal rate (1,319 examples, 0 truncated)
- Exemplars suppress that refusal to 0% on the identical questions
- Base materially outperforms M0 on GSM8K 8-shot (+14.1pp; interval +13.9 … +16.3)
- Corrected-budget MMLU-Pro shows substantial Base → M0 degradation (+12.0pp; interval +6.4 … +33.1)
- M0 → M1-v2 is behaviourally near-flat on the measured suite
- MMLU-Pro @768 is invalid for checkpoint comparison
- M0 is the exact weight-level parent of M1-v2

**SUPPORTED BUT QUALIFIED:**

- The refusal is prompt-regime conditioned (strong across three benchmarks; mechanism not established)
- Few-shot context moves the model out of the refusal regime (behavioural, not mechanistic)
- The regression first appears at Base → M0 *within the verified primary lineage* (single lineage,
  no replicate); M1-v1, DPO directly on the base, also refuses 70.7% of zero-shot GSM8K, so it is
  not specific to SFT

**SUGGESTIVE:**

- The internal mechanism of the refusal behaviour
- That refusal-as-`ANSWER` supervision caused it (association only; ablation not run)
- "Catastrophic specialization" as a mechanistic explanation

**UNKNOWN:**

- Corrected-budget MMLU-Pro for M1-v1
- Whether one specific source dataset induced the pattern
- Whether DPO could repair it if explicitly targeted
- Whether the effect generalizes to other model families

---

## 11. Runtime parity

| metric | value |
|---|---|
| decision agreement (vLLM vs llama.cpp BF16) | **0.983555** |
| decision flips | **21** of 1,277 |
| flips among known tokenizer-divergent prompts | 3 |
| flips attributable to engine numerics | 18 |
| exact output agreement | 0.870 |

Supported statement: **runtime choice produces measurable non-zero per-example behavioural
differences.**

A controlled comparison against quantization does exist — llama.cpp BF16 → Q6_K gives **25 flips**
at 0.980423 agreement on the *same 1,277 examples* with the *same* metric. The magnitudes are
comparable (21 vs 25). **However**, the engine arm also changed hardware and kernels (H200 vs A100),
so it is a runtime-stack comparison rather than a pure engine intervention. Any stronger phrasing
than "comparable in magnitude, with hardware confounded" is unsupported.

**This is held entirely separate from the checkpoint study.** Every capability stage ran on one
engine and version.

---

## 12. Scoring bugs found and fixed

| bug | impact | fix |
|---|---|---|
| GSM8K positional `last_number` fallback extracted incidental digits from refusal text ("…over a period of **5** weeks") | 2 refusals counted as attempted wrong answers; understated refusal rate; polluted `accuracy_given_answer` | Refusal only counts as an attempt under an **explicit** answer marker (`####`, "the answer is", `\boxed{}`) |
| MMLU-Pro `trailing_letter` fallback could capture an incidental option letter from refusal text | same class | Same rule applied |
| IFEval scorer non-deterministic (±0.2pp) | scores not reproducible | `random` and `langdetect.DetectorFactory` seeded; vendored checkers untouched |
| JSONL written with `ensure_ascii=False` | MMLU-Pro contains U+0085, which `splitlines()` treats as a line break — files were not reliably line-delimited | All writers escape non-ASCII; regression test added |
| Corpus audit took the last assistant turn in multi-turn records | 9,257 records falsely appeared to be mislabelled `CALL` | Mislabelling claim restricted to single-exchange records; CALL count fell to **zero** |

All are covered by regression tests in `tests/evaluation/test_capability_benchmarks.py`.

---

## 13. Protocol amendments

Both were amendments to reduce a truncation confound, applied uniformly to every stage, not
undisclosed parameter tweaks. Only the IFEval change is recorded as decided before scoring. The
768-token MMLU-Pro pass was scored first (38.0 / 36.8 / 36.9) and the budget was raised after
observing its truncation, as the table says. Neither amendment removed truncation entirely (§4, §6).

| amendment | trigger | timing |
|---|---|---|
| IFEval 1280 → 2560 tokens | 65/541 responses hit the cap on the first stage | Decided from `finish_reason` counts **before any response was scored**; applied uniformly to all stages, so no stage is compared across budgets |
| MMLU-Pro 768 → 2048 tokens | Stage-dependent truncation: Base 37.6% vs M0 7.2% | Decided after observing truncation counts; entire pass discarded and re-run; superseded artifacts retained |

---

## 14. Cost audit

Canonical source: `results/benchmarks/h200/capability_v1/gpu_runs.jsonl` (append-only, one row per
GPU allocation). The previous approach — scanning per-stage `run_summary.json` — was **invalid for
campaign accounting**, because that file lives at a fixed path per stage and is overwritten by any
re-run, double-counting some runs and erasing others.

| category | USD |
|---|---:|
| retained runs (includes $0.757 for seq 13, which reported no results) | 5.01 |
| superseded runs (counted in full) | 3.89 |
| INFRASTRUCTURE_WASTE | **0.00** |
| continuation subtotal | **8.90** |
| prior H200 run (preserved, unchanged) | 2.49 |
| **campaign total** | **11.39** |

- **13 GPU runs**, **3 failed launches with `gpu_allocated: false`** — all died client-side or at
  module import, so $0.00, asserted from the log rather than assumed.
- **1 estimated row** (seq 13, M1-v1 MMLU-Pro): terminated mid-run, never reported
  `container_seconds`; flagged `usd_is_estimate: true` with its basis recorded.
- **$3.89 of $8.90 (44%) is superseded work**, almost all the 768-token MMLU-Pro pass. Adding the
  $0.757 terminated M1-v1 MMLU-Pro run, **$4.64 (52%) of the continuation produced no reported
  result**.

Campaign spend was **$11.39 against an approximately $22.70 internal planning envelope**. That
envelope was **never a queried account balance**, and treating it as one led to over-committing
against actual credit. `REMAINING_CREDIT_BALANCE = NOT_QUERYABLE` — the Modal API exposes
consumption, not a balance.

---

## 15. Claims explicitly rejected

| rejected claim | why |
|---|---|
| "The model simply forgot math" | It solves 55.5% of GSM8K with exemplars and never refuses MMLU-Pro. |
| "The issue is purely refusal" | −14.1pp on 8-shot GSM8K and −21.3pp conditional accuracy on MMLU-Pro (−17.7pp on items both stages attempted) occur where refusal is ~0%. |
| "The issue is purely capability loss" | 100% → 0% refusal from exemplars alone, content held constant, cannot be a capability fact. |
| "The M0 → M1-v2 DPO stage caused the regression" | Every signal is first observable on `Base → M0`; `M0 → M1-v2` is flat and 82% byte-identical. This rules out only that preference stage: M1-v1 — DPO directly on the base, no SFT — refuses 70.7% of zero-shot GSM8K, so DPO on When2Call-derived data can produce the refusal too. |
| "DPO did nothing / barely changed the weights" | Only behaviour on this suite was measured. No weight-space claim is supported. |
| "MMLU-Pro showed no degradation" | That was the @768 artifact. Corrected: −12.0pp. |
| "The promoted checkpoint is strictly better than Base" | It is better on tool policy and materially worse on every general-capability measure here. |
| "The 7-case sentinel proves general capability" | 7 cases; one case is 14pp. Base and M1-v2 both score 5/7 with opposite failure modes. |
| "$22.70 was an available balance" | It was a planning envelope. Never queried. |
| "Capability is intact" | Substantial arithmetic remains *accessible* under 8-shot, at ~14pp below Base. Not intact. |
| "The refusal is a literal surface-token pattern" | Stereotypy is consistent with it; the internal mechanism was not measured. |

---

## 16. Reproducibility status

| check | result |
|---|---|
| validation gate | `tests/evaluation/test_capability_benchmarks.py` — see §17 |
| preserved artifacts | 18, verified no drift (`scripts/preserve_h200_state.py --verify`) |
| checkpoint ladder | re-resolves; 2 expected failures documented as upstream properties |
| per-example evidence records | **48,840** across 15 benchmark×checkpoint combinations |
| append-only findings ledger | `results/benchmarks/capability_findings.jsonl`, 15 rows |
| GPU run log | `gpu_runs.jsonl`, 13 runs + 3 failed launches |
| IFEval scorer | deterministic under seed (verified ×3) |
| dataset revisions | IFEval `966cd89545d6`, GSM8K `740312add88f`, MMLU-Pro `b189ec765aa7` |
| vendored checkers | byte-identical to upstream; digests asserted in tests |

Every aggregate in this report recomputes from `capability_v1/evidence/*.jsonl` without a GPU —
**in a working tree that has them**. Those files, the raw generations and the per-example scores are
bulk (~720 MB, single files up to 97 MB) and are **not committed**, following the same rule the run
directories already use: canonical record in, bulk it was derived from out. See `.gitignore`.

What a fresh clone *does* get: every `*_scores.json`, `capability_findings.jsonl`, `gpu_runs.jsonl`,
`regression_analysis.json`, `final_campaign_audit.json`, the checkpoint ladder and the verdict —
which together carry every number quoted in this report. What it does **not** get is the per-example
layer beneath them. Request sets rebuild byte-identically from pinned HF revisions
(`scripts/prepare_capability_benchmarks.py`, hash asserted by test); **generations do not — they
require GPU time.** Publishing them as a Hugging Face dataset, rather than committing them, is the
open item.

---

## 17. Remaining uncertainties and known gaps

1. **M1-v1 corrected-budget MMLU-Pro is UNMEASURED.** Not interpolated.
2. **Causation of the refusal pattern is not established.** The corpus audit is an association.
3. **Single lineage, no replicate.** One base model, one SFT run. No claim generalizes to other
   model families.
4. **Base MMLU-Pro remains truncation-disadvantaged** at 21.8% vs M0's 6.3%. Handled by interval,
   not eliminated.
5. **The refusal detector is HEURISTIC** (`HEURISTIC_REGEX_v1`). Its precision has not been
   measured against hand labels. It never feeds an official benchmark metric.
6. **11 benchmarks remain `BLOCKED_NO_DATASET`**; Speculative Replay remains
   `BLOCKED_MISSING_MTP_COMPONENT`.
7. **Base is a multimodal artifact evaluated text-only** — a caveat on Base deltas only.
8. **No ANSWER-capability gate existed** at promotion time. That is the design lesson, now recorded
   in `docs/PROMOTION_POLICY.md` and `docs/EVALUATION.md`.
