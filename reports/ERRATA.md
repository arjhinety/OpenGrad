# Errata

**Date:** 2026-09-13

The files listed below are hash-pinned evidence, or are bound to evidence that is. Their bytes are
recorded in `reports/data/m0-final-freeze.json`, `reports/data/m0-v2-minus-xlam-freeze.json` or
`results/benchmarks/h200/PRESERVED_STATE_v1.json`. `reports/PTQ_PHASE_CLOSURE.md` is rendered from
the pinned `manifests/quantization/ptq_phase_closure_v1.json`. `reports/H200_BENCHMARK_RUN.md` is
append-only, with its original section pinned by `original_sha256`. Files under `release/` are
generated handoff bundles. Editing any of them would break the chain that lets a reader check the
numbers, so they are left byte-for-byte as frozen.

This file supersedes the statements listed here. Where a frozen file and this file disagree, this
file is the correct one. No frozen number is edited. Each entry either changes what a number is
said to show, or replaces a number that the frozen artifacts contradict.

These entries come from a claim audit of the repository, the Hugging Face cards and the project
site, which checked about 990 claims against the committed artifacts and found 93 that did not hold.
The full list, with the resolution of each finding (including those fixed in unfrozen files), is in
the [claim audit PDF](https://opengrad.arjhinety.com/claim-audit.pdf).

## How to verify

Each entry names the artifact its correction comes from. Line numbers refer to the frozen files as
committed. Every number below was recomputed from artifacts rather than copied from another report:

- routing metrics use `opengrad.evaluation.routing.routing_metrics` over committed predictions, or
  are read from the committed `curve.json` / `selection--dev.json` / `score_*.json` files;
- the M0 gate (`tool_use_promotion_v3`) is `opengrad.promotion.tool_use_policy.PromotionPolicyV2`,
  evaluated against the DEV B0 baseline stored in each run's `eval/dev/selection--dev.json`;
- the M1 gate (`tool_use_promotion_v4`) is `opengrad.promotion.m1_calibration.M1CalibrationPolicy`;
- the quantization gate uses the thresholds in `results/quantization/quantization_preservation_v1.json`;
- training exposure is `examples_seen` and `supervised_tokens_seen` at each step of
  `runs/<run>/metrics/train_log.jsonl`.

Where a source file is present only in the author's working tree and not committed, the entry says
so.

---

## 1. M0 on Canonical-v2 (final)

### `reports/M0_CANONICAL_V2_FINAL_EVALUATION.md`

- `reports/M0_CANONICAL_V2_FINAL_EVALUATION.md:44-46`. **As written:** confirmatory-table rows
  "M0-v1 (corpus v1)", "M1-DPO-v1" and "M0 partial-v2 @1200" (partial-v2: "0.6278 | 0.7610 |
  0.5342 | 0.0922 | 0.7951 | 0.6225"). **Correction:** these three rows can't be checked from the
  repository. No per-partition metrics artifact for them is committed. The committed
  `runs/qwen35_2b_m0_sft_v2corpus/eval/checkpoint-*/metrics.json` files are full 3,650-example
  scores (checkpoint 1200: recall 0.5050, `call_f1` 0.5995), not confirmatory ones. The per-example
  predictions the rows would be rescored from are excluded by `.gitignore` (`runs/**/predictions.jsonl`),
  and the table does not name the M0-v1 or M1-v1 checkpoints. The partial-v2 values appear only as
  prose in status notes in `runs/central_ledger.jsonl`; no committed artifact contains the M0-v1 or
  M1-DPO-v1 rows. Treat all three rows as unverified. The B0 and M0-final rows can be checked.
  **Source:** `git ls-files runs/`, `.gitignore:56`, `runs/qwen35_2b_m0_sft_v2corpus/eval/checkpoint-1200/metrics.json`.

- `reports/M0_CANONICAL_V2_FINAL_EVALUATION.md:75-78`. **As written:** "The rule instead selected
  **1800** on a tie-break … **the tie-break decided this selection**". **Correction:** 1800 has the
  highest macro score of the eligible checkpoints (0.6757 against 2400's 0.6744), so it wins with or
  without the tie-break. The tie-break was applied (`tie_break_applied: true`) but did not change
  the selection. The report's caution that 1800 and 2400 are effectively tied still holds. This is
  the same defect class as the minus-xLAM and M1 tie-break findings below, and was found while
  checking them. **Source:** `runs/m0_sft_canonical_v2_final/eval/dev/selection--dev.json` (`macro`, `ineligible`).

- `reports/M0_CANONICAL_V2_FINAL_EVALUATION.md:95`. **As written:** "slightly more favourable on
  four of six dimensions". **Correction:** five of six. From DEV to confirmatory, `call_f1`
  (0.7092 → 0.7470), precision (0.6990 → 0.7350), recall (0.7197 → 0.7594), over-call
  (0.1705 → 0.1505, lower is better) and clarification (0.7576 → 0.7682) are all more favourable.
  Only unsupported (0.5499 → 0.5430) is worse. **Source:** the table at lines 86-93 of the same
  report; `runs/m0_sft_canonical_v2_final/eval/{dev,confirmatory}/curve.json`.

- `reports/M0_CANONICAL_V2_FINAL_EVALUATION.md:105-107`. **As written:** "The policy caps over-call
  at 0.20 **and** forbids a recall drop beyond 0.10; a calibrated model cannot satisfy both. Every
  checkpoint on this run, and the historical partial-v2 checkpoint, fail the same constraint."
  **Correction:** there are two errors.
  1. Not every checkpoint fails `regression.call_recall`. Checkpoint 600 fails only
     `over_call_rate` (0.3436 > 0.20). Its DEV recall of 0.9121 is −0.059 against B0's 0.9715,
     inside the 0.10 allowance. Checkpoint 1200 fails both checks; 1800 and 2400 fail recall only.
  2. "Cannot satisfy both" is not established. Recall is computed over gold-`CALL` examples and
     over-call over gold non-`CALL` examples. These are disjoint sets, so one model can hold
     recall ≥ 0.8715 and over-call ≤ 0.20 at the same time. What the data show is only that no
     checkpoint in this run met both.

  **Source:** `runs/m0_sft_canonical_v2_final/eval/dev/selection--dev.json` (`ineligible`,
  `policy_verdicts`); `PromotionPolicyV2` recomputed per checkpoint; `src/opengrad/evaluation/routing.py`.

### `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`

- `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md:97, 117-118, 121-122`. **As written:**
  "Examples seen | 38,400"; "Examples seen at 2,400 × 16 | 38,400 | 38,400"; "Epochs over the
  corpus | **0.377** | **0.237**"; "it consumes 7.96M of 33.57M supervised tokens against
  partial-v2's 11.08M of 29.38M". **Correction:** 38,400 is the nominal 2,400 steps × 16. Batches
  are token-budgeted (`micro_batch_tokens: 4096`), so fewer examples fit per step. The logged values
  are:

  | | partial-v2 | final-v2 |
  |---|---:|---:|
  | Examples seen | 27,672 | 27,372 |
  | Supervised tokens seen | 8,011,435 | 5,678,531 |
  | Passes over trainable records | 0.272 (27,672 / 101,785) | 0.169 (27,372 / 161,966) |
  | Share of available supervised tokens | 27.3% of 29.38M | 16.9% of 33.57M |

  Line 96 of the same report already gives the 5,678,531 figure. The stated consequence still holds
  in direction: the final-v2 run saw a smaller fraction of a larger corpus. The magnitudes are the
  ones above. **Source:** `runs/m0_sft_canonical_v2_final/metrics/train_log.jsonl`,
  `runs/qwen35_2b_m0_sft_v2corpus/metrics/train_log.jsonl` (step 2400), and the `corpus_ready`
  events in each run's `events.jsonl`.

- `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md:142-143`. **As written:** "**Selected:
  `checkpoint-1800`, by tie-break.**" **Correction:** as for the evaluation report, lines 75-78:
  1800 is the outright macro maximum among eligible checkpoints (0.6757 against 0.6744), so the
  tie-break did not decide the selection. **Source:** `runs/m0_sft_canonical_v2_final/eval/dev/selection--dev.json`.

- `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md:197-199`. **As written:** "Every checkpoint,
  including the selected one, fails `regression.call_recall`". **Correction:** checkpoint 600 fails
  only `over_call_rate`. Its recall delta is −0.059, inside the allowance. 1200, 1800 and 2400 fail
  `regression.call_recall`. **Source:** as for the evaluation report, lines 105-107.

- `reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md:206-209`. **As written:** "Any model that fixes
  over-calling necessarily loses raw recall … As written, those two constraints cannot both be
  satisfied by a calibrated model." **Correction:** this is not established. The two constraints
  are measured on disjoint example sets (gold-`CALL` for recall, gold non-`CALL` for over-call), and
  a model that calls on exactly the gold-`CALL` examples would satisfy both. The supported statement
  is narrower: no checkpoint of this run, or of the ablation arms, met both. **Source:**
  `src/opengrad/evaluation/routing.py`; the DEV curves under `runs/*/eval/dev/`.

## 2. Joint xLAM + `CALL_PREDICTION` removal ablation

### `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md`

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md:55`. **As written:** "Launch commits |
  fixed `95a8225`, matched `0807fa3` (both arms from clean trees)". **Correction:** the
  matched-exposure arm was launched at 2026-09-11 12:49:03 UTC from `500cb4e`, with `git_dirty: true`.
  `0807fa3` is the commit at 13:26:57 UTC that recorded the finished run ("chore(ablation): record
  the matched-exposure arm's completed training run"). The fixed-compute arm's record is as stated:
  `95a8225`, and `git_dirty: false` in `experiment.json`. The trainer's `run_start` event records
  `git_dirty: true` for the fixed arm and for M0-final as well, because it runs after the
  experiment has appended to the tracked `runs/central_ledger.jsonl`. The `experiment.json` capture,
  taken before that append, is the one that tells the two arms apart. **Source:**
  `runs/m0_v2_final_minus_xlam_matched_exposure/experiment.json` (`git_commit`, `git_dirty`,
  `launch_timestamp`), the same run's `events.jsonl` (`run_start`), `git log -1 0807fa3`.

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md:29`. **As written:** matched-exposure arm
  "Held fixed | measured supervised-token exposure". **Correction:** exposure was not held fixed.
  The matched arm saw 6,743,788 supervised tokens against the reference's 5,678,531, which is 1.19×,
  not ~1.0×. The horizon formula at line 40 scales steps by the ratio of available supervised
  tokens. That holds exposure fixed only if every step consumes the same number of supervised
  tokens. Batches are token-budgeted, and without xLAM's short call-only targets the retained corpus
  averages 280 supervised tokens per record against the reference corpus's 207. So the matched arm
  consumed about 3,183 supervised tokens per step against the reference's 2,366. The report's own
  §5 table (line 81) records both token counts. **Source:** step 2119 of
  `runs/m0_v2_final_minus_xlam_matched_exposure/metrics/train_log.jsonl`; step 2400 of
  `runs/m0_sft_canonical_v2_final/metrics/train_log.jsonl`; the `corpus_ready` events.

### `reports/M0_V2_FINAL_ABLATION_DESIGN.md`

- `reports/M0_V2_FINAL_ABLATION_DESIGN.md:18`. **As written:** matched-exposure arm "What is fixed |
  measured supervised-token exposure". **Correction:** this was the design intent, and the
  executed run did not meet it: the arm saw 1.19× the reference's supervised tokens (6,743,788
  against 5,678,531). The step-scaling rule cannot hold tokens fixed when batches are token-budgeted
  and the removed source has shorter targets than the retained ones. **Source:** as above.

### `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md`

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:28`. **As written:** "| B0 | 0.6191 |
  0.4542 | 0.9722 | 0.6425 | 0.1009 | 0.0131 | — |" in a table of confirmatory results. **Correction:**
  these are B0's full 3,650-example numbers. On the 1,277-example confirmatory partition, B0 scores
  `call_f1` **0.6264**, precision **0.4618**, recall **0.9735**, over-call **0.6238**, clarification
  **0.1186**, unsupported **0.0177**. Against the correct row, fixed-compute's `call_f1` (0.6030) is
  0.023 below B0 and matched-exposure's (0.5557) is 0.071 below. Line 40's "well below … B0" is
  accurate for the matched arm and marginal for the fixed arm. **Source:** recomputed from
  `reports/baselines/qwen35_2b_baseline/predictions.jsonl` restricted to the confirmatory ids in
  `reports/evaluation/behavioral-heldout-v2-partition.json`.

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:30`. **As written:** fixed-compute @1200
  "parse_ok 1.0000". **Correction:** 0.999217 (1 of 1,277 not parsed). This is still above the
  0.99 floor. **Source:** `runs/m0_v2_final_minus_xlam_fixed_compute/eval/confirmatory/curve.json` (`parse_valid_rate`).

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:38`. **As written:** "matched − reference"
  over_call "−0.105". **Correction:** −0.095 (0.0558 − 0.1505 = −0.0947). Every other delta in both
  rows checks. **Source:** the table at lines 29-31 of the same report.

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:60, 65, 69-72`. **As written:** "the
  exposure confound does not explain the result"; "Reference exposure over retained corpus | ~1.53× |
  ~1.0×"; "The arm that trains **more** … is the arm that does better … The recall loss therefore
  tracks the missing supervision, not the training budget." **Correction:** as stated, this is not
  supported.
  1. The matched arm's ~1.0× is wrong. Its supervised tokens are 1.19× the reference's. As a
     fraction of its own corpus it saw 1.35× the reference's share by tokens (22.8% of 29.63M
     against 16.9% of 33.57M) and 1.36× by examples (0.231 passes against 0.169). The fixed arm's
     ~1.53× matches the token-fraction measure (1.53×). In absolute supervised tokens it is 1.35×.
  2. The table describes full horizons, but the metrics being compared come from the selected
     checkpoints. Fixed-compute @1200 had seen 3,801,605 supervised tokens and matched-exposure
     @1060 had seen 3,363,940. Both are fewer than the reference's selected @1800 at 4,270,591
     (0.89× and 0.79×). At the compared checkpoints the ablation arms had seen less supervised
     data than the reference, not more. "The arm that trains more does better" compares 1200 with
     1060, not the horizons in the table.
  3. The curves do support a narrower statement. The reference's lowest DEV recall at any
     checkpoint (0.7031 at 2400) is above the highest DEV recall of any ablation checkpoint
     (0.5036, fixed @600). At similar exposure the gap is the same: reference @600 recalls 0.9121
     after 1,403,293 supervised tokens, and fixed-compute @600 recalls 0.5036 after 1,880,934. On
     this evidence the recall loss is not an artifact of seeing fewer tokens. That is the argument
     the curves support; it is not the one the report makes. Single seed, no interval.

  **Source:** step 1060, 1200, 1800, 2119 and 2400 rows of each run's `metrics/train_log.jsonl`;
  `runs/*/eval/dev/curve.json` for the three runs.

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:91, 97-98`. **As written:** "selected
  (tie-break)"; "matched-exposure's 1060 won a tie-break". **Correction:** 1060 is the outright
  macro maximum (0.6213 against 0.6155, 0.6140 and 0.5583), so it is selected with or without the
  tie-break. `tie_break_applied: true` is recorded because 1590 falls within the 0.01 tolerance, but
  the tie-break did not change the selection. **Source:**
  `runs/m0_v2_final_minus_xlam_matched_exposure/eval/dev/selection--dev.json` (`macro`, `tie_break_applied`).

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:101-103`. **As written:** "Across steps both
  arms sharpen precision and suppress over-calling while recall keeps falling". **Correction:** this
  holds for the fixed-compute arm only. There, DEV recall goes 0.5036 → 0.4454 → 0.3373 → 0.3444,
  precision 0.7199 → 0.8169 and over-call 0.1078 → 0.0425. The matched-exposure arm moves the other
  way at first. Recall rises from 0.2352 (530) to 0.3884 (1060) and 0.3943 (1590), then 0.3812
  (2119). Precision falls from 0.8285 to 0.7804, then partly recovers to 0.7985. Over-call rises
  from 0.0268 to 0.0601, then eases to 0.0529. **Source:** `runs/m0_v2_final_minus_xlam_*/eval/dev/curve.json`.

- `reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md:107-109`. **As written:** "The failing check
  is `regression.call_recall` against B0's 0.9715, the same structural constraint every earlier
  checkpoint fails." **Correction:** `call_f1_retention` (≥ 0.90 of B0's DEV `call_f1` 0.6153) also
  fails:
  - on all four matched-exposure checkpoints (retention 0.595, 0.843, 0.855, 0.839 at 530, 1060,
    1590, 2119);
  - on fixed-compute 1800 (0.776) and 2400 (0.788).

  Fixed-compute 600 and 1200 fail recall only. "Every earlier checkpoint" is also not true: M0-final
  @600 did not fail `regression.call_recall` (see section 1). **Source:** `PromotionPolicyV2`
  recomputed over each arm's `eval/dev/selection--dev.json`.

## 3. Quantization and the H200 run

### `reports/QUANTIZATION_PTQ_EVALUATION.md`

- `reports/QUANTIZATION_PTQ_EVALUATION.md:89-91`. **As written:** "with an explicit warning that its
  margin is thin"; "Rungs passing the frozen gate: `Q6_K`, `Q8_0`." **Correction:** the margin is
  one example, and the gate's pass/fail is inside run-to-run noise.
  - Q6_K has 357 correct calls among 490 predicted calls. The precision floor needs 356.96, so it
    passes by one example. Its over-call (133 of 824 non-call examples) is also one example inside
    the 134.2 allowed.
  - The unquantized reference itself fails `quantization_preservation_v1` when rerun. The H200 vLLM
    BF16 rerun of the same weights scores recall 0.7638 (346 of 453) against the 0.7671 floor
    (347.5), 5 correct calls fewer than the frozen reference's 351.

  "Only Q6_K and Q8_0 pass" is a correct reading of the frozen gate. It is not evidence that Q6_K
  or Q8_0 preserve behaviour better than a rerun of BF16 does. **Source:**
  `results/quantization/gguf/score_m1-v2-Q6_K_confirmatory.json` (`decision_counts`, `metrics`);
  `results/benchmarks/h200/score_vllm-bf16_confirmatory.json`; thresholds in
  `results/quantization/quantization_preservation_v1.json`.

- `reports/QUANTIZATION_PTQ_EVALUATION.md:138-140`. **As written:** "No per-example vLLM ↔ llama.cpp
  agreement is claimed anywhere in this study." **Correction:** this is still true of the frozen
  vLLM BF16 reference, whose per-example predictions were never preserved. It is no longer true of
  the repository. `reports/H200_BENCHMARK_RUN.md:107-115` and `reports/FINAL_CAMPAIGN_AUDIT.md`
  report decision agreement 0.9836 (1,256 of 1,277; 21 flips). That figure compares a new vLLM BF16
  run on an H200 with llama.cpp BF16 on an A100. It is not the frozen reference: the H200 run's
  aggregates differ from it (`call_f1` 0.7505 against 0.7548), and engine, hardware and kernels all
  differ. Every per-example agreement number in this report remains llama.cpp QX ↔ llama.cpp BF16.
  **Source:** `results/benchmarks/h200/vllm_vs_llamacpp_agreement.json` (`note`), `results/benchmarks/h200/score_vllm-bf16_confirmatory.json`.

### `reports/QUANTIZATION_ENGINE_PARITY.md`

- `reports/QUANTIZATION_ENGINE_PARITY.md:149-151`. **As written:** "Because the frozen vLLM per-example
  predictions do not exist, **no per-example vLLM ↔ llama.cpp agreement is claimed anywhere in this
  study.**" **Correction:** same as the entry above. It remains true of the frozen reference. A later
  report claims 0.9836 agreement between a separate H200 vLLM run and llama.cpp BF16, and that is a
  runtime-stack comparison, not a comparison with the frozen reference. **Source:** as above.

### `reports/PTQ_PHASE_CLOSURE.md`

- `reports/PTQ_PHASE_CLOSURE.md:21-22, 28-29`. **As written:** "`Q6_K` | 1.45 |
  `MEMORY_OPTIMIZED_RELEASE`" and "`Q8_0` | 1.87 | `RECOMMENDED_RELEASE`". **Correction:** the
  recommended release is **Q6_K (1.45 GiB)**. The pre-registered rule is "the smallest format that
  passes the gate wins" (`reports/QUANTIZATION_STUDY_PLAN.md:230`). The closure manifest this report
  is rendered from records `selection.recommended_release_rung: Q6_K`. Q8_0 (1.87 GiB) also passes
  the primary gate and is a larger alternative, not the recommendation. The role labels were
  hard-coded after the ladder was scored (`scripts/close_ptq_phase.py:47-55`, `ROLES`), and they
  contradict the rule and the manifest. Both rungs fail the secondary release bar, as the report
  says. **Source:** `manifests/quantization/ptq_phase_closure_v1.json` (`selection`);
  `reports/QUANTIZATION_STUDY_PLAN.md:230`.

- `reports/PTQ_PHASE_CLOSURE.md:31-32`. **As written:** "`Q5_K_M` and every rung below it were
  **scored and failed the primary gate**." **Correction:** this is correct, and needs the caveat
  given for `QUANTIZATION_PTQ_EVALUATION.md:89-91` above. Q6_K clears precision by one example
  (357 of 490 against 356.96 required), and the H200 BF16 rerun of the unquantized reference fails
  the same gate on recall (0.7638 < 0.7671). **Source:** as above.

- `reports/PTQ_PHASE_CLOSURE.md:84-85`. **As written:** "so no per-example vLLM ↔ llama.cpp agreement
  is claimed anywhere in this phase." **Correction:** true of this phase. See the
  `QUANTIZATION_PTQ_EVALUATION.md:138-140` entry for the later 0.9836 figure and why it doesn't
  compare against the frozen reference.

### `reports/H200_BENCHMARK_RUN.md`

- `reports/H200_BENCHMARK_RUN.md:90-91`. **As written:** "This regenerates the per-example vLLM
  predictions that were never preserved, removing the standing caveat from the entire quantization
  study." **Correction:** the rerun is a new H200 execution. It does not recover the frozen
  reference's predictions: its aggregates differ (`call_f1` 0.7505 against 0.7548, recall 0.7638
  against 0.7748). The standing caveat therefore still applies to every comparison against the
  frozen reference. The rerun's own per-example predictions
  (`results/benchmarks/h200/predictions_vllm-bf16_confirmatory.json`) are excluded by `.gitignore`
  (`results/benchmarks/h200/predictions_*.json`). Only its scores and the agreement summary are
  committed. **Source:** `results/benchmarks/h200/score_vllm-bf16_confirmatory.json`, `.gitignore:136`.

- `reports/H200_BENCHMARK_RUN.md:103-105`. **As written:** "Every delta is within +/-0.011 … but the
  reference is confirmed." **Correction:** the rerun does not confirm the reference at the
  resolution the quantization gate uses. Scored under `quantization_preservation_v1`, the H200 vLLM
  BF16 rerun of the unquantized reference fails: recall 0.7638 (346 of 453 calls) against the
  0.7671 floor (347.5). Q6_K clears precision by one example (357 of 490 against 356.96 required).
  Gate pass/fail is therefore inside run-to-run noise. **Source:**
  `results/benchmarks/h200/score_vllm-bf16_confirmatory.json`,
  `results/quantization/quantization_preservation_v1.json`,
  `results/quantization/gguf/score_m1-v2-Q6_K_confirmatory.json`.

- `reports/H200_BENCHMARK_RUN.md:117-120`. **As written:** "the engine change alone (vLLM to
  llama.cpp) produces 21 per-example decision flips. Q6_K quantization produced 25." **Correction:**
  the 21 flips are not an engine-only effect. They compare vLLM BF16 on an H200 with llama.cpp BF16
  on an A100, so engine, hardware and kernels all differ. 3 of the 21 fall on the 6 known
  tokenizer-divergent prompts, as the table at lines 111-115 already shows. This is a runtime-stack
  comparison. The 25 Q6_K flips are llama.cpp Q6_K against llama.cpp BF16 on the same stack.
  **Source:** `results/benchmarks/h200/vllm_vs_llamacpp_agreement.json` (`note`,
  `flips_among_known_tokenizer_divergent`); `manifests/quantization/ptq_phase_closure_v1.json` (Q6_K `decision_flips`).

- `reports/H200_BENCHMARK_RUN.md:200, 205`. **As written:** "accuracy_given_answer falls by 21.3pp on
  at least one measured edge -- the ability itself is worse". **Correction:** the label still holds,
  but the figure is inflated by truncation. The 21.3pp is MMLU-Pro Base 59.7% against M0 38.4%, each
  over the items that stage attempted. Base's unattempted items include 2,136 truncations, which are
  excluded from its denominator. On the 9,637 items both stages attempted, the drop is **17.7pp**
  (60.0% → 42.4%). No interval is given. **Source:**
  `results/benchmarks/h200/capability_v1/final_campaign_audit.json`
  (`unattempted_that_are_truncations`); the joint-attempt figure is from
  `results/benchmarks/h200/capability_v1/{BASE,M0_SFT}/mmlu_pro_scores_per_example.jsonl`, which
  are present in the author's working tree and not committed.

- `reports/H200_BENCHMARK_RUN.md:223`. **As written:** "conditional math accuracy falls | not
  materially observable". **Correction:** it can't be measured at 0-shot on the primary path,
  because M0 and M1-v2 attempted none of the 1,319 questions. It can be measured with 8 exemplars.
  There, every primary-path stage answers all 1,319, so conditional accuracy equals accuracy: Base
  70.4% (929) → M0 56.3% (743), −14.1pp on `BASE->M0_SFT`. That is the drop reported one row below
  as "fewshot math accuracy". On the `BASE->M1_DPO_HISTORICAL` edge, 0-shot conditional accuracy is
  measurable and falls from 67.4% to 59.8% (228 of 381 attempted). **Source:**
  `results/final_campaign_verdict.json` (`canonical_benchmarks.gsm8k.per_stage`).

- `reports/H200_BENCHMARK_RUN.md:240`. **As written:** "MMLU-Pro | `BLOCKED_NO_DATASET` | **COMPLETE**
  ×4 stages". **Correction:** ×3 stages at the corrected 2,048-token budget (Base, M0, M1-v2).
  `M1_DPO_HISTORICAL`'s corrected-budget run was terminated for budget (`TERMINATED_BUDGET`, no
  results retained), so its MMLU-Pro is **UNMEASURED**. Lines 263-265 of the same report describe
  the termination. IFEval and GSM8K are ×4 as stated. **Source:**
  `results/final_campaign_verdict.json` (`limitations`), `results/benchmarks/h200/capability_v1/cost_ledger.json` (`gpu_runs`, seq 13).

- `reports/H200_BENCHMARK_RUN.md:249`. **As written:** "retained runs (results reported) | $5.01".
  **Correction:** the $5.01 includes $0.757 (an estimate) for the terminated
  `M1_DPO_HISTORICAL` MMLU-Pro run, which returned no results. Runs whose results are reported cost
  **$4.25**. Spend that produced no reported result is **$4.64** ($3.89 superseded + $0.757
  terminated), which is 52% of the $8.90 continuation, not the 44% the superseded share alone
  suggests. **Source:** `results/benchmarks/h200/capability_v1/cost_ledger.json` (`gpu_runs`,
  `retained_runs_usd`, `superseded_runs_usd`).

### `reports/OPENWEIGHTS_TRANSFER_EVALUATION.md`

This is Phase 2 of the H200 run. The file is pinned by `results/benchmarks/h200/PRESERVED_STATE_v1.json`.

- `reports/OPENWEIGHTS_TRANSFER_EVALUATION.md:12-14`. **As written:** "That is an over-refusal /
  alignment-tax signature, not a tool-policy failure." **Correction:** both failing outputs were
  parsed as `UNSUPPORTED`, the routing class the tool policy is trained to emit. The failures are
  the tool policy's refusal behaviour applied to requests that aren't tool requests, so they can't
  be separated from the tool policy. **Source:** `results/benchmarks/openweights_parity_opengrad_v1.json`
  (`parsed_decision` for `multi-step-change` and `format-constraint`).

- `reports/OPENWEIGHTS_TRANSFER_EVALUATION.md:56-59`. **As written:** "**General capability
  regressed.** `multi-step-change` … and `format-constraint` … are well within a 2B model's ability;
  the model declines both." **Correction:** this suite does not show a regression. The Qwen3.5-2B
  base model, run later on the same seven cases, also scores 5/7 (general 3/5, tool 2/2). It fails
  `multi-step-change` itself (wrong content), and it fails `trap-arithmetic`, which OpenGrad passes.
  Relative to the base, OpenGrad trades one general case for another. The general-capability
  regression is established by IFEval, GSM8K and MMLU-Pro (see `reports/H200_BENCHMARK_RUN.md` from
  line 187 on, and section 7 below), not by this sentinel. **Source:**
  `results/benchmarks/h200/capability_v1/BASE/sentinel_scores.json`.

- `reports/OPENWEIGHTS_TRANSFER_EVALUATION.md:68`. **As written:** "No Qwen3.5-2B baseline exists in
  the OpenWeights results, so the upstream-vs-OpenGrad delta cannot be computed from these files
  alone." **Correction:** this is true of the OpenWeights files, but a Qwen3.5-2B baseline on this
  suite now exists in the repository. The upstream-vs-OpenGrad delta is 0 cases (5/7 against 5/7).
  **Source:** `results/benchmarks/h200/capability_v1/BASE/sentinel_scores.json`.

## 4. Release bundles

- `release/gguf/manifest.json:42`. **As written:** `"primary_mobile_target": "Q4_K_M"`.
  **Correction:** Q4_K_M is `REJECTED_ACCURACY` under `quantization_preservation_v1`. It fails
  `call_precision` (0.6612 < 0.7285), `clarification_accuracy` (0.6388 < 0.7578),
  `unsupported_accuracy` (0.5011 < 0.5332) and `over_call_rate` (0.2512 > 0.1629). It must not be
  used as a deployment target for this model. Q5_K_M, also in this manifest's `ladder`, is rejected
  as well. The recommended rung is Q6_K (1.45 GiB); Q8_0 (1.87 GiB) also passes. The noise caveat in
  section 3 applies to both. **Source:** `results/quantization/gguf/score_m1-v2-Q4_K_M_confirmatory.json`
  (gate recomputed against the thresholds in `results/quantization/quantization_preservation_v1.json`);
  `manifests/quantization/ptq_phase_closure_v1.json` (`selection`).

- `release/executorch/cpu/README.md:36-39`. **As written:** "the Snapdragon target additionally needs
  two source patches to ExecuTorch." **Correction:** the two patches are necessary but not
  sufficient. Both were applied (`lift_constant_scalar_operands.py`, `export_llama_lib.py`), and
  both Qualcomm HTP exports for SM8650, unquantized and `qnn_16a4w`, still failed with a
  graph-ordering `RuntimeError` ("Argument '_lifted_tensor_constant…' … was used before it has been
  defined"). Status is `REJECTED_EXPORT` and `exported: false`. No Snapdragon artifact exists.
  **Source:** `results/quantization/executorch/export_qnn_.json`, `results/quantization/executorch/export_qnn_qnn_16a4w.json`.

- `release/executorch/cpu/README.md:33, 41-43`. **As written:** "`qwen3_5_2b_8da4w.pte` | 8da4w |
  2.89 GiB"; "exported cleanly, 3.09× smaller than fp32". **Correction:** the size ratio is correct
  (3.087×). "8da4w" does not mean all weights are int4. Int4 covers 99.98% of the named-data
  weights, but only 78.7% of stored parameters. The embedding table (508,559,360 parameters) stays
  fp32 in the program constants, and at 2,034,237,440 bytes (1.89 GiB) it is 65.7% of the 8da4w `.pte`. The tied `lm_head`
  copy is int4. **Source:** `results/quantization/executorch/quantization_audit_8da4w.json`
  (`coverage`, `program_constants`).

## 5. Configs

- `configs/releases/toolpolicy_canonical_v2_final.yaml:57, 71`. **As written:** Glaive
  `source_revision: 7e7e32f0466699513b38923bc4991858e192c9274df22e243b9e77eb2775b6e8`; ToolACE
  `source_revision: 7a7a6a2c3b1003c7`. **Correction:** these two values are not Hugging Face
  revisions, and cards that display them as "Pinned revision" are mislabelled. The Glaive value is
  the SHA-256 of the downloaded raw source file, which is 64 hex digits; a Hub revision is a 40-digit
  commit id. The ToolACE value is the first 16 hex digits of that file's SHA-256 (`7a7a6a2c3b1003c789bb…`).
  The Hub snapshot revisions recorded for the same sources are `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`
  (`glaiveai/glaive-function-calling-v2`) and `6bda777c88d21e5a204703c1ee45597a8fa4f734`
  (`Team-ACE/ToolACE`). The config is pinned by both M0 freezes and is left unchanged. Read the
  field as a raw-file content hash for these two sources. No statement is made here about the xLAM
  and When2Call values. **Source:** `registry/datasets.yaml` (`source_revision.value` for both
  sources); `configs/releases/toolpolicy_canonical_v1.yaml:31, 41`; `config.source_sha256` in
  `data/processed/normalization-v1/{glaive,toolace}/manifest.json` (working tree only;
  `data/processed/` is gitignored).

## 6. Preserved state

- `results/benchmarks/h200/PRESERVED_STATE_v1.json`. **As written:** the manifest pins sha256 digests
  for the H200 campaign state, verified by `scripts/preserve_h200_state.py --verify`.
  **Correction (known reproducibility limitation):** the digests were computed over the author's
  Windows working-tree bytes (CRLF), not the committed blobs. A clone checks out LF
  (`.gitattributes` sets `eol=lf` for `*.json`, `*.md` and `*.yaml`), so the state can't be verified
  off the author's machine. Of the 18 pins:
  - 15 match only the CRLF rendering of the committed blob. These are 14 current artifacts,
    including `reports/QUANTIZATION_PTQ_EVALUATION.md`, `reports/QUANTIZATION_ENGINE_PARITY.md`,
    `reports/OPENWEIGHTS_TRANSFER_EVALUATION.md` and `manifests/quantization/ptq_phase_closure_v1.json`,
    plus the original section of `reports/H200_BENCHMARK_RUN.md` (`original_sha256`, 9,090 CRLF bytes).
  - 1 (`results/benchmarks/openweights_parity_cases_v1.json`) matches the committed LF bytes.
  - 2 are gitignored and absent from any clone: `results/benchmarks/h200/predictions_vllm-bf16_confirmatory.json`
    and `results/quantization/frozen_behavioral_eval_v1.jsonl`.

  The M0 freezes (`reports/data/m0-final-freeze.json`, `reports/data/m0-v2-minus-xlam-freeze.json`)
  have the opposite property: they pin the committed LF blobs, so they verify on an LF checkout and
  fail on this Windows working tree. Until a verifier normalizes line endings, or a successor
  manifest pins committed blobs, a CRLF-pinned file can be checked from an LF checkout with
  `python -c "import sys,hashlib;print(hashlib.sha256(open(sys.argv[1],'rb').read().replace(b'\n',b'\r\n')).hexdigest())" <path>`.
  This works for files that contain no CRLF of their own. **Source:** `git show HEAD:<path>`
  compared with each `artifacts.*.sha256` and `append_only_updates.report_h200.original_sha256`;
  `.gitattributes`; `.gitignore:134, 136`.

## 7. Append-only ledgers and publication records

These records are historical: a ledger row or a publication record states what was believed when it
was written, so it is corrected here rather than rewritten.

- `results/benchmarks/capability_findings.jsonl` rows 4, 8 and 15 (`M1_DPO_HISTORICAL`: GSM8K,
  IFEval, sentinel). **As written:** `ladder_role` "earlier DPO lineage on a different SFT parent
  (CorpusV2, not CanonicalV2-Final)". **Correction:** M1-v1 (`qwen35_2b_m1_dpo_v1`, Hub
  `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO` checkpoint-300) is DPO applied directly to the base model,
  with `parent_experiment_id: null` and `reference: initial_policy`. There is no SFT parent. The
  ledger is append-only; rows written from now on read the corrected `role` from
  `results/benchmarks/checkpoint_ladder.json`. **Source:** `runs/qwen35_2b_m1_dpo_v1/experiment.json`;
  the Hub `checkpoint-300/checkpoint_metadata.json`.
- `reports/releases/hf-publication-2026-09-11.json:76`. **As written:** "NOT PROMOTED — fails
  regression.call_recall against B0, as does every checkpoint of the run". **Correction:** checkpoint
  600 does not fail `regression.call_recall`; its recall delta is −0.059, inside the −0.10 allowance.
  It fails only `over_call_rate`. Every other checkpoint of the run does fail
  `regression.call_recall`. **Source:** `runs/m0_sft_canonical_v2_final/eval/dev/selection--dev.json`.
- `results/benchmarks/h200/capability_v1/gpu_runs.jsonl` line 8 (`seq` 7, `superseded_detail`).
  **As written:** "100% of its unattempted examples were truncations rather than refusals".
  **Correction:** 99.6%: 4,292 of Base's 4,309 unattempted MMLU-Pro items at the 768-token budget
  were truncations. The conclusion (truncation, not refusal) is unchanged, and
  `reports/GENERAL_CAPABILITY_REGRESSION.md` and `reports/FINAL_CAMPAIGN_AUDIT.md` already give 99.6%.
  **Source:** `results/benchmarks/h200/capability_v1/final_campaign_audit.json`
  (`mmlu_pro_superseded_768.BASE`: `unattempted`, `unattempted_that_are_truncations`).

## 8. Consequences for the headline findings

Several frozen reports above state the pre-correction framing. These four statements take precedence
wherever the headline findings are summarised.

### C1. The regression is not only an SFT effect

It is a general-capability regression associated with tool-policy post-training on When2Call-derived
data. It appears after SFT (M0) and also after DPO applied directly to the base (M1-v1), so it is not
specific to SFT. Causation is not established: one lineage, one seed, no replicate.

- M0 (`m0_sft_canonical_v2_final` @1800) refuses 100% of GSM8K zero-shot (1,319 of 1,319). M1-v2
  (DPO on M0) inherits 100%. Base refuses 0%.
- M1-v1 (`qwen35_2b_m1_dpo_v1`, checkpoint-300, benchmarked as `M1_DPO_HISTORICAL`) is DPO applied
  directly to the base model: `parent_experiment_id: null`, `reference: initial_policy`, preference
  data `when2call_pref_v1`. It refuses 70.7% (933 of 1,319) of GSM8K zero-shot.
- Capability, Base → M0 / M1-v2:

  | benchmark | Base | M0 | M1-v2 |
  |---|---:|---:|---:|
  | GSM8K 0-shot | 67.4% | 0.0% | 0.0% |
  | GSM8K 8-shot | 70.4% | 56.3% | 55.5% |
  | IFEval prompt-strict | 67.8% | 45.1% | 45.8% |
  | MMLU-Pro | 49.0% | 37.0% | 37.0% |

  M1-v1's MMLU-Pro at the corrected budget is UNMEASURED.

Source: `results/final_campaign_verdict.json`, `runs/qwen35_2b_m1_dpo_v1/experiment.json`.

### C2. What M1-v2's promotion means

M1-v2 is promoted under a parent-relative gate (v4) introduced after M0 was evaluated; M0 also clears
v4, and M1-v2 fails the v3 gate that rejected M0. The promotion reflects the gate change; the measured
difference from M0 (+0.0078 call_f1, 7 of 453 calls, single seed) is within noise.

- `tool_use_promotion_v4` was committed on 2026-09-11 at 17:10 UTC (`92ca2b9`). M0's confirmatory
  results were committed at 08:58 UTC (`26234c2`).
- Under `tool_use_promotion_v3`, all four M1-v2 DEV checkpoints are REJECT on `regression.call_recall`.
- M0 checkpoint 1800's confirmatory metrics clear every v4 floor.
- On the confirmatory partition, M0 → M1-v2:

  | metric | M0 | M1-v2 |
  |---|---:|---:|
  | `call_f1` | 0.7470 | 0.7548 (+0.0078) |
  | correct calls (of 453) | 344 | 351 (+7) |
  | recall | 0.7594 | 0.7748 |
  | precision | 0.7350 | 0.7358 |
  | over-call | 0.1505 | 0.1529 (slightly worse) |
  | clarification | 0.7682 | 0.7655 |
  | unsupported | 0.5430 | 0.5386 |

  No interval is available.

Source: `runs/m1_dpo_canonical_v2_final_v2/eval/dev/selection--dev.json`,
`runs/{m0_sft_canonical_v2_final,m1_dpo_canonical_v2_final_v2}/eval/confirmatory/curve.json`,
`M1CalibrationPolicy`, `git log -1 92ca2b9`, `git log -1 26234c2`.

### C5. Quantization

- The pre-registered rule is that the smallest passing format wins. The recommended release is
  **Q6_K (1.45 GiB)**, and Q8_0 (1.87 GiB) also passes. The closure manifest already records
  `recommended_release_rung: Q6_K`.
- The gate is inside run-to-run noise. The H200 vLLM BF16 rerun of the unquantized reference itself
  fails `quantization_preservation_v1` (recall 0.7638 < 0.7671 floor). Q6_K clears precision by one
  example (357 of 490 against 356.96 required).
- Over-call through DPO rose slightly (0.1505 → 0.1529). DPO did not lower it.

### C6. Minus-xLAM ablation

- The matched-exposure arm was launched at 12:49 from `500cb4e` with `git_dirty: true`. `0807fa3` is
  the 13:26 commit that recorded the finished run.
- Logged supervised tokens are 6,743,788 for the matched arm and 5,678,531 for the reference
  (1.19×), not ~1.0×. The selected checkpoints (1060 matched, 1200 fixed) saw 3.36M and 3.80M
  supervised tokens, against 4.27M for the reference's selected 1800. "Exposure does not explain
  the result" is not supported as stated. The narrower statement the curves do support is in
  section 2.

## 9. Publication provenance

**Added 2026-09-14.** `scripts/verify_publication.py` and the checks it drives
(`src/opengrad/registry/provenance.py`, `docs/PROVENANCE.md`) were added because OpenGrad had a
release discipline and no publication discipline: every rule governed the payload, while the act of
publishing had no gate. Running the new checks over the existing registry found defects no earlier
check could see. All are corrected forward. No frozen artifact was edited.

- **`canonical_v2` licence evidence did not exist.** `license.source` pointed at
  `.release/hf/toolpolicy-canonical-v2/source-licenses.md`, and `sample_count.verified_from` at
  `.release/hf/toolpolicy-canonical-v2/release-manifest.json`. `.release/**` is build output under
  `.gitignore`; neither path resolves. Both now point at the tracked card directory and the tracked
  publication record.
- **`canonical_v1` licence evidence was uncommitted.** `license.source` pointed at
  `.release/hf/toolpolicy-canonical-v1/source-licenses.md`, which still existed on the machine that
  built the release. The claim therefore looked sound while being unreproducible from a fresh
  clone — a worse failure mode than a missing file, because nothing draws attention to it.
  Repointed to the tracked card directory.
- **The `qwen3.5-2b` licence evidence was a moving target.**
  `https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE` returns whatever the licence text
  happens to be at fetch time, so it pinned nothing about what was verified. Repointed to the
  revision the record already pins (`15852e8c…`).
- **The v2-final licence file described the wrong corpus.** It was byte-identical to the
  three-source partial snapshot's file: it stated the release contained three sources only and
  listed `Salesforce/xlam-function-calling-60k` under "Sources not in this release", while the
  payload carries 57,342 xLAM records. Its `CITATIONS.bib` omitted the APIGen entry the release
  manifest declares as required for xLAM. Corrected in the repository and republished; recorded in
  `reports/releases/hf-publication-2026-09-14-canonical-v2-licence-correction.json`. The payload is
  unchanged: 176 shards, 173,237 records, fingerprint `8ced403b…`.
- **Two publication events had left no record.** `459bc01b` (card correction, 2026-09-13) and the
  commits of the licence correction were published without publication records, leaving every
  recorded revision stale. All are now recorded, including the intermediate revision in which the
  licence file was corrected and the citation was not, and including the retroactive entry, which
  is transcribed from the Hub commit history rather than re-derived.
- **The minus-xLAM derived fingerprint moved with no recorded input identity.** Kept as the single
  named legacy exception (`provenance_version: legacy_single_digest_v1`) rather than rewritten. Its
  superseded and current fingerprints are both preserved.

## 10. Closure: the calibration identity, and a freeze check that was doing nothing

**Added 2026-09-14.** Section 9 closed with one record whose derived identity could not be
re-derived, and two publication events reconstructed from Hub history. Both are resolved here.

**Two checks were reporting PASS by doing nothing.**

`validate_freeze` obtained the artifact a digest was the identity of via
`processed_dataset_hash.source` or `findings.release_manifest`. Neither field is present on
`canonical_v1`, `canonical_v2` or `canonical_v2_final` — they name their manifest in
`source_repository` — so `_local_manifest_for` returned nothing for all three and the check skipped
them. The strongest freeze assertion in the repository executed nothing. It now reads
`source_repository`, and a record whose identity cannot be re-derived must either be re-derivable
from a committed artifact or declare `identity_artifact_unavailable` with a tracked record
corroborating the same digest. A declaration with no corroboration is rejected.

`src/opengrad/registry/validate.py` also had no `if __name__ == "__main__"` guard, so
`python -m opengrad.registry.validate` imported the module, ran nothing and exited **0** — which is
indistinguishable from a passing validation. Only the `opengrad-validate` console script actually ran
the check. Any earlier report that this validation passed via `-m` was therefore vacuous, including
in this errata's own section 9. The guard is added, and both entry points now report the real result.

Turning the freeze check on produced one open finding and confirmed two identities:

- **`m1_calibration_preference_pairs_v1` — `RESOLVED_IDENTITY`.** `d3916894…` is the SHA-256 of
  `data/processed/m1_calibration_preference_pairs_v1.jsonl`. The artifact was already named by the
  record (`source_repository`), and `reports/data/m1-calibration-preference-pairs-v1.json` records
  the same path as its `output` with the same digest. Searching every one of the 987 committed blobs
  found exactly one match, and the check now re-derives it. Nothing was recomputed into a
  replacement; the identity was demonstrated.
  The digest is over the **LF** bytes of the artifact. The Windows working tree holds CRLF for the
  same file, which is why hashing the file on disk gave `0907ac32…`. `.gitattributes` covers
  `*.yaml`, `*.json` and `*.md` but not `*.jsonl`, and `core.autocrlf=true`. Content digests are now
  verified against committed blobs, so a `source_sha256` is a claim about the artifact rather than
  about the verifier's platform, and evidence paths must be tracked by Git.
- **`canonical_v2_final` — `RESOLVED_IDENTITY`.** `8ced403b…` is the SHA-256 of
  `.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`, which is committed. Re-derived,
  not assumed.
- **`canonical_v1` — corroborated.** Its manifest is uncommitted build output and cannot be
  re-derived. The record now says so and names the tracked publication record whose
  `release_manifest_sha256` carries the same `181b3fba…`, so the identity is backed by a second
  record rather than by assertion.
- **`canonical_v2` — `UNRECOVERABLE_IDENTITY`, OPEN.** No committed artifact reproduces
  `09018d26…`. Separately, the tracked publication record for the same corpus stores
  `release_manifest_sha256 = 277a0ae4…` and records the manifest at commit `ad70a8ba…`, while the
  registry records `source_revision = 7a2a8e7b…`. Two records therefore disagree about the identity
  of the same published corpus, and the repository does not contain the evidence needed to say which
  is right. **This is not resolved by choosing one.** It is left open deliberately: publication
  provenance is not PASS while it stands.

**Publication impact.** `d3916894…` is in the evidence chain of a current publication-visible claim:
the published model card at `arjhinety/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` states it,
as do the M1 publication record, `results/registry.jsonl`, and the registry entry. Because the
identity resolves, it does not block. `canonical_v2` is also publication-visible
(`arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`), and it does block.

**Reconstructed events.** The retroactively recorded 2026-09-13 correction now carries
`event_occurred_at`, `entry_recorded_at`, `recorded_retroactively: true` and
`reconstruction_evidence` naming the Hub revision it was transcribed from — so the record states
that it did not exist contemporaneously with the event. Validation rejects an entry whose dates
differ while claiming contemporaneity, one that claims reconstruction without dates or evidence, and
one whose `entry_recorded_at` disagrees with the record it lives in.

How to verify: `python scripts/verify_publication.py` exits 1 with one FAIL — the open `canonical_v2`
identity — and does not exit 0. With no network it also reports `BLOCKED_NETWORK`. It will exit 0
only once `canonical_v2`'s identity is resolved from evidence or its claim is corrected.

How to verify: `python scripts/verify_publication.py` must exit 0. At the time this section was
written it did; section 10 supersedes that, because the freeze check was not yet actually running.
With no network it exits 2 with `BLOCKED_NETWORK`, which is not a pass.

## 11. Closure: the canonical_v2 identity was recovered, and the verifier contract moved to v2

**Added 2026-09-14.** Section 10 left `canonical_v2` as `UNRECOVERABLE_IDENTITY` with publication
provenance failing. It is now `RESOLVED_IDENTITY`, and the failure was in the search rather than in
the evidence. Nothing was chosen because it looked newer or more plausible; both digests were
recovered and each was reproduced from the published artifact.

**The two digests are not competing identities.** They are two editions of the same
`release-manifest.json`, and both were reproduced from
`arjhinety/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot` at full 40-character revisions:

| digest | bytes | revisions carrying it |
|---|---:|---|
| `09018d26…` | 23522 | `bb70545cd486c080c4337fa2fd465df6d14fb0ed`, and `b0de031fdf86694d0e093c75628e4b4862dc658f` (HEAD) |
| `277a0ae4…` | 23525 | `ad70a8ba7a0bb3d9bee6a5a7b5de7433ce019f61`, `5130fb4eb198cc846a42d601dd111dfdcc2e26d5`, `8b43caebf2b96259f8f86611d39f75bcaa3ee130` |

The publication record is therefore self-consistent: both revisions it names (`8b43caeb` as
`hub_revision`, `ad70a8ba` as `manifest_revision_at_payload_upload`) carry `277a0ae4…`. The registry
is consistent with HEAD. **The two editions differ in exactly two fields** — `hub_repository`
(`unpublished/OpenGrad-ToolPolicy-Canonical-v2-partial` versus
`arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`) and `opengrad_git_commit`
(`7a2a8e7b…` versus `84d1de61…`) — and their 104 `output_shards` lists are **identical**, with
`record_count` 103036 in both. So the "conflict" was a record that did not say which revision each
digest belonged to.

**What was searched.** Version control: `git cat-file --batch-all-objects` hashed all 3648 objects
(1783 blobs) including the 39 unreachable ones — no blob reproduces either digest, so no manifest was
ever committed and deleted. All refs, tags, branches, stashes, and full history for any
`*manifest*` path. The Hub: full commit history and the file list at **every** revision for the
M0-snapshot repo (6 commits, 109 files at HEAD), with both digests reproduced by download and local
hash. Also checked: the release config, the publication record, `results/registry.jsonl`,
`docs/publishing/`, the completion report, and the CI workflow (which has no manifest or upload
step). Not accessible: the gitignored local staging directory, the training host that built the
corpus, and any CI artifact.

**A previous misreading is corrected here.** An earlier attempt to reproduce the digests reported
`RepositoryNotFoundError` for every revision. That was a defect in the attempt, not a fact about the
artifact: `hf_hub_download` defaults to `repo_type="model"`. With `repo_type="dataset"` both
download fine. Two further traps are worth recording because they are the same rule the provenance
module enforces: **abbreviated revisions are not immutable identifiers** (12-character prefixes were
rejected; full 40-character revisions were not), and `repo_info` succeeding does not imply the file
is retrievable.

**Non-vacuity is now a first-class invariant.** Contract v2 requires every gate to prove both that
its assertions passed *and* that they executed. Two checks in this repository were doing neither:

- `validate_freeze` looked for the manifest in `processed_dataset_hash.source` and
  `findings.release_manifest`. The canonical records name it in `source_repository`, so every corpus
  was skipped and the gate reported PASS by executing nothing. (Already recorded in section 10.)
- `python -m opengrad.registry.validate` had no `__main__` guard, so it imported, ran nothing and
  exited 0. **Any earlier report in this errata, or in review, that this validation passed via `-m`
  was vacuous.**

Every gate now reports `discovered / checked / passed / failed / blocked / skipped`, asserts
`discovered == checked + blocked + skipped` and `checked == passed + failed`, and fails when a
required population discovers nothing or when a counter does not add up. That assertion caught two
further defects during this work: a deferred candidate was excluded from the census it was skipped
from, and a sentinel skip was recorded without being discovered. A gate cannot skip what it never
saw.

**Verifier contract.** `publication_verifier_contract: 2`, defined in
`src/opengrad/verification/__init__.py` and reported by `scripts/verify_publication.py --json`. It
adds executable module validation, correct canonical freeze discovery, non-vacuity enforcement,
immutable provenance enforcement, reconstructed-event validation, and explicit skipped/blocked
accounting. **A PASS recorded under v1 does not mean what a PASS under v2 means**, because v1 could
pass without running. Historical results are left as written and are not re-stated as v2 results.

**What changed forward.** `canonical_v2`'s identity is anchored to the immutable published artifact
(`processed_dataset_hash.remote_identity`: repository, revision, path, digest), replacing a
`source_repository` that pointed at a gitignored path present in no revision. The local freeze gate
defers that one corpus and names the deferral; a new **remote freeze-identity gate** downloads the
named file at the named revision and hashes it. Offline it is `BLOCKED_NETWORK` and the run is not a
pass. No frozen artifact was rewritten, and the four forward corrections from section 9 are
unchanged.

**Still open, and not blocking the identity.** The published manifest at HEAD declares
`hub_repository: unpublished/OpenGrad-ToolPolicy-Canonical-v2-partial` — the live published artifact
names itself as unpublished. Nothing in the repository accounts for which step rewrote
`hub_repository` from `unpublished/…` to `arrochi112/…`; the commit message that documents the
rebuild (`30b0e28`) mentions only `opengrad_git_commit`. This does not affect which artifact each
digest identifies, so it is recorded here rather than resolved by preference.

How to verify: `python scripts/verify_publication.py` exits 0, and `--json` reports
`verifier_contract: 2`. Offline it exits 2 with `BLOCKED_NETWORK`, because two required populations
cannot be resolved without the network. The previous section's instruction that it must exit 1 is
superseded by this one.

## 12. Amendment `study_002_prereg_v2`: P-DET reference labels from a declared model

**Added 2026-09-15.** Study 002's P-DET protocol names "human P-DET gold" as the reference the ETL
classifier is validated against
([`docs/research/study-002/22-PDET-PROTOCOL.md`](../docs/research/study-002/22-PDET-PROTOCOL.md) §6–§7).
With current resources, human annotation of all 581 items is not feasible, so it is deferred.

**The reference is now a composite:**

- human labels where they exist: Pass A, items #1–#11;
- labels from the declared model annotator `model.claude-opus-5` everywhere else.

**What stays the same:** no frozen file changes, and no threshold, population or rubric changes.

**What changes for claims:**

- Model labels are recorded and exported as model judgments, never as human labels.
- Every P-DET metric states its label sources.
- A classifier qualification measured against this reference is `MODEL_REFERENCE` and provisional.
- No inter-annotator agreement may be claimed.

No classifier had been scored on P-DET when this was decided, so there is no earlier result to re-state.

The full record, including the model procedure and its limits, is
[`docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md`](../docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md).
Where this file and 22 disagree about the source of P-DET reference labels, 28 is the correct one.


## 13. Amendment `study_002_prereg_v3`: the P-DET rationale becomes optional

**Added 2026-09-15.** The P-DET annotation instrument
([`docs/research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md`](../docs/research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md) §2)
marks `annotator_rationale`, one sentence saying why, as required. Writing it for every item made
continued human review too slow, so it is now **optional**. An UNKNOWN label still needs an ambiguity
status.

**This is an ergonomics change, not a taxonomy change.**

- No frozen file changes, and 23 is not edited.
- Labels, definitions, fields, values, constraints and the population are unchanged.
- Every rationale already written is kept: the 11 human labels and the 570 model judgments.
- A label without a rationale is not lower-confidence because no rationale was written.

**How it was recorded.** The task store accepts only a declared relaxation. Its hash-chained definition
history records the change from definition `0f060bb3…` to `c00eab8d…` (only `annotator_rationale.required`
differs), with 581 annotation records existing at the time.

**Also, as tooling rather than protocol:**

- **Review order.** A pinned `priority-review` queue sets the order in which the human pass may review
  items. Model judgments stay hidden from the human pass.
- **Audit trail.** The model-label audit trail is archived in `reports/pdet/provenance/model-a/`.

Nothing is frozen, and no classifier had been scored when this was decided.

The full record is
[`docs/research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md`](../docs/research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md).
Where this file and 23 §2 disagree about whether a rationale is required, 29 is the correct one.

## 14. Finding: P-DET-v1 contains no CALL and no DIRECT item, so it cannot authorise C1

**Added 2026-09-15**, after Pass A labeled all 581 P-DET-v1 items (one annotator, not frozen). The human
labels are:

| Label | Items |
|---|---:|
| UNSUPPORTED | 298 |
| CLARIFY | 281 |
| UNKNOWN | 2 |
| DIRECT | 0 |
| CALL | 0 |

The report on the frozen population
([`docs/research/study-002/24-PHASE-3-REPORT.md`](../docs/research/study-002/24-PHASE-3-REPORT.md) §7)
left the per-mode distribution **UNKNOWN** until annotation. It is now known, and it rules out two modes.

**Cause: the source.** The whole candidate population is When2Call `train_sft`, 14,829 records
([`22-PDET-PROTOCOL.md`](../docs/research/study-002/22-PDET-PROTOCOL.md) §5).

- **No tool calls.** 0 of those 14,829 carry a tool call. The frozen builder's call-payload predicate
  also matches 0 responses.
- **No direct answers.** Every reply asks for information or declines.
- **By design.** In When2Call's own test set, the correct answer is never `direct`.

The sampling and the annotation did not remove anything.

**Consequence under the frozen rules.** No classifier can be granted balancing permission for CALL or
DIRECT on P-DET-v1, because the coverage rule (22 §5) is not met for either mode. The DIRECT and CALL
thresholds of 22 §6 cannot be measured. "DIRECT … failing → C1 is not authorised" therefore applies:
**P-DET-v1 cannot authorise C1.**

**What would cover these modes.** A separately frozen validation population drawn from sources that
contain tool calls and direct answers. That is a design decision that has not been made.

The details, and the 11 items where the human and the superseded model labels differ, are in
[`docs/research/study-002/README.md`](../docs/research/study-002/README.md).


## 15. Amendment `study_002_prereg_v4`: P-DET-COVERAGE-v1 adopted as a second validation population

**Added 2026-09-16.** §14 found that P-DET-v1 contains no CALL and no DIRECT item, so no classifier can
be qualified for those modes on it, and C1 cannot be authorised. The study owner adopted
[`docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`](../docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md)
as the population that covers them.

**What it fixes:**
- the population design: seed, strata, quotas, dedup, exclusions and blinding;
- the acceptance rules, with minimum sizes of 50 gold (recall), 50 predictions (precision), 30 hard gold
  (challenge) and 20 DIRECT predictions per source;
- its input: normalization-v3 fingerprint `60d3123e…`.

**What stays the same:**
- P-DET-v1 and its labels;
- 22 and 23;
- every study threshold, partition, arm, seed count and arm corpus.

No frozen file changes. Adopted before the population was drawn and before any label existed, and no
classifier had been scored on either population.

The full record is 30, and the amendment entry is in `docs/research/study-002/03-PREREGISTRATION.md`.


## 16. Amendment `study_002_prereg_v5`: P-DET-COVERAGE-v1's reference comes from a three-model consensus

**Added 2026-09-17, before any P-DET-COVERAGE-v1 label existed.** §15 adopted P-DET-COVERAGE-v1 with a
human-only reference (30 §10). The study owner will not annotate, and no human annotator is available.

**The reference is now a three-model consensus:**
- three declared non-Claude models label every item independently and blind: Gemini 3.8 Flash (High) via
  the Antigravity CLI, gpt-5.6-sol via the Codex CLI, and deepseek-v4.1-flash via the Cline CLI;
- an item's reference label is the one at least two give;
- items where all three differ are `NO_CONSENSUS`, reported and excluded from metrics.

Claude is excluded because it builds the classifier under test.

**What stays the same:** the population, the rubric, the blinding, the acceptance rules and minimum sizes,
and P-DET-v1.

**What changes for claims:** metrics name the three-model reference and are never called human-validated;
qualifications are `MODEL_REFERENCE` and provisional; agreement is reported as model–model agreement.

The full record is
[`docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md`](../docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md).


## 17. Amendment `study_002_prereg_v6`: first replies, P-DET-COVERAGE-v2 and classifier v2

**Added 2026-09-17, after `prose-decision-classifier-v1`'s one-shot test, before anything was drawn under it.**
v1 qualified for UNSUPPORTED and CLARIFY but not DIRECT, so C1 stayed unauthorised
(`docs/research/study-002/33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md` §8). Counts-only audits then showed
why no single-exchange population could fix that: about 22 DIRECT items remain in the unused single-exchange
pool, while Study 001's training corpus holds thousands of direct answers as the first reply of conversations
that continue, a shape the classifier input contract excluded (35 §4–§6).

**What changes:**
- contract `prose-decision-input-v2` admits the first assistant reply of any record, with the same features;
  later turns are never read;
- P-DET-COVERAGE-v2, 420 first replies, becomes the validation population for `prose-decision-classifier-v2`;
- P-DET-v1 and P-DET-COVERAGE-v1 are development-exposed for v2.

**What stays the same:** every threshold and minimum size, the C1 rule, the arms, P-DET-v1, P-DET-COVERAGE-v1,
contract v1, and classifier v1 with its recorded result.

**Correction recorded alongside:** 35 §5 first claimed the whole corpus held almost no direct answers. It had
counted single exchanges only; 35 §6 corrects it.

The full record is
[`docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md`](../docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md).


## 18. Amendment `study_002_prereg_v7` (recorded late)

**Added 2026-09-24.** `study_002_prereg_v7` (2026-09-18) was recorded in
`docs/research/study-002/03-PREREGISTRATION.md` but not here, although step 2 of the amendment procedure
(03, "Amendments") requires both. It was also placed under "Registration of the unit of analysis" rather than
under "Amendments"; 03 now points to it from there.

**What changed under v7:** the two permissions 22 §6 and 35 §1 left to the study owner were taken under the
owner's explicit delegation (38): balancing permission, provisionally, for UNSUPPORTED and CLARIFY on both
layer B sources, DIRECT on glaive only, and not CALL; and C1 (the canonical-v3 build workstream) authorised
past its classifier gate, under 38 §3's conditions.

**What stays the same:** every threshold and minimum, the C1 rule, the arms, the seeds, the partition, both
input contracts and every population. No training run is authorised; no gold is frozen.

The full record is
[`docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md`](../docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md).

## 19. Study 002 gate contract 2, and corrections to Study 002 prose

**Added 2026-09-24,** from an adversarial review of the repository. No arm has been scored, so no verdict
changes.

### `src/opengrad/verification/study_002_gate.py`

- **As written (contract 1):** the fourteen checks of `11-THRESHOLDS.md`. **Correction:** contract 1 did not
  implement them as written. Probed with one-field mutations of its own healthy fixture, it returned PASS for
  candidates `tool_use_promotion_v5` returned REJECT on (clarification accuracy 0.10 against a 0.50 floor;
  unsupported accuracy 0.05 against 0.30; answer rate 0.98 → 0.60, a 0.38 drop against 0.30; call recall
  0.99 → 0.77 against a 0.10 regression bound), because it read seven v5 dimensions and never v5's
  `decision`. It also passed an `ANSWER` mode of n = 5 (below the n ≥ 200 floor of 06 §C2), a census that
  scored nothing or reported 500 failed items, a bundle that declared its own one-item sentinel and
  provenance lists, a comparison row with no margin or n ≤ 0, and every comparison `WITHIN_NOISE`. On an
  empty bundle it returned FAIL, not the `BLOCKED_INPUT_MISSING` that 11 and 16 stated. It never returned
  PASS on an empty bundle.
  **Contract 2** fails whenever v5 does not promote, enforces the floor, pins the sentinel list (08) and
  provenance fields (15 V3, V9) in code, requires the census to score every gold item once, fails unresolved
  rows and a comparison set that resolves nothing, and fails a missing or non-numeric metric instead of
  crashing. Check 12's "declared factor" and the `P-UNANS` size behind check 4 are declared nowhere in the
  preregistration, so the gate blocks on them; `docs/research/study-002/40-PREREG-V8-DRAFT.md` proposes values
  (not adopted). **Source:** `tests/verification/test_study_002_gate.py`.
- `src/opengrad/promotion/tool_use_policy.py`, `PromotionPolicyV5`. **As written:** "v4 plus". **Correction:**
  it subclasses the v3 class (`PromotionPolicyV2`); v4's parent-relative floors are not part of it. v5's
  behaviour is unchanged. Three ways v5 can promote without measuring (no confusion matrix; an absent
  `refusal_correctness` or baseline `answer_rate`; float error at a threshold, where `0.90 − 0.60` fails
  `≤ 0.30`) are closed in a new `tool_use_promotion_v6`, which no gate uses unless the owner adopts the v8
  draft. **Source:** `tests/evaluation/test_tool_use_promotion_v6.py`.

### `docs/research/study-002/11-THRESHOLDS.md`

- §`tool_use_promotion_v5`. **As written:** "v5 is v4 **plus**", with a "v3/v4 value" column. **Correction:**
  the column holds v3 values. v4 (`src/opengrad/promotion/m1_calibration.py`) sets `min_call_precision` 0.65,
  `min_call_recall` 0.60, `min_macro_recall` 0.60, `min_clarification_accuracy` 0.60 and
  `min_unsupported_accuracy` 0.40. v5's values, and the code, match the v5 column exactly.
- Threshold register and the paragraph under it. **As written:** `max_over_call_rate` on 824 items has a
  6.9pp resolvable margin, "below the 10pp floor", so small changes are "not adjudicable"; CLARIFY at 371
  (10.2pp) is "adjudicable: yes". **Correction:** `resolvable_margin(824)` is **6.83pp** (6.8pp). The
  direction is reversed: a smaller resolvable margin is finer resolution. Under the rule "no claim on a row
  that cannot resolve 10pp", over-call on 824 items is adjudicable and CLARIFY at n = 371 (10.18pp) is not; it
  needs n ≥ 385.

### `docs/research/study-002/06-SPLIT-SPEC.md`

- §C2. **As written:** the resolvable margin is "twice the Wilson half-width at p = 0.5". **Correction:** the
  formula used, `z·sqrt(0.25/n)`, is the normal-approximation (Wald) half-width. Wilson's is slightly smaller
  (13.73pp against 13.86pp at n = 200), so every margin stated is conservative, and no conclusion changes.
- §C2 status note. **As written:** "≈384 for the 10pp floor". **Correction:** 385 (n = 384 gives 10.002pp).
- Same note. **As written:** `frozen_behavioural_eval_v1.jsonl`. **Correction:** the file is
  `results/quantization/frozen_behavioral_eval_v1.jsonl`.
- Same note. **As written:** "≈37,000 first-exchange `DIRECT` prompts in canonical-v2-final".
  **Correction:** 37,512 is the number of prose first answers with *no tools offered*, not of DIRECT items.
  Frozen classifier v1 *predicts* DIRECT on 33,831 of them (33,756 distinct user messages), and on 9,455
  first answers with tools offered: about 43,300 in total. These are classifier predictions, not gold. The
  conclusion, that the only large DIRECT pools are the training corpora and the disjointness rule bars them,
  is unchanged. **Source:** `docs/research/study-002/35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md` §6.
- Same note. **As written:** the QAD corpus (140) and its calibration sibling (100) are "the largest
  non-training `ANSWER`-gold pools" and, in the same sentence, "both training artifacts". **Correction:**
  they are training-derived pools; the survey found no non-training pool at or above n = 200. The survey
  also did not list `m1_v2_imatrix_calibration_v2` (ANSWER 300), which is training-derived and equally
  barred.

### `README.md`, `ROADMAP.md`

- README "+0.0078 call_f1, 7 of 453 calls" (and ROADMAP step 10). **Correction:** the 7 of 453 is the
  *recall* difference (0.0155 × 453 = 7.0 more gold calls recalled); +0.0078 is the `call_f1` difference.
  Both are within noise, as stated.
- ROADMAP step 16. **As written:** "A CPU audit of the supervision M0 trained on — the published
  Canonical-v2 final corpus — finds 18,114 of 173,237 records". **Correction:** 173,237 is the released
  corpus; M0 trained on 161,966 of them. The counts are exact for M0's training only for When2Call (4,038 of
  6,505) and xLAM (0); Glaive's 14,066 and ToolACE's 10 are counted over the release, of which 97,112 and
  2,259 records were trainable. **Source:** `m0_training_admission` in
  `results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json`.

## 20. Study 001: an in-place edit after the freeze

**Added 2026-09-24.** Study 001 was frozen on 2026-09-13 at tag `study-001`. Commit `c59bc16`
(2026-09-18) edited `reports/FINAL_CAMPAIGN_AUDIT.md` §4 in place: the heading "Superseded results —
preserved, with one gap" became "Superseded results — preserved", and M1-v1's MMLU-Pro @768 accuracy cell
changed from "—" to **33.0%**, after the pass was recovered from the Modal volume (claim audit #41).
`docs/research/STUDIES.md` says a number in a frozen study is corrected here, not in place. The file is not
in `PRESERVED_STATE_v1.json`, and the commit message disclosed the edit and the recomputation, but no entry
was made here.

**The numbers, as recomputed by `scripts/score_mmlu_pro.py`:** accuracy 33.0%, answer rate 52.8%, accuracy
given answer 62.5%, 5,978 of 12,032 truncated (49.7%) at `max_tokens` 768, matching the cost ledger's run
seq 9. M1-v1's MMLU-Pro at the corrected 2,048 budget remains **UNMEASURED**. The tag `study-001` keeps the
text as frozen. **Source:** the `superseded_mmlu_768/mmlu_pro_scores.json` that commit added.


## 21. Preserved state: a successor manifest over committed blobs, and the confirmatory deltas

**Added 2026-09-24.**

### `results/benchmarks/h200/PRESERVED_STATE_v1.json`

§6 recorded that v1 pinned the author's CRLF working tree and could not be verified from a clone, "until
… a successor manifest pins committed blobs". Nothing ran `--verify` in CI, and since commit `d28beff`
rewrote the Windows working copies of `*.jsonl` to their committed LF bytes, `--verify` failed on
`results/quantization/findings.jsonl` on the author's machine too. No artifact changed: the committed
blob of every file is the one v1 preserved.

**The successor:** `results/benchmarks/h200/PRESERVED_STATE_v2.json` pins the committed blob of each
tracked artifact and records, for each, which line-ending rendering reproduces its v1 digest: 14 CRLF and
1 LF (`results/benchmarks/openweights_parity_cases_v1.json`), exactly as §6 counted. The two gitignored
artifacts are recorded `in_clone: false`. `reports/H200_BENCHMARK_RUN.md` stays append-only: v1's
`original_sha256` is the CRLF rendering of the first 8,909 LF bytes of the committed file, now pinned as
that prefix. v1 is unchanged and remains the historical record.
`python scripts/preserve_h200_state.py --verify` now checks v2 and runs in CI through
`tests/results/test_preserved_state.py`, which also checks every tracked `.sha256` sidecar (43) against
its committed file. `--verify-v1` keeps the old working-tree check.

### `results/registry.jsonl`

- **As written:** each confirmatory row's `deltas`. **Correction:** they are measured against the
  **full-set** B0 `call_f1` 0.6191, not the confirmatory B0 0.6264, because they are copied from each run's
  `baseline_comparison`. M1-v2's confirmatory `call_f1` delta is therefore recorded as +0.1357; against the
  confirmatory B0 it is about +0.128. M0 final's is recorded as +0.1279 (about +0.121). The file is pinned by
  `reports/data/m0-final-freeze.json` and is not edited. No prose quotes these deltas; the README and reports
  quote the absolute confirmatory values, which are correct.


## 22. A credential string in upstream data, and local paths in pinned audit trails

**Added 2026-09-24.** Neither changes a number. Both are in hash-pinned evidence, so the bytes are kept and
the facts are recorded here.

### `reports/pdet/pdet-v1.population.jsonl`, line 271

- **What it is:** a string in GitHub personal-access-token format, the whole value of
  `tools[0].parameters.properties.key.default` in a When2Call row (`pdet_id` `when2call-sft:736b54c6…`,
  a tool described as reading a GitHub repository folder). It came with the upstream NVIDIA When2Call
  training data. It is not an OpenGrad credential, and the prompt and response do not use it.
- **Where else:** the same row, shown to the model annotator, is in two members of
  `reports/pdet/provenance/model-a/pdet-v1.model-a.audit-trail.tar.gz` (`model-a/batch-06.json`, `.md`). It has
  been in history since `308b4ca`. No other tracked file or revision carries a live-format credential.
- **What was done:** the token was reported upstream on 2026-09-24 by the study owner. The population is
  frozen P-DET evidence whose sha256 is pinned in about fifteen places (its manifest, the task config, three
  code constants, tests and derived reports), and redaction would not remove it from history, so the owner
  kept the bytes. `scripts/repo/check_publication_hygiene.py` now reads `.jsonl` files and tarball members and
  recognises `ghp_`, `github_pat_`, `hf_`, `AKIA` and private-key patterns; these three occurrences are listed
  in `scripts/repo/publication_hygiene_allowlist.yaml` with exact counts, so any new credential fails CI.

### Local paths

- **As written:** nine audit-trail manifests under `reports/prose-classifier/*/provenance/` record the Read
  tool's absolute `file_path` (149 occurrences), and five audit-trail tarballs under `reports/pdet*/` and
  `reports/prose-classifier/v2-devcheck-1/` record each external CLI run's `argv` (255), both under the
  author's Windows user folder. `.gitignore` kept the raw transcripts out because they carry local paths; the
  manifests and tarballs that are tracked carried them anyway. The hygiene scanner reported the manifests and
  could not read the tarballs, and CI never ran it.
- **What was done:** the files are pinned by `.sha256` sidecars and kept. The writers now record portable
  paths (`portable_path` in `src/opengrad/registry/provenance.py`, used by
  `scripts/archive_devset_model_labels.py` and `scripts/run_external_annotation.py`): repo-relative inside the
  repository, `<outside-repo>/<name>` outside it. The existing occurrences are allowlisted with exact counts,
  and the scan runs in CI.


## 23. Licensing records: the registry, the model cards and the M1 preference pairs

**Added 2026-09-24.** No released file changes; the corrections are to the registry and to the local copies of
the model cards. The copies on Hugging Face still carry the old text until they are re-uploaded (G15), which is
a separate, owner-approved step.

- `registry/datasets.yaml`. **As written:** `redistribution: NOT_ASSESSED` for When2Call, ToolACE, BUTTON,
  LoopTool-23k and Glaive. **Correction:** all five were assessed on 2026-09-04 in
  `docs/publishing/source-redistribution-audit.md` as `REDISTRIBUTION_WITH_ATTRIBUTION`, and the published
  `source-licenses.md` files say so. The registry, which is meant to be the source of truth, was never updated;
  it now records the status and its basis. BUTTON and LoopTool were licence-checked at their pinned revisions on
  2026-09-02; their upstreams later became unavailable (HTTP 401 / not located), which is why they are absent
  from v2, not a change in their licences.
- `release/huggingface/qwen35-2b-*/README.md` (five weight cards). **As written:** `license: other`,
  `license_name: composite-per-source`, with no licence link and no statement of the base model's terms.
  **Correction:** the weights derive from `Qwen/Qwen3.5-2B` (Apache-2.0), whose license and notices carry to a
  derivative. Each card now links the licence at the pinned revision (`license_link`) and states the per-source
  data terms. The GGUF card's generator (`scripts/build_gguf_card.py`) renders the same section.
- `release/huggingface/qwen35-2b-m0-sft-corpusv1-evaluation/README.md`. **As written:** `license: apache-2.0`.
  **Correction:** the record's predictions contain Canonical-v1 prompts, including CC-BY-4.0 sources, so the
  card is `other` / `composite-per-source` with the per-source terms stated.
- `data/processed/m1_calibration_preference_pairs_v1.jsonl`. **As written:** `redistribution:
  INTERNAL_RESEARCH_ARTIFACT`, `downstream_access_requirement: private`. **Correction:** the file has been in the
  public repository since `92ca2b9`, so it is public in fact. Its rows derive from When2Call (CC-BY-4.0) and from
  prompts of the canonical sources, all redistributable with attribution, and 5 rows carrying credential-like
  strings were excluded when it was built. The registry now says `PER_SOURCE` and `public_allowed`.
- `hf/MODEL_CARD_TEMPLATE.md` gains two required fields: general-capability regressions against the base model,
  and license and attribution. **Source:** `tests/publication/test_model_cards.py`.

## 24. `study_002_prereg_v8` adopted, and git history kept as it is

**Added 2026-09-24.** Two owner decisions, recorded together because both change what the record may be read as.

- **`study_002_prereg_v8` adopted, items A–D as drafted** (`docs/research/study-002/40-PREREG-V8-DRAFT.md`). It
  declares the two values the preregistration required but never quantified: check 12's truncation factor
  (imbalanced when two arms differ by more than 2pp **and** by more than 2.0×) and `P-UNANS` n ≥ 385. It switches
  `study_002_gate_v1` from `tool_use_promotion_v5` to v6 at gate contract 3. Recorded in
  `docs/research/study-002/03-PREREGISTRATION.md` at the end of the file (so cited line numbers hold) and here, in the
  same commit (G15). No candidate was scored and no arm launched under v7's gate, so nothing is re-run. **Code:**
  `ADOPTED_PARAMETERS` and `STUDY_002_GATE_CONTRACT = 3` in `src/opengrad/verification/study_002_gate.py`.
  - **As written** in §19 and 11/16: "the gate blocks on them until the owner adopts values" and "wraps v5". Both
    were true until this entry.
- **Git history is not rewritten** to remove the upstream credential string of §22 or the local paths. Rewriting
  would have to change the bytes of `reports/pdet/pdet-v1.population.jsonl`, whose sha256 (`6ab92087…`) 24 tracked
  files pin, so P-DET-v1 would stop being the population they describe. It would also give every commit from
  `308b4ca` onward a new id. That moves the pushed tag `prose-decision-classifier-v1`, and breaks 16 short commit
  ids cited in study-002 documents 33, 37 and 38, in this file and in a test. Several of those ids are the evidence that a
  rule was committed before the check set it governs was drawn: commit order is the preregistration's proof. The
  string is NVIDIA's, it is public in their When2Call data, and it has been reported to them; a rewrite would
  not remove it from existing clones. The publication hygiene scan (`scripts/repo/check_publication_hygiene.py`)
  fails on any new occurrence.

## 25. `bfcl-v4` recorded as contamination-`CLEAN` from a scan of placeholders

**Added 2026-09-24.** `reports/data/benchmark_contamination_registry.json` is a living registry, not a pinned
artifact, so it is corrected in place and the correction recorded here (G15).

- **As written** (since 2026-09-10): `bfcl-v4` had `scan_status: CLEAN`, every level including level 5
  `COMPLETED`, and `training_corpus_fingerprint: "sample-or-materialized-fingerprint"`.
- **What that scan was.** `opengrad-benchmark contamination-scan` loaded ten tasks from the BFCL adapter, which
  synthesizes placeholder tasks (`docs/evaluation/BENCHMARK_STRATEGY.md` section 6 already says no real BFCL data
  has been scored). Given no `--training-data`, it compared them with two built-in fixture prompts. The scanner
  then set level 5, the human audit, to `COMPLETED` itself, and called an empty queue `CLEAN`. No real benchmark
  was compared with any real training corpus.
- **A second defect in the same scanner.** Its level 1 ("exact") normalised whitespace and case before hashing,
  so it was not the level 1 of the policy or of the held-out screen, which hashes the raw text.
- **Correction.** `bfcl-v4` is `UNSCANNED`, and every entry carries the same five level names, all `NOT_RUN`.
  The benchmark scan is now a layer over the held-out screen's engine (`src/opengrad/contamination/levels.py`);
  it reads real benchmark and training files and records their sha256. The registry refuses an entry with a
  non-sha256 fingerprint, with level 5 anything but `NOT_RUN`, or with a status of `CLEAN`.
  **Source:** `tests/benchmarks/test_contamination_scanner.py`.
- **What does not change.** No claim in a report rested on this entry: Study 001's contamination evidence is the
  behavioural held-out screen and its human audit (`reports/data/behavioral-heldout-v2-contamination*.json`),
  whose report the refactor leaves byte-identical (`tests/contamination/test_levels.py`).
