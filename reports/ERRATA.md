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
