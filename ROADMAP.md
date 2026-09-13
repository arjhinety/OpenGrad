# OpenGrad roadmap

OpenGrad proceeds from infrastructure to controlled measurement. A later phase is not considered successful without reproducible evidence and regression analysis.

## Completed pre-GPU foundation

0. Repository and research infrastructure — COMPLETE
0.5. CPU-only pre-experiment validation — COMPLETE
0.6. Deterministic CPU baseline pipeline hardening — COMPLETE (GPU execution has since run; see steps 6–10)
1. Data source audit and provenance registration — COMPLETE
2. Canonical normalization and quarantine policy — COMPLETE
3. Accessible corpus materialization, overlap, contamination, and coverage audits — COMPLETE
4. Exact Qwen3.5-2B rendering and token audit — COMPLETE
5. Frozen held-out evaluation preparation — COMPLETE

Current evidence (Canonical-v2 era):

- **Canonical-v2 final** is the active training corpus: 173,237 canonical records, 161,966 trainable under the two supervision contracts, across four sources, corpus fingerprint `8ced403b…`. BUTTON and LoopTool stay excluded because their upstreams are unavailable.
- The frozen behavioral held-out contains 3,652 distinct items; 3,650 were scored after two quarantines. The v1-era planning figures — 213,951 canonical records and a 3,952 pre-deduplication split sum — are historical, and are labelled as such in the publication milestone below.
- Model inference, training, and model-quality results now exist: the B0 baseline and **seven post-training interventions**, including the promoted M1-v2 DPO (see [Results](README.md#latest-results)). The machine-checkable per-record view is [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md).
- Canonical dataset publication — v1 (213,951 records, six sources) and Canonical-v2 final (173,237 records, four sources) are published and verified; the 103,036-record partial v2 snapshot is retained as historical evidence, not as the current corpus.

## Publication milestone

Canonical dataset publication — CANONICAL_DATASET_PUBLISHED

- [`arrochi112/OpenGrad-ToolPolicy-Canonical-v1`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v1) (historical) is public and verified at Hub commit `bb295d8a4ad64f7e8161044ad2fa34f873ede418`. The release contains 213,951 legally cleared canonical SFT records. xLAM is included under CC BY 4.0 with attribution and APIGen citation; its upstream access gate is not reproduced downstream. Publication metadata is recorded in `reports/releases/toolpolicy-canonical-v1-publication.json`.
- [`arrochi112/OpenGrad-ToolPolicy-Canonical-v2`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2) (current) carries 173,237 canonical records and 161,966 trainable ones across four sources, with fingerprint `8ced403b…` proven reproducible by a delete-and-rebuild. Publication metadata is recorded in `reports/releases/hf-publication-2026-09-11.json`.
- [`arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot) (historical, partial) is the 103,036-record, three-of-six-source corpus behind the first successful M0. It is distinguished from the final v2 by name, fingerprint and source count, and is not the current corpus. Its publication metadata is recorded in `reports/releases/toolpolicy-canonical-v2-publication.json`.

## Empirical sequence

6. B0 unmodified Qwen3.5-2B baseline inference — **REAL_RESULT**
   Executed on an A100 (engine vLLM 0.29.0) over the frozen behavioral-heldout-v2 evaluation:
   3,650 distinct examples, 196 s, from a clean traced tree. See
   [the B0 result](reports/baselines/qwen35_2b_baseline/RESULT.md). External benchmark
   families were FROZEN_NOT_EXECUTED at this step; IFEval, GSM8K and MMLU-Pro have since been
   executed in the H200 capability diagnosis (step 16). The other 14 remain frozen.

7. Freeze B0 evidence and generate the residual profile — **COMPLETE**
   Predictions, metrics, environment, residual profile, and the canonical experiment record
   are recorded and committed. The engine, renderer, evaluator, generation configuration,
   and failure taxonomy are pinned alongside them.

8. Controlled SFT comparison — **EXECUTED — CLOSED**
    The frozen M0 lineage includes the v1 negative, partial-v2, definitive final-v2, and the paired
    fixed-compute/matched-exposure minus-xLAM arms. Final-v2 recovered call recall but was not
    promoted; both minus-xLAM arms substantially lost recall. Because xLAM is currently the only
    CALL_PREDICTION source, those arms measure joint removal of xLAM and that supervision channel,
    not a pure xLAM-content effect. See [M0 phase closure](reports/M0_PHASE_CLOSURE.md).

9. Full post-training evaluation and diagnosis — **EXECUTED**
    M0 and both minus-xLAM arms were fully evaluated on the frozen DEV partition, selected under
    the frozen balanced rule, and scored once on the pre-registered internal confirmatory partition.
    External benchmark families were `FROZEN_NOT_EXECUTED` at this step; IFEval, GSM8K and
    MMLU-Pro were later executed on Base → M0 → M1-v2 (step 16), and the other 14 remain frozen.
    Tool-selection and argument/schema validity are still unmeasured by the current evaluator.

10. Preference optimization or on-policy distillation — **DPO EXECUTED / DISTILLATION NOT JUSTIFIED**
     M1-v2 DPO started from the selected M0-final checkpoint, preserved its calibrated frontier, and
     was promoted under `tool_use_promotion_v4`. M1-v2 is promoted under a parent-relative gate
     (v4) introduced after M0 was evaluated; M0 also clears v4, and M1-v2 fails the v3 gate that
     rejected M0. The promotion reflects the gate change; the measured difference from M0 (+0.0078
     call_f1, 7 of 453 calls, single seed) is within noise. The first M1 identity is preserved as
     a 119/120-step failure. The existing M2 path is mock-only: it has no live teacher, student
     update, parent checkpoint loading, or valid prompt-state dataset. M1 already passes the
     intended measured calibration policy, so launching M2 would neither answer a distinct valid
     question nor justify its infrastructure gaps. See the [M1 evaluation report](reports/M1_DPO_EVALUATION.md)
     and [M2 decision](reports/M2_DECISION.md).

11. Cross-model replication — PLANNED

12. Quantization and runtime evaluation — GGUF PTQ EXECUTED AND CLOSED; ExecuTorch EXPORTED
    A nine-rung GGUF PTQ ladder was built from a BF16 GGUF and scored on the confirmatory partition.
    Q6_K (1.45 GiB) is the pre-registered release and Q8_0 (1.87 GiB) also passes
    `quantization_preservation_v1`, but the gate sits inside rerun noise
    ([PTQ closure](reports/PTQ_PHASE_CLOSURE.md), [errata](reports/ERRATA.md)). ExecuTorch
    CPU/XNNPACK fp32 and 8da4w were exported and audited, with no behavioural verdict; Snapdragon is
    `REJECTED_EXPORT` and MediaTek `BLOCKED_PORT_INCOMPLETE`. The separate optimization producer
    interface ([`src/opengrad/optimization/`](docs/optimization/README.md)) is still INTERFACE_ONLY:
    no ModelOpt optimization has been executed. See
    [the ModelOpt integration report](reports/MODELOPT_INTEGRATION_REPORT.md).

13. OpenWeights-derived downstream deployment studies — PARITY SUITE RUN; NO DEVICE STUDY
    OpenWeights is an independent project; its observations motivate hypotheses but are not OpenGrad results.
    The 7-case OpenWeights ParitySuite was run on the H200 against the promoted checkpoint and the base
    ([report](reports/OPENWEIGHTS_TRANSFER_EVALUATION.md), [errata](reports/ERRATA.md)); both score 5/7,
    so it shows no transfer difference. No post-training on-device study has been executed, so no
    OpenGrad result rests on device measurements.

14. Speculative decoding and inference research — PLANNED / GPU_REQUIRED
    Future comparisons may include autoregressive decoding, external draft speculation, Medusa, EAGLE-3, DFlash, DSpark, and native MTP where supported. A reserved configuration exists; no method is currently implemented, benchmarked, or supported by OpenGrad.

15. Joint capability-efficiency optimization — PLANNED

16. Refusal-supervision ablation — **PLANNED / BLOCKED_ON_PREFLIGHT**

    Diagnosis (executed, see [general-capability regression](reports/GENERAL_CAPABILITY_REGRESSION.md)):
    the post-SFT checkpoints refuse **100% of bare GSM8K questions** while solving 55.5% of the
    *same* questions with 8 exemplars, and refusing **0%** of 5-shot MMLU-Pro. The regression is
    not specific to SFT: M1-v1, DPO applied directly to the base with no SFT parent, refuses
    **70.7%** (933/1,319) of the same zero-shot questions. It is a general-capability regression
    associated with tool-policy post-training on When2Call-derived data; causation is not
    established (one lineage, one seed, no replicate).

    A CPU audit of the supervision M0 trained on — the published Canonical-v2 final corpus — finds
    **18,114 of 173,237 records (10.5%) whose single-exchange supervised target is a refusal and
    whose decision label is `ANSWER`**: When2Call 4,038 of 6,505 (62.1%), Glaive 14,066 of 98,339
    (14.3%), ToolACE 10, xLAM 0, and **zero** labelled `CANNOT_ANSWER`. Evidence:
    `results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json`. The
    original audit
    `results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit.json` (21,749 of 217,903,
    10.0%; `when2call-sft` 7,490 of 14,829) measured the normalization-v1 sources, not M0's corpus,
    and should not be quoted for M0; see the [errata](reports/ERRATA.md).

    The hypothesis is that this supervision teaches "answering looks like declining", and that the
    behaviour over-generalises to questions the model can answer. **It is a hypothesis. Nothing
    below may be executed as though it were established.**

    **The defect is the LABEL, not the refusal.** Many of these refusals are correct — a model
    genuinely cannot report today's trending topics or query a VIN database. Deleting them would
    trade over-refusal for hallucination, which is harder to detect and worse. The intervention
    under consideration is *relabelling* refusal-targeted records away from `ANSWER`, not removing
    refusals from the corpus.

    **Pre-flight gate — every item must pass before any GPU time is spent:**

    1. **Refusal-detector precision.** The 18,114 count comes from `HEURISTIC_REGEX_v1`. Hand-label
       a random sample (n >= 200) of flagged records and measure precision and recall. A detector
       at, say, 80% precision means ~3,600 records would be relabelled wrongly. Proceed only with a
       measured precision figure, not an assumed one.
    2. **Answerability triage.** Partition the flagged records into *correctly refused* (needs
       real-time data, external action, private state) and *wrongly refused* (answerable from
       parametric knowledge or arithmetic). This partition is the actual intervention design and
       does not exist yet. Without it there is no defensible relabelling rule.
    3. **Correct target label.** Decide what a correctly-refused record should be labelled and what
       its target text should be, consistent with the existing supervision contract. `ANSWER` is
       wrong; `CANNOT_ANSWER` may also be wrong if the taxonomy means something narrower.
    4. **Tool-policy regression guard.** Pre-register that the ablation must not degrade
       `call_f1`, `call_precision`, `over_call_rate` or `clarification_accuracy` beyond the frozen
       `quantization_preservation_v1`-style thresholds. The tool policy is the thing that works;
       an intervention that fixes refusal by breaking it is a net loss.
    5. **Fresh held-out evaluation.** The frozen confirmatory partition has **no ANSWER examples**,
       which is precisely why it could not detect this. A new held-out set covering ANSWER
       behaviour must be constructed and frozen **before** training, not after.
    6. **Budget.** A full M0 retrain is materially more expensive than this entire diagnosis
       ($11.39 for the campaign, of which the continuation was $8.90). Confirm actual available
       credit — not a planning envelope — before committing.
       See `cost_ledger.json:envelope_is_not_a_balance`.

    **Only then**, the experiment: retrain M0 SFT with the relabelling applied, every other factor
    held fixed (same base, seed, hyperparameters, schedule, tokenizer, template), and re-measure
    GSM8K zero-shot refusal rate, IFEval, MMLU-Pro **and** the tool-policy metrics against the
    pre-registered thresholds. A null result — "relabelling did not reduce zero-shot refusal" —
    is a publishable outcome and would falsify the hypothesis.

    **Scope discipline.** This is a new experiment family. It does not modify, supersede or
    relabel any existing checkpoint, corpus release, or frozen evaluation artifact. Canonical-v2
    and every published checkpoint stay exactly as they are.

The dependency order was intentional; the status of each stage is now:

B0 baseline                                  -> EXECUTED (REAL_RESULT)
    -> frozen baseline evidence              -> COMPLETE
    -> residual profile                      -> COMPLETE
    -> M0/M1/M2 decision                     -> M0 closed; M1 evaluated; M2 not run (mock-only path, not justified)
    -> SFT                                   -> EXECUTED (5 arms: 1 negative on corpus v1, 1 partial recovery, 1 definitive, 2 negative joint-removal ablations; 3 corpus-v1 launches FAILED before an evaluated model)
    -> post-SFT evaluation                   -> EXECUTED (see step 9)
    -> preference optimization               -> first DPO identity (on Base) rejected/not reproducible; M1-v2 DPO EXECUTED and PROMOTED under v4 (fails v3; within noise of M0)
    -> distillation                          -> NOT RUN (scaffold only; live training path unimplemented)
    -> quantization/runtime                  -> GGUF PTQ EXECUTED and CLOSED (Q6_K recommended; gate inside rerun noise); ExecuTorch exported, no behavioural verdict
    -> OpenWeights device validation         -> ParitySuite run on H200 (Base and M1-v2 both 5/7); no on-device study executed
    -> speculative decoding                  -> PLANNED / GPU_REQUIRED (no runtime support or benchmark executed)
    -> joint capability-efficiency studies   -> PLANNED
    -> general-capability diagnosis          -> EXECUTED (IFEval/GSM8K/MMLU-Pro across BASE -> M0 -> M1-v2)
    -> refusal-supervision ablation          -> PLANNED / BLOCKED_ON_PREFLIGHT (step 16)

> **Status update (2026-09-13).** Steps 12 and 13 previously read `INTERFACE_ONLY` / `no study
> executed`, which was accurate when written. As the earlier drift warning here asked, they were
> updated deliberately in their own change, to record the executed and closed GGUF PTQ ladder
> (`reports/PTQ_PHASE_CLOSURE.md`) and the OpenWeights ParitySuite run
> (`reports/OPENWEIGHTS_TRANSFER_EVALUATION.md`). The previous wording is in git history.

> **Study boundary (2026-09-13).** Everything executed above is Study 001, frozen at tag
> `study-001`. The refusal-supervision ablation (step 16), on-policy distillation (the unexecuted
> branch of step 10) and speculative decoding (step 14) are Study 002, which has no results yet.
> Cross-model replication (step 11) and joint capability-efficiency optimization (step 15) are not
> assigned to a study. See [`docs/research/STUDIES.md`](docs/research/STUDIES.md).
