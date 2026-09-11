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

Current evidence:

- 213,951 canonical valid SFT records across the current accessible sources.
- 210,874 tokenizer-rendered SFT candidates; 3,077 LoopTool records remain explicit renderer exclusions because they contain no user query.
- xLAM and BUTTON are now materialized and included in the corpus audit.
- The frozen held-out evaluation contains 3,952 records.
- Model inference, training, and model-quality results now exist: the B0 baseline and three post-training interventions (see [Results](README.md#results)).
- Canonical dataset publication — v1 published and verified; the Canonical-v2 RC snapshot is published as a partial (3-of-6) rebuild.

## Publication milestone

Canonical dataset publication — CANONICAL_DATASET_PUBLISHED

`arrochi112/OpenGrad-ToolPolicy-Canonical-v1` is public and verified at Hub commit `bb295d8a4ad64f7e8161044ad2fa34f873ede418`. The release contains 213,951 legally cleared canonical SFT records. xLAM is included under CC BY 4.0 with attribution and APIGen citation; its upstream access gate is not reproduced downstream. Publication metadata is recorded in `reports/releases/toolpolicy-canonical-v1-publication.json`.

## Empirical sequence

6. B0 unmodified Qwen3.5-2B baseline inference — **REAL_RESULT**
   Executed on an A100 (engine vLLM 0.29.0) over the frozen behavioral-heldout-v2 evaluation:
   3,650 distinct examples, 196 s, from a clean traced tree. See
   [the B0 result](reports/baselines/qwen35_2b_baseline/RESULT.md). External benchmark
   families remain FROZEN_NOT_EXECUTED.

7. Freeze B0 evidence and generate the residual profile — **COMPLETE**
   Predictions, metrics, environment, residual profile, and the canonical experiment record
   are recorded and committed. The engine, renderer, evaluator, generation configuration,
   and failure taxonomy are pinned alongside them.

8. Controlled SFT comparison — **EXECUTED — 2 NEGATIVE, 1 PARTIAL RECOVERY**
   Two runs reached real optimizer steps against the canonical corpus. M0 on corpus v1
   (`qwen35_2b_m0_sft_full_v3`) collapsed tool calling (`call_f1` 0.6191 → 0.0000); M1 DPO on the
   When2Call preference pairs (`qwen35_2b_m1_dpo_v1`) removed over-calling but collapsed call
   recall and did not reproduce on a repeat. M0 on the corrected corpus v2
   (`qwen35_2b_m0_sft_v2corpus`) reached `call_f1` 0.5995 and macro recall 0.6416. M2 was not
   run; `onpolicy_prompts_v1` was never materialized. See the
   [M0 SFT execution report](reports/M0_SFT_EXECUTION_REPORT.md).

9. Full post-SFT evaluation and diagnosis — EXECUTED (partial)
   Every saved checkpoint was measured against the frozen behavioral-heldout-v2 set with the same
   engine and parser that produced B0; the failure/regression diagnosis is in the M0 report (§3
   for the v1 collapses, §8 for the v2 recovery). External benchmark families remain
   `FROZEN_NOT_EXECUTED`, and the best v2 checkpoint (1200) is still selection on the existing
   evaluation set and needs checkpoint-selection-disjoint confirmation.

10. Preference optimization or on-policy distillation — EXECUTED (DPO) / NOT ATTEMPTED (distillation)
    DPO was run on the When2Call preference pairs and regressed on the promotion metric; all three
    checkpoints were rejected, and the trajectory did not reproduce (INC-0001). On-policy
    distillation was explicitly out of scope and not attempted — not merely conditional. See the
    [M0 SFT execution report](reports/M0_SFT_EXECUTION_REPORT.md).

11. Cross-model replication — PLANNED

12. Quantization and runtime evaluation — PLANNED
    An isolated, optional optimization producer interface exists
    ([`src/opengrad/optimization/`](docs/optimization/README.md)): a trained checkpoint
    plus a recipe in, an optimized checkpoint with full provenance out. It is
    INTERFACE_ONLY — no optimization has been executed, ModelOpt is not a dependency,
    and no capability is established (all `UNKNOWN`). See
    [the ModelOpt integration report](reports/MODELOPT_INTEGRATION_REPORT.md).

13. OpenWeights-derived downstream deployment studies — PLANNED
    OpenWeights is an independent project; its observations motivate hypotheses but are not OpenGrad results.

14. Speculative decoding and inference research — PLANNED / GPU_REQUIRED
    Future comparisons may include autoregressive decoding, external draft speculation, Medusa, EAGLE-3, DFlash, DSpark, and native MTP where supported. No method is currently benchmarked or supported by OpenGrad.

15. Joint capability-efficiency optimization — PLANNED

The dependency order was intentional; the status of each stage is now:

B0 baseline                                  -> EXECUTED (REAL_RESULT)
    -> frozen baseline evidence              -> COMPLETE
    -> residual profile                      -> COMPLETE
    -> M0/M1/M2 decision                     -> M0 and M1 evaluated; M2 not run
    -> SFT                                   -> EXECUTED (2 negative, 1 partial recovery)
    -> post-SFT evaluation                   -> EXECUTED (partial; see step 9)
    -> conditional preference/distillation   -> DPO run and rejected; distillation not attempted
    -> quantization/runtime                  -> INTERFACE_ONLY (no execution)
    -> OpenWeights device validation         -> PLANNED
    -> speculative decoding                  -> PLANNED
    -> joint capability-efficiency studies   -> PLANNED
