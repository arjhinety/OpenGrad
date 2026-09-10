# OpenGrad roadmap

OpenGrad proceeds from infrastructure to controlled measurement. A later phase is not considered successful without reproducible evidence and regression analysis.

## Completed pre-GPU foundation

0. Repository and research infrastructure — COMPLETE
0.5. CPU-only pre-experiment validation — COMPLETE
0.6. Deterministic CPU baseline pipeline hardening — COMPLETE; GPU execution not run
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
- No model inference, training, or model-quality result exists yet.
- Canonical dataset publication architecture — PREPARED; upload not run.

## Publication milestone

Canonical dataset publication — CANONICAL_DATASET_PUBLISHED

`arrochi112/OpenGrad-ToolPolicy-Canonical-v1` is public and verified at Hub commit `bb295d8a4ad64f7e8161044ad2fa34f873ede418`. The release contains 213,951 legally cleared canonical SFT records. xLAM is included under CC BY 4.0 with attribution and APIGen citation; its upstream access gate is not reproduced downstream. Publication metadata is recorded in `reports/releases/toolpolicy-canonical-v1-publication.json`.

## Next empirical sequence

6. B0 unmodified Qwen3.5-2B baseline inference — **REAL_RESULT**
   Executed on an A100 (engine vLLM 0.29.0) over the frozen behavioral-heldout-v2 evaluation:
   3,650 distinct examples, 196 s, from a clean traced tree. See
   [the B0 result](reports/baselines/qwen35_2b_baseline/RESULT.md). External benchmark
   families remain FROZEN_NOT_EXECUTED.

7. Freeze B0 evidence and generate the residual profile — **COMPLETE**
   Predictions, metrics, environment, residual profile, and the canonical experiment record
   are recorded and committed. The engine, renderer, evaluator, generation configuration,
   and failure taxonomy are pinned alongside them.

8. Decide and run the controlled SFT comparison — NOT STARTED
   Compare M0 source-oriented control, M1 behavior-balanced hypothesis, and M2 residual-driven mixture only after B0 evidence. M2 requires a real residual profile and remains unresolved.
   The readiness contract for SFT is satisfied (`opengrad readiness configs/experiments/m0_sft.yaml`
   reports PASS with no blocking gates) and the dry-run plumbing is verified, but no SFT run
   has been launched. M1 and M2 additionally reference datasets that have not been
   materialized (`when2call_pref_v1`, `onpolicy_prompts_v1`), so their contracts do not yet
   resolve.

9. Full post-SFT evaluation and diagnosis — NOT STARTED

10. Preference optimization or on-policy distillation — CONDITIONAL
    Run only when a measured residual justifies the objective.

11. Cross-model replication — PLANNED

12. Quantization and runtime evaluation — PLANNED

13. OpenWeights-derived downstream deployment studies — PLANNED
    OpenWeights is an independent project; its observations motivate hypotheses but are not OpenGrad results.

14. Speculative decoding and inference research — PLANNED / GPU_REQUIRED
    Future comparisons may include autoregressive decoding, external draft speculation, Medusa, EAGLE-3, DFlash, DSpark, and native MTP where supported. No method is currently benchmarked or supported by OpenGrad.

15. Joint capability-efficiency optimization — PLANNED

The dependency order is intentional:

B0 baseline
    -> frozen baseline evidence
    -> residual profile
    -> M0/M1/M2 decision
    -> SFT
    -> post-SFT evaluation
    -> conditional preference/distillation
    -> quantization/runtime
    -> OpenWeights device validation
    -> speculative decoding
    -> joint capability-efficiency studies
