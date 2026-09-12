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
- [`arrochi112/OpenGrad-ToolPolicy-Canonical-v2`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2) (current) carries 173,237 canonical records and 161,966 trainable ones across four sources, with fingerprint `8ced403b…` proven reproducible by a delete-and-rebuild. Publication metadata is recorded in `reports/releases/toolpolicy-canonical-v2-publication.json`.
- [`arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot) (historical, partial) is the 103,036-record, three-of-six-source corpus behind the first successful M0. It is distinguished from the final v2 by name, fingerprint and source count, and is not the current corpus.

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

8. Controlled SFT comparison — **EXECUTED — CLOSED**
    The frozen M0 lineage includes the v1 negative, partial-v2, definitive final-v2, and the paired
    fixed-compute/matched-exposure minus-xLAM arms. Final-v2 recovered call recall but was not
    promoted; both minus-xLAM arms substantially lost recall. Because xLAM is currently the only
    CALL_PREDICTION source, those arms measure joint removal of xLAM and that supervision channel,
    not a pure xLAM-content effect. See [M0 phase closure](reports/M0_PHASE_CLOSURE.md).

9. Full post-training evaluation and diagnosis — **EXECUTED**
    M0 and both minus-xLAM arms were fully evaluated on the frozen DEV partition, selected under
    the frozen balanced rule, and scored once on the pre-registered internal confirmatory partition.
    External benchmark families remain `FROZEN_NOT_EXECUTED`; tool-selection and argument/schema
    validity are still unmeasured by the current evaluator.

10. Preference optimization or on-policy distillation — **DPO EXECUTED / DISTILLATION NOT JUSTIFIED**
     M1-v2 DPO started from the selected M0-final checkpoint, preserved its calibrated frontier, and
     was promoted under prospective `tool_use_promotion_v4`. The first M1 identity is preserved as
     a 119/120-step failure. The existing M2 path is mock-only: it has no live teacher, student
     update, parent checkpoint loading, or valid prompt-state dataset. M1 already passes the
     intended measured calibration policy, so launching M2 would neither answer a distinct valid
     question nor justify its infrastructure gaps. See the [M1 evaluation report](reports/M1_DPO_EVALUATION.md)
     and [M2 decision](reports/M2_DECISION.md).

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
    The adapter and the versioned Tier C `openweights` benchmark definition exist, but no post-training
    device study has been executed, so no OpenGrad result rests on device measurements.

14. Speculative decoding and inference research — PLANNED / GPU_REQUIRED
    Future comparisons may include autoregressive decoding, external draft speculation, Medusa, EAGLE-3, DFlash, DSpark, and native MTP where supported. A reserved configuration exists; no method is currently implemented, benchmarked, or supported by OpenGrad.

15. Joint capability-efficiency optimization — PLANNED

The dependency order was intentional; the status of each stage is now:

B0 baseline                                  -> EXECUTED (REAL_RESULT)
    -> frozen baseline evidence              -> COMPLETE
    -> residual profile                      -> COMPLETE
    -> M0/M1/M2 decision                     -> M0 closed; M1 evaluated; M2 not run (mock-only path, not justified)
    -> SFT                                   -> EXECUTED (2 negative, 1 partial recovery, 1 definitive, 2 negative joint-removal ablations)
    -> post-SFT evaluation                   -> EXECUTED (see step 9)
    -> preference optimization               -> first DPO identity rejected/not reproducible; M1-v2 DPO EXECUTED and PROMOTED
    -> distillation                          -> NOT RUN (scaffold only; live training path unimplemented)
    -> quantization/runtime                  -> INTERFACE_ONLY (no execution)
    -> OpenWeights device validation         -> PLANNED (integration and benchmark definition only; no study executed)
    -> speculative decoding                  -> PLANNED / GPU_REQUIRED (no runtime support or benchmark executed)
    -> joint capability-efficiency studies   -> PLANNED
