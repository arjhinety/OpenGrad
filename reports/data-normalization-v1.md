# OpenGrad pre-GPU normalization and readiness report

Status: BASELINE_INFERENCE_READY

> **Status note (2026-09-11):** this report describes the pre-GPU corpus state. GPU work has since
> been executed on an NVIDIA A100-SXM4-80GB — the boundary smoke, the B0 baseline, and the M0/M1
> post-training runs — so the "no model scores" and "no GPU" framing below is historical, not
> current. The corpus-normalization evidence is retained. See the
> [B0 result](baselines/qwen35_2b_baseline/RESULT.md) and the
> [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md).

This report preserves the historical parser progression below and separates it from the current full accessible-corpus evidence. All currently accessible sources have been normalized and rendered; no external source blocker remains for the pre-GPU baseline. At the time of writing, no model weights had been downloaded and no CUDA context, model inference, or training had been run.

## Historical engineering progression

- ToolACE: 27/100 -> 79/100 -> 974/1000 -> final 11,190 retained from 11,300 raw (107 parse failures, 3 canonical duplicates; the 24 syntax cases and 2 unknown-tool cases remain quarantined in the historical 1K evidence).
- LoopTool: 2/100 -> 24/100 -> 1000/1000 -> final 20,827 retained from 23,040 raw; 2,213 parse/semantic failures remain quarantined.
- BUTTON: 20/100 -> 100/100 -> final 7,941 retained from 8,000 raw (59 duplicate-tool failures quarantined).
- When2Call: 997/1000 before duplicate policy -> duplicate policy -> final SFT 14,829 retained from 15,000 raw (47 parse failures, 124 canonical duplicates).
- Glaive: 970 unique + 30 duplicates in the 1K check -> final 99,794 retained from 112,960 raw (13,166 canonical duplicates).

## Full accessible corpus status

| source | raw | valid | quarantined/parse failures | canonical duplicates | eligible | status | manifest |
|---|---:|---:|---:|---:|---:|---|---|
| ToolACE | 11,300 | 11,190 | 107 | 3 | 11,190 SFT | FULL_DATA_VALIDATED | data/processed/normalization-v1/toolace/manifest.json |
| LoopTool-23k | 23,040 | 20,827 | 2,213 | 0 | 20,827 SFT | FULL_DATA_VALIDATED; 3,077 renderer exclusions | data/processed/normalization-v1/looptool/manifest.json |
| Glaive v2 | 112,960 | 99,794 | 0 | 13,166 | 99,794 SFT | FULL_DATA_VALIDATED | data/processed/normalization-v1/glaive/manifest.json |
| When2Call SFT | 15,000 | 14,829 | 47 | 124 | 14,829 SFT | FULL_DATA_VALIDATED | data/processed/normalization-v1/when2call-sft/manifest.json |
| When2Call preference | 9,000 | 9,000 | 0 | 0 | preference-only | FULL_DATA_VALIDATED | data/processed/normalization-v1/when2call-preference/manifest.json |
| When2Call MCQ | 3,652 | 3,652 | 0 | 0 | evaluation-only | FULL_DATA_VALIDATED | data/processed/normalization-v1/when2call-mcq/manifest.json |
| When2Call LLM judge | 300 | 300 | 0 | 0 | evaluation-only | FULL_DATA_VALIDATED | data/processed/normalization-v1/when2call-llm-judge/manifest.json |
| xLAM | 60,000 | 59,370 | 259 | 371 | 59,370 SFT | FULL_DATA_VALIDATED | data/processed/normalization-v1/xlam/manifest.json |
| BUTTON | 8,000 | 7,941 | 59 duplicate-tool failures | 0 canonical duplicates after rejection | 7,941 SFT | FULL_DATA_VALIDATED | data/processed/normalization-v1/button.manifest.json |

> **Glaive caveat (added 2026-09-13).** `FULL_DATA_VALIDATED` and "0 quarantined" for Glaive describe
> schema validation at normalization, which does not run the trajectory checks. The v1 Glaive
> adapter never parsed Glaive's `<functioncall>` markers into `tool_calls`: it left them in the
> assistant text and emitted an orphaned `tool` message. At the training boundary 50,900 of the
> materialized Glaive records were rejected as `SEM_ORPHAN_RESULT`, none of the surviving ones
> carried a tool call, and the published v1 corpus had 9 tool-call targets in 55,719 trainable
> records. That defect, not a data-access blocker, is why M0 on v1 collapsed. It was fixed by the
> separate `glaive_v2` adapter used for Canonical-v2 (`M0_SFT_EXECUTION_REPORT.md` §3.1, §8).

Accessible SFT candidate count currently audited/rendered: 210,874 tokenizer-rendered records (213,951 canonical records before the explicit 3,077 LoopTool template exclusions). Preference and evaluation artifacts remain separate and cannot enter ordinary SFT materialization.

## Audits

Machine-readable outputs are under reports/artifacts/:

- attrition-full.json: measured raw/valid/quarantine percentages for every accessible artifact.
- real-corpus-audit-final.json: real pairwise fingerprints across xLAM, BUTTON, ToolACE, LoopTool, Glaive, and When2Call SFT; source quality, behavior counts, contamination result, held-out count, and candidate index reference.
- Overlap pairs (raw, canonical, conversation, user-prompt, tool catalogue): xLAM/LoopTool (0, 0, 0, 160, 0); xLAM/When2Call SFT (0, 0, 0, 7,596, 0); ToolACE/LoopTool (0, 0, 0, 2,436, 1); ToolACE/Glaive (0, 0, 0, 1, 1); LoopTool/When2Call SFT (0, 130, 130, 140, 169); all remaining pair metrics are 0 except measured catalogue overlaps recorded in the JSON matrix. Prompt and catalogue overlap are not called contamination.
- Contamination against the frozen When2Call MCQ + LLM-judge namespace: 0 exact conversation matches; no training records were excluded by confirmed exact overlap.
- Behavior distribution across accessible SFT: 213,951 records; CALL 147,699 and ANSWER 66,252; `argument_grounding` 73,508, `single_tool_selection` 73,508, and `consume_tool_result` 74,191. Unsupported capability fields remain explicitly unknown rather than inflated.
- clean-sft-candidate-index.jsonl and its manifest preserve canonical ID, source, split, behavior labels, quality/duplicate/contamination state, structural counts, eligibility, and artifact lineage. Final behavioral reweighting has not been performed.

## Qwen rendering and token audit

Pinned contract: Qwen/Qwen3.5-2B; revision 15852e8c16360a2fea060d615a32b45270f8a8fc; template hash 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80; renderer qwen3_5_2b_v1; thinking disabled.

Golden fixtures cover ANSWER, single call, multiple calls, observation->answer, observation->subsequent call, multi-turn call sequence, CLARIFY, and UNSUPPORTED. Runtime tokenizer inspection reproduced the pinned template hash without loading model weights.

Full SFT token statistics are in reports/artifacts/token-statistics-full.json:

- xLAM: n=59,370; mean 616.60; P50/P75/P90/P95/P99/max = 576/727/913/1,040/1,303/3,180.
- ToolACE: n=11,190; mean 1,382.89; P50/P75/P90/P95/P99/max = 1,297/1,737/2,180/2,483/3,124/5,615.
- LoopTool: n=20,827; rendered=17,750; mean 4,064.25; P50/P75/P90/P95/P99/max = 4,138/5,452/6,677/7,479/9,194/15,969; 3,077 explicit Qwen template failures (no user query), not repaired.
- Glaive: n=99,794; mean 625.46; P50/P75/P90/P95/P99/max = 541/698/1,013/1,238/1,704/11,803.
- BUTTON: n=7,941; mean 2,442.06; P50/P75/P90/P95/P99/max = 2,168/2,868/3,740/4,422/6,872/26,057.
- When2Call SFT: n=14,829; mean 520.37; P50/P75/P90/P95/P99/max = 508/679/882/1,030/1,281/2,500.

Candidate sequence lengths are evidence-based proposals only: 2048 covers the large majority of ToolACE/Glaive/When2Call examples but truncates long trajectories. The baseline execution contract now freezes `max_input_tokens: 4096` with `overflow: reject`; training mixtures may measure a separate bucket later, but they cannot silently truncate this baseline.

Rendered lineage is in reports/artifacts/render-manifest-lineage.json and per-source JSONL manifests. All 3,077 LoopTool failures are not silently converted into training text; all other retained SFT records have canonical parent IDs and rendered checksums.

## Evaluation and experiments

reports/evaluation/behavioral-heldout-v2.manifest.json is MATERIALIZED and frozen with 3,952 evaluation-only records (MCQ 3,652; LLM judge 300), the exact renderer/model contract, source hashes, and an explicit exclusion policy covering all six SFT sources. At the time of writing no model scores existed; they have since been measured (see the [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md)). The behavioral evaluator contract includes CALL, ANSWER, CLARIFY, UNSUPPORTED, confusion matrix, call precision/recall/F1, under/over-call, wrong-tool, argument, clarification, and unsupported errors.

The baseline config is prepared for the unmodified Qwen3.5-2B instruct checkpoint. M0 is the source-oriented hypothesis control; M1 is the behavior-balanced HYPOTHESIS_ONLY config; M2 is SCHEMA_READY and UNRESOLVED_UNTIL_BASELINE with no fabricated weights. Residual failure schema and experiment lineage are prepared but unpopulated.

GPU resource preflight is now a `READY` record grounded in the executed A100 boundary smoke (`configs/hardware/gpu_preflight_v1.yaml`, `reports/hardware/qwen_gpu_smoke.json`); no exact throughput claim is made.

ModelOpt, vLLM Speculators, and DSpark are REFERENCE_ONLY / NOT_TESTED. NVIDIA ModelOpt is CUDA_ONLY by its upstream dependency boundary; vLLM, Transformers, and training support remain BACKEND_DEPENDENT or UNVERIFIED until the selected runtime is tested. No speculative method is claimed supported, and all remain GPU_REQUIRED / PLANNED.

## Reproducibility and remaining limits

The resumable materializer uses bounded Parquet batches, deterministic source order, atomic shard replacement, per-shard checksums, manifest checkpoints, and corruption-triggered rebuild. Clean rerun and completed-shard reuse are covered by tests and were exercised on real cached Parquet sources. Different worker-count execution is not applicable to the single-process canonical materializer; canonical shard order is deterministic. Corrupted/missing shard tests pass. The generated reports are local ignored artifacts by policy; manifests and reports are reproducibility outputs, not raw corpus commits.

All currently accessible sources, including xLAM and BUTTON, are normalized and audited locally. No external data-access blocker remains for the first unmodified-model baseline.
