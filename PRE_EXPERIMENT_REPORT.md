# Phase 0.5 — Pre-Experiment Validation Report

Historical phase report. Its statements describe the Phase 0.5 checkpoint; current accessible-corpus completion is recorded in `reports/data-normalization-v1.md` and `ROADMAP.md`.

## Current gate (post-review)

The repository is `DATA_READY / BASELINE_PIPELINE_HARDENING`, not
`BASELINE_INFERENCE_READY`. `opengrad baseline --dry-run` now runs the B0
pipeline on CPU with a deterministic backend. A real model or benchmark score
still does not exist. The frozen experiment uses `behavioral-heldout-v2`, pins
the evaluator and benchmark revisions, and fixes the 4096-token overflow policy.

## 1. Completed work

Implemented CPU-safe canonical tool schema validation, six source adapter boundaries with fixtures, model-family renderer seams, strict tool-call parser states, synthetic contamination methods, benchmark mock harnesses, normalized evaluation results, taxonomy mapping, experiment definition, lineage and stage gates, data statistics and mixture analysis, configuration validation, report generation, environment capture, CLI preflight, and Phase 1/2 protocols.

## 2. Remaining unresolved metadata

The bibliography and dataset registry now use field-level `VERIFIED`, `PARTIAL`, or `UNRESOLVED` records with checked sources. Remaining unresolved items are explicitly limited to metadata not established by the checked primary sources: the canonical identity of the STAR queue entry, Toolathlon canonical identity/release, some LoopTool paper metadata, some benchmark evaluator revisions, and per-split counts unavailable without materialization or authoritative published counts.

## 3. Benchmark harness status

CPU mock smoke harnesses accept deterministic predictions and emit `SMOKE_TEST_ONLY` result envelopes for BFCL, When2Call, τ-bench/τ², ToolSandbox, MCPMark, and Toolathlon. In addition, `opengrad baseline --dry-run` exercises the real B0 pipeline from the frozen manifest through rendering, native Qwen parsing, behavioral scoring, predictions, residuals, and environment capture. No real model or benchmark score was produced.

## 4. Dataset-adapter status

Fixture adapters exist for xLAM, When2Call, ToolACE, BUTTON, LoopTool, and Glaive. They preserve source identity/split metadata, reject malformed records, and do not materialize full datasets.

## 5. Parser/schema status

Canonical conversations support system/user/assistant/tool roles, multiple and parallel calls, call IDs, results, sequential turns, no-call/clarification/impossible-tool responses, and strict rejection. Parser output distinguishes `RAW_VALID` and `INVALID`; no automatic repair is performed.

## 6. Contamination-tool status

Synthetic tests cover normalized exact hashes, canonical JSON signatures, n-gram Jaccard, edit similarity, and MinHash. Real corpora were not scanned. Semantic embeddings remain optional and unimplemented.

## 7. Experiment registry status

`tool_calling/qwen35_2b/baseline` remains `PLANNED` for the real model run, but its execution contract is frozen. The native parser has adversarial golden tests, generation settings/runtime versions are pinned, and the GPU backend is the only unexercised implementation boundary.

## 8. Tests executed

Final verification passed: Ruff, format check, pytest (29 passed), registry validation, mypy, CLI preflight, six benchmark mock smoke harnesses, and diff checks. All are CPU-only and GPU-free.

## 9. Known limitations

No GPU, model inference, large checkpoint download, full benchmark run, training, dataset materialization, real contamination scan, GGUF/ExecuTorch export, or speculative decoding was performed. Those are later-phase activities.

## 10. Tasks requiring GPU access

Future Qwen3.5-2B baseline inference and full benchmark reproduction; training/post-training; teacher/student distillation; GPU runtime comparisons; and any accelerator-specific throughput work.

## 11. CPU-only but intentionally deferred

Further primary-source reading for unresolved STAR/Toolathlon identities, exact per-split metadata where not publicly exposed, evaluator-specific adapters beyond the generic smoke contract, model-native renderer validation against exact tokenizer revisions, semantic contamination embeddings, and additional benchmark revision pinning when canonical repositories are identified.

## 12. Readiness decision

READY_FOR_PHASE_1

This means the repository preflight and CPU validation infrastructure pass. It does not authorize Phase 1 or begin any model work. Phase 1 still requires explicit authorization and must follow `docs/experiments/BASELINE_REPRODUCTION_PROTOCOL.md`.

## Methodology revision addendum

This historical report describes the prior source-adapter/bootstrap state. The subsequent methodology revision preserves that provenance work but adds a dual-axis behavioral model, M0/M1/M2 mixture classes, deterministic residual-to-mixture infrastructure, and coverage auditing. It does not retroactively claim that historical source proportions were behaviorally measured. See `docs/data/tool-use-mixture-methodology.md`.
