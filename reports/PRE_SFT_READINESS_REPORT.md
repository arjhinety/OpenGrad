# Pre-SFT readiness report

**Date:** 2026-09-11
**Status:** `CONSUMED` — readiness was evaluated, the frozen experiment ran, and the results are
in [`M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`](M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md) and
[`M0_CANONICAL_V2_FINAL_EVALUATION.md`](M0_CANONICAL_V2_FINAL_EVALUATION.md).

This report is left as it stood at launch, because a readiness record that is edited after the
run stops being evidence that readiness was established *before* it. The gap it documents in §9
— no confirmatory held-out set at the time of writing — was closed before training by
pre-registering a DEV/confirmatory partition instead (see `docs/evaluation/CHECKPOINT_SELECTION_RULE.md`).

This report is the evidence for one statement: the next legitimate project action is a fresh
controlled M0 SFT run from the immutable `Qwen/Qwen3.5-2B` base model. It is not a summary of the
project; it is the gate record.

---

## 1. Canonical-v2 status

**Frozen.** Four sources, 173,237 canonical records, 176 shards. Distinct from the historical
partial snapshot in three measured ways: xLAM is included and contributes 56,090 trainable
records where the partial snapshot contributed 0 gradients; every record declares an explicit
supervision contract; and the partial snapshot's When2Call artifact was an *interrupted*
materialization that had read 9,162 of the raw file's 15,000 rows and was never finalized, so it
published 4,000 records where this build retains 6,505.

The partial snapshot is unmodified and remains published separately.

## 2. Identity

| Field | Value |
|---|---|
| Corpus fingerprint (release manifest sha256) | `8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161` |
| Determinism | **proven** — two independent builds produce this manifest and all 176 shards byte-identical (`0` mismatches), and a *delete-and-rebuild* reproduces the fingerprint exactly because the release config pins `build_commit` |
| Total canonical records | **173,237** |
| Total trainable records | **161,966** |
| Tool-call targets | **105,140** |
| Non-tool targets | 56,826 |
| `CALL_PREDICTION` trainable | **56,090** (34.6% of trainable) |
| `COMPLETE_TRAJECTORY` trainable | **105,876** (65.4%) |
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Training steps | **2,400** (unchanged from partial-v2) |
| Training config sha256 | `24dc5f523bbd0ac2cb4e5720576ed9172c9757f17e1c6daeb7edc32279c9f344` |
| Yield report | `reports/data/canonical-v2-final-yield.json` |
| Promotion policy | `tool_use_promotion_v2` |
| Schema / supervision contract | `tool_use_ir_v1` / `supervision_contract_v1` |

## 3. Per-source trainability

Measured by rendering every record, not by counting canonical rows.

| Source | Canonical | Trainable | Yield | Tool-call targets | Kind |
|---|---:|---:|---:|---:|---|
| glaive-function-calling-v2 | 98,339 | 97,112 | 0.988 | 48,723 | `COMPLETE_TRAJECTORY` |
| xlam-function-calling-60k | 57,342 | **56,090** | 0.978 | 56,090 | `CALL_PREDICTION` |
| toolace | 11,051 | 2,259 | 0.204 | 327 | `COMPLETE_TRAJECTORY` |
| when2call | 6,505 | 6,505 | 1.000 | 0 | `COMPLETE_TRAJECTORY` |

The mixture is trained under **natural sampling**. No kind is reweighted, and none is excluded:
`supervision.include` lists both kinds, and the ablation (all vs one kind) runs by changing that
block alone.

## 4. xLAM

| Stage | Records |
|---|---:|
| Upstream (reconstructed from the published v1 derivative) | 59,370 |
| Schema-valid | 57,342 |
| **Trainable** | **56,090 (94.5%)** |

Reconstruction is verified record by record: `tools` compared against the derivative verbatim,
`query` checked non-empty, **59,370/59,370 with 0 mismatches**. Upstream is access-gated (HTTP 401
at the pinned revision) and no substitute revision was used.

**Remaining quarantine, by cause:**

| Cause | Records | Attribution |
|---|---:|---|
| `XLAM_TYPE_UNSUPPORTED_UNION` | 1,216 | our adapter refuses `Union[int, float]` |
| `XLAM_TYPE_UNSUPPORTED_CALLABLE` | 529 | our adapter refuses `Callable[...]` |
| `XLAM_TYPE_UNSUPPORTED_SET` | 283 | our adapter refuses `set` (no JSON equivalent) |
| `SEM_ARGUMENT_INVALID` | 1,231 | **upstream data quality** — the gold call's value contradicts the type the same record declares |
| `TARGET_TRUNCATED` | 21 | exceeds the 2,048-token window |

3.4% is unrepresentable at the schema layer and was never approximated. The 1,231 argument
failures are genuine upstream defects and are not recoverable without inventing data.

## 5. Records deliberately left quarantined

| Source | Subset | Decision |
|---|---|---|
| ToolACE | 8,476 records ending on an unanswered call | **quarantined**, not reclassified. Call-prediction-shaped, but the bytes do not establish intended next-call supervision versus truncation. |
| LoopTool-23k | entire source | **not fetched** — upstream not located (excluded from the release) |
| BUTTON | 2,359 dangling calls; entire source | calls never answered by any tool message; the terminal turn is prose so there is no call-prediction target either. Invalid under **both** contracts. Source itself is access-gated. |
| xLAM | 2,028 schema-unrepresentable, 1,231 argument-invalid | see §4 |

No record was rescued by inferring semantics from shape. This is the discipline the whole
supervision-contract design exists to enforce, and it costs real yield: ToolACE retains 20.4% of
its canonical records and the gate **reports it as `ANOMALY` on every run** rather than having its
threshold fitted to the measurement.

That choice is deliberate and was contested. Setting the floor below 0.204 would have turned the
gate green and made the run look cleaner; it would also have meant the threshold was fitted to the
number it exists to judge, so the next degradation would pass unnoticed. `ANOMALY` does not block
readiness, so the honest reading costs nothing operationally: a reader sees that 79.6% of this
source is quarantined **by decision**, instead of a green tick that hides it.

A second measurement did change an *expectation* rather than a threshold. When2Call was declared
`expects_tool_calls: true` on the assumption that its `<TOOLCALL>` markers became structured calls.
Measured, 0 of its 6,505 canonical records carries a structured call, a tool result, or even the
marker text. What it supervises is the **decision** — decline, ask for the missing detail, or answer
directly — in natural language, which is precisely the behaviour B0 fails worst
(`unsupported_accuracy` 0.0131, `clarification_accuracy` 0.1009). Requiring structured calls of it
would have failed a healthy source rather than detected a broken one, so the expectation was
corrected to `false` and `min_yield_ratio` retained as its collapse guard.

## 6. Contamination

| Level | Status |
|---|---|
| 1 exact canonical conversation hash | MEASURED |
| 2 normalized prompt hash | MEASURED |
| 3 near-duplicate ngram minhash | MEASURED |
| 4 semantic similarity | MEASURED |
| 5 manual audit | **COMPLETE** — 2 findings, 2 adjudicated, 2 `CONTAMINATED`, 0 pending |

Both candidates were re-verified against the **final** corpus bytes before adjudication, not
re-stamped from the previous verdict: the held-out prompt *"What is the current time?"* occurs in
**8** Glaive training records (up from 5), and *"What is the current weather?"* occurs in **1**
When2Call record. Both are exact-prompt matches whose training labels contradict the held-out
gold decision, so both are quarantined and the benchmark is 3,950 distinct examples.

Contamination status: `SEMANTIC_REVIEW_COMPLETE`. Evidence artifacts are corpus-scoped
(`…--toolpolicy-canonical-v2-final.json`) so they cannot be confused with the v1 corpus's.

## 7. Held-out evaluation identity

| | |
|---|---|
| Manifest | `reports/evaluation/behavioral-heldout-v2.manifest.json` |
| sha256 | `8bb6ad2e37613c996476bb734b93ef792eae807ec2cccf6b200d96a47332c640` |
| Freeze revision | `2d97c7d5a8de0b16…` |
| Contents | `when2call-mcq` 3,652 + `when2call-llm-judge` 300, less 2 quarantined = **3,950** |
| Role | **development / checkpoint selection** |

## 8. Gate status

Every gate below is **active** — it evaluates measured evidence, not the existence of
infrastructure. None is disabled or dormant for this configuration.

| Gate | Status | Note |
|---|---|---|
| `repository_validation` | PASS | registries valid |
| `config_validation` | PASS | experiment schema |
| `model_revision` / `model_identity` | PASS | pinned base revision |
| `tokenizer_revision` | PASS | pinned |
| `chat_template_contract` | PASS | native renderer |
| `evaluation_manifest` / `evaluation_materialization` | PASS | 3,950 examples materialized |
| `dataset_revision` / `dataset_snapshot` | PASS | `canonical_v2_final` pinned in the registry |
| `contamination_gate` | PASS | Level 5 complete, corpus-scoped |
| `evaluation_leakage` | PASS | evaluation-only manifest excludes the training sources |
| `native_parser` | PASS | golden fixtures |
| `gpu_probe` / `gpu_boundary` | PASS | A100 receipt |
| `real_b0` / `baseline_artifacts` | PASS | B0 immutable, predictions + metrics present |
| `training_data_policy` | PASS | no evaluation-only dataset IDs |
| `renderability_yield` | **ACTIVE** | reads a real measurement: `reports/data/canonical-v2-final-yield.json`. No source collapses; xLAM contributes 56,090 tool-call targets. Reports ToolACE as `ANOMALY` (0.204 against a floor of 0.3) — visible, non-blocking, and deliberately not threshold-fitted |
| `supervision_composition` | **ACTIVE** | reads the same report; verifies per-kind composition and that the config's selection exists in the corpus |
| `experiment_preflight` | PASS | |

**`supervision_composition` is not a formality.** It fails when a config selects a kind the corpus
does not contain, and when a trainable record carries no kind. Both are checked against the
measured report.

## 9. The one documented gap

**The confirmatory held-out set does not exist.** `behavioral-heldout-v2` is where the partial-v2
checkpoint 1200 was selected, so it is development/selection evidence and cannot serve as clean
test evidence for a *close* comparison. All 3,952 upstream test rows are already in use and there
is no spare held-out data, so materializing a disjoint confirmatory set requires either a
deterministic split of the existing 3,950 or new upstream data.

This does **not** block SFT, and the config does not pretend otherwise: it names
`behavioral-heldout-v2` explicitly as the selection set, and `CHECKPOINT_SELECTION.md` records
the distinction. What it does limit is the strength of any future *close* claim — a peak
`call_f1` chosen over four checkpoints on this set must be reported as selected, with the number
of checkpoints compared stated alongside. Cross-corpus claims should prefer metrics insensitive
to a four-way maximum: per-class behaviour at a fixed step, or the trajectory shape across steps.

## 10. Training recipe preserved

Verified field by field against `qwen35_2b_m0_sft_v2corpus` — all 14 trainer fields, the model
triple, generation settings and reproducibility settings are **identical**. `call_f1` is not the
sole promotion gate: `tool_use_promotion_v2` requires macro per-class recall, absolute floors on
the behaviours B0 fails, an over-call ceiling, a parse-validity floor, and per-dimension
non-regression.

**Epoch difference, reported not compensated:** 2,400 steps now covers fewer epochs than on
partial-v2, because the corpus grew from 103,036 to 173,237 canonical records. The step count is
held fixed on purpose so the two runs differ in one respect.

## 11. Next command

Readiness, immediately before training:

```bash
opengrad readiness configs/experiments/m0_sft_canonical_v2_final.yaml
```

Then the definitive SFT — **not executed as part of this work**:

```bash
opengrad train configs/experiments/m0_sft_canonical_v2_final.yaml
```

Both are plain shell commands requiring no agent harness.
