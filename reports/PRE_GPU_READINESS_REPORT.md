# Pre-GPU Readiness Report

**Scope:** harness-agnostic agent integration, the pre-GPU baseline/readiness contract, and
held-out contamination screening.
**Status:** `PRE_GPU` — no real model result exists.
**Repository commit at time of writing:** the commit that adds this file; `git log -- reports/PRE_GPU_READINESS_REPORT.md` pins it exactly.

> This report intentionally does **not** claim `READY_FOR_SFT`, a real B0, or a successful GPU boundary. None of those conditions is met. See [Remaining blockers](#11-remaining-blockers).

---

## 1. Integration versions

| Component | Version |
| --- | --- |
| OpenGrad MCP server (`integrations/opengrad-mcp`) | 0.1.0, private, no third-party runtime dependencies |
| Node.js | v24.21.0 (server requires ≥ 20) |
| Python | 3.12.3 (project requires ≥ 3.11) |
| `@opengrad/mcp-server` install | one command, stdio transport, no vendor SDK |

The previous vendor-specific DeepSeek Harness plugin was removed. The integration is now a plain stdio MCP server plus the documented `opengrad … --json` CLI, so any MCP-capable harness (or none at all) can drive OpenGrad.

## 2. What was built

- `integrations/opengrad-mcp/src/core.js` — argv-only subprocess bridge (no shell), repository path containment, recursive + inline secret redaction, canonical OpenGrad error codes, bounded output.
- `integrations/opengrad-mcp/src/guardian.js` — permission tiers, monotonic invariants, static guards, and the shared high-impact gate evaluator.
- `integrations/opengrad-mcp/src/tools.js` — the typed tool catalog and the policy-then-execute dispatcher.
- `integrations/opengrad-mcp/src/workflows.js` — B0 and post-SFT orchestration.
- `integrations/opengrad-mcp/src/server.js`, `src/index.js` — MCP stdio JSON-RPC (`initialize`, `tools/list`, `tools/call`, `ping`).
- `integrations/opengrad-mcp/install.sh`, `mcp.example.json`, `README.md`, `tests/`.

Supporting OpenGrad changes: `src/opengrad/readiness.py` (new), plus hardening in `evaluation/runner.py`, `agent_cli.py`, `cli.py`, `experiments/store.py`, `checkpoints/registry.py`, `data/materialize.py`.

## 3. Tools exposed

Read-only: `opengrad_status`, `opengrad_doctor`, `opengrad_validate`, `opengrad_validate_data`, `opengrad_inspect_template`, `opengrad_baseline_status`, `opengrad_experiment_list|show|diff`, `opengrad_checkpoint_list|inspect`, `opengrad_failures`, `opengrad_eval_compare`.

Safe write: `opengrad_readiness`, `opengrad_gpu_smoke`, `opengrad_preflight`.

High impact: `opengrad_baseline_run`, `opengrad_sft`, `opengrad_eval_run`, `opengrad_b0_workflow`, `opengrad_post_sft_workflow`.

DPO, preference generation, on-policy distillation, RL, judge orchestration, checkpoint promotion, and arbitrary shell execution are deliberately absent.

## 4. Guard usage and monotonic invariants

Guards are enforced in two layers, independent of the model:

1. **Static guards** (`evaluateStaticGuard`) run before dispatch: evaluation-only configs/records can never enter SFT, and real SFT requires an explicit immutable config.
2. **Gate checks** (`evaluateHighImpactGate`) run for high-impact tools unless an explicit dry-run is requested, consulted against the authoritative `opengrad readiness --json` projection.

Invariants: `NO_REAL_SFT_WITHOUT_VALID_B0`, `NO_TRAINING_WITH_FAILED_PREFLIGHT`, `NO_TRAINING_ON_EVAL_DATA`, `NO_FLOATING_MODEL_REVISION`, `NO_FLOATING_DATASET_REVISION`, `NO_EXPERIMENT_ID_MUTATION`, `NO_CHECKPOINT_OVERWRITE`, `NO_SECRET_LOGGING`, `NO_PROMOTION_WITHOUT_REQUIRED_EVALUATION`.

Operations that *establish* B0 are not required to already own B0's post-run artifacts, so the B0 workflow is not deadlocked by the boundary it produces. SFT remains fail-closed on the full contract. Unknown high-impact operations fail closed, and there is no `--force` bypass.

## 5. Workflows

- `opengrad_b0_workflow`: prerequisite readiness → GPU boundary smoke → real baseline inference. Never runs SFT.
- `opengrad_post_sft_workflow`: evaluation → regression comparison → residual/failure analysis, deferring stages whose artifact paths do not exist yet. Does not train or promote.

## 6. Verification executed

| Command | Result |
| --- | --- |
| `python3 -m py_compile <7 modules>` | PASS |
| `.venv/bin/python -m pytest` | **164 passed** |
| `.venv/bin/python -m ruff check .` | **All checks passed** |
| `.venv/bin/opengrad-validate` | `registry validation: OK` |
| `cd integrations/opengrad-mcp && npm run check` | PASS |
| `cd integrations/opengrad-mcp && npm test` | **26 passed, 0 failed** |
| `.venv/bin/opengrad-contamination heldout-screen` | levels 1–4 `MEASURED`; 2-item audit queue |
| `cmd mcp get opengrad` | registered, project scope, stdio, enabled |
| `cmd config get permissions.defaultMode` | `bypass` (user scope) |

The end-to-end test `test_baseline_plumbing_runs_over_a_real_materialized_split` materializes real Parquet with the real materializer and runs the real manifest loader, content-hash verifier, example constructor, prediction path, and artifact writer. Only the tokenizer-dependent renderer is stubbed.

## 6b. Held-out materialization and re-freeze

The withheld evaluation splits are evaluation-only and are deliberately absent from both the canonical release and Git, so they were rebuilt from the pinned upstream revision `0582f7749df63a96fdc3070932e83e72396ace53` of `nvidia/When2Call` (`scripts/rebuild_eval_splits.py`, idempotent):

| Split | Upstream file | Rows | Manifest declares |
| --- | --- | --- | --- |
| `when2call-mcq` | `test/when2call_test_mcq.jsonl` | 3,652 | 3,652 |
| `when2call-llm-judge` | `test/when2call_test_llm_judge.jsonl` | 300 | 300 |

`opengrad readiness` now reports `evaluation_materialization: PASS`. The frozen manifest's
`content_hash` values were re-frozen from these artifacts with recorded provenance
(`hash_provenance`), and the baseline experiment's `dataset_hash` was re-synced — see §7.

## 6c. Contamination screening

`screening levels 1–4 are now machine-measured` over the held-out set against the 213,951-record canonical training corpus. Headline result: **2 distinct held-out questions are exact prompt matches** with training records:

- `"What is the current time?"` — matches 5 Glaive Function-Calling v2 records.
- `"What is the current weather?"` — matches 1 When2Call SFT record.

Both are short, generic tool-use queries. Whether they are substantive contamination or incidental lexical coincidence is precisely the human judgment reserved for level 5; this report does **not** decide it. Level 3 near-duplicate and level 4 SequenceMatcher screening found nothing above threshold once trivially short prompts were excluded.

## 7. Defects found and fixed in this pass

- **Content-hash byte drift (would have blocked real B0 on GPU day).** `evaluation/runner.py` joined rows with a literal two-character `\n` instead of a newline byte, while the materializer and the readiness gate used a real newline. Every correctly materialized split would have failed with *"content hash does not match materialized rows"*. Fixed and pinned by two regression tests (materializer↔evaluator and materializer↔readiness).
- **Unreproducible frozen content hashes (hard B0 blocker).** The frozen manifest's `content_hash` values were authored by hand: `content_hash` did not exist in `materialize.py` at any commit up to and including `908f1be`, the commit that froze the manifest, and nothing in the repository writes that manifest. With the real splits materialized and row counts exactly matching, the gate still failed on the hash alone. ~26 serialization/row-shape/split-name combinations were probed; none reproduced the frozen value. Resolved by re-freezing from the real artifacts with recorded `hash_provenance` (§6b) rather than by weakening the gate.
- **Hand-authored contamination report.** `behavioral-heldout-v2-contamination.json` likewise had no generator — nothing in the repository writes it. Levels 1–4 are now produced by `opengrad.contamination.heldout`; level 5 remains human.
- **Contamination scan identity bug (caught by its own invariant).** `opengrad_id` and `source_record_id` are the literal string `"unknown"` for all 20,827 LoopTool rows, collapsing them into one bucket and producing impossible metrics (negative Jaccard, containment ≈ 11.7) and a 3,905-item false-positive queue. Training identity now uses the unique-per-row `canonical_hash`, an explicit invariant refuses to emit an impossible overlap, and the queue fell to its true size of 2.
- **Short-prompt containment false positives.** A 3-shingle prompt ("ok thanks") is trivially contained in anything, so containment alone flagged noise. Containment can no longer flag a held-out prompt shorter than `min_shingles`.
- **Tests reran the real GPU smoke and rewrote committed evidence.** `tests/config/test_readiness.py` called `gpu_smoke(ROOT)`, which performed a real model load/generation on this A100 and rewrote `reports/hardware/qwen_gpu_smoke.json` on every `pytest` run. It now uses an isolated root and a stubbed no-accelerator probe; the receipt hash is asserted stable across the suite.
- **Lint failures (CI parity).** 15 ruff errors introduced by earlier work: unsorted imports, four useless `if/else` conditionals, two `ValueError`-on-type-error cases, and five blind `except` clauses now annotated with justification. CI runs `ruff check .`, so this suite would have failed.
- **Dry-run training collision.** Dry-run SFT previously required and mutated a real experiment record; it now writes to a scratch namespace and is labeled `evidence: false`.
- **`baseline --dry-run` polluted the canonical evidence paths.** With the held-out data present, a dry run wrote deterministic-mock artifacts into `reports/baselines/qwen35_2b_baseline/` and `reports/failures/...`. Because the real run refuses to overwrite existing evidence, a routine dry run *permanently blocked the real B0* behind its own overwrite guard. Dry runs are now redirected to `runs/.dry-run/...` unless the config gives an explicit absolute destination, and a regression test asserts the canonical paths stay free and a real run still succeeds afterwards.
- **Successful dry runs exited non-zero.** `opengrad baseline --dry-run` returned exit 1, so an agent bridge reported a completed plumbing run as `COMMAND_FAILED`. A finished dry run is now a successful command; the `status` field still carries `DRY_RUN` vs `EXECUTED`.
- **Silent experiment-ID reuse.** A second real launch with an existing ID now fails with `EXPERIMENT_ID_COLLISION` instead of silently resuming.

## 8. Real Qwen GPU smoke status — `INCOMPLETE`

`reports/hardware/qwen_gpu_smoke.json` (kind `GPU_BOUNDARY_VERIFIED`, `status: INCOMPLETE`) records a real A100 attempt:

| Check | Result |
| --- | --- |
| `model_access` | PASS (`Qwen/Qwen3.5-2B` @ `15852e8c…8a8fc`) |
| `native_template` | PASS |
| `model_load` | PASS (`cuda:0`, BF16) |
| `one_generation` | PASS |
| `native_parser` | **FAIL** — `FORMAT_ERROR: ['UNCLOSED_TOOL_CALL']` |
| `runtime` | **FAIL** — `GPU_SMOKE_FAILED` |

Hardware: `NVIDIA A100-SXM4-80GB` (A100_80GB), BF16 supported.

The boundary is **not** verified: the native parser rejected the smoke generation. This receipt was not rerun or modified during this work, and its byte hash is unchanged.

## 9. Real B0 status — not run

No real baseline exists. `real_b0` and `baseline_artifacts` are both `FAIL`, because B0 has not been executed — not because its inputs are missing. The frozen held-out splits are now materialized and hash-verified, so `baseline --dry-run` proceeds through manifest loading and fails only where it should: the deterministic mock is a plumbing check, and a real run still requires the GPU boundary.

`data/processed/` is gitignored by design. It can be regenerated at any time from the pinned upstream revision with `python scripts/rebuild_eval_splits.py`.

## 10. SFT readiness — `false`

`ready_for_sft: false`. `ready_for_baseline: false` (the contamination gate is a baseline prerequisite). CPU deterministic output never satisfies this contract.

## 11. Remaining blockers

`opengrad readiness --json` → `status: FAIL`, blocking gates:

| Gate | Status | Code | Note |
| --- | --- | --- | --- |
| `contamination_gate` | FAIL | `CONTAMINATION_FAILURE` | `pending_levels=['5_manual_audit']` only; levels 1–4 `MEASURED` |
| `gpu_boundary` | FAIL | `GPU_SMOKE_FAILED` | native parser `FORMAT_ERROR` |
| `real_b0` | FAIL | `BASELINE_NOT_FOUND` | not run |
| `baseline_artifacts` | FAIL | `BASELINE_NOT_FOUND` | not run |

`evaluation_materialization` now **PASSES** (both frozen held-out splits are materialized and hash-verified). Passing gates also include `repository_validation`, `config_validation`, `model_revision`, `model_identity`, `tokenizer_revision`, `chat_template_contract`, `evaluation_manifest`, `evaluation_leakage`, `artifact_storage`, `native_parser` (module presence), and `gpu_probe`.

To unblock, in order: (1) a human completes level-5 audit of the 2-item queue and adjudicates the two exact prompt matches; (2) resolve the native-parser smoke failure; (3) execute real B0. Step 1 is irreducibly human — no amount of code can close it, and it is deliberately not marked complete.

## 12. Known limitations

- The MCP server has no third-party runtime dependencies and no vendor plugin, so it cannot reuse a harness's native approval UI. High-impact tools are gated by the server's own readiness checks plus the harness's permission model.
- Levels 1–2 compare user-visible prompt text, not a whole-conversation hash: held-out evaluation examples and training trajectories do not share a conversation schema. No whole-conversation equality claim is made.
- Level 3 prunes shingles whose training document frequency exceeds `max_df` (default 1000 of 213,951). Pruned count is reported; a paraphrase built from ubiquitous n-grams would be missed.
- Level 4 candidate generation is prefiltered by level-3 Jaccard and scored with `difflib.SequenceMatcher`. It is **not** an exhaustive semantic search and no embedding similarity was computed, so it can miss meaning-level reuse with no lexical overlap. This is the level most likely to need strengthening before a generalization claim.
- Level 5 is human and unperformed. Until it completes, the two exact prompt matches stand unadjudicated and `contamination_gate` remains FAIL by design.
- The `content_hash` re-freeze and `dataset_hash` re-sync are documented in `hash_provenance`; anyone who considers the original frozen placeholders authoritative should treat this as a contract change rather than a fix.
- The native parser rejects an unclosed tool call in the bounded smoke. Until that is diagnosed, the GPU boundary cannot pass and real B0/SFT stay blocked.
- Post-SFT comparison and residual analysis remain deferred until artifact paths exist; the workflow reports `DEFERRED_UNTIL_ARTIFACT_PATHS` rather than inventing values.

## 13. Confirmation

No real SFT was run. No real B0 was run. No successful GPU boundary is claimed. No metric, score, or artifact in this repository was fabricated.
