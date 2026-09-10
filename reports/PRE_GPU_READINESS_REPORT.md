# Pre-GPU Readiness Report

**Scope:** harness-agnostic agent integration and the pre-GPU baseline/readiness contract.
**Status:** `PRE_GPU` — no real model result exists.
**Repository commit at time of writing:** `e7b2306` (this report is added by the immediately following commit; `git log -- reports/PRE_GPU_READINESS_REPORT.md` pins it exactly).

> This report intentionally does **not** claim `READY_FOR_SFT`, a real B0, or a successful GPU boundary. None of those conditions is met. See [Remaining blockers](#remaining-blockers).

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
| `.venv/bin/python -m pytest` | **153 passed** |
| `.venv/bin/python -m ruff check .` | **All checks passed** |
| `.venv/bin/opengrad-validate` | `registry validation: OK` |
| `cd integrations/opengrad-mcp && npm run check` | PASS |
| `cd integrations/opengrad-mcp && npm test` | **26 passed, 0 failed** |
| `cmd mcp get opengrad` | registered, project scope, stdio, enabled |
| `cmd config get permissions.defaultMode` | `bypass` (user scope) |

The end-to-end test `test_baseline_plumbing_runs_over_a_real_materialized_split` materializes real Parquet with the real materializer and runs the real manifest loader, content-hash verifier, example constructor, prediction path, and artifact writer. Only the tokenizer-dependent renderer is stubbed.

## 7. Defects found and fixed in this pass

- **Content-hash byte mismatch (would have blocked real B0 on GPU day).** `evaluation/runner.py` joined rows with a literal two-character `\n` instead of a newline byte, while the materializer and the readiness gate used a real newline. Every correctly materialized split would have failed with *"content hash does not match materialized rows"*. Fixed and pinned by two regression tests (materializer↔evaluator and materializer↔readiness).
- **Tests reran the real GPU smoke and rewrote committed evidence.** `tests/config/test_readiness.py` called `gpu_smoke(ROOT)`, which performed a real model load/generation on this A100 and rewrote `reports/hardware/qwen_gpu_smoke.json` on every `pytest` run. It now uses an isolated root and a stubbed no-accelerator probe; the receipt hash is asserted stable across the suite.
- **Lint failures (CI parity).** 15 ruff errors introduced by earlier work: unsorted imports, four useless `if/else` conditionals, two `ValueError`-on-type-error cases, and five blind `except` clauses now annotated with justification. CI runs `ruff check .`, so this suite would have failed.
- **Dry-run training collision.** Dry-run SFT previously required and mutated a real experiment record; it now writes to a scratch namespace and is labeled `evidence: false`.
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

No real baseline exists. `real_b0` and `baseline_artifacts` are both `FAIL`. `opengrad baseline --dry-run` is blocked, correctly, because the frozen held-out Parquet materialization is absent:

```text
evaluation materialization manifest is missing:
data/processed/normalization-v1/when2call-mcq/manifest.json
```

`data/processed/` is gitignored by design, so the held-out splits are not in the checkout. This blocker is reported, not bypassed.

## 10. SFT readiness — `false`

`ready_for_sft: false`. `ready_for_baseline: false`. CPU deterministic output never satisfies this contract.

## 11. Remaining blockers

`opengrad readiness --json` → `status: FAIL`, blocking gates:

| Gate | Status | Code |
| --- | --- | --- |
| `evaluation_materialization` | FAIL | `DATASET_NOT_FOUND` |
| `contamination_gate` | FAIL | `CONTAMINATION_FAILURE` |
| `gpu_boundary` | FAIL | `GPU_SMOKE_FAILED` |
| `real_b0` | FAIL | `BASELINE_NOT_FOUND` |
| `baseline_artifacts` | FAIL | `BASELINE_NOT_FOUND` |

Passing gates include `repository_validation`, `config_validation`, `model_revision`, `model_identity`, `tokenizer_revision`, `chat_template_contract`, `evaluation_manifest`, `evaluation_leakage`, `artifact_storage`, `native_parser` (module presence), and `gpu_probe`.

To unblock, in order: materialize the held-out splits from their pinned sources; complete the contamination review; resolve the native-parser smoke failure; then execute real B0.

## 12. Known limitations

- The MCP server has no third-party runtime dependencies and no vendor plugin, so it cannot reuse a harness's native approval UI. High-impact tools are gated by the server's own readiness checks plus the harness's permission model.
- Verification of the real held-out splits has not been executed here, because the artifacts are absent; only the gate logic and an end-to-end synthetic materialization are exercised.
- The native parser rejects an unclosed tool call in the bounded smoke. Until that is diagnosed, the GPU boundary cannot pass and real B0/SFT stay blocked.
- Post-SFT comparison and residual analysis remain deferred until artifact paths exist; the workflow reports `DEFERRED_UNTIL_ARTIFACT_PATHS` rather than inventing values.

## 13. Confirmation

No real SFT was run. No real B0 was run. No successful GPU boundary is claimed. No metric, score, or artifact in this repository was fabricated.
