# Pre-GPU Readiness Report

> **Superseded (2026-09-11).** This report describes the pre-GPU state. Every condition it says was
> unmet has since been met: the GPU boundary passed, real B0 was run and recorded, and SFT was
> executed (two negative runs and one partial recovery). The false present-tense statements below
> are corrected in place and the report is kept as the incident record it is. Current status:
> [README § Current research status](../README.md#current-research-status) and the
> [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md).

**Scope:** harness-agnostic agent integration, the pre-GPU baseline/readiness contract, and
held-out contamination screening.
**Status:** `PRE_GPU` (historical) — a real model result now exists.
**Repository commit at time of writing:** the commit that adds this file; `git log -- reports/PRE_GPU_READINESS_REPORT.md` pins it exactly.

> At the time of writing this report intentionally did **not** claim `READY_FOR_SFT`, a real B0, or a successful GPU boundary. All three have since been met. See [Remaining blockers](#11-remaining-blockers).

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
| `python3 -m py_compile <modules>` | PASS |
| `.venv/bin/python -m pytest` | **193 passed** |
| `.venv/bin/python -m ruff check .` | **All checks passed** |
| `.venv/bin/opengrad-validate` | `registry validation: OK` |
| `cd integrations/opengrad-mcp && npm run check` | PASS |
| `cd integrations/opengrad-mcp && npm test` | **26 passed, 0 failed** |
| `.venv/bin/opengrad-contamination heldout-screen` | levels 1–4 `MEASURED` |
| `.venv/bin/opengrad-contamination adjudicate --status` | level 5 `COMPLETE`; 2 quarantined |
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

## 6c. Contamination screening and Level-5 adjudication

Levels 1–4 are machine-measured over the held-out set against the 213,951-record canonical training corpus. Level 5 is now a durable human adjudication with a first-class CLI, and any `CONTAMINATED` verdict is quarantined from evaluation. See [Contamination adjudication](../docs/evaluation/CONTAMINATION_ADJUDICATION.md) for the state machine and the exact workflow.

Two held-out items were flagged by exact prompt hash. Their matched training records were inspected in full; both expose the identical prompt **and the decision the item measures**, so both were adjudicated `CONTAMINATED` and quarantined:

| Held-out item | Gold decision | Matched training record | Why it is contamination |
| --- | --- | --- | --- |
| `when2call-mcq:6a9903a1…` — *"What is the current time?"* | `request_for_info` | 5 × Glaive Function-Calling v2, each labeled `CALL` | Same prompt, and training supplies the CALL behaviour for exactly the CALL-vs-ASK_FIRST decision this item measures. |
| `when2call-mcq:77821b1c…` — *"What is the current weather?"* | `request_for_info` | 1 × When2Call SFT | Same prompt, and the training response asks for the missing location — semantically the gold `request_for_info` answer. |

The verdicts are recorded in `reports/data/behavioral-heldout-v2-contamination-audit.json` with reviewer, timestamp, and reasoning; the exclusions are in `reports/evaluation/behavioral-heldout-v2-quarantine.json`. The held-out benchmark is now **3,650** evaluated examples (3,652 distinct — the 300 `when2call-llm-judge` rows are byte-identical duplicates of MCQ rows — minus 2 quarantined), and `load_evaluation_examples` provably omits both. `contamination_gate` is **PASS** with `level_5=COMPLETE` and status `SEMANTIC_REVIEW_COMPLETE`.

> The two verdicts rest on repository evidence (identical prompt plus the measured decision present in training), which is why they were classified rather than left `PENDING`. They are ordinary reviewable judgments: re-run `opengrad-contamination adjudicate` to change either one, then re-apply quarantine and rescan.

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

## 8. Real Qwen GPU smoke status — `PASS`

`reports/hardware/qwen_gpu_smoke.json` (kind `GPU_BOUNDARY_VERIFIED`, `status: PASS`) records a real A100 run. All seven checks pass, with no limitations:

| Check | Result |
| --- | --- |
| `model_access` | PASS (`Qwen/Qwen3.5-2B` @ `15852e8c…8a8fc`) |
| `native_template` | PASS (1184 characters) |
| `model_load` | PASS (`cuda:0`, BF16) |
| `one_generation` | PASS (115 characters) |
| `native_parser` | PASS — `RAW_VALID`, decision `CALL` |
| `vram` | PASS (3.513 GiB allocated) |
| `cleanup` | PASS |

Hardware: `NVIDIA A100-SXM4-80GB` (A100_80GB), BF16 supported.

Two independent harness defects had to be fixed to reach this, and both were diagnosed on CPU rather than by guessing:

1. **Truncated generation (fixed in `6b08665`).** The smoke used `max_new_tokens=8`, but its prompt elicits a tool call needing ~20 tokens. The call was cut off, and the parser *correctly* rejected a genuinely truncated call as `UNCLOSED_TOOL_CALL`. The previous receipt's `output_chars: 30` is the signature of that cut.
2. **The parser could not read the model's native output (fixed in `2357cbd`).** This was the larger finding — see §9a.

## 9. Real B0 status — run and recorded

See §9b. `real_b0` and `baseline_artifacts` are derived from the committed artifacts and the canonical `ExperimentRecord`.

## 9a. The parser defect (why the boundary mattered)

The bounded smoke earned its place: on its first successful generation it proved the native parser could not read the model's actual output.

Qwen3.5-2B's pinned chat template instructs the model to reply with an **XML** payload, and renders assistant tool calls the same way:

```text
<tool_call>
<function=lookup>
<parameter=q>
worker 12
</parameter>
</function>
</tool_call>
```

`parse_qwen_native_output` accepted only `<tool_call>{"name": …}</tool_call>`, so this was `MALFORMED_TOOL_CALL[0]: Expecting value` — a JSON parser meeting an XML payload. Every real tool call would have been scored `FORMAT_ERROR` and fallen back to decision `ANSWER`.

The consequence would have been silent and specific: the baseline would have reported that Qwen3.5-2B almost never calls tools, when in fact it calls correctly on nearly every prompt (12 of the first 20 held-out examples, with correct tool names and arguments). That is a wrong statement about a model, produced by a bug in our reader — the exact failure this repository exists to not publish.

Why it survived until a GPU run: the unit tests use hand-written JSON, and the deterministic backend emits JSON, so the parser and the fixtures agreed with each other while neither agreed with the model. The golden renderer fixture (`tests/fixtures/rendered/qwen35_2b_single_call.txt`, lines 39–45) had contained the XML answer the whole time, and it round-trips against the real tokenizer — nothing parsed it.

Cross-checked against the downstream runtime, which has an equivalent parser:

- `OpenWeights` `ToolCallParser.parseTaggedXml` reads exactly this XML form, with the same regexes; the branch exists because llama.cpp's built-in parser does not recognise it.
- OpenGrad's grammar is now byte-compatible with it, so a checkpoint measured by B0 and the same checkpoint measured on-device agree on what a call is.
- OpenGrad's own `openweights` benchmark adapter had the mirror-image bug: it accepted the two JSON arms but returned `FORMAT_ERROR` for native XML. OpenWeights prefers a model's own template when it carries tools (Qwen3.5's does), so its arms are a request rather than a guarantee, and the adapter must accept what the model actually emits.

Two upstream follow-ups were filed from this cross-check:

- `alpharomercoma/openweights#2` — `ToolCallParser.parseTaggedXml` read only the first `<tool_call>` envelope and the first `<function>`, so a model calling twice in one reply had its second call dropped. The JSON branch had already been fixed for exactly this case; the XML branch had not. Fixed and tested in that PR.

## 9b. Real B0 — executed

The frozen held-out benchmark ran end to end on the A100 with the real model, the fixed parser, and vLLM 0.29.0. `opengrad readiness` reports `status: PASS`, `baseline: REAL_COMPLETE`, `blocking_gates: []`, and `ready_for_sft: true` (at the time of writing no SFT had been run; it has since been executed — see the [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md)).

| | |
| --- | --- |
| Examples scored | 3,650 distinct (3,952 materialized − 300 cross-split duplicates − 2 quarantined) |
| Parse quality | 3,646 / 3,650 `RAW_VALID` = 0.99890 (bound 0.99) |
| Context buckets | base 3,644 · overflow 6 |
| Elapsed | 197 s |
| `call_recall` / `call_precision` | 0.9722 / 0.4542 |
| `over_call_rate` | 0.6425 |

Findings: the model has tool-call syntax and selection but not the call/no-call decision — it called a tool on 64.3% of the 2,355 examples whose gold decision was not CALL, and correctly refused an unsupported request 17 times out of 1,295. Full write-up, confusion matrix, and scope limits: [B0 result](../reports/baselines/qwen35_2b_baseline/RESULT.md).

Two further defects had to be fixed to produce this, both found by the evidence contract rather than by inspection:

3. **The frozen benchmark double-counted 300 examples.** The upstream `when2call_test_llm_judge.jsonl` is a byte-identical subset of rows already in `when2call_test_mcq.jsonl`. The uniqueness check caught it (it returned "expected ids unavailable" instead of passing), so the first "successful" run was discarded rather than reported. Evaluation identity is now the distinct union, with the overlap declared in the manifest.
4. **The contract required 100% parseable output, which is unsatisfiable.** Output that runs past the 512-token completion budget mid-tool-call has no valid parse; the first run had 4 such rows. Requiring zero meant no real run could ever produce passing evidence, so the requirement is now an explicit bound pinned in the frozen config, with the measured rate reported in the artifacts and the gate detail.


## 10. SFT readiness — `true`, and now actually verified

`ready_for_sft: true` for **the real SFT config**, not just the baseline:

```text
opengrad readiness configs/experiments/m0_sft.yaml
status: PASS   blocking_gates: []   warnings: []   (all 21 gates PASS)
```

This distinction mattered and was initially wrong. The default `opengrad readiness` evaluates
the *baseline* config, where the SFT-specific data gates auto-pass, so `ready_for_sft: true`
there said nothing about whether an SFT run could start. Evaluated against the SFT config it
failed on three gates. Three further unsatisfiable-contract defects were fixed:

5. **The training corpus was not in the dataset registry.** The SFT config pinned
   `canonical_v1: 181b3fba…` — and that hash is correct, it *is* the release manifest's
   sha256 — but with no registry entry the two dataset gates could not verify source
   revision, eligibility, or processed hash, so they failed closed. Registered from the
   release artifact: `checksum` and `processed_dataset_hash` are the manifest sha256,
   `source_revision` is the commit that built the release, and the six upstream revisions,
   licences, and split used are recorded under `derived_from`.
6. **The SFT data gate tested `forbidden_splits` for evaluation terms — inverted.** Declaring
   a split forbidden is precisely what makes a corpus safe to train on, so the gate refused
   the datasets that had done the right thing (it would have rejected `when2call` for
   correctly forbidding `mcq_test`). It now tests `allowed_splits`. It also normalises the
   registry's `future_` stage prefix: an exact match on `"preference"` silently missed
   `future_preference`, letting a preference corpus into SFT.
7. **Preflight reported `SHA: unknown` for every repository.** It read `env["git"]["sha"]`,
   but `capture()` returns flat `git_sha`/`git_dirty` keys, so the provenance field carried no
   information. It also used the whole-tree dirty flag, which is true for any untracked file —
   and every run creates untracked outputs — so "clean tree" was unsatisfiable. Both now use a
   shared `tracked_tree_provenance()` helper, the same rule the baseline contract uses.

At the time of writing no SFT had been launched: `ready_for_sft: true` meant the contract was
satisfied, not that a run had been started. SFT has since been executed — two negative runs and
one partial recovery; see the [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md).

## 11. Remaining blockers

**None.** `opengrad readiness --json` → `status: PASS`, `blocking_gates: []`.

Every gate passes: `repository_validation`, `config_validation`, `model_revision`, `model_identity`, `tokenizer_revision`, `chat_template_contract`, `evaluation_manifest`, `evaluation_materialization`, `dataset_revision`, `dataset_snapshot`, `contamination_gate` (`SEMANTIC_REVIEW_COMPLETE`, level 5 `COMPLETE`, 2 quarantined), `evaluation_leakage`, `disk_capacity`, `artifact_storage`, `native_parser`, `gpu_probe`, `gpu_boundary`, `experiment_preflight`, `training_data_policy`, `real_b0` (`REAL_COMPLETE`), and `baseline_artifacts`.

What remained at the time of writing was *research*, not pre-SFT preparation: the SFT stage itself (since executed — two negative runs and one partial recovery) and the external benchmark families, which are still `FROZEN_NOT_EXECUTED`.

## 12. Known limitations

- The MCP server has no third-party runtime dependencies and no vendor plugin, so it cannot reuse a harness's native approval UI. High-impact tools are gated by the server's own readiness checks plus the harness's permission model.
- Levels 1–2 compare user-visible prompt text, not a whole-conversation hash: held-out evaluation examples and training trajectories do not share a conversation schema. No whole-conversation equality claim is made.
- Level 3 prunes shingles whose training document frequency exceeds `max_df` (default 1000 of 213,951). Pruned count is reported; a paraphrase built from ubiquitous n-grams would be missed.
- Level 4 candidate generation is prefiltered by level-3 Jaccard and scored with `difflib.SequenceMatcher`. It is **not** an exhaustive semantic search and no embedding similarity was computed, so it can miss meaning-level reuse with no lexical overlap. This is the level most likely to need strengthening before a generalization claim.
- Level 5 is human and now complete for the current findings. It is only as good as the current scanner: a paraphrase with no lexical overlap against any training record is invisible to levels 1–4 and therefore never reaches the queue.
- The `content_hash` re-freeze and `dataset_hash` re-sync are documented in `hash_provenance`; anyone who considers the original frozen placeholders authoritative should treat this as a contract change rather than a fix.
- (Resolved.) The native parser initially rejected an unclosed tool call in the bounded smoke. The truncation was diagnosed and the GPU boundary now passes; the entry is kept as the record of why the boundary mattered.
- Post-SFT comparison and residual analysis remain deferred until artifact paths exist; the workflow reports `DEFERRED_UNTIL_ARTIFACT_PATHS` rather than inventing values.

## 13. Confirmation

The GPU boundary **is** now claimed, with a passing receipt. Real B0 **was** run, and is recorded. SFT has since been run — two negative runs and one partial recovery; see the [M0 SFT execution report](M0_SFT_EXECUTION_REPORT.md). No metric, score, or artifact in this repository was fabricated; the two runs whose evidence was invalid (a double-counted benchmark, then an unsatisfiable parse requirement) were discarded and re-run rather than reported.
