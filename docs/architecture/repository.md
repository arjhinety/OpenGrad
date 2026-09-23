# Repository architecture

OpenGrad is a provenance-first empirical research repository. The current pre-GPU pipeline has four boundaries: registries (declarative identity and versions), canonical data (source-independent semantics), adapters/renderers (source and model protocol boundaries), and evidence (manifests, reports, runs, and artifacts).

## Current data flow

```text
upstream dataset
    -> source-specific adapter
    -> canonical semantic IR
    -> semantic validation and quarantine
    -> quality classification and deduplication
    -> contamination boundary
    -> behavioral metadata
    -> clean SFT candidate index
    -> exact-model tokenizer/chat-template renderer
    -> training or evaluation artifact
```

Universalize the meaning; specialize the representation. Canonical records remain model-independent. Qwen rendering is a separate exact-checkpoint operation using the pinned tokenizer/template contract.

## Canonical boundaries

`CanonicalSFTExample`, `CanonicalPreferenceExample`, and `CanonicalEvaluationExample` are separate contracts. Preference records preserve context, chosen, and rejected responses; rejected responses never silently become SFT targets. Evaluation records preserve expected decisions and candidates in an evaluation-only namespace; they never silently enter training.

## Current evidence boundary

The accessible pinned sources are normalized, audited, and rendered into local ignored artifacts. Manifests preserve source revisions, adapter versions, checksums, canonical schema, behavioral taxonomy, and retained counts. The frozen held-out evaluation is separate from training. Large data and rendered files remain outside Git; tracked schemas, configurations, small fixtures, and summary reports define the reproducible interface.

The first empirical model action is B0: unmodified `Qwen/Qwen3.5-2B` inference against the frozen
held-out evaluation. B0 is executed. The M0 SFT lineage has since closed with five completed arms
(one negative on corpus v1, one partial recovery on corpus v2, the definitive final-v2, and the
paired minus-xLAM joint-removal ablations), and the M1-v2 DPO parented on M0-final-v2 is promoted
under the parent-relative `tool_use_promotion_v4` gate, introduced after M0 was evaluated (M0 also
clears v4, M1-v2 fails the v3 gate that rejected M0, and its difference from M0 is within noise).
M2 on-policy distillation has not run — its scaffold has no live training path. Three external
benchmarks — IFEval, GSM8K and MMLU-Pro — were executed in the H200 capability campaign
([`results/final_campaign_verdict.json`](../../results/final_campaign_verdict.json)); no
speculative-decoding result exists. See
[`docs/EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) and
[`docs/EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md).

## Heterogeneous supervision

> OpenGrad does not require all datasets to share one conversational trajectory shape. It
> requires every dataset to declare what it supervises. Different supervision contracts may
> coexist in one experiment, but they remain explicitly typed, independently validated,
> separately measurable, and reproducibly mixable.

Not every legitimate post-training corpus has the same shape. xLAM/APIGen is
`query + tools -> assistant tool_call`: the supervised objective is next-call prediction, and the
corpus structurally contains no tool-result turn. Rejecting it as an unresolved trajectory
discards a useful dataset; accepting unresolved calls in general corrupts datasets that do claim
full trajectories. The resolution is a declared type, not a heuristic.

```text
UPSTREAM DATASET
      │
      ▼
SOURCE ADAPTER          normalizes the representation and states the upstream semantics
      │
      ▼
CANONICAL RECORD        messages + tools + provenance + metadata.supervision
      │
      ▼
CONTRACT-AWARE VALIDATION
      ├── COMPLETE_TRAJECTORY   every call resolved; terminal turn is an assistant response
      ├── CALL_PREDICTION       the terminal assistant call is the target and needs no result
      └── future contracts      added by declaring them, never by widening an existing one
      │
      ▼
RENDERER + LOSS MASK    the contract says which turns are context and which carry loss
      │
      ▼
TRAINING MIXTURE        measurable per kind, filterable, optionally weightable
```

The only rule a contract may change is whether a **terminal** call requires a future environment
response. Undeclared tools, invalid arguments, malformed calls, duplicate ids, orphaned results,
FIFO order violations, and a call appearing before a required result all remain invalid under
every contract, and a malformed call stays quarantined even when its *shape* is a valid
call-prediction example. An undeclared supervision kind is a failure, never a fallback; absence
of the field on a record that predates it is read as the stricter `COMPLETE_TRAJECTORY`.

OpenGrad explicitly rejects these:

```text
NO: dropping a useful dataset because it lacks an arbitrary preferred turn
NO: fabricating missing tool results
NO: treating terminal calls as unresolved trajectories without considering task semantics
NO: globally accepting orphaned calls
NO: source-name conditionals scattered through validators
NO: hiding supervision differences in renderer code
NO: reporting call-prediction records as complete trajectories
NO: collapsing all supervision kinds into one undifferentiated trainable count
```

Measured composition and the per-source mapping live in
[the supervision contract report](../../reports/SUPERVISION_CONTRACT_REPORT.md).

## Experiment state: authoritative versus derived

Experiment state has exactly one owner per value, and the discovery index is not one of them:

```text
Authoritative — owns the value
    runs/<experiment_id>/experiment.json    identity, configuration, dataset hashes, model and
                                            tokenizer revisions, environment, lifecycle status
    runs/<experiment_id>/eval/              per-checkpoint metrics, predictions, benchmark artifacts
    runs/central_ledger.jsonl               append-only lifecycle transitions

Derived — a rebuildable projection
    results/registry.jsonl                  one discovery/summary row per experiment
```

`ExperimentStore` is the only writer of `experiment.json` and the ledgers. The index is refreshed
only *after* an authoritative transition commits, is written atomically, and may be deleted and
regenerated byte-for-byte with `opengrad results rebuild-registry`. It therefore cannot become a
second source of truth: `opengrad results validate-registry` reports drift between the projection
and the artifacts rather than letting the two diverge silently. See
[the results namespace](../../results/README.md).

## Repository layout

Every tracked top-level directory, with one role each. `tests/results/test_generated_indexes.py` fails when a
tracked top-level directory is missing from this table, so the map cannot fall behind the tree.

| Path | Role |
|---|---|
| `src/` | The `opengrad` package: canonical data and adapters, rendering and parsing, training, evaluation, benchmarks, contamination, promotion gates, verification, registry and reporting. All library code lives here. |
| `tests/` | CPU-safe tests; the tree mirrors `src/opengrad/`. |
| `scripts/` | Campaign, audit, freeze and reporting tooling that is run by hand or by CI. Indexed in [`scripts/README.md`](../../scripts/README.md) (generated). Reusable logic belongs in `src/`. |
| `configs/` | Versioned definitions: experiments, data, releases, evaluation, benchmarks, annotation tasks, training, inference. Directories holding only a README are reserved names, not implementations. |
| `registry/` | Dataset, model, benchmark, runtime and hardware registries and their JSON schemas; validated by `opengrad-validate`. |
| `runs/` | **Authoritative** experiment state (`runs/<id>/experiment.json`, `eval/`, ledgers), written only by `ExperimentStore`. |
| `results/` | The derived experiment index (`results/registry.jsonl`) **plus** benchmark and quantization campaign results, which have no `runs/` identity. Under `results/benchmarks/` and `results/quantization/` the per-benchmark scores and ledgers are authoritative; see [`results/README.md`](../../results/README.md). |
| `reports/` | Written analyses, audits and closures, and the evidence behind Study 002 (populations, annotations, audit trails, provenance). Indexed in [`reports/README.md`](../../reports/README.md) (generated), with a study column. `reports/ERRATA.md` corrects frozen files. |
| `docs/` | Living specifications and methodology; the studies under `docs/research/`. Indexed in [`docs/README.md`](../README.md) (generated). Terms: [`docs/GLOSSARY.md`](../GLOSSARY.md). |
| `manifests/` | Pinned input manifests for quantization (the PTQ closure and calibration sets). |
| `data/` | Local data, git-ignored except two small preference-pair files under `data/processed/` that are tracked on purpose (`registry/datasets.yaml`). |
| `release/` | Generated Hugging Face, GGUF and ExecuTorch release bundles and model cards, as published. |
| `hf/` | Card and report templates for Hugging Face releases. |
| `.release/` | Local release build output, git-ignored except one release manifest a freeze pins. |
| `integrations/` | Harness-facing integrations over the `opengrad … --json` boundary: `opengrad-mcp/` (stdio MCP server), `annotate-ui/` (the annotation web UI), and ExecuTorch/Hugging Face/W&B/OpenPapers adapters. |
| `plugins/` | The `opengrad` Claude Code plugin: the development skills, kept current by `tests/skills/`. |
| `third_party/` | Vendored code, byte-exact and hash-pinned (the IFEval checkers). |
| `experiments/` | A pointer only: experiment records live in `runs/` (see its README). |
| `assets/` | Images for the README. |
| `.github/` | CI workflow, issue and PR templates. |
| `.claude/`, `.claude-plugin/` | Project settings that register the plugin marketplace, and the marketplace manifest. |

**Where a study's material lives.** Study 001's written record is the flat `reports/*.md` files plus
`docs/EXPERIMENT_RESULTS.md`, frozen at tag `study-001`. Study 002's text is `docs/research/study-002/`; its
evidence is under `reports/pdet*/`, `reports/prose-classifier/`, `reports/normalization-v3/` and
`reports/canonical-v3/`. Frozen evidence is not moved into per-study folders, because its paths are pinned;
the generated `reports/README.md` records the study of every file instead. **New studies** write evidence to
`reports/study-00N/` and name files in kebab-case with the version last (`population-v2.jsonl`).

Training and inference have been executed on an A100. Large data and checkpoints remain outside Git
and must be referenced by immutable revisions and hashes.
