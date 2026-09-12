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
paired minus-xLAM joint-removal ablations), and the M1-v2 DPO parented on M0-final-v2 is promoted.
M2 on-policy distillation has not run — its scaffold has no live training path. No
speculative-decoding or external-benchmark result exists. See
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

| Path | Role |
|---|---|
| `registry/` | Dataset, benchmark, model, runtime, hardware, provenance, and experiment contracts |
| `src/opengrad/` | Canonical data, adapters and renderers, parsing, contamination tools, evaluation schemas, lineage, stage gates, reporting utilities |
| `configs/` | Data, evaluation, model, training, inference, and planned experiment configurations |
| `runs/` | Authoritative experiment state (`runs/<id>/experiment.json`, `eval/`, ledgers) |
| `results/` | Derived, rebuildable result index (`results/registry.jsonl`) |
| `reports/` | Written experiment analyses, baselines, releases, and incident records |
| `experiments/` | Evidence namespace for experiment definitions and closures |
| `docs/` | Methodology, architecture, data, benchmark, inference, reproducibility, contribution, and publication protocols |
| `integrations/` | Harness-facing integrations over the `opengrad … --json` boundary (`opengrad-mcp/` stdio server) |
| `release/`, `hf/` | Tracked Hugging Face release definitions and card/report templates |
| `scripts/` | Execution, evaluation, freezing, and reporting tooling |
| `tests/` | CPU-safe fixture and contract tests |

Training and inference have been executed on an A100. Large data and checkpoints remain outside Git
and must be referenced by immutable revisions and hashes.
