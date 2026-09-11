---
pretty_name: OpenGrad ToolPolicy Canonical v2
language:
- en
license: other
task_categories:
- text-generation
configs:
- config_name: canonical
  data_files:
  - split: train
    path: "*.parquet"
---

> This is a provenance-preserving canonical candidate corpus. It is a pre-training canonical release, not an empirically selected or recommended training mixture.

## What this release is

OpenGrad ToolPolicy Canonical v2 is a provenance-preserving, model-independent normalization of
public tool-use datasets in which **every record declares what it supervises**. It exists because
not every legitimate post-training corpus has the same conversational trajectory shape, and
discarding a corpus for lacking an arbitrary turn is a loss of real supervision.

This is the **completed** v2 release: four sources, 173,237 canonical records, 176 shards. It is
distinct from the historical partial snapshot
([`OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`](https://huggingface.co/datasets/arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot)),
which is unchanged and still published. The two differ in three measured ways:

* this release includes xLAM/APIGen, which contributed **0 trainable records** to the partial
  snapshot and contributes **56,090** here;
* every record carries a supervision contract, so call-prediction and complete-trajectory targets
  are validated, rendered and measured separately;
* the partial snapshot's When2Call artifact was an **interrupted materialization** — it had read
  9,162 of the raw file's 15,000 rows and was never finalized — so it published 4,000 records
  where this build retains 6,505.

## Supervision contracts

Two contracts are implemented, and exactly one rule differs between them: whether a **terminal**
tool call requires a future environment response.

| Contract | Terminal target | Result required after the terminal call | Records here |
|---|---|---|---:|
| `COMPLETE_TRAJECTORY` | assistant final response | yes | 105,876 trainable |
| `CALL_PREDICTION` | assistant tool call | **no** | 56,090 trainable |

A `CALL_PREDICTION` record is a `query + tools -> assistant tool_call` example whose supervised
target *is* the call. No tool result is fabricated to make it look like a trajectory, and no
placeholder observation is inserted. The contract changes which turns are *expected*, never which
turns *exist*.

Everything else fails closed under both contracts: undeclared tools, invalid arguments against the
effective schema, malformed calls, duplicate call ids, orphaned results, FIFO order violations, and
a non-terminal call with no result. A malformed call stays quarantined even when its shape is a
valid call-prediction example.

## Source manifest

| Source | Role | Upstream | Pinned revision | Raw count | Canonical retained | Published count | License/terms | Adapter version |
|---|---|---|---|---|---:|---:|---:|---|---|
{{SOURCE_TABLE}}

## Composition and trainability

Measured at the training boundary by rendering every record, not by counting canonical rows.

| Source | Canonical | Trainable | Yield | Tool-call targets | Contract |
|---|---:|---:|---:|---:|---|
| glaive-function-calling-v2 | 98,339 | 97,112 | 0.988 | 48,723 | `COMPLETE_TRAJECTORY` |
| xlam-function-calling-60k | 57,342 | 56,090 | 0.978 | 56,090 | `CALL_PREDICTION` |
| toolace | 11,051 | 2,259 | 0.204 | 327 | `COMPLETE_TRAJECTORY` |
| when2call | 6,505 | 6,505 | 1.000 | 0 | `COMPLETE_TRAJECTORY` |
| **total** | **173,237** | **161,966** | | **105,140** | |

The mixture is **natural**: no supervision kind is reweighted and none is excluded.

### What each source actually supervises

* **xLAM/APIGen** is next-tool-call prediction. It contains no tool-result turn at all, which is
  why it needs `CALL_PREDICTION`; its upstream card documents `answers` as the call to make.
* **Glaive** supplies real multi-turn trajectories: every call is answered by a function-response
  turn.
* **ToolACE** retains 20.4% of its canonical records. 8,476 end on an assistant call that no tool
  message answers. Those are quarantined rather than reclassified, because the available bytes do
  not establish whether they are intended next-call supervision or truncated trajectories, and
  guessing would train on incomplete conversations while labelling them deliberate.
* **When2Call** supervises the *decision* in prose — decline, ask for the missing detail, or
  answer directly. It carries no structured call and no tool result, and that is its contribution
  rather than a defect.

## What is not included

| Source | Reason |
|---|---|
| `BUTTON` | Upstream repository is access-gated (HTTP 401). Not fetched, not approximated. |
| `LoopTool-23k` | Upstream source was not located. Not fetched, not approximated. |

xLAM's upstream is also access-gated: its records are reconstructed from the published Canonical-v1
derivative, which retains `tools[].parameters` unmodified. The derivation is representational and
verified record by record (59,370/59,370 with zero mismatches); it invents no field and repairs
nothing.

## Quarantine

| Cause | Records | Attribution |
|---|---:|---|
| Unrepresentable at the schema layer | 2,028 | Adapter refuses `Union`, `Callable`, `set` — no exact JSON Schema form |
| `SEM_ARGUMENT_INVALID` | 1,231 | **Upstream data quality**: the gold call's value contradicts the type the same record declares |
| Unresolved non-terminal calls | 8,476 | ToolACE records quarantined under the conservative classification |
| `TARGET_TRUNCATED` | 700 | Exceed the 2,048-token window after rendering |
| `UNRENDERABLE` | 2,095 | Trajectory contract violations, counted rather than repaired |

Quarantine is a result, not a shortfall. A record that cannot be interpreted under its declared
contract is reported rather than coerced into training text.

## Evaluation boundary

The frozen When2Call held-out namespace (`when2call-mcq`, `when2call-llm-judge`, and the preference
split) is **excluded** from this payload and remains separate in the OpenGrad repository. Two
held-out examples were adjudicated `CONTAMINATED` against this corpus and quarantined, leaving
3,950 distinct evaluation examples. Contamination review is complete for this corpus and its
evidence artifacts are corpus-scoped.

## Provenance and versioning

The release manifest records the OpenGrad commit, source manifest hashes, source revisions, adapter
versions, output shard hashes and release filters. The recorded commit contains the code that
produced this payload, so the release is reproducible from it: two independent builds at that
commit produce an identical manifest and 176 byte-identical shards.

This release is frozen. A future change is a new version with a new identity, never an in-place edit.

## Licensing and citations

OpenGrad source code is Apache-2.0. Upstream dataset terms remain source-specific and are documented
in `source-licenses.md`; this release does not relicense upstream data. See `CITATIONS.bib` for source
references.

## Responsible use

Use the data in accordance with each upstream source's terms and attribution requirements. Do not
infer that a valid tool call demonstrates reliable tool-use policy or task completion. Records
supervised under `CALL_PREDICTION` teach next-call prediction; they are not evidence of tool-result
interpretation, multi-turn recovery, or multi-step planning, and should not be described as complete
tool-use trajectories.
