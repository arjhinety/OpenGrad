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

| Source | Role | Upstream | Pinned revision or input-file SHA-256 † | Raw count | Canonical retained | Published count | License/terms | Adapter version |
|---|---|---|---|---:|---:|---:|---|---|
{{SOURCE_TABLE}}

† For xLAM and When2Call this is a Hub commit. For Glaive and ToolACE it is **not** a Hub
revision: it is the SHA-256 of the local input file the adapter read (for ToolACE, the first 16
hex characters of `7a7a6a2c3b1003c789bb…`), and it does not match the Hub file's LFS hash either.
The Hub revisions are `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` (Glaive) and
`6bda777c88d21e5a204703c1ee45597a8fa4f734` (ToolACE): Canonical-v1 records the same input files,
by SHA-256, against those revisions, and the Hub history shows neither upstream data file has
changed since it was uploaded.

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

### Refusal text under `ANSWER` labels

The refusal detector used to score model outputs (`HEURISTIC_REGEX_v1`) finds refusal-shaped
supervised targets that carry the decision label `ANSWER`. Counting single-exchange records only
(a multi-turn record's label describes its first exchange, not its last turn), **18,114 of the
173,237 records (10.5%)** have one, and all 18,114 are labelled `ANSWER`:

| Source | Refusal-shaped `ANSWER` targets |
|---|---:|
| when2call | **4,038 of 6,505 (62.1%)** |
| glaive-function-calling-v2 | 14,066 of 98,339 |
| toolace | 10 of 11,051 |
| xlam-function-calling-60k | 0 of 57,342 |

A record labelled `ANSWER` whose target declines teaches that answering can look like declining.
M0, trained on this corpus, and M1-v2, trained from M0, decline all 1,319 zero-shot GSM8K
questions; M1-v1, DPO applied directly to the base model without this corpus, declines 70.7%.
The count is descriptive and does not establish that these records caused that behaviour; that
would need a retrain without them. When2Call entered M0's training in full, so its count applies
exactly; Glaive's and ToolACE's are over release records, some of which M0's renderer
quarantined. Source:
`results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json`, which
supersedes an audit of the normalization-v1 sources (21,749 of 217,903) that was not this corpus.

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
| Unrepresentable at the schema layer | 2,028 | xLAM, at canonicalization (59,370 → 57,342), so not among the 173,237 canonical records. Adapter refuses `Union`, `Callable`, `set` — no exact JSON Schema form |
| `SEM_ARGUMENT_INVALID` | 1,231 | xLAM. **Upstream data quality**: the gold call's value contradicts the type the same record declares |
| Unresolved non-terminal calls | 8,476 | ToolACE records quarantined under the conservative classification |
| `TARGET_TRUNCATED` | 700 | Exceed the 2,048-token window after rendering (Glaive 363, ToolACE 316, xLAM 21) |
| `UNRENDERABLE` | 864 | Glaive records that fail the argument or trajectory contract, counted rather than repaired |

The last four rows are the training-boundary drops: 1,231 + 8,476 + 700 + 864 = **11,271** of the
173,237 canonical records, leaving 161,966 trainable. An earlier version of this table gave
`UNRENDERABLE` as 2,095, which counted xLAM's 1,231 a second time.

Quarantine is a result, not a shortfall. A record that cannot be interpreted under its declared
contract is reported rather than coerced into training text.

## Evaluation boundary

The frozen When2Call held-out namespace (`when2call-mcq`, `when2call-llm-judge`, and the preference
split) is **excluded** from this payload and remains separate in the OpenGrad repository. It
holds 3,952 records but 3,652 distinct examples, because all 300 LLM-judge rows duplicate MCQ
rows. Two held-out examples were adjudicated `CONTAMINATED` against this corpus and quarantined,
leaving 3,650 distinct evaluation examples. Contamination review is complete for this corpus and
its evidence artifacts are corpus-scoped.

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
