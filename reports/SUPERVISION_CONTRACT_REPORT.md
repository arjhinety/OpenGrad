# Supervision contract report

**Status:** `IMPLEMENTED` and `MEASURED`. xLAM is rebuilt under the contract and audited over all
59,370 retained records. Canonical-v2 is **not** final — see §8.

**Date:** 2026-09-11
**Code:** `src/opengrad/data/supervision.py`, `src/opengrad/data/semantic.py`
**Machine-readable evidence:** `reports/data/xlam-normalization-forensics.json`

---

## 1. The problem this solves

Not every legitimate post-training dataset has the same trajectory shape, and OpenGrad was
discarding useful supervision because of it.

xLAM/APIGen is `query + tools -> answers`, where an answer names the call to make. It contains no
tool-result turn, because the supervised objective is *next-tool-call prediction*. OpenGrad read
that as an unresolved trajectory and rejected the source: **0 of 59,370 records were trainable**,
while every canonical-count gate passed.

The alternative — accepting unresolved calls in general — would corrupt datasets that do claim
full trajectories, and would mean training on incomplete conversations as though they were
complete. Neither option is acceptable, so the distinction is now explicit and typed.

**Governing principle:** dataset heterogeneity is allowed; semantic ambiguity is not.

## 2. The contracts

| Kind | Terminal target | Result required after terminal call | Policy |
|---|---|---|---|
| `COMPLETE_TRAJECTORY` | `assistant_final_response` | **yes** | `complete_target_v1` |
| `CALL_PREDICTION` | `assistant_tool_call` | **no** | `call_prediction_v1` |

`SupervisionContract` carries the rules — `kind`, `terminal_target_type`,
`tool_result_required_after_terminal_call`, `allow_intermediate_calls`,
`require_intermediate_results`, `supervised_message_roles`, `validation_policy` — in one registry
keyed by the `SupervisionKind` enum. Adding a kind means declaring it there; there is no path that
widens an existing contract.

### The relaxation is exactly one rule

Everything else fails closed under **every** contract, and this is tested rather than asserted:

* undeclared tool names → `SEM_UNDECLARED_TOOL`
* invalid arguments against the effective schema → `SEM_ARGUMENT_INVALID`
* malformed argument JSON → `MALFORMED_TOOL_ARGUMENTS`
* duplicate call ids → `SEM_DUPLICATE_CALL_ID`
* orphaned tool results → `SEM_ORPHAN_RESULT`
* FIFO order violations and messages before a required result → `SEM_RESULT_ORDER`
* a call that is *not* the terminal target and has no result → `SEM_UNRESOLVED_CALL`

That last rule is the anti-pattern guard: the exemption covers a declared terminal target, never a
stray call. A record whose declared kind is undeclared is rejected; a record predating the field
is read as the stricter contract and reported as `legacy_default` rather than as declared.

## 3. Source → supervision-kind mapping

Measured from the 213,951 records of the published Canonical-v1 release. "Unresolved (terminal)"
counts records whose only unanswered calls are in the final assistant turn.

| Source | Records | Ends on assistant call | Has tool result | Unresolved calls | Unresolved & terminal | Assigned kind | Evidence |
|---|---:|---:|---:|---:|---:|---|---|
| xLAM/APIGen | 59,370 | 59,370 | **0** | 59,370 | 59,370 | **`CALL_PREDICTION`** | Verified: upstream card documents `answers` as the call to make; 0 records have a tool-result turn; every gold argument key is a parameter name |
| Glaive FC v2 | 99,794 | 0 | 51,009 | **0** | 0 | `COMPLETE_TRAJECTORY` | Every call is answered by a function-response turn. 48,785 records have no calls at all and end in prose |
| ToolACE | 11,190 | 8,543 | 793 | 8,573 | 8,543 | `COMPLETE_TRAJECTORY` (**mixed, unresolved**) | 8,543 records end on an unanswered call, which is call-prediction shaped. Not reclassified — see §4 |
| LoopTool-23k | 20,827 | 20,192 | 14,448 | 9,288 | 9,288 | `COMPLETE_TRAJECTORY` (**mixed, unresolved**) | Same shape, 9,288 records. 635 records have no calls |
| BUTTON | 7,941 | 0 | 7,941 | 2,359 | **0** | `COMPLETE_TRAJECTORY` | 5,582 fully resolved; 2,359 ask for several tools in one turn and never answer all of them |
| When2Call | 14,829 | 0 | 0 | 0 | 0 | `COMPLETE_TRAJECTORY` | No structured calls and no results; the call/answer/clarify decision is expressed as prose |

Only xLAM's classification is new. **No other source's classification changed**, which the task
required.

## 4. The open question I deliberately did not resolve

ToolACE (8,543 records) and LoopTool (9,288 records) contain a large call-prediction-shaped
subset: the conversation ends on an assistant call that is never answered. That shape is
*consistent* with intended next-call supervision, and it is equally consistent with truncated
trajectories whose results were dropped upstream.

**The available bytes cannot distinguish those.** Reclassifying on structure alone would be
guessing, and the guess is not harmless: if the records are truncated, calling them
`CALL_PREDICTION` would train on incomplete conversations while reporting them as intentional
targets. Leaving them as `COMPLETE_TRAJECTORY` keeps them quarantined, which is the
least-assumptive reading and the status quo.

Resolving this needs upstream evidence — a documented format, a maintainer statement, or a
construction pattern that shows the terminal call is the intended target. Until then the counts
above are the honest characterisation, and the yield report counts them explicitly rather than
folding them into a percentage.

### A second finding: BUTTON's dangling calls

2,359 BUTTON records emit calls that no tool message answers. Inspecting one:

```text
[2] assistant  calls=[call_0000, call_0001, call_0002]
[3] tool       tool_call_id=call_0000
[4] assistant  calls=[call_0003]
[5] tool       tool_call_id=call_0001
[6] assistant  prose (terminal)
```

An assistant turn requests three tools and the conversation returns results for only two, so
`call_0002` and `call_0003` are never answered. These are invalid under **both** contracts,
because the terminal turn is prose and there is therefore no call-prediction target either. They
stay quarantined. This is the correct outcome: the trajectory is genuinely incomplete, and the
absence of a contract clause that rescues it is the point.

An earlier adapter comment claimed every BUTTON call had a result. That was measured by presence
of a `tool` message rather than by call/result identity, and it was wrong; the comment is
corrected.

## 5. Provenance

Every canonical record carries `metadata.supervision`:

```json
{
  "kind": "CALL_PREDICTION",
  "assignment": "upstream_declared",
  "adapter": "xlam_function_calling_60k_v2",
  "adapter_version": "1.0.0",
  "contract_version": "supervision_contract_v1",
  "validation_policy": "call_prediction_v1",
  "note": "upstream card documents answers as the call to make; the corpus has no tool-result turn"
}
```

`assignment` distinguishes `upstream_declared`, `source_adapter`, `derived`, `legacy_default` and
`manual`, so a reader can tell a classification that upstream stated from one an adapter inferred
and from one inherited for compatibility. The block survives canonical serialization and the
materialization round-trip, which is tested.

## 6. Rendering and loss masking

**No new rendering path was needed and none was added.** Assistant-span masking already covers
every assistant turn, so a `CALL_PREDICTION` record renders exactly its messages and its terminal
call is supervised. Verified on real xLAM bytes:

* the rendered text contains no `<|im_start|>tool` turn — no result was fabricated;
* the supervised span equals the terminal call turn (`target_span` `[304, 366]`, 62 supervised
  tokens for the sampled record);
* `build_sample` now *asserts* the terminal call falls inside a supervised span and records
  `target_span`, so a future masking change cannot silently drop the target and leave a sample
  that looks trainable while supervising nothing.

Both directions are tested: a `CALL_PREDICTION` record renders deterministically and its target
carries loss; a `COMPLETE_TRAJECTORY` record still supervises every assistant turn and not the
tool results.

## 7. Mixture visibility

A single trainable count cannot distinguish 50k call-prediction records from 50k complete
trajectories, so composition is reported per kind:

* `SupervisedSample.supervision_kind`, from the canonical record;
* `supervision_composition()` → per-kind records, targets and share of trainable;
* the yield gate reports `supervision_kinds_trainable` and `supervision_kinds_canonical` per
  source, and prints the breakdown in its table;
* readiness gains a `supervision_composition` gate that fails when a config selects a kind the
  corpus does not contain (`SUPERVISION_SELECTION_MISMATCH`) or when a trainable record carries no
  kind at all (`SUPERVISION_UNCLASSIFIED`). It is dormant for configs that declare no yield
  report, and says so.

`supervision.include` supports the ablation the task asked for — full corpus vs one kind — without
rebuilding canonical data, and it is part of the sample-cache identity so a filtered run cannot
reuse an unfiltered cache.

`supervision.sampling_weights` is **rejected, not accepted**: weighting is not implemented, and a
declared weight the trainer ignores would change an experiment's meaning without changing its
output. That is a deliberate gap rather than a silent one.

## 8. Canonical-v2 status

**Not final.** The contract changes the semantic meaning of canonical records, so the rebuilt
corpus is a new artifact with a new fingerprint, and this session did not rebuild all sources.
The historical facts are unchanged and remain stated on the published snapshot:

```text
Corpus-v2 snapshot used for the successful M0:
    xLAM training contribution = 0
```

Canonical-v2 becomes final only after: the xLAM rebuild is included in a corpus build, the
ToolACE/LoopTool classification question is resolved or explicitly deferred in writing, the
renderability gate is declared by the SFT config so it runs rather than idles, and contamination
is re-audited for the new composition.

## 9. Whether another SFT is justified

**Not yet, and not by this change alone.** The architecture now admits call-prediction
supervision, which is a real capability, but no corpus has been rebuilt and released with it, so
there is nothing new to train on. A next controlled M0 is justified once Canonical-v2 is frozen
with a measured supervision composition — at which point the interesting comparison is
deliberately available: the same procedure on a corpus with call-prediction supervision versus the
corpus without it, reported on call recall, tool selection and argument accuracy separately from
tool-result interpretation and multi-turn recovery.
