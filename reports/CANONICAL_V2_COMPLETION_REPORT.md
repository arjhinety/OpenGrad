# Canonical-v2 completion report

**Status:** `FROZEN` — four sources, 173,237 canonical records, fingerprint
`8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161`, proven deterministic by two
independent builds. BUTTON and LoopTool-23k remain excluded because their upstreams are
unavailable, and that is stated rather than worked around.

**Date:** 2026-09-11
**Supersedes:** the "v2 candidate" framing in `configs/releases/toolpolicy_canonical_v2.yaml`

This report states what Canonical-v2 can and cannot claim after the xLAM normalization work.
It does not restate the M0/M1 experiment results, which are in
[`M0_SFT_EXECUTION_REPORT.md`](M0_SFT_EXECUTION_REPORT.md).

---

## 1. Headline

The corpus is frozen at **173,237 canonical records / 161,966 trainable**, from four sources, with
every record interpreted under an explicit supervision contract.

Three independent defects were found and fixed on the way here, in three different layers. Two
were ours; one was upstream; and one of the findings was that the *previous* partial snapshot was
itself incomplete in a way nobody had noticed.

| Fix | Layer | Effect |
|---|---|---|
| xLAM property-map normalization | our adapter | canonical acceptance 33 → 57,342 |
| Supervision contract (`CALL_PREDICTION`) | our policy | xLAM trainable 0 → 56,090 |
| `required: null` treated as a conflicting value | our validator | ToolACE accepted 697 → 11,051 |
| Interrupted When2Call materialization | our pipeline | when2call published 4,000 → 6,505 |

The xLAM source-adapter defect was verified across all 59,370 retained records: canonical
acceptance moved from **33 → 57,342** (0.06% → 96.6%).

That was the first blocker. The second was a *policy* problem rather than a parsing one: every one
of those 57,342 records was then rejected at the training boundary with `SEM_UNRESOLVED_CALL`,
because xLAM is single-turn call prediction with no tool result and the trajectory policy required
every call to be resolved.

Both are now fixed, and they were genuinely different problems in different layers. The schema
blocker needed an adapter-level transformation; the trainability blocker needed an explicit
statement of *what the corpus supervises*, which is now the
[supervision contract](SUPERVISION_CONTRACT_REPORT.md):

| xLAM | Before | After |
|---|---:|---:|
| Schema-valid | 33 | 57,342 |
| `SEM_UNRESOLVED_CALL` | 56,111 | **0** |
| Trainable | **0** | **56,090 (94.5%)** |
| Tool-call targets (trainable) | 0 | 56,090 |

The remaining 1,252 non-trainable records are 1,231 genuine upstream argument defects and 21 that
exceed the token window. None is attributable to OpenGrad's parsing or validation.

---

## 2. Per-source state

**The frozen corpus**, measured by rendering every record at the training boundary rather than by
counting canonical rows:

| Source | Canonical | Trainable | Yield | Tool-call targets | Contract |
|---|---:|---:|---:|---:|---|
| glaive-function-calling-v2 | 98,339 | 97,112 | 0.988 | 48,723 | `COMPLETE_TRAJECTORY` |
| xlam-function-calling-60k | 57,342 | **56,090** | 0.978 | 56,090 | `CALL_PREDICTION` |
| toolace | 11,051 | 2,259 | 0.204 | 327 | `COMPLETE_TRAJECTORY` |
| when2call | 6,505 | 6,505 | 1.000 | 0 | `COMPLETE_TRAJECTORY` |
| **total** | **173,237** | **161,966** | 0.935 | 105,140 | |

### The v1 corpus, for comparison

The v1 corpus's own record already shows the defect class is not an xLAM problem. Its training
boundary yielded, per source:

| Source | Published records | Trainable in v1 | Yield | Excluded because |
|---|---:|---:|---:|---|
| glaive-function-calling-v2 | 99,794 | 48,387 | 48.5% | `SEM_ORPHAN_RESULT` — call format unparsed |
| when2call | 14,829 | 6,505 | 43.9% | schema / trajectory |
| toolace | 11,190 | 673 | 6.0% | schema (`type: "dict"`) |
| looptool-23k | 20,827 | 145 | 0.7% | schema (`optional` marker) |
| button | 7,941 | 9 | 0.1% | schema (**bare property map**) |
| xlam-function-calling-60k | 59,370 | **0** | 0.0% | schema (**bare property map**) + argument validation |

**Three sources were effectively collapsed in the released corpus and no canonical-count gate
noticed.** BUTTON's exclusion reason is the same bare-property-map defect as xLAM's, and
LoopTool's is the same `, optional` annotation convention. ToolACE fails on a `dict` alias. So
the defect fixed for xLAM is shared, and fixing xLAM alone would have left three sources still
contributing almost nothing.

`xlam_types.py` implements the shared shape (bare property maps, `, optional` annotations, scalar
aliases). It is wired into the xLAM adapter only, because BUTTON and LoopTool cannot be fetched
here to verify a classification against real bytes. Generalizing it is the highest-value next
step and must not be done by widening the generic validator.

## 3. xLAM: what was wrong, and what is fixed

**Fixed, twice, in two layers.** xLAM stores `tools[].parameters` as a property-definition map rather than JSON
Schema. The upstream dataset card documents that contract explicitly, and it is confirmed
against the retained derivative by the call arguments: 3,150 of 3,150 call-argument key sets are
subsets of the corresponding parameter-map key names, with zero counterexamples. The
transformation is implemented at the xLAM source-adapter boundary, so the generic canonical
validator still rejects a bare map — which matters, because the corpus contains tools whose
parameters are literally named `type` (2,551), `format` (2,179) and `items` (670).

**Second layer, also fixed.** Trainability failed with `SEM_UNRESOLVED_CALL` from the trajectory
policy: a call with no tool result, and xLAM structurally has no tool-result turn. That is not a
parsing defect and was not repairable at the adapter.

It was fixed by making the trajectory policy *supervision-aware* rather than by special-casing the
source: xLAM declares `CALL_PREDICTION`, so its terminal call is the supervised target and needs no
result, while `COMPLETE_TRAJECTORY` still requires every call to be resolved. This is a versioned
semantic change with its own contract and evidence, not something slipped in under a
data-normalization fix. See [`SUPERVISION_CONTRACT_REPORT.md`](SUPERVISION_CONTRACT_REPORT.md).

Result: xLAM went from 0 to **56,090 trainable records (94.5%)**, with no fabricated tool result
and no fabricated message.

Full evidence: [`XLAM_NORMALIZATION_FORENSICS.md`](XLAM_NORMALIZATION_FORENSICS.md) and
`reports/data/xlam-normalization-forensics.json`.

## 4. Why two sources cannot be fetched

`Salesforce/xlam-function-calling-60k` and BUTTON are access-gated; a bounded HTTP range read of
the pinned revision returns **401**, and substituting a different revision would not be acceptable.
LoopTool's upstream was not located. No substitute revision was used and no source was
approximated. Both are absent from the frozen corpus and named as excluded in the release config
and on its card, rather than silently missing.

xLAM is nonetheless included, because the published Canonical-v1 release retains its 59,370 records
with `parameters` unmodified. A reconstruction script recovers the source-shaped input from that
derivative and **verifies it record by record**: `tools` is compared against the derivative
verbatim and `query` is checked non-empty, for all 59,370 records, with **0 mismatches**. The
derivation is recorded in `data/raw/xlam/derivation.json`. This is a representational recovery,
not a semantic inference.

## 5. Contamination

**Complete.** All five levels measured; Level 5 adjudicated with 2 candidates, 2 verdicts and 0
pending. Status `SEMANTIC_REVIEW_COMPLETE`.

Unchanged and **incomplete for the partial-v2 candidate**: the v1 corpus completed a semantic
review with a human-adjudicated audit artifact, but the partial-v2 candidate was never
re-reviewed, and its registry entry says `NOT_REVIEWED_FOR_V2` rather than implying otherwise.
That remains true of the partial snapshot and is why it is a snapshot.

For the **frozen** corpus, the review is done. Both candidates were re-verified against the final
corpus bytes rather than re-stamped from the previous verdict, because the evidence changed: the
held-out prompt *"What is the current time?"* now occurs in **8** Glaive training records where
the partial corpus had 5, and *"What is the current weather?"* occurs in **1** When2Call record.
Both are exact-prompt matches whose training labels contradict the held-out gold decision, so both
are `CONTAMINATED` and quarantined, leaving 3,650 held-out examples (3,652 distinct; the 300
`when2call-llm-judge` rows duplicate MCQ rows).

The newly recovered Glaive records — the ones the corrected adapter now parses — went through this
review; the increase from 5 to 8 matches is exactly why re-verification was required rather than
inherited.

Contamination evidence is now **corpus-scoped**. Previously the scanner report, the human audit
artifact and the quarantine list were single fixed paths, so scanning a second corpus silently
overwrote the first corpus's evidence — which is what happened, and which turned the B0 readiness
gate red by comparing v1 verdicts against a v2 fingerprint. Artifacts carry a corpus slug, with the
historical v1 corpus keeping the original paths so recorded B0 results resolve unchanged.

## 6. The v2 RC snapshot

Published and immutable at `arrochi112/OpenGrad-ToolPolicy-Canonical-v2-M0-snapshot`, with the
fingerprint recorded in the training lineage, `09018d26…`, verified to resolve to the published
manifest bytes.

This report found and fixed a defect that had broken exactly that: rebuilding the release to fix
its dataset card rewrote `opengrad_git_commit` in the manifest, moving the hash while all 104
Parquet shards stayed byte-identical, so the recorded dataset hash no longer matched the
published bytes. The manifest has been restored and re-uploaded, and the builder now preserves an
established manifest whenever the payload is unchanged. See commit `30b0e28`.

The snapshot remains a **snapshot**, not final Canonical-v2, and states its incompleteness on its
card. A completed Canonical-v2 must receive a new identity and fingerprint.

## 7. Gates

Readiness integrates two gates that evaluate **measured evidence**, not the presence of
infrastructure, and both are now live on the definitive config:

* `renderability_yield` fails with `DATASET_TRAINABILITY_COLLAPSE` when a source expected to teach
  tool calling yields no tool-call targets, and a declared-but-missing report fails closed.
* `supervision_composition` fails when a config selects a kind the corpus does not contain, or when
  a trainable record carries no kind at all.

A third defect class was found while wiring these up: the gate's coverage check required a fixed
six-source set and could therefore never be satisfied by a corpus built from four sources. The
expected set is now derived from the corpus being evaluated, so the check means what it should —
every source this corpus contains was scanned, and nothing else was.

The yield gate would have caught both known corpus defects, and it does catch the third: ToolACE
reports `ANOMALY` at a measured 0.204 against a floor of 0.3. That floor is deliberately **not**
fitted below the measurement — a threshold set to the value it is meant to judge stops being a
gate — and `ANOMALY` is visible and non-blocking, so the 79.6% quarantine is reported rather than
hidden behind a green tick.

## 7b. Frozen identity

| Field | Value |
|---|---|
| Release name / version | `OpenGrad ToolPolicy Canonical (v2 final)` / `v2.0.0` |
| Fingerprint (release manifest sha256) | `8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161` |
| Determinism | two independent builds → identical manifest **and all 176 shards byte-identical** |
| Canonical / trainable | 173,237 / 161,966 |
| Sources | glaive-function-calling-v2, xlam-function-calling-60k, toolace, when2call |
| Corpus fingerprint (contamination binding) | `c12964b0…` |
| Schema / contract | `tool_use_ir_v1` / `supervision_contract_v1` |
| Builder | `configs/releases/toolpolicy_canonical_v2_final.yaml` |

A frozen release must not be mistaken for the partial snapshot: different name, different
fingerprint, four sources rather than three, and a different When2Call artifact.

## 8. Readiness and whether a new SFT is justified

`opengrad readiness configs/experiments/m0_sft_canonical_v2_final.yaml` reports:

```text
status: PASS
ready_for_sft: True
blocking_gates: []
warnings: []
```

All 23 gates are **active** — the two new ones read a real measurement rather than checking that
infrastructure exists:

```text
renderability_yield      PASS   4 sources measured from reports/data/canonical-v2-final-yield.json; collapse=none
supervision_composition  PASS   {'CALL_PREDICTION': 56090, 'COMPLETE_TRAJECTORY': 105876}
contamination_gate       PASS   level_5=COMPLETE; scanned == corpus sources exactly; quarantined=2
```

**A new SFT is now justified**, which it was not before this work: the corpus contains 56,090
trainable call-prediction records where it contained zero, every record declares what it
supervises, contamination review is complete for this corpus, and the config preserves the
partial-v2 recipe field for field so the corpus is the only variable.

It is not, however, a *confirmatory* experiment. `behavioral-heldout-v2` is where the partial-v2
checkpoint was selected, so a peak `call_f1` chosen over four checkpoints on it must be reported
as selected rather than as an unbiased measurement. See
[`PRE_SFT_READINESS_REPORT.md`](PRE_SFT_READINESS_REPORT.md) §9.

## 9. Remaining work, in order

1. **Materialize a confirmatory held-out set** disjoint from checkpoint selection. All 3,652
   distinct upstream test examples are already in use, so this needs a deterministic split of the
   existing 3,650 or new upstream data. Until then close claims stay development-set claims.
   *Done (2026-09-11):* a pre-registered 2,373 DEV / 1,277 confirmatory split of the 3,650
   (`reports/evaluation/behavioral-heldout-v2-partition.json`). The confirmatory side is internal
   evidence, not an untouched benchmark.
2. **Obtain access** to the gated xLAM upstream and BUTTON, or record them as permanently
   unavailable. xLAM's *inclusion* no longer depends on it (the derivative reconstruction is
   verified), but its provenance does.
3. **Generalize the parameter-map normalizer to BUTTON and LoopTool.** BUTTON's v1 exclusion
   reason is the identical bare-property-map defect and LoopTool's is the same `, optional`
   convention, so `xlam_types.py` is the right shape. It is deliberately wired only into the xLAM
   adapter, because neither source could be fetched to verify against real bytes.
4. **Resolve the ToolACE and LoopTool terminal-call question** with upstream evidence, one way or
   the other. 8,476 ToolACE records are quarantined by that decision, which is why its yield is
   0.204 and its gate status is `ANOMALY`.
5. **Publish the frozen corpus** as `arrochi112/OpenGrad-ToolPolicy-Canonical-v2` once its card is
   written. It is built and frozen locally; the partial snapshot is untouched and remains
   distinguishable by name, fingerprint and source count.
