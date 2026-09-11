# Canonical-v2 completion report

**Status:** `INCOMPLETE` — xLAM canonical normalization is fixed, but xLAM is not trainable,
and two of six sources cannot be fetched in this environment. Canonical-v2 must not be called
final.

**Date:** 2026-09-11
**Supersedes:** the "v2 candidate" framing in `configs/releases/toolpolicy_canonical_v2.yaml`

This report states what Canonical-v2 can and cannot claim after the xLAM normalization work.
It does not restate the M0/M1 experiment results, which are in
[`M0_SFT_EXECUTION_REPORT.md`](M0_SFT_EXECUTION_REPORT.md).

---

## 1. Headline

The xLAM source-adapter defect is fixed, and the fix is verified across all 59,370 retained
records: canonical acceptance moved from **33 → 57,342** (0.06% → 96.6%).

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
| Tool-call targets | 0 | 57,342 |

The remaining 1,252 non-trainable records are 1,231 genuine upstream argument defects and 21 that
exceed the token window. None is attributable to OpenGrad's parsing or validation.

---

## 2. Per-source state

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
the defect fixed for xLAM is shared, and fixing xLAM alone leaves three sources still
contributing almost nothing:

| Source | Raw upstream | Canonical records | Trainable | Status |
|---|---|---:|---:|---|
| glaive-function-calling-v2 | local | 98,339 eligible | 97,112 | in the v2 RC |
| toolace | local | 697 | 673 | in the v2 RC |
| when2call | local | 4,000 | 4,000 | in the v2 RC |
| xlam-function-calling-60k | **gated (HTTP 401)** | 57,342 | **0** | canonical fixed; blocked by trajectory policy |
| button | **gated** | 0 | 0 | **NOT FETCHED** — same bare-map defect documented for v1 |
| looptool-23k | **not located** | 0 | 0 | **NOT FETCHED** |

The v2 RC is a **three-of-six** rebuild. Its totals are 103,036 canonical / 101,785 trainable /
48,723 tool-call targets, all of which verify from `runs/qwen35_2b_m0_sft_v2corpus/`.

Per-source *tool-call* targets are not broken out in the existing RC evidence — the rendering
report records the `CALL`/`ANSWER` split in aggregate (48,723 / 53,062). Breaking it out per
source requires a full re-render and is listed as remaining work rather than estimated here.

What this means for scope: the `xlam_types` normalizer is the right shape for BUTTON and
LoopTool too (bare property maps, `, optional` annotations, scalar aliases), but it is wired into
the xLAM adapter only, and those two sources cannot be fetched here to verify against real bytes.
Generalizing it is the highest-value next step and must not be done by widening the generic
validator.

## 3. xLAM: what was wrong, and what is still wrong

**Fixed.** xLAM stores `tools[].parameters` as a property-definition map rather than JSON
Schema. The upstream dataset card documents that contract explicitly, and it is confirmed
against the retained derivative by the call arguments: 3,150 of 3,150 call-argument key sets are
subsets of the corresponding parameter-map key names, with zero counterexamples. The
transformation is implemented at the xLAM source-adapter boundary, so the generic canonical
validator still rejects a bare map — which matters, because the corpus contains tools whose
parameters are literally named `type` (2,551), `format` (2,179) and `items` (670).

**Not fixed.** Trainability. The failure is `SEM_UNRESOLVED_CALL` from the trajectory policy: a
call with no tool result. xLAM structurally has no tool-result turn, so the policy rejects 100%
of the source. This is not a parsing defect and it is not repairable at the adapter.

Two things this implies:

* The briefing's expectation that xLAM would become healthy is only half met. Canonically it is
  healthy; as training data it contributes nothing.
* Making xLAM trainable requires an explicit, **versioned** change to the trajectory policy —
  admitting a terminal call as a valid supervised target for call-prediction sources. That is a
  semantic change to what the model is trained to do, it affects every source, and it is not
  something to slip in under a data-normalization fix. It is **not** done here.

Full evidence: [`XLAM_NORMALIZATION_FORENSICS.md`](XLAM_NORMALIZATION_FORENSICS.md) and
`reports/data/xlam-normalization-forensics.json`.

## 4. Why two sources cannot be fetched

`Salesforce/xlam-function-calling-60k` and BUTTON are access-gated; a bounded HTTP range read of
the pinned revision returns **401**, and the task's own instruction was not to silently
substitute a different revision. LoopTool's upstream was not located in the earlier v2 build
either. No substitute revision was used and no source was approximated.

xLAM is nonetheless auditable because the published Canonical-v1 release retains its 59,370
records with parameters unmodified, which is why the audit could run over exact bytes at all.

## 5. Contamination

Unchanged and **incomplete for v2**. The v1 corpus completed a semantic review with a
human-adjudicated audit artifact; the v2 candidate has not been re-reviewed, and its registry
entry says `NOT_REVIEWED_FOR_V2` rather than implying otherwise. Its evaluation-side artifacts
are excluded by the same policy as v1, which is necessary but not the same as a contamination
verdict.

The newly recovered Glaive records — the ones the corrected adapter now parses — have not been
through a semantic review. Re-running contamination auditing for v2 is remaining work.

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

Readiness integrates a per-source trainability gate. A configuration that declares
`datasets.yield_report` fails with `DATASET_TRAINABILITY_COLLAPSE` when a source that is
supposed to teach tool calling yields no tool-call targets; a declared-but-missing report fails
closed. It is dormant for configurations that predate the measurement, and says so.

The gate would have caught both known corpus defects. It is not yet declared by any config,
because no config's corpus has been measured in full at the training boundary — wiring it into
the v2 config is remaining work and is listed below.

## 8. Readiness and whether a new SFT is justified

`opengrad readiness` reports `PASS` / `ready_for_sft: true` for the existing SFT configs, with
the new gate passing as dormant.

**A new SFT is not justified by this continuation.** Nothing here changes the training data:
xLAM adds zero trainable records, two sources are unavailable, and v2 contamination review is
incomplete. The M0-v2 result stands as it was, and its checkpoint 1200 remains a candidate
selected on the existing evaluation set — not promoted, and still requiring
checkpoint-selection-disjoint confirmation before promotion.

## 9. Remaining work, in order

1. **Decide the trajectory policy** for terminal calls, as an explicit versioned change with its
   own evidence. Until then xLAM contributes nothing and should not be counted as recovered.
   `SEM_UNRESOLVED_CALL` accounts for 56,111 of xLAM's 57,342 canonically valid records, so this
   decision is the whole of xLAM's value as training data.
2. **Obtain access** to the gated xLAM and BUTTON revisions, or record them as permanently
   unavailable and scope Canonical-v2 to four sources.
3. **Break out per-source tool-call targets** in the yield report and declare
   `datasets.yield_report` on the SFT config so the gate is live rather than dormant.
4. **Re-run contamination auditing** for v2, including the newly recovered Glaive records.
   The supervision contract also changes the composition, so the audit must cover the new
   mixture rather than the old one.
5. **Re-audit the corrected Glaive adapter** against the training boundary per source, since
   canonical retention and trainability are distinct.
6. **Generalize the parameter-map normalizer to BUTTON and LoopTool.** BUTTON's v1 exclusion
   reason is the identical bare-property-map defect and LoopTool's is the same `, optional`
   convention, so `xlam_types.py` is the right shape. It is deliberately wired only into the xLAM
   adapter, because neither source could be fetched here to verify against real bytes.
7. **Freeze final Canonical-v2** with a new identity and fingerprint once 1–6 are done.
