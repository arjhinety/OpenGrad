# Contamination adjudication (Level 5) and held-out quarantine

Levels 1–4 of the [benchmark contamination policy](./BENCHMARK_STRATEGY.md#4-benchmark-contamination-policy)
are machine-measured. **Level 5 is a human judgment**, and this document is the exact workflow
for making and recording it before B0.

Two artifacts are involved, and only one of them is ever written by a human:

| Artifact | Owner | Purpose |
| --- | --- | --- |
| `reports/data/behavioral-heldout-v2-contamination.json` | **machine** (`opengrad-contamination heldout-screen`) | Levels 1–4 evidence and the audit queue. Regenerated on every scan — never hand-edit it. |
| `reports/data/behavioral-heldout-v2-contamination-audit.json` | **human** (`opengrad-contamination adjudicate`) | Durable Level-5 verdicts. The scanner never writes this file. |
| `reports/evaluation/behavioral-heldout-v2-quarantine.json` | derived from the audit | Examples excluded from B0 after a `CONTAMINATED` verdict. |

## The Level-5 state machine

```text
                       heldout-screen
                             │
                             ▼
        generated report  (levels 1-4 MEASURED, level 5 NOT_RUN)
                             │
                      adjudicate (syncs the queue)
                             │
             ┌───────────────┴────────────────┐
             │                                │
        no findings                    N findings
             │                                │
             ▼                                ▼
      level 5 = COMPLETE            every finding PENDING
      status  = CLEAN                        │
                                             │  record a verdict per finding
                                             ▼
                            ┌────────────────┴─────────────────┐
                            │                                  │
                     INCIDENTAL_OVERLAP                  CONTAMINATED
                            │                                  │
                            │                          quarantine --apply
                            │                          (excluded from B0)
                            └────────────────┬─────────────────┘
                                             ▼
                                  level 5 = COMPLETE
                                  status  = SEMANTIC_REVIEW_COMPLETE
```

Blocking rules enforced by `src/opengrad/readiness.py`:

- **Any** current finding without a valid verdict, or in `PENDING`, keeps level 5 `NOT_RUN` and the
  `contamination_gate` **FAIL**.
- **Any** `CONTAMINATED` verdict that is not in the quarantine list keeps the gate **FAIL**.
- A judgment whose supporting evidence changed is **stale** and blocks until re-adjudicated. This is
  triggered by a changed finding (different matched records or levels), a changed benchmark
  fingerprint (held-out splits, counts, content hashes), or a changed training-corpus fingerprint
  (release shard set and record count).
- Level 5 becomes `COMPLETE` only when there are no problems and nothing is pending.
- Status is `SEMANTIC_REVIEW_COMPLETE` when the queue is empty because examples were quarantined,
  and `CLEAN` only when nothing was ever flagged.

## Verdicts

| Verdict | Meaning | Effect |
| --- | --- | --- |
| `PENDING` | Not yet reviewed. | Blocks. |
| `INCIDENTAL_OVERLAP` | The overlap is generic phrasing, not leakage of the measured decision. | Clears the finding. |
| `CONTAMINATED` | The training record exposes the same prompt, tool-selection behaviour, label, or the equivalent decision being measured. | Must be quarantined; otherwise blocks. |

Adjudicate conservatively: losing an evaluation example is cheaper than retaining known or
behaviourally equivalent leakage.

## Exact pre-B0 workflow

Run from the repository root.

```bash
# 1. Measure levels 1-4 against the canonical training corpus (~2 minutes).
.venv/bin/opengrad-contamination heldout-screen

# 2. See how many findings still need a human verdict.
.venv/bin/opengrad-contamination adjudicate --status
.venv/bin/opengrad-contamination adjudicate --list

# 3. Review and record verdicts, one at a time, side by side.
.venv/bin/opengrad-contamination adjudicate
#    [c] Contaminated   [i] Incidental overlap   [s] Skip/Pending   [q] Quit

#    Or non-interactively / reproducibly:
.venv/bin/opengrad-contamination adjudicate \
  --id when2call-mcq:<uuid> \
  --verdict contaminated \
  --reason "exact prompt and equivalent tool-use behaviour"

# 4. Exclude every CONTAMINATED example from B0 and all later held-out runs.
.venv/bin/opengrad-contamination quarantine --apply

# 5. Rescan so the report reflects the excluded examples, then confirm.
.venv/bin/opengrad-contamination heldout-screen
.venv/bin/opengrad-contamination adjudicate --status
.venv/bin/opengrad-contamination quarantine --status

# 6. Confirm the gate.
.venv/bin/opengrad readiness --json
```

The gate passes when `contamination_gate` is `PASS` with `level_5=COMPLETE`. Remaining blockers
after that are the GPU boundary and B0 itself.

Reviewer identity comes from `--reviewer`, else `$OPENGRAD_REVIEWER`, else `$USER`. Timestamps are
recorded automatically. Re-running `adjudicate` or `heldout-screen` never erases a recorded verdict.

## Non-negotiables

- Never edit the generated report by hand; it is overwritten by the next scan.
- Never delete the audit artifact to make the gate pass — that is exactly the review the gate exists
  to require.
- Never mark level 5 complete by hand. It is derived from the audit artifact and the quarantine list.
