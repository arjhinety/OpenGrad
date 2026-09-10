# Checkpoint Registry Specification

**Building in Public.** Checkpoints in OpenGrad are tracked entities with immutable lineage, step counters, loss records, and explicit lifecycle states.

---

## 1. Checkpoint Lifecycle States

Every checkpoint is assigned one of the following states:

- **`TRAINING`**: Intermediate unfinalized weights being updated.
- **`CANDIDATE`**: Completed training step; registered for evaluation.
- **`EVALUATING`**: Actively executing benchmark evaluation suites.
- **`PROMOTED`**: Checkpoint has passed all quality and non-regression gates.
- **`REJECTED`**: Checkpoint failed regression tolerances or safety criteria.
- **`ARCHIVED`**: Historical checkpoint retained for reproducibility.

> **Rule:** The newest checkpoint is **never** automatically considered "best". "Best" requires passing a formal promotion policy against the baseline.

---

## 2. Checkpoint Commands

List all registered checkpoints:
```bash
opengrad checkpoint list
```

Inspect a specific checkpoint:
```bash
opengrad checkpoint inspect checkpoint-5
```

Manually promote or reject with an auditable justification note:
```bash
opengrad promote checkpoint-5 --reason "Passes tool calling gates with no regression on IFEval"
opengrad reject checkpoint-5 --reason "Over-calling on ambiguous requests increased by 3.2%"
```
All state changes append immutable events to `runs/central_ledger.jsonl`.

---

## 3. Retention: upload before you delete

**Never delete local checkpoint weights before every one of them is uploaded and the upload
verified.** Deleting first and uploading later is not a valid order, because the window between
the two has no copy anywhere.

Checkpoint weights are not guaranteed reproducible byte-for-byte, even from a pinned config, the
same seed, and identical data: kernel nondeterminism, library versions, and hardware differ across
runs. A deleted checkpoint must therefore be treated as **gone**, not *regenerable*. A re-run is
a replacement, not the original, and it may or may not reproduce the original's measurements. If
it does not, its weights are not the model behind any number the original published, and it must
be labelled as a reproduction with its own results rather than passed off as the original.

### The rule

1. Upload **all** retained checkpoints, including intermediate steps — not just the final one.
   The informative part of a training curve is usually the trajectory, not the endpoint.
2. Verify what landed: every checkpoint present, weights present inside each, sizes non-zero.
3. Only then delete locally.
4. Keep the load-bearing restorable: optimizer state, config, and the run's `events.jsonl`,
   `resolved_config.yaml`, and preference/dataset identity hash.

### Why this matters more for negative results

A negative result whose per-step numbers are reported but whose weights are absent cannot be
independently checked — a reader can see the claim but not test it. When a run's best measured
state is the one that was discarded, the published artifact is its *worst* state, and the
headline number has no corresponding model.

This has already cost this project once: the M1 DPO run (`qwen35_2b_m1_dpo_v1`) reported three
checkpoints and published only step 300, while step 100 held the best measured metrics
(`call_f1` 0.1715 against 0.0258 at step 300). Steps 100 and 200 were deleted after the run and
had never been uploaded; the recovery was a full re-run. See
`docs/TRAINING_LIFECYCLE.md` for the run-directory layout this applies to.

### Disk pressure is not a reason to skip the upload

When disk is short, the correct order is still upload → verify → delete. Uploading the retained
checkpoint set is bounded and known in advance: a full-parameter 2B checkpoint without optimizer
state is ~3.8 GiB, so four retained checkpoints are ~15 GiB. If that does not fit, delete
something else, raise `max_checkpoints`, or stop the run — never delete weights that exist
nowhere else.
