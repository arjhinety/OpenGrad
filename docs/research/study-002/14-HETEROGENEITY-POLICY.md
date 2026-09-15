# 14 — Heterogeneity policy

[13](13-HARDWARE-AGNOSTIC-EXECUTION.md) makes the study runnable on any provider. This document states
precisely where provider and hardware variation is allowed to matter, how it is measured, and what happens
to a claim when it does.

## The rule

> A claim may be made **within** a device class. It may be made **across** device classes only after a
> heterogeneity check has shown the arm-level deltas agree to within their resolvable margins.

The default is therefore *within-provider claims are valid; cross-provider claims are not, unless earned*.
This is the opposite of the usual assumption that hardware is a detail, and it is chosen because Study 001
already paid for the lesson: an engine comparison was reported as an engine effect while hardware differed
too, and three of the 21 flips were tokenizer-caused (`#34`, `#43`).

## The heterogeneity check

Its design is fixed here and is deliberately cheap:

| Item | Value |
|---|---|
| arms | `C0` and `R1` only |
| seeds | the full `k = 3`, same seed values |
| providers | two, being the primary class and one other with a `READY` preflight |
| endpoints | `answer_rate` and `refusal_rate` on `ANSWER`-gold, plus `call_f1` |
| statistic | `delta(R1 − C0)` computed **within each provider**, then compared across providers |

The comparison is on **deltas**, not on levels. Levels legitimately differ across providers — a different
kernel or dtype can shift a raw score — whereas the mechanism claim is about a difference between arms, and
that is the quantity the study's conclusion rests on. Comparing levels would flag a harmless shift as
heterogeneity and invite exactly the sloppy "hardware doesn't matter much" reasoning the check exists to
avoid.

**Verdict:**

- `HOMOGENEOUS` — every provider's delta agrees with the primary's to within the resolvable margin
  ([06](06-SPLIT-SPEC.md)). Cross-provider claims are permitted, and the two provider rows are still printed
  separately.
- `HETEROGENEOUS` — any delta differs by more than the resolvable margin. Cross-provider claims are
  **withdrawn**, the provider rows are printed separately, the discrepancy is reported as a finding in its
  own right, and every downstream table gains the marker `DEVICE_CLASS_SPECIFIC`.
- `NOT_RUN` — the second provider was unavailable. Cross-provider claims are not made, and the limitation is
  recorded in the `limitations` array of the preflight record rather than in prose.

`HETEROGENEOUS` is a publishable outcome, not a failure. A corpus-level mechanism that only reproduces on
one vendor's kernels is a materially weaker claim, and saying so is more useful than averaging it away.

## Determinism classes

Every run declares one, and the declaration is checked rather than trusted:

| Class | Meaning | Evidence required |
|---|---|---|
| `DECLARED_DETERMINISTIC` | same seed, same device class, identical outputs | `REP-A` ([05](05-SEED-AND-REPRODUCIBILITY-POLICY.md)) reproduces the metric values exactly |
| `NON_DETERMINISTIC_KERNEL` | a kernel or reduction is non-deterministic | `REP-A` residual measured and reported; it becomes the noise floor every margin in [11](11-THRESHOLDS.md) is compared against |

A run that declares `DECLARED_DETERMINISTIC` and fails `REP-A` is a **defect**, not a reclassification: the
declaration was checked and found false, and the run's metrics are reported with the residual as an interval
while the defect is recorded.

## Factors that are never varied inside a comparison

| Factor | Treatment |
|---|---|
| inference engine | fixed for the whole study; a change re-runs the whole comparison set ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)) |
| dtype / precision | part of `device_class`; a change is a different row, not a variant |
| quantization | not an arm; if present it is recorded and the run is classified by it |
| device count / parallelism | recorded; a difference in topology between two compared arms is a confound and blocks the comparison |
| generation budget | fixed per benchmark; changes are pre-scoring amendments only ([09](09-BENCHMARK-PLAN.md)) |

Any of these differing between two rows that a table asks a reader to compare is an `INVALID_COMPARISON`
failure ([11](11-THRESHOLDS.md), check 12).

## Reporting

Every table states the device class of each row, or states that all rows share one. A cross-provider table
prints the `HOMOGENEOUS`/`HETEROGENEOUS` verdict above it. Where the verdict is `NOT_RUN`, the table says so
in the table, because a reader who has to reach the limitations section to discover that two rows came from
two different providers has effectively been misled.