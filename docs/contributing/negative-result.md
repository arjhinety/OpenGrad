# Negative result template

State the hypothesis, controlled comparison, failed improvement, regressions, uncertainty, contamination risk, and what should not be tried again without new evidence.

## Why negative results are kept

OpenGrad retains successful runs, failed runs, regressions, null results, non-reproductions, and
rejected hypotheses. This prevents duplicated failed work, exposes unstable recipes and model-family
differences, makes sensitivity visible, and reduces cherry-picking. A lower score can be useful
evidence if the comparison and failure analysis are reproducible.

The project's own operational mistakes are part of that record. [The incident log](../INCIDENT_LOG.md)
documents errors that changed what can be claimed — including INC-0001, where checkpoint weights were
deleted before upload and could not be reproduced. Reports affected by an incident carry a correction
pointing at the entry, and past entries are never rewritten to look better.

Two rules follow:

- **Upload before you delete.** A rejected checkpoint is still evidence; see
  [Checkpoints § Retention](../CHECKPOINTS.md).
- **A correction is appended, never substituted.** Experiment records annotate `validity` and append
  a ledger event; the append-only history keeps the original transition intact. See
  [`docs/EXPERIMENT_STATUS.md`](../EXPERIMENT_STATUS.md) for how invalid and scaffold records are
  marked without rewriting them.
