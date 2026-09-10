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
