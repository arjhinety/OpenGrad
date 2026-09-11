# On-policy distillation configuration namespace

**Deliberately not executed.** The trainer foundations exist (`src/opengrad/training/distillation.py`,
with `TeacherProvider` and `RolloutProvider` seams), but no distillation run has been performed and
none is planned for the current sequence.

That is a decision, not an omission. The M0/M1 line ended at DPO because the diagnosis identified
the corpus as the binding constraint, and distillation reweights or transfers behaviour a policy can
already produce — the same limitation that made DPO ineffective on corpus v1. See
[`reports/M0_SFT_EXECUTION_REPORT.md`](../../../reports/M0_SFT_EXECUTION_REPORT.md) §6.

| File | Status |
|---|---|
| `qwen35_2b_qwen38_27b_a100.yaml` | A prepared configuration, **not executed**. It also names a hardware arrangement the current machine cannot provide: a 27B bf16 teacher co-resident with the student needs more VRAM than is available, which was one of the reasons the stage was stopped |

Nothing here is evidence. If distillation is attempted later it needs its own pre-registration and
its own confirmatory evaluation, not this file's hypothesis.
