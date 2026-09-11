# Foundation reports

Historical records of the architecture that was built before any model was trained. They are
archived here rather than kept at the repository root because they describe *foundations*, and the
project has since moved past them into experiments.

Read them as a record of what was implemented at the time, not as current status. Several of the
statements in them have been superseded by executed work — see
[`M0_SFT_EXECUTION_REPORT.md`](../../reports/M0_SFT_EXECUTION_REPORT.md) and the
[supervision contract report](../../reports/SUPERVISION_CONTRACT_REPORT.md) for what actually ran.

| Report | What it records |
|---|---|
| [BOOTSTRAP_REPORT.md](BOOTSTRAP_REPORT.md) | The Phase 0 repository bootstrap: schemas, fixtures, and structure, before any model existed |
| [PRE_EXPERIMENT_REPORT.md](PRE_EXPERIMENT_REPORT.md) | Phase 0.5 CPU-safe validation: adapters, renderers, parsers, contamination methods, mock harnesses |
| [EXPERIMENT_FOUNDATION_COMPLETION_REPORT.md](EXPERIMENT_FOUNDATION_COMPLETION_REPORT.md) | The experiment operating system: preflight, trainers, checkpoint registry, evaluation, promotion |
| [DPO_OPD_IMPLEMENTATION_REPORT.md](DPO_OPD_IMPLEMENTATION_REPORT.md) | The preference-optimization and on-policy-distillation foundations |

## Known-stale claims

These documents are preserved unedited, so a reader should know which of their statements no
longer hold:

- **"No model inference, training, or model-quality result exists"** (BOOTSTRAP_REPORT). Superseded:
  B0 and four post-training interventions have executed, and the M0-final run is reported in
  `reports/`.
- **"GPU branches remain UNVERIFIED"** (EXPERIMENT_FOUNDATION_COMPLETION_REPORT). Superseded: the
  GPU boundary smoke passes and real training and evaluation have run on an A100.
- **On-policy distillation "supported"** (DPO_OPD_IMPLEMENTATION_REPORT). The foundations exist, but
  OPD was explicitly not attempted — it is out of scope by decision, not by omission.
