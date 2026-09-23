# Model card

- Base model / exact revision:
- Datasets, licenses, source composition, and preprocessing:
- Mixture class: source-oriented / behavior-balanced / residual-driven
- Behavioral target composition (decisions, capabilities, retention, catalogues):
- Baseline experiment and residual profile motivating weighting (if applicable):
- Training method / tokens / hyperparameters:
- Hardware / software:
- Benchmarks / versions / contamination procedure:
- CALL/ANSWER/CLARIFY/UNSUPPORTED confusion matrix and directional routing metrics:
- **General-capability regressions against the base model (required, even when none were found):**
  benchmark, prompt mode, n, base and model scores, delta and interval; link the report and any
  errata. Study 001's promoted card omitted its regression (docs/research/GUARDRAILS.md).
- License and attribution: the base model's license at its pinned revision (`license_link`), and the
  per-source data terms. `license: other` never implies relicensing a component.
- Limitations / runtime / parser:
- Intended and out-of-scope use:

No result should be reported as an OpenGrad result without an executed, reproducible evaluation. OpenWeights observations are external motivation unless independently reproduced.
