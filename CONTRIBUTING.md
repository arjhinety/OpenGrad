# Contributing

OpenGrad welcomes reproductions, alternative seeds, model-family adapters, source adapters, behavioral annotations, coverage audits, hard negatives, counterfactual pairs, held-out evaluations, residual-driven mixtures, alternate mixture algorithms, benchmark discrepancy reports, hardware measurements, quantization studies, and negative results.

Before opening a data or experiment PR:
- keep changes within the currently documented research phase; phase transitions require maintainer approval;
- preserve source revision, split, license, processing, and contamination provenance;
- label behavioral annotations as known, derived, heuristic, or unknown;
- add a machine-readable experiment or mixture record when applicable;
- never commit checkpoints, bulk datasets, credentials, anonymous dumped JSON, or fabricated results;
- distinguish deterministic fixture tests from live provider/device evidence;
- for M2, reference the executed baseline, residual profile, evidence count, and algorithm revision; never hard-code model-family assumptions;
- include regressions, contamination risk, and limitations.
- for any experiment or published claim, follow the checklists in `docs/research/GUARDRAILS.md`, which record the mistakes Study 001 made and how to avoid repeating them.

Capability data and speculative/MTP continuation data have separate objectives and configurations. A behavior column is not evidence that a source has been fully measured. Use the canonical taxonomy in `registry/tool_behaviors.yaml` and the methodology in `docs/data/tool-use-mixture-methodology.md`.

Use the issue and PR templates. A failure to reproduce an existing finding is a valuable contribution.
