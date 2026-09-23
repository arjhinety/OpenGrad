# Contributing

OpenGrad welcomes reproductions, alternative seeds, model-family adapters, source adapters, behavioral annotations, coverage audits, hard negatives, counterfactual pairs, held-out evaluations, residual-driven mixtures, alternate mixture algorithms, benchmark discrepancy reports, hardware measurements, quantization studies, and negative results.

## Set up and check a change the way CI does

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run mypy src                                        # strict
uv run pytest                                          # network tests are deselected; run with -m network
uv run opengrad-validate
uv run python scripts/preserve_h200_state.py --verify  # frozen H200 pins, on committed blobs
uv run python scripts/repo/check_publication_hygiene.py
(cd integrations/opengrad-mcp && npm run check && npm test)   # if you touch the MCP server
```

Optional: `uv run pre-commit install` runs ruff on each commit (its version matches `uv.lock`).
Python is 3.11 (`.python-version`). CI runs on Linux; see
[`plugins/opengrad/skills/opengrad-development/SKILL.md`](plugins/opengrad/skills/opengrad-development/SKILL.md)
for the Windows notes (line endings change hashes).

## Before opening a data or experiment PR

- Keep changes within the current research phase. The current phase and its open decisions are in
  [Study 002 — Current state](docs/research/study-002/README.md#current-state); phase transitions require
  maintainer approval.
- Preserve source revision, split, license, processing, and contamination provenance.
- Label behavioral annotations as known, derived, heuristic, or unknown.
- Add a machine-readable experiment or mixture record when applicable.
- Never commit checkpoints, bulk datasets, credentials, anonymous dumped JSON, or fabricated results.
- Never edit a frozen or hash-pinned file; correct it in [`reports/ERRATA.md`](reports/ERRATA.md).
- Distinguish deterministic fixture tests from live provider/device evidence.
- Include regressions, contamination risk, and limitations.
- For any experiment or published claim, follow the checklists in `docs/research/GUARDRAILS.md`, which record
  the mistakes Study 001 made and how to avoid repeating them.

Capability data and speculative/MTP continuation data have separate objectives and configurations. A behavior column is not evidence that a source has been fully measured. Use the canonical taxonomy in `registry/tool_behaviors.yaml` and the methodology in `docs/data/tool-use-mixture-methodology.md`. Terms are defined in the [glossary](docs/GLOSSARY.md).

Use the issue and PR templates. A failure to reproduce an existing finding is a valuable contribution.
