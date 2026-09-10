# Baseline artifacts

Real baseline measurements live here, one directory per run identity.

- [`qwen35_2b_baseline/`](./qwen35_2b_baseline) — the executed `Qwen/Qwen3.5-2B` behavioral
  baseline: `metrics.json`, `predictions.jsonl`, `environment.json`, and the
  [result write-up](./qwen35_2b_baseline/RESULT.md). Residual analysis is in
  `reports/failures/qwen35_2b_baseline/`, and the canonical experiment record is
  `runs/tool_calling/qwen35_2b/baseline/experiment.json`.

A real run is immutable: it refuses to overwrite existing evidence or an existing experiment
record, so a new measurement needs a new run identity. A `--dry-run` writes to
`runs/.dry-run/` instead and is never evidence.
