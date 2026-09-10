# Failure Analysis & Clustering Specification

**Building in Public.** OpenGrad records and analyzes individual evaluation failures rather than reporting only aggregate accuracy numbers.

---

## 1. Normalized Failure Records

When a benchmark task fails, a normalized record is saved in `failures.json`:
- `benchmark`: Benchmark ID.
- `sample_id`: Test case identifier.
- `prompt`: Input prompt text.
- `expected`: Ground truth decision or argument structure.
- `actual`: Output emitted by the model.
- `score`: Partial or zero score.
- `failure_category`: Canonical error code from `src/opengrad/benchmarks/taxonomy.py`.

---

## 2. Failure Clustering

The `FailureAnalyzer` (`src/opengrad/failures/analyzer.py`) automatically clusters failures across taxonomy categories:
- `wrong_tool`
- `missed_tool`
- `unnecessary_tool`
- `malformed_tool_call`
- `missing_argument`
- `wrong_argument`
- `failed_clarification`
- `tool_loop`
- `parser_failure`
- `observation_ignored`
- `recovery_failure`

---

## 3. Failure Diffing Between Checkpoints

OpenGrad can answer **"What new failures appeared after M1?"** by diffing baseline and candidate failure lists:
- **`fixed_failures`**: Errors present in baseline that were corrected by candidate.
- **`new_failures`**: Regressions that appeared exclusively in the candidate.
- **`persistent_failures`**: Errors unresolved across both runs.

Run failure analysis on a run:
```bash
opengrad failures runs/qwen35_2b_m0_sft/eval
```
Emits category percentages and representative failing examples.
