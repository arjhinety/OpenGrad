# OpenGrad NVIDIA ModelOpt integration report

**Date:** 2026-09-11
**Scope:** implement an isolated, optional NVIDIA ModelOpt optimization-producer interface;
establish what is implemented, what is tested, and what is explicitly not attempted.

---

## 1. Status

| Dimension | Result |
| --- | --- |
| Interface implemented | **YES** — `src/opengrad/optimization/` (protocol, capabilities, ModelOpt backend, mock backend, selector) |
| CPU tests written and passing | **YES** — 25 passed, 1 skipped (`pytest tests/optimization`) |
| Registry entry updated | **YES** — `nvidia-modelopt` is `INTERFACE_ONLY` |
| ModelOpt installed | **NO** — never installed; not a required or installed dependency |
| ModelOpt executed on any model | **NO** — `execution_status = NOT_ATTEMPTED` |
| Capability for `Qwen/Qwen3.5-2B` GPU-validated | **NO** — every probed technique is `UNKNOWN` |
| FP8 PTQ / NVFP4 / other PTQ | **NOT ATTEMPTED** |
| QAT / QAD | **NOT ATTEMPTED** |
| Distillation | **NOT ATTEMPTED** |
| Pruning / sparsity | **NOT ATTEMPTED** |
| Speculative-decoding training (EAGLE/DFlash-style) | **NOT ATTEMPTED** |

The headline is deliberately narrow: OpenGrad now has a *boundary* through which an
optimized checkpoint can be produced, validated, and recorded with full provenance. No
optimization has been run, nothing has been installed, and no capability has been
claimed. Every capability answer for the study model is `UNKNOWN`.

---

## 2. The boundary this implements

```
trained checkpoint  ->  optimization recipe  ->  optimized checkpoint
     ->  existing inference backend  ->  existing benchmark / evaluation system
```

Optimization *changes artifacts*; the existing inference backend *executes* them and the
existing benchmark/evaluation system *measures* them. The optimization layer does not
train, does not generate, and does not score. Keeping those responsibilities separate is
what lets a quantization study reuse the same frozen evaluation the baseline used rather
than inventing a parallel measurement path.

The layer consumes two read-only inputs:

* a **trained checkpoint** (`SourceCheckpoint`), identified by checkpoint id, hash,
  experiment id, and experiment lineage; and
* an **`OptimizationRecipe`** (technique, target format, calibration identity, parameters),
  pinned by a deterministic `recipe_hash`.

It produces one frozen `OptimizationResult` carrying the full provenance below, and one
artifact on disk. The source checkpoint is guarded against overwrite (§5).

---

## 3. What was implemented

`src/opengrad/optimization/`:

| Module | Contents |
| --- | --- |
| `protocol.py` | `OptimizationBackend` Protocol (`name`, `capabilities()`, `validate_recipe()`, `optimize()`, `export_artifact()`); frozen `OptimizationRecipe` with deterministic `recipe_hash`; frozen `OptimizationResult` with full provenance; `SourceCheckpoint`; `build_optimization_result()`; `ensure_distinct_output()` overwrite guard |
| `capabilities.py` | `CapabilityStatus` (SUPPORTED / UNSUPPORTED / PARTIAL / UNKNOWN); `CapabilityProbe` and `CapabilityMatrix`; `probe_capabilities()` which discovers per exact model and never infers across models |
| `modelopt_backend.py` | `ModelOptBackend` — the one real, optional backend; every ModelOpt symbol resolved by lazy `importlib.import_module`, raising a `RuntimeError` naming the `optimization` extra when absent |
| `mock_backend.py` | `MockOptimizationBackend` — deterministic CPU artifact so the boundary is exercisable with no GPU and no ModelOpt |
| `__init__.py` | exports plus `select_optimization_backend()` (hard-coded selector mirroring `_select_trainer`) |
| `__main__.py` | `python -m opengrad.optimization capability-matrix` and `... optimize` |

The result carries the complete provenance the task requires: source checkpoint id +
hash, source experiment id + lineage, backend + version, technique, recipe + recipe hash,
calibration dataset identity/fingerprint/sample count/seed, quantization format,
hardware, cuda/torch/transformers versions, export format, output artifact path + hash,
runtime used, benchmark ids, capability deltas, efficiency deltas, and
failures/unsupported features. `build_optimization_result()` assembles it centrally so a
backend cannot omit a field; `to_dict()`/`from_dict()` round-trip losslessly.

### Constraints honoured

* **ModelOpt is not a required dependency.** The package imports, the selector works, the
  mock runs, and capability discovery returns `UNKNOWN` with ModelOpt absent. A test
  simulates absence by making `importlib.import_module("modelopt")` fail.
* **No canonical-data, trainer, or inference-backend behaviour changed.** The layer is
  additive and self-contained; no SFT/DPO/distillation code, no data layer, and no
  existing inference backend was modified.
* **No optimization executed.** Execution status is `NOT_ATTEMPTED`.

---

## 4. Capability discovery (what is recorded)

`probe_capabilities()` looks up a fact by the exact `(model_id, technique)` key in an
explicit table that is currently **empty**. There is no model-family or size fallback,
and a technique ModelOpt documents at the library level does not promote a per-model
answer. Anything not explicitly recorded returns `UNKNOWN` with an evidence string that
names the backend revision.

The recorded matrix for `Qwen/Qwen3.5-2B` (revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`) against ModelOpt is in
`reports/optimization/modelopt-capability-matrix.json`:

| Technique | Status | Backend revision |
| --- | --- | --- |
| `fp8_ptq` (FP8 PTQ) | UNKNOWN | absent |
| `nvfp4` | UNKNOWN | absent |
| `ptq_other` (INT8/INT4/AWQ/GPTQ) | UNKNOWN | absent |
| `qat` (QAT) | UNKNOWN | absent |
| `qad` (QAD) | UNKNOWN | absent |
| `distillation` | UNKNOWN | absent |
| `pruning_sparsity` | UNKNOWN | absent |
| `speculative_decoding` (EAGLE/DFlash-style) | UNKNOWN | absent |
| `hf_export` (Hugging Face export) | UNKNOWN | absent |
| `vllm_deployment` (vLLM-compatible deployment) | UNKNOWN | absent |

`backend_revision` is `null` because ModelOpt is not installed. The evidence strings also
record this host's accelerator presence; that is a statement that a GPU exists, **not**
that any ModelOpt capability was validated on it.

A specific rule is enforced and tested: a fact recorded for `Qwen/Qwen3.5-7B` must not
answer for `Qwen/Qwen3.5-2B`, or vice versa. `test_capabilities.py` injects a knowledge
entry for one model and asserts the other stays `UNKNOWN`.

---

## 5. What was tested

```
pytest tests/optimization -q
```

| File | Coverage |
| --- | --- |
| `test_protocol.py` | recipe hash determinism (including dict key-order independence), hash sensitivity to intent changes, round-trip with stable hash, recipe validation refusals, full-provenance completeness + result round-trip, deterministic artifact hash, overwrite guard (identical/nested paths and via the mock), source-untouched assertions, selector mapping |
| `test_capabilities.py` | every required technique probed, `UNKNOWN` default, evidence names the backend revision, no cross-size inference in either direction, matrix round-trip |
| `test_optional_import.py` | package importable and selector works with ModelOpt absent (simulated), mutating calls raise `RuntimeError` naming the extra, capabilities stay `UNKNOWN` and non-raising, and a GPU-only path **skipped with an explicit reason** |

Result: **25 passed, 1 skipped** (the skip is the GPU-only path, which requires ModelOpt
and CUDA and is not authorized to run).

The overwrite guard is covered directly: `MockOptimizationBackend().optimize(...)` with
an output inside the source raises, and the source file bytes are asserted unchanged
afterwards.

---

## 6. What was *not* attempted

Nothing in this list was run, and none of it is claimed:

* **No ModelOpt install.** The `optimization` extra was added to `pyproject.toml` but not
  installed; ModelOpt is not in core dependencies and is not present in this environment.
* **No optimization execution.** FP8 PTQ, NVFP4, other PTQ formats, QAT, QAD,
  distillation, pruning/sparsity, and speculative-decoding training were **NOT attempted**.
  The ModelOpt `optimize()`/`export_artifact()` paths are written against ModelOpt's
  documented surface but are marked UNVERIFIED and have never been executed.
* **No GPU validation.** No ModelOpt capability for `Qwen/Qwen3.5-2B` (or any model) has
  been GPU-validated. The capability matrix is `UNKNOWN` across the board.
* **No benchmark or efficiency measurement.** `capability_deltas` and `efficiency_deltas`
  are empty by construction; no artifact was produced to measure.

---

## 7. Deterministic smoke command and config for later execution

The config is `configs/optimization/qwen35_2b_fp8_ptq_smoke.yaml` (status
`SMOKE_NOT_EXECUTED`; it pins the source checkpoint, revision, recipe, calibration
identity, and seed). Two commands run today with **no GPU and no ModelOpt**:

```bash
# Discovery only: regenerate the per-model capability matrix.
python -m opengrad.optimization capability-matrix \
    --output reports/optimization/modelopt-capability-matrix.json

# Deterministic CPU plumbing through the boundary (MOCK, never evidence).
python -m opengrad.optimization optimize --backend mock \
    --config configs/optimization/qwen35_2b_fp8_ptq_smoke.yaml \
    --output-dir runs/.dry-run/optimization
```

When a GPU host with the `optimization` extra becomes available, the same config drives
the real path; substitute the real trained-checkpoint directory for `source.path` first.
This command is the deferred step and is **not** run here:

```bash
pip install '.[optimization]'
python -m opengrad.optimization capability-matrix --backend modelopt   # re-probe with a real revision
python -m opengrad.optimization optimize --backend modelopt \
    --config configs/optimization/qwen35_2b_fp8_ptq_smoke.yaml \
    --output-dir runs/optimization/qwen35_2b_fp8_ptq --runtime vllm
```

A successful real run must then be measured by the existing inference backend and
benchmark/evaluation system against the same frozen held-out set that produced B0; the
optimization layer produces an artifact, it does not score one.

---

## 8. Invariants

* **Additive only.** No canonical-data file, SFT/DPO/distillation trainer, or existing
  inference backend was modified. `opengrad train`, dataset commands, readiness, and
  evaluation are unaffected and keep working without NVIDIA packages.
* **Fail closed.** A missing backend raises naming the extra; an output that would
  overwrite its source raises; a recipe missing its format or calibration identity raises.
* **No optimistic claims.** Every capability answer is `UNKNOWN`; no result is marked
  `EXECUTED`; no metric is fabricated.
* **Registry honest.** `nvidia-modelopt` is `INTERFACE_ONLY` (not `SUPPORTED_AFTER_VERIFICATION`).

---

## 9. What could not be verified

* **The ModelOpt execution path.** ModelOpt is not installed and no optimization is
  authorized, so `ModelOptBackend._execute()` and `export_artifact()` are unexecuted. They
  are written to the documented API but their correctness is **unverified** and they must
  not be treated as working until a real, reviewed run exists.
* **Per-model capability.** No capability for `Qwen/Qwen3.5-2B` (or any model) is
  validated; the matrix is `UNKNOWN` by design, not by omission.
* **Deployment/round-trip.** vLLM-compatibility and the Hugging Face export round-trip are
  unmeasured; both require producing a real artifact first.
