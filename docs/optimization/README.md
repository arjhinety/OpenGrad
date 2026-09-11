# Optimization layer

The optimization layer is an isolated **producer**: it turns a trained checkpoint into a
new, separately-identified optimized checkpoint. It does not train, it does not execute,
and it does not measure.

```
trained checkpoint  ->  optimization recipe  ->  optimized checkpoint
        ->  existing inference backend  ->  existing benchmark / evaluation system
```

The split matters. Optimization **changes artifacts**. The existing inference backend
**executes** them. The existing benchmark and evaluation system **measures** them against
the same frozen held-out contract the baseline used. Nothing in this layer re-implements
generation, parsing, or scoring, so a quantization, pruning, or distillation study is
comparable to every other result in the repository.

## Inputs and outputs

* A **`SourceCheckpoint`** (checkpoint id, path, model id + revision, hash, experiment id,
  and experiment lineage) is read-only. A backend must never write into, over, or above it,
  and `ensure_distinct_output()` enforces that.
* An **`OptimizationRecipe`** (technique, target format, calibration identity, parameters)
  is the model-independent description of what to produce, pinned by a deterministic
  `recipe_hash`.
* An **`OptimizationResult`** is the frozen output: the recipe and its hash, the source
  identity and lineage, the backend and its version, calibration identity/fingerprint/
  sample count/seed, the quantization format, hardware and library versions, the export
  format, the output artifact path and hash, the runtime used, benchmark ids, capability
  and efficiency deltas, failures, and unsupported features.

The result's `execution_status` and `evidence` fields are the honesty boundary: a mock or
dry-run is never evidence, and `NOT_ATTEMPTED` means exactly that.

## Backends

Backends are selected by a small, hard-coded selector (`select_optimization_backend`) that
mirrors `_select_trainer`.

* `MockOptimizationBackend` (`mock`) is deterministic and CPU-only. It writes a
  deterministic provenance artifact and reports `MOCK`, `evidence=False`. It exists so the
  boundary can be exercised with no GPU and no optional dependency.
* `ModelOptBackend` (`modelopt`) is the one real backend and is optional. See
  [modelopt-integration.md](modelopt-integration.md).

Capability discovery is separate from execution and is described in
[modelopt-integration.md](modelopt-integration.md).

## Commands

```bash
# Discovery only. Works with no GPU and no ModelOpt.
python -m opengrad.optimization capability-matrix \
    --output reports/optimization/modelopt-capability-matrix.json

# Deterministic CPU plumbing through the boundary. Never evidence.
python -m opengrad.optimization optimize --backend mock \
    --config configs/optimization/qwen35_2b_fp8_ptq_smoke.yaml \
    --output-dir runs/.dry-run/optimization
```

## Status

INTERFACE_ONLY. No optimization has been executed. Capability discovery reports `UNKNOWN`
for every technique, and no ModelOpt capability has been GPU-validated. See
[the integration report](../../reports/MODELOPT_INTEGRATION_REPORT.md).
