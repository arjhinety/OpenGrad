# Pre-GPU configuration surfaces

Accelerator work has started. The bounded Qwen boundary smoke ran on an NVIDIA A100-SXM4-80GB and is recorded in `reports/hardware/qwen_gpu_smoke.json`; `configs/hardware/gpu_preflight_v1.yaml` now records that executed preflight (`status: READY`, `COMPATIBLE`) rather than a `NOT_RUN` placeholder. It is not a general claim that every GPU, driver, CUDA, ROCm, runtime, or quantization path works — quantization and speculative decoding remain unexecuted.

A future preflight must record requested and observed device counts, VRAM, driver/runtime versions, provider (`nvidia` or `amd`), and a compatibility result with its basis. `UNKNOWN`, `NOT_TESTED`, and `INCOMPATIBLE` must not be collapsed into support. The contract is `registry/gpu_preflight.schema.json`.

## Reserved runtime/component references

`registry/runtime_components.yaml` records upstream references, plus the one interface
boundary that now exists:

- NVIDIA ModelOpt: optional optimization-producer interface (`INTERFACE_ONLY`); no execution, NVIDIA `NOT_TESTED`, AMD/CPU `UNKNOWN`. See [the optimization layer](../optimization/README.md).
- vLLM Speculators: speculative-decoding reference; NVIDIA and AMD `NOT_TESTED`.
- DSpark: inference-runtime reference; NVIDIA and AMD `NOT_TESTED`.

These entries provide provenance and an explicit compatibility matrix. They do not install dependencies, download weights, run inference, or establish support. Their machine-readable contract is `registry/runtime_components.schema.json`; `registry/runtimes.yaml` retains the runtime-level index.
