# Inference

How models are executed for measurement, and what that execution costs.

| Document | Covers |
|---|---|
| [efficiency.md](efficiency.md) | The efficiency axis: latency, throughput, memory, and why it is kept separate from behavioural correctness |
| [pre-gpu-surfaces.md](pre-gpu-surfaces.md) | The accelerator surface: what is declared, probed, and verified |

vLLM is the engine of record for evaluation, and its version is recorded as provenance on every
measurement. The engine is declared in the frozen config and pinned, so two runs cannot be
compared if they were measured by different runtimes. The backend abstraction lives in
`src/opengrad/evaluation/runner.py`.
