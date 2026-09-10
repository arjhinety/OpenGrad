# How to Add an Inference Backend to OpenGrad

OpenGrad benchmarks are runtime-agnostic. The harness relies on the `InferenceBackend` protocol so that evaluations can run across Hugging Face Transformers, vLLM, llama.cpp, SGLang, custom native MTP runtimes, or DSpark without changing benchmark logic.

---

## 1. The InferenceBackend Protocol

Defined in `src/opengrad/benchmarks/backends/protocol.py`:

```python
class InferenceBackend(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult: ...
```

---

## 2. The GenerationResult Contract

Every backend must return a structured `GenerationResult`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | The decoded generated completion string. |
| `token_ids` | `list[int]` | Sequence of generated completion token IDs. |
| `prompt_tokens` | `int` | Number of tokens in the prefill prompt. |
| `completion_tokens`| `int` | Number of tokens generated in completion. |
| `latency` | `float` | Wall-clock generation time in seconds. |
| `ttft` | `float` | Time to first token in seconds. |
| `token_timestamps` | `list[float]`| Timestamp for each token decode event. |
| `backend_metadata` | `dict[str, Any]`| Runtime details (peak VRAM, batch size, engine version). |
| `speculative_metadata` | `dict[str, Any] \| None`| Speculative draft/verifier telemetry (if applicable). |

---

## 3. Emitting Speculative / MTP Metadata

When implementing speculative decoding engines (e.g., native MTP or DSpark), populate `speculative_metadata`:

```python
speculative_metadata = {
    "speculation_mode": "mtp",  # "mtp", "speculative_draft", "dspark"
    "speculation_depth_requested": 3,
    "speculation_depth_achieved": 3,
    "proposed_tokens": 120,
    "accepted_tokens": 96,
    "rejected_tokens": 24,
    "acceptance_rate": 0.80,
    "accepted_tokens_per_step": 2.6,
    "verifier_steps": 40,
    "rollback_count": 6,
    "recomputed_tokens": 12,
    "verification_overhead_ms": 16.4,
    "memory_overhead_mb": 450.0,
    "mtp_per_depth_acceptance": {
        "depth_1": 0.85,
        "depth_2": 0.71,
        "depth_3": 0.48,
        "depth_4": 0.22,
    }
}
```

---

## 4. Implementing a New Backend

Example skeleton for a vLLM / SGLang backend:

```python
from opengrad.benchmarks.backends.protocol import GenerationResult

class VLLMInferenceBackend:
    name = "vllm"

    def __init__(self, model_id: str, speculative_model: str | None = None) -> None:
        self.model_id = model_id
        self.speculative_model = speculative_model
        # Initialize LLM engine lazily...

    def generate(
        self,
        prompt: str,
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        # 1. Execute generation with engine
        # 2. Extract timing and speculative counters
        # 3. Return GenerationResult
        ...
```

---

## 5. Verification

Verify the backend using the internal performance microsuite:
```bash
opengrad benchmark run --benchmark performance_microsuite --backend <your_backend>
```
Ensure that:
1. Deterministic generation is reproducible.
2. Latency and token counts match server telemetry.
3. Quality parity is verified against standard autoregressive decoding.
