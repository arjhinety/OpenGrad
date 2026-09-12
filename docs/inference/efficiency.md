# Efficiency research

OpenGrad's systems research centers on generation methods: speculative decoding, native multi-token prediction (MTP) heads, external draft/target models, and architecture-aware decoding. OpenWeights is the downstream device execution environment for compatible GGUF/llama.cpp and ExecuTorch artifacts.

Future measurements will capture TTFT, prefill/decode throughput, memory, quantization, device/backend, and thermal or power context where measurable. Behavioral tool correctness remains a separate axis. No efficiency result exists yet.

## Planned measurements

Runtime studies may record: time to first token; prefill and decode throughput; end-to-end latency;
VRAM/RAM; checkpoint size; quantization effects; speculative acceptance rate and accepted length;
drafted and accepted tokens per step; draft/target verification cost; total model footprint; added
parameters; KV-cache use; load time; output equivalence; parser/EOS/tool failures; and energy or
thermal behavior where reliable instrumentation exists.

## Target-attached vs external speculation

OpenGrad uses **target-attached/native speculative decoding** to mean a speculative mechanism
trained into or closely attached to the target model, such as an MTP or Medusa-style head, an
architecture-permitted EAGLE-like method, a self-speculative method, or another attached mechanism.
It does not mean that external draft-model speculation is inherently bad, and no approach is
presumed faster. Where feasible, the comparison is ordinary autoregressive decoding versus external
draft-model speculation versus target-attached/native speculation. An efficiency gain is not
automatically desirable if capability or reliability falls substantially.

Current status: speculative decoding is planned, not implemented. A reserved configuration exists at
`configs/inference/speculative/`, and no method is benchmarked or supported. The metric taxonomy and
quality-parity principle are specified in [Speculative Decoding & MTP](../evaluation/SPECULATIVE_DECODING.md).
The optional optimization producer layer (quantization recipes, INTERFACE_ONLY) is described in
[optimization](../optimization/README.md).
