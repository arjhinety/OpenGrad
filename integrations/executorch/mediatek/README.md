# Qwen3.5 → MediaTek NeuroPilot (staged, SDK-blocked)

Status: **`BLOCKED_SDK_ACCESS`**. Nothing here has been compiled, run, or validated, and that is
stated rather than implied.

## Why it is blocked

MediaTek's ExecuTorch backend needs the NeuroPilot Express SDK, and `mtk_converter` is what turns a
PyTorch module into a MediaTek representation. There is no substitute for it. Probed from a clean
container on 2026-09-12:

| check | result |
|---|---|
| `neuropilot.mediatek.com` docs | HTTP 200, reachable |
| `pip download mtk-converter` | `No matching distribution found` — *"from versions: none"* |
| `pip download mtk-neuron` | `No matching distribution found` |

So the wheels are portal-gated behind registration, not publicly distributed. Evidence:
`results/quantization/executorch/probe_mediatek_sdk.json`.

Two further constraints, independent of the SDK:

- **No model support.** ExecuTorch has no `qwen3_5` definition for MediaTek;
  `examples/mediatek/models/llm_models/` covers Qwen2/2.5/3 dense only, and there is no
  `export_llm` integration for this backend at all.
- **No way to evaluate.** Unlike Qualcomm, which ships an x86 HTP emulator, MediaTek has no host
  emulator. Even a successful build could not be behaviourally scored without a Dimensity
  9300/9400 device.

## What is here

| file | status |
|---|---|
| `configuration_qwen3_5.py` | **complete** — pure config parsing, no SDK needed |
| `modeling_qwen3_5.py` | **scaffold** — structure only; unimplemented paths raise |
| `IMPLEMENTATION_SPEC.md` | **complete** — what to build, exact shapes, op risks, how to validate |
| `export_qwen3_5.sh` | **complete** — one-command runner for when the wheels exist |

`modeling_qwen3_5.py` deliberately raises `NotImplementedError` on the Gated DeltaNet path instead
of returning something plausible. An NPU model that silently produces wrong numbers is worse than
one that refuses to build, because the first kind ships.

## To unblock

1. Register at the [NeuroPilot portal](https://neuropilot.mediatek.com/) and download NeuroPilot
   Express. You need `mtk_converter` (cp310 wheel — **pin Python 3.10**), `mtk_neuron`,
   `libneuronusdk_adapter.mtk.so`, `libneuron_buffer_allocator.so`, and `NeuronAdapter.h`.
2. Copy `NeuronAdapter.h` into `backends/mediatek/runtime/include/api/`.
3. `pip install mtk_neuron-*.whl mtk_converter-*.whl`
4. Copy `configuration_qwen3_5.py` and `modeling_qwen3_5.py` into
   `examples/mediatek/models/llm_models/`.
5. Implement the DeltaNet layer per `IMPLEMENTATION_SPEC.md`.
6. `./export_qwen3_5.sh <path-to-checkpoint>`

## Validate before believing any export

Steps 1–3 of the spec's validation ladder need **only PyTorch** — no SDK, no device — and are the
highest-value work available on this branch today:

1. per-layer numerics against HF `Qwen3_5ForCausalLM`
2. full-model logits against the BF16 reference
3. state threading: one-pass vs split-prompt must agree

Only then run the frozen confirmatory set through `release/executorch/*/score_generations.py` and
apply `quantization_preservation_v1`. A completed export is not evidence of preserved behaviour.

## Parent

`arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` @ `f33d2030…`, checkpoint
`dpo-checkpoint-30`. Note its `config.json` declares `mtp_num_hidden_layers: 1` while containing no
`mtp.*` tensors — `configuration_qwen3_5.py` defaults that field to 0 so a checkpoint without MTP
weights cannot silently receive a randomly-initialised head.
