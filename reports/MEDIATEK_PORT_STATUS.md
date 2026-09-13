# MediaTek / NeuroPilot port status — Qwen3.5-2B

**Status: `BLOCKED_PORT_INCOMPLETE`** (previously `BLOCKED_SDK_ACCESS`).

The blocker moved. SDK access is solved; the architecture port is not.

## What changed

The NeuroPilot Express SDK was obtained directly (the portal is registration-gated and an
automated probe could not obtain it: the SDK is not on PyPI and is not downloadable without a
login. The committed probe, `results/quantization/executorch/probe_mediatek_sdk.json`, records only
two public documentation URLs, both HTTP 200; no 404 is recorded).

| component | version |
|---|---|
| package | `neuropilot-express-sdk-8.0.8-build20250925` |
| `mtk_converter` | 8.13.0+public, **cp310** manylinux_2_17_x86_64 wheel |
| `mtk_neuron` | 8.2.23, py3-none-linux_x86_64 |
| `libneuronusdk_adapter.mtk.so` | 13,621,224 bytes |
| `libneuron_buffer_allocator.so` | 771,056 bytes |
| `api/NeuronAdapter.h` | 103,800 bytes |
| archive sha256 | `42155c3b25e9f9af1bef55534d0700835f32564a289e8cb1e44f0b2433ddab48` |

`mtk_converter` exposes a **`torch.export` frontend**
(`converters/pytorch/exported_program_utils.py`, `converter_v2`) plus a custom-op compiler, so an
`ExportedProgram` is an accepted input path.

## Licensing — binding

NeuroPilot Express is **proprietary and licensed** (`LICENSE AGREEMENT.pdf` ships in the archive).

- **Not** committed to this repository
- **Not** uploaded to HuggingFace or any public artifact
- **Not** vendored
- **No** extracted SDK source published

The repository documents how a user with legitimate SDK access installs it locally. That is the
only permitted form.

## Why the export is still blocked

`integrations/executorch/mediatek/modeling_qwen3_5.py` is an honest scaffold:

- the **full-attention path is real** — 6 of 24 layers
- the **Gated DeltaNet path raises `NotImplementedError`** — 18 of 24 layers

It raises rather than emitting a model that would produce plausible-looking wrong numbers. A
partial implementation is not an acceptable export when it covers 75% of the layers.

### What Gated DeltaNet requires (from the verified architecture)

Measured directly from the promoted checkpoint, not assumed:

| weight | shape | per layer |
|---|---|---|
| `in_proj_qkv` (fused) | 6144 × 2048 | 1 |
| `in_proj_z` (output gate) | 2048 × 2048 | 1 |
| `out_proj` | 2048 × 2048 | 1 |
| `a_proj`, `b_proj` (per-head gate scalars) | 16 × 2048 | 2 |
| causal depthwise `conv1d` | 6144 × 1 × 4 | 1 |

Plus recurrent state across the sequence, `linear_num_key_heads`/`value_heads` = 16 at head dim
128, and **no RoPE on DeltaNet layers** — applying it would be a silent correctness bug.

## Required order of work

Architecture correctness is established **before** the compiler is involved. Debugging an
architecture through NeuroPilot compiler errors conflates two independent failure sources.

1. implement `Qwen3_5LinearAttention` (Gated DeltaNet) faithfully
2. load canonical weights (`dpo-checkpoint-30`, sha `903f9b11…`)
3. verify deterministic forward execution
4. compare logits / hidden states against the known-good HF implementation
5. test representative sequence lengths (the frozen set reaches 5,235 prompt tokens)
6. validate cache / recurrent-state behaviour
7. `torch.export` independently — must succeed on its own
8. **only then** hand the `ExportedProgram` to `mtk_converter`

## Torch-version incompatibility — to be handled scientifically

`exported_program_utils.py` gates on `TORCH_2_1` / `TORCH_2_2` / `TORCH_2_3` only. The current
ExecuTorch exports ran on torch 2.14.0+cpu (`results/quantization/executorch/export_cpu_*.json`).
This is unverified, not known-broken.

**Do not patch the SDK's version guard as a first move.** Instead, stand up an isolated environment
matching the SDK's supported torch family and determine separately:

- **A.** can Qwen3.5 be exported at all under the SDK-supported torch stack?
- **B.** can the resulting MediaTek artifact be consumed independently of the modern ExecuTorch build?
- **C.** can an interchange format bridge the two environments?

Patching the guard conflates "the SDK refuses this torch version" with "the SDK cannot handle this
graph", and those have completely different remedies.

## What exists today

| item | state |
|---|---|
| `configuration_qwen3_5.py` | staged |
| `modeling_qwen3_5.py` | scaffold; DeltaNet raises |
| `export_qwen3_5.sh` | one-command runner with prerequisite checks |
| `IMPLEMENTATION_SPEC.md` | what the DeltaNet layer must do |
| SDK | obtained, licensed, local only |
| Android NDK r26.3.11579264 | not obtained |
| device or emulator | **none exists for this target** |

## Honest ceiling

Even a fully successful export yields **no runtime numbers**. There is no MediaTek device and no
emulator in this setup, so the deliverable would be a compiled artifact plus a correctness
argument — not a performance result. Any device measurement belongs to OpenWeights.

This track is **independent of QAD** and must not be mixed into that experiment.
