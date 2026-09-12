# OpenGrad × ExecuTorch — custom export path for Qwen3.5

Exporting the promoted OpenGrad model to ExecuTorch is **not a stock one-command export**. Getting
a `.pte` out requires configuration that deviates from the shipped defaults and, for the Snapdragon
target, source patches to ExecuTorch itself. This directory records all of it so the exports are
reproducible by someone who does not have this conversation.

Parent model: `arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2` @ `f33d2030…`,
checkpoint `dpo-checkpoint-30`.
ExecuTorch pinned at **`df6147afadf106a0ef3a74f65d80b5c589e63e44`** — the `qwen3_5` example landed
after release v1.4.1, so a release wheel cannot build this.

## Why it is not all-in-one

Qwen3.5 is a **hybrid**: 18 of its 24 layers are Gated DeltaNet with recurrent state, 6 are gated
full attention. ExecuTorch's LLM export path was built around homogeneous KV-cache attention, and
almost every problem below traces back to that single fact.

| Target | Outcome | Size | What was needed |
|---|---|---:|---|
| CPU / XNNPACK fp32 | **Exported** | 8.91 GiB | config deviations only |
| CPU / XNNPACK 8da4w | **Exported** | 2.89 GiB | config deviations only |
| Snapdragon / QNN | see ledger | — | 1 image dependency + 2 source patches |
| MediaTek | `BLOCKED_SDK_ACCESS` | — | SDK is portal-gated; see `mediatek/` |

Both CPU artifacts are `EXPORTED_PENDING_EVALUATION`: exported is not evaluated, and neither
carries a preservation verdict until scored on the frozen confirmatory set.

### 8da4w works, despite the upstream README

`examples/models/qwen3_5/README.md` states:

> `q8da4w` quantization for Qwen3.5 is intentionally deferred to a follow-up.

It was attempted anyway, on the principle that a README can lag its code — and it exported cleanly
(`rc=0`, 937s, `group_size=32`), producing a **3.09× smaller** artifact than fp32. The documentation
is stale, not the capability.

The size is consistent with the weights genuinely being quantized rather than the pass silently
no-op'ing: fp32 weights are ~1.88B × 4B ≈ 7.0 GiB of the 8.91 GiB total, and at ~4 bits plus scales
those become ~1.0 GiB, leaving ~1.9 GiB of static-shape buffers — which lands at ≈2.9 GiB, matching
the measured 2.89 GiB. That is corroboration, not proof; per-layer quantized/skipped counts are the
real check and are still owed.

## Configuration deviations (all targets)

Both are required by the frozen measurement contract, and both are applied explicitly rather than
inherited:

| setting | stock | OpenGrad | why |
|---|---|---|---|
| `export.max_seq_length` | 2048 | **5760** | longest frozen confirmatory prompt is 5,235 tokens + 512 completion budget; 2048 would truncate the evaluation into a different measurement |
| `export.max_context_length` | 2048 | **5760** | same |
| Hydra override prefix | `+` | **`++`** | the stock yaml already defines these keys, and `+` refuses an existing key |

Not patched, but recorded: the stock metadata declares `get_bos_id: 248045`, which is
`<|im_start|>` — the token every OpenGrad-rendered prompt already begins with, on a tokenizer whose
`add_bos_token` is `False`. A runner honouring that metadata doubles the token and silently measures
a different model. The handoff packages therefore ship **pre-tokenized** prompts; see
`release/executorch/*/PARITY_NOTES.md`.

## Source patches (Snapdragon / QNN only)

Both are genuine upstream defects triggered by hybrid attention, not fudges to force an export.
They are applied at runtime by `scripts/modal/executorch_export.py::_patch_executorch_for_hybrid`,
which records before/after SHA-256 in the artifact provenance under `toolchain_patch` and
**refuses to apply if upstream source has drifted** rather than forcing a match.

### `0001-qnn-lift-pass-skip-higher-order-ops.patch`

`LiftConstantScalarOperands._create_tensor_args` dereferences `node.target._schema` for every
`call_function` node. A `HigherOrderOperator` has none:

```
AttributeError: 'AutoFunctionalizedV2' object has no attribute '_schema'
```

`auto_functionalized_v2` is PyTorch's wrapper for ops that mutate their inputs, so this fires for
**any** model with in-place state mutation — here, DeltaNet's recurrent state. A HOP has no liftable
constant scalars, so skipping is correct.

### `0002-qnn-hybrid-attention-cache-shape.patch`

`_to_edge_and_lower_llama` reads the `TagQuantIO` cache shape off `model.layers[0].attention`,
assuming layer 0 has a KV cache. Qwen3.5's `layer_types[0]` is `linear_attention`:

```
AttributeError: 'AttentionGatedDeltaNet' object has no attribute 'max_context_len'
```

Fixed by selecting the first attention module that actually has a cache. **This fixes the crash,
not the semantics** — the resulting shape describes only the full-attention KV cache, and the
recurrent-state I/O remains untagged for quantization. A complete upstream fix would tag both.

### Environment

The QNN x86 libraries link against LLVM's libc++, not libstdc++. Without `libc++1` / `libc++abi1`,
`libQnnHtp.so` fails to `dlopen` and the export aborts at backend init — *before the partitioner
ever sees the model*, which makes it look like a model rejection when it is a missing package.

## Honesty note on patch count

Two patches is a port. Ten would be a different finding.

If this list keeps growing, the correct conclusion is that ExecuTorch's QNN path assumes
homogeneous attention throughout, and the study should report that rather than claiming "Qwen3.5
works on QNN" after stacking fixes until something emits a file. The count is tracked deliberately
for that reason, and every artifact records which patches were applied to produce it.

## Reproducing

```bash
modal run scripts/modal/executorch_export.py --target cpu                 # fp32 XNNPACK
modal run scripts/modal/executorch_export.py --target cpu --quantize 8da4w
modal run scripts/modal/executorch_export.py --target qnn                 # applies both patches
modal run scripts/modal/mediatek_sdk_probe.py                             # SDK availability
```

Results land in `results/quantization/executorch/*.json`, each carrying `versions`,
`toolchain_patch`, the full command, and the artifact hash. Handoff packages for evaluation are
built by `scripts/build_executorch_handoff.py`.

## Upstreaming

Both patches are written against the pinned commit and are suitable for submission to
pytorch/executorch as-is. Patch 2's semantic gap (untagged recurrent-state I/O) should be mentioned
in any upstream discussion rather than presented as complete.
