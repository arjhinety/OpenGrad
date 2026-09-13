# Qwen3.5 model definition for the MediaTek NeuroPilot ExecuTorch backend.
#
# STATUS: structural scaffold. The full-attention path is real; the Gated DeltaNet path is NOT
# implemented and raises rather than returning plausible-looking wrong numbers.
#
# Read IMPLEMENTATION_SPEC.md before touching this. The short version: 18 of Qwen3.5-2B's 24 layers
# are Gated DeltaNet with recurrent state, and modeling_common.py has no conv1d, no recurrent state
# and no delta rule to build them from. Writing that blind -- with no SDK to compile it, no device
# to run it and no emulator to test it -- would produce code whose correctness nobody could check.
#
# Drop into examples/mediatek/models/llm_models/ in an ExecuTorch checkout, beside
# modeling_qwen.py, together with configuration_qwen3_5.py.

import torch
from models.llm_models.configuration_qwen3_5 import Qwen3_5Config
from models.llm_models.modeling_common import MLP, Attention, DecoderLayer, ModelChunk

SPEC = "integrations/executorch/mediatek/IMPLEMENTATION_SPEC.md (OpenGrad repository)"

REFERENCE = (
    "executorch/examples/models/llama/attention.py::AttentionGatedDeltaNet and "
    "_recurrent_gated_delta_rule -- the authority for numerics. Proven to trace: the OpenGrad QNN "
    "export lowered DeltaNet successfully before failing for an unrelated backend reason."
)


class Qwen3_5MLP(MLP):
    """Standard SwiGLU MLP. Identical in both layer kinds, so nothing special is needed."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)


class Qwen3_5FullAttention(Attention):
    """The 6 gated full-attention layers.

    These are close to Qwen3 dense attention and are expressible with modeling_common's Attention,
    but three details differ and each changes the output silently if missed:

      * q_norm / k_norm are applied BEFORE rope (`qk_norm_before_rope`), not after;
      * rotary is partial -- `partial_rotary_factor` 0.25, so only the first 25% of head_dim is
        rotated and the remainder passes through unmodified;
      * mrope is interleaved with sections [11, 11, 10] rather than plain rope;
      * `attn_output_gate` multiplies the attention output by a learned gate before out_proj.

    modeling_common.Attention supports qk-norm and rope, but NOT partial rotary, NOT interleaved
    mrope and NOT the output gate. Those three are the additions required here; they are small and
    local compared with the DeltaNet work, but they are still additions.
    """

    def __init__(self, config: Qwen3_5Config, jit_trace=False):
        super().__init__(config, jit_trace)
        self.partial_rotary_factor = config.partial_rotary_factor
        self.mrope_section = config.mrope_section
        self.mrope_interleaved = config.mrope_interleaved
        self.attn_output_gate = config.attn_output_gate
        if self.partial_rotary_factor != 1.0 or self.attn_output_gate:
            raise NotImplementedError(
                "Qwen3.5 full attention needs partial rotary "
                f"({self.partial_rotary_factor}), interleaved mrope "
                f"({self.mrope_section}) and attn_output_gate={self.attn_output_gate}; "
                f"modeling_common.Attention provides none of the three. See {SPEC}"
            )


class Qwen3_5LinearAttention(torch.nn.Module):
    """The 18 Gated DeltaNet layers. NOT IMPLEMENTED.

    Parameter inventory and exact 2B shapes are in IMPLEMENTATION_SPEC.md. The load-bearing facts:

      * DeltaNet layers do NOT use rope. Applying it is a silent correctness bug.
      * Two state tensors must cross MediaTek ModelChunk boundaries as explicit graph I/O:
        conv_state (batch, conv_dim=6144, 4) and the delta-rule carry
        (batch, 16, 128, 128) per layer -- 1 MiB per layer at fp32, 18 MiB across the model.
        modeling_common threads only a KV cache, whose lifetime semantics differ: it grows, these
        are fixed-size and updated in place.
      * The recurrence is sequential over the sequence dimension. Whether mtk_converter can express
        a sequence-carried recurrence at all, or must unroll it to max_seq_len (catastrophic at
        5760), is an empirical question that needs the SDK to answer.
    """

    def __init__(self, config: Qwen3_5Config, jit_trace=False):
        super().__init__()
        self.config = config
        raise NotImplementedError(
            "Gated DeltaNet is not implemented for the MediaTek backend.\n"
            f"  spec      : {SPEC}\n"
            f"  reference : {REFERENCE}\n"
            "  blocked by: BLOCKED_SDK_ACCESS -- mtk_converter and mtk_neuron are not on PyPI and "
            "require NeuroPilot portal registration, so no implementation here can be compiled, "
            "run or validated."
        )


class Qwen3_5DecoderLayer(DecoderLayer):
    """Dispatches on `layer_types[layer_idx]` rather than on an index formula.

    The HF config ships `layer_types` explicitly and it is authoritative. Re-deriving the interleave
    from `full_attention_interval` produces a model that loads, runs, and generates nonsense when
    the two disagree -- the worst possible failure shape.
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        return_attn=False,
        jit_trace=False,
    ):
        kind = config.layer_types[layer_idx]
        attn_class = (
            Qwen3_5FullAttention if kind == "full_attention" else Qwen3_5LinearAttention
        )
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=attn_class,
            mlp_class=Qwen3_5MLP,
        )
        self.layer_idx = layer_idx
        self.layer_kind = kind


class Qwen3_5ModelChunk(ModelChunk):
    """A slice of layers for the NPU.

    Unfinished by construction: ModelChunk threads a KV cache between chunks, and a hybrid model
    must additionally thread conv_state and the delta-rule carry for every linear layer in the
    chunk. That plumbing is step 3 of the validation ladder in the spec and is the single most
    likely place for a silent bug -- a model with mis-threaded state produces correct output for a
    one-pass prompt and wrong output when the prompt is split.
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        num_blocks,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            num_blocks,
            chunk_idx,
            dtype,
            include_tail,
            return_attn,
            jit_trace,
            decoder_class=Qwen3_5DecoderLayer,
        )
        linear_in_chunk = [
            index
            for index in range(chunk_idx * num_blocks, (chunk_idx + 1) * num_blocks)
            if index < config.num_hidden_layers
            and config.layer_types[index] == "linear_attention"
        ]
        if linear_in_chunk:
            raise NotImplementedError(
                f"chunk {chunk_idx} contains linear_attention layers {linear_in_chunk}, whose "
                f"conv_state and recurrent state must be threaded across chunk boundaries. "
                f"See {SPEC}"
            )
