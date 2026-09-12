# Qwen3.5 configuration for the MediaTek NeuroPilot ExecuTorch backend.
#
# Drop into examples/mediatek/models/llm_models/ in an ExecuTorch checkout. This file is complete
# and needs no SDK: it is pure config parsing, matching the BaseConfig contract that
# configuration_qwen.py and configuration_gemma.py follow.
#
# The model definition beside it (modeling_qwen3_5.py) is deliberately NOT complete -- see
# IMPLEMENTATION_SPEC.md for exactly what is missing and why it could not be written blind.

from models.llm_models.configuration_base import BaseConfig


# flake8: noqa: C901


class Qwen3_5Config(BaseConfig):
    """Qwen3.5 hybrid text config.

    Qwen3.5 is not a dense transformer. Of its 24 decoder layers, 18 are Gated DeltaNet
    ("linear_attention") carrying recurrent state and 6 are gated full attention, interleaved every
    `full_attention_interval`. `layer_types` in the HF config is authoritative and is preserved
    verbatim rather than re-derived, because a mis-derived interleave silently produces a model
    that loads and generates nonsense.
    """

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        head_dim=None,
        max_position_embeddings=None,
        norm="RMSNorm",
        position_embedding="rope",
        norm_eps=1e-6,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=248046,
        unk_token_id=None,
        use_stable_embedding=False,
        tie_word_embeddings=True,
        combine_qkv=False,
        response_handler=None,
        model_type="qwen3_5_text",
        # Hybrid / linear-attention geometry.
        layer_types=None,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        # Attention detail.
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_parameters=None,
        rope_theta=10000000.0,
        # Multi-token prediction. Declared by the base repo; see the note below.
        mtp_num_hidden_layers=0,
        **kwargs,
    ):
        super().__init__()

        self.model_type = model_type
        self.vocab_size = vocab_size
        if self.vocab_size is None:
            raise KeyError("vocab_size is required but missing from config.json")
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            raise KeyError("hidden_size is required but missing from config.json")
        self.intermediate_size = intermediate_size
        if self.intermediate_size is None:
            raise KeyError("intermediate_size is required but missing from config.json")
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers is None:
            raise KeyError("num_hidden_layers is required but missing from config.json")
        self.num_attention_heads = num_attention_heads
        if self.num_attention_heads is None:
            raise KeyError("num_attention_heads is required but missing from config.json")

        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        # Qwen3.5 does NOT satisfy head_dim == hidden_size // num_attention_heads:
        # 2048 // 8 == 256 happens to match at 2B, but the field is explicit upstream and other
        # sizes break the assumption, so it is read rather than computed.
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.max_position_embeddings = max_position_embeddings or 262144
        self.norm = norm
        self.norm_eps = norm_eps
        self.position_embedding = position_embedding

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.use_stable_embedding = use_stable_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.combine_qkv = combine_qkv
        self.use_qk_norm = True  # Qwen3.5 applies q_norm/k_norm before rope

        # --- hybrid geometry -------------------------------------------------------------
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            # Derived only as a fallback; the HF config always ships layer_types explicitly.
            layer_types = [
                "full_attention"
                if (index + 1) % full_attention_interval == 0
                else "linear_attention"
                for index in range(self.num_hidden_layers)
            ]
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )
        unknown = set(layer_types) - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"unknown layer types: {sorted(unknown)}")
        self.layer_types = list(layer_types)

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.attn_output_gate = attn_output_gate
        rope_parameters = rope_parameters or {}
        self.partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor", partial_rotary_factor
        )
        self.rope_theta = rope_parameters.get("rope_theta", rope_theta)
        self.mrope_section = rope_parameters.get("mrope_section")
        self.mrope_interleaved = rope_parameters.get("mrope_interleaved", True)

        # The promoted OpenGrad checkpoint declares mtp_num_hidden_layers: 1 while containing no
        # mtp.* tensors -- both trainers load through AutoModelForCausalLM, which never builds the
        # head. Defaulting to 0 here means a checkpoint without MTP weights does not silently get a
        # randomly-initialised MTP block. Set it explicitly only when the tensors are present.
        self.mtp_num_hidden_layers = mtp_num_hidden_layers

        self.tokenizer = "pretrained_fast"

    @property
    def linear_layer_indices(self):
        return [i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"]

    @property
    def full_attention_layer_indices(self):
        return [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]

    def print_config(self, response_handler=None):
        emit = print if response_handler is None else response_handler
        emit(f"model_type: {self.model_type}")
        emit(f"vocab_size: {self.vocab_size}")
        emit(f"hidden_size: {self.hidden_size}")
        emit(f"intermediate_size: {self.intermediate_size}")
        emit(f"num_hidden_layers: {self.num_hidden_layers}")
        emit(f"num_attention_heads: {self.num_attention_heads}")
        emit(f"num_key_value_heads: {self.num_key_value_heads}")
        emit(f"head_dim: {self.head_dim}")
        emit(f"norm: {self.norm} (eps {self.norm_eps})")
        emit(f"position_embedding: {self.position_embedding} (theta {self.rope_theta})")
        emit(f"partial_rotary_factor: {self.partial_rotary_factor}")
        emit(f"mrope_section: {self.mrope_section} interleaved={self.mrope_interleaved}")
        emit(f"attn_output_gate: {self.attn_output_gate}")
        emit(
            f"hybrid: {len(self.linear_layer_indices)} linear_attention / "
            f"{len(self.full_attention_layer_indices)} full_attention "
            f"(interval {self.full_attention_interval})"
        )
        emit(
            f"linear heads: k={self.linear_num_key_heads}x{self.linear_key_head_dim} "
            f"v={self.linear_num_value_heads}x{self.linear_value_head_dim} "
            f"conv_kernel={self.linear_conv_kernel_dim}"
        )
        emit(f"tie_word_embeddings: {self.tie_word_embeddings}")
        emit(f"mtp_num_hidden_layers: {self.mtp_num_hidden_layers}")
