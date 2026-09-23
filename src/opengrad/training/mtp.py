"""Full-model loading for training, and the native Qwen3.5 multi-token-prediction (MTP) layer.

Why this exists
---------------
Transformers 5.16.1 implements Qwen3.5's language model and vision encoder but not its MTP layer:
every Qwen3.5 class lists ``^mtp.*`` under ``_keys_to_ignore_on_load_unexpected``, and
``Qwen3_5ForCausalLM`` also drops ``^model.visual.*``. Loading through those classes, as the
trainers did before ``full-model-components-v1``, silently produced text-only checkpoints. This
module loads every declared component, attaches the MTP layer under the checkpoint's own tensor
names so ``save_pretrained`` writes it back, and computes the MTP training loss.

The MTP layer, as inference engines run it
------------------------------------------
Matched to vLLM's ``Qwen3_5MultiTokenPredictor`` and its speculative-decoding proposer, because a
layer trained any other way would not draft correctly at inference:

* the draft pair at position ``t`` is ``(embed(x[t+1]), h[t])``, where ``h`` is the main model's
  **post-final-norm** hidden state (the input of the output head) and the embedding is the main
  model's own (``mtp_use_dedicated_embeddings: false``);
* ``fc(concat(pre_fc_norm_embedding(e), pre_fc_norm_hidden(h)))`` feeds one full-attention decoder
  layer at text position ``t``, then ``norm``, then the shared output head;
* the result at position ``t`` predicts ``x[t+2]``.

The tensor names (``mtp.fc.weight``, ``mtp.pre_fc_norm_embedding.weight``,
``mtp.pre_fc_norm_hidden.weight``, ``mtp.layers.0.*``, ``mtp.norm.weight``) are the pinned base
checkpoint's, and the layer is loaded from them with ``strict=True``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from opengrad.training.model_components import (
    ComponentSettings,
    ModelComponentError,
    carried_components,
    check_same_language_model,
    declared_components,
    partition_keys,
)

IGNORE_INDEX = -100
MTP_PREFIX = "mtp."
VISION_SEGMENT = "visual."


class Qwen35MultiTokenPredictor(nn.Module):  # type: ignore[misc]
    """Qwen3.5's native MTP layer, parameter-for-parameter the checkpoint's ``mtp.*`` tensors."""

    def __init__(self, text_config: Any) -> None:
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        count = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
        if count < 1:
            raise ModelComponentError("the text config declares no MTP layers")
        if getattr(text_config, "mtp_use_dedicated_embeddings", False):
            raise ModelComponentError(
                "mtp_use_dedicated_embeddings is not implemented; this layer reuses the main "
                "model's embeddings, which is what the pinned checkpoint declares"
            )
        config = copy.deepcopy(text_config)
        # Every MTP layer is a full-attention layer, whatever the main stack's interleaving.
        config.layer_types = ["full_attention"] * count
        config.num_hidden_layers = count
        hidden = int(config.hidden_size)
        self.config = config
        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(Qwen3_5DecoderLayer(config, index) for index in range(count))
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        # Non-persistent buffers only, so it adds no tensor to the checkpoint.
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask

        combined = torch.cat(
            [self.pre_fc_norm_embedding(input_embeddings), self.pre_fc_norm_hidden(hidden_states)],
            dim=-1,
        )
        states = self.fc(combined)
        position_embeddings = self.rotary_emb(states, position_ids)
        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        layer = self.layers[step % len(self.layers)]
        states = layer(
            states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            position_ids=position_ids,
        )
        return self.norm(states)  # type: ignore[no-any-return,unused-ignore]


class FinalHiddenCapture:
    """Records the input of the output head: the post-final-norm hidden state MTP consumes.

    A forward pre-hook rather than ``output_hidden_states``, so the main forward pass and its loss
    are computed exactly as they were before MTP existed.
    """

    def __init__(self, model: Any) -> None:
        self._hidden: torch.Tensor | None = None
        head = model.get_output_embeddings()
        self._handle = head.register_forward_pre_hook(self._hook)

    def _hook(self, module: Any, args: tuple[Any, ...]) -> None:
        self._hidden = args[0]

    def take(self) -> torch.Tensor:
        hidden, self._hidden = self._hidden, None
        if hidden is None:
            raise RuntimeError("the output head did not run, so there is no hidden state for MTP")
        return hidden

    def remove(self) -> None:
        self._handle.remove()


def mtp_targets(labels: torch.Tensor) -> torch.Tensor:
    """What the MTP output at position ``t`` is scored against: ``labels[t + 2]``.

    Shape ``(batch, length - 2)``, aligned with MTP outputs ``0 .. length - 3``. A position whose
    target is not supervised stays ``IGNORE_INDEX``.
    """
    return labels[:, 2:]


def mtp_loss(
    *,
    model: Any,
    mtp: Qwen35MultiTokenPredictor,
    final_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    gradient_scope: str,
    has_images: bool = False,
) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy of the MTP layer's ``x[t+2]`` predictions over supervised targets.

    Returns ``(loss, supervised_targets)``. With no supervised target the loss is a zero that is
    still connected to the MTP parameters, so accumulation and backward stay uniform.

    ``head_only`` detaches everything the main model owns -- the hidden state, the embedding
    lookup and the output-head weight -- so the gradient reaches ``mtp.*`` and nothing else.
    """
    if has_images:
        raise ModelComponentError(
            "the MTP loss over batches carrying images needs the multimodal position ids of the "
            "draft pairs, which are not implemented; refusing to train MTP on wrong positions"
        )
    if gradient_scope not in ("joint", "head_only"):
        raise ModelComponentError(f"unknown MTP gradient scope {gradient_scope!r}")
    batch, length = input_ids.shape
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if length < 3:
        zero = sum(parameter.sum() for parameter in mtp.parameters()) * 0.0
        return zero, 0

    next_ids = input_ids[:, 1:]
    hidden = final_hidden[:, :-1]
    if gradient_scope == "head_only":
        hidden = hidden.detach()
        input_embeddings = F.embedding(next_ids, embedding.weight.detach())
    else:
        input_embeddings = embedding(next_ids)
    input_embeddings = input_embeddings.to(hidden.dtype)

    pair_mask = attention_mask[:, 1:]
    position_ids = torch.arange(length - 1, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    outputs = mtp(input_embeddings, hidden, pair_mask, position_ids)

    targets = mtp_targets(labels)
    scored = outputs[:, :-1]
    selected = targets != IGNORE_INDEX
    count = int(selected.sum())
    if count == 0:
        return scored.sum() * 0.0, 0
    states = scored[selected]
    if gradient_scope == "head_only":
        weight = getattr(head, "weight", None)
        if weight is None or getattr(head, "bias", None) is not None:
            raise ModelComponentError("head_only MTP needs a plain linear output head")
        logits = F.linear(states, weight.detach())
    else:
        logits = head(states)
    loss = F.cross_entropy(logits.float(), targets[selected], reduction="mean")
    return loss, count


# ── loading ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class LoadedModel:
    """A training model with every carried component, and where each one's weights came from."""

    model: Any
    mtp: Qwen35MultiTokenPredictor | None
    model_type: str
    declared: dict[str, bool]
    carried: dict[str, bool]
    initialized_from: dict[str, str | None] = field(default_factory=dict)


def weight_files(directory: Path) -> list[Path]:
    """The safetensors files of a checkpoint directory, single-file or sharded."""
    single = directory / "model.safetensors"
    if single.is_file():
        return [single]
    shards = sorted(directory.glob("model-*.safetensors")) or sorted(
        directory.glob("*.safetensors")
    )
    return shards


def read_tensors(files: list[Path], predicate: Callable[[str], bool]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                if predicate(key):
                    tensors[key] = handle.get_tensor(key)
    return tensors


def tensor_names(files: list[Path]) -> list[str]:
    from safetensors import safe_open

    names: list[str] = []
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            names.extend(handle.keys())
    return names


def _base_directory(base_id: str, base_revision: str | None) -> Path:
    local = Path(base_id)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=base_id,
            revision=base_revision,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
        )
    )


def _graft_vision(model: Any, files: list[Path]) -> int:
    """Copy the vision encoder from ``files`` into ``model``, matching names by their suffix."""
    source = read_tensors(files, lambda key: VISION_SEGMENT in key)
    by_suffix = {key.split(VISION_SEGMENT, 1)[1]: tensor for key, tensor in source.items()}
    state = model.state_dict()
    targets = [key for key in state if VISION_SEGMENT in key]
    missing = [key for key in targets if key.split(VISION_SEGMENT, 1)[1] not in by_suffix]
    if not targets or missing:
        raise ModelComponentError(
            f"cannot graft the vision encoder: {len(missing)} of {len(targets)} tensors have no "
            "source in the base checkpoint"
        )
    with torch.no_grad():
        for key in targets:
            state[key].copy_(by_suffix[key.split(VISION_SEGMENT, 1)[1]])
    return len(targets)


def load_training_model(
    transformers: Any,
    *,
    base_id: str,
    base_revision: str | None,
    checkpoint: Path | None,
    dtype: Any,
    settings: ComponentSettings,
    device_map: Any = "auto",
) -> LoadedModel:
    """Load every component the base declares and the settings carry.

    ``checkpoint`` is where training resumes or starts from; ``None`` starts from the base. A
    checkpoint that lacks a carried component (every pre-policy checkpoint lacks ``vision`` and
    ``mtp``) has it grafted from the pinned base revision, and ``initialized_from`` says so.
    """
    import json

    base_config = transformers.AutoConfig.from_pretrained(
        base_id, revision=base_revision, trust_remote_code=False
    )
    model_type = str(base_config.model_type)
    declared = declared_components(base_config.to_dict())
    carried = carried_components(declared, settings, model_type=model_type)
    base_label = f"base:{base_id}@{base_revision}" if base_revision else f"base:{base_id}"

    checkpoint_names: dict[str, list[str]] = {}
    checkpoint_files: list[Path] = []
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if (checkpoint / "adapter_config.json").is_file():
            raise ModelComponentError(
                f"{checkpoint} is a LoRA adapter, not a full checkpoint. Resuming LoRA runs is not "
                "implemented under full-model-components-v1: the adapter needs a base loaded with "
                "its components attached, and guessing that base would change the run"
            )
        checkpoint_files = weight_files(checkpoint)
        if not checkpoint_files:
            raise ModelComponentError(f"no safetensors weights in {checkpoint}")
        check_same_language_model(
            base_config.to_dict(),
            json.loads((checkpoint / "config.json").read_text(encoding="utf-8")),
        )
        checkpoint_names = partition_keys(tensor_names(checkpoint_files))
        if not checkpoint_names["language_model"]:
            raise ModelComponentError(f"{checkpoint} holds no language-model tensors")

    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": False,
        "output_loading_info": True,
    }
    if checkpoint is None:
        load_kwargs["revision"] = base_revision
    source = str(checkpoint) if checkpoint is not None else base_id
    if carried["vision"]:
        # The base config, so a text-only checkpoint still loads into the composite architecture.
        model, info = transformers.AutoModelForImageTextToText.from_pretrained(
            source, config=base_config, **load_kwargs
        )
    else:
        model, info = transformers.AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

    missing = partition_keys(info.get("missing_keys") or [])
    if missing["language_model"]:
        raise ModelComponentError(
            f"language-model tensors missing from {source}: {missing['language_model'][:5]}"
        )

    initialized_from: dict[str, str | None] = {
        "language_model": str(checkpoint) if checkpoint is not None else base_label,
        "vision": None,
        "mtp": None,
    }
    base_files: list[Path] | None = None

    def base_weight_files() -> list[Path]:
        nonlocal base_files
        if base_files is None:
            base_files = weight_files(_base_directory(base_id, base_revision))
        return base_files

    if carried["vision"]:
        if checkpoint is not None and not checkpoint_names["vision"]:
            _graft_vision(model, base_weight_files())
            initialized_from["vision"] = base_label
        elif missing["vision"]:
            raise ModelComponentError(
                f"vision tensors missing from {source}: {missing['vision'][:5]}"
            )
        else:
            initialized_from["vision"] = str(checkpoint) if checkpoint is not None else base_label

    mtp: Qwen35MultiTokenPredictor | None = None
    if carried["mtp"]:
        # The loaded model's config, not `base_config`: loading is what sets the attention
        # implementation, and the MTP layer must attend the way the main stack does.
        mtp = Qwen35MultiTokenPredictor(model.config.get_text_config())

        def is_mtp(key: str) -> bool:
            return key.startswith(MTP_PREFIX)

        tensors: dict[str, torch.Tensor] = {}
        if checkpoint is not None and checkpoint_names["mtp"]:
            tensors = read_tensors(checkpoint_files, is_mtp)
            initialized_from["mtp"] = str(checkpoint)
        else:
            tensors = read_tensors(base_weight_files(), is_mtp)
            initialized_from["mtp"] = base_label
        if not tensors:
            raise ModelComponentError(
                "the base config declares an MTP layer but its weights hold no mtp.* tensors"
            )
        mtp.load_state_dict(
            {key[len(MTP_PREFIX) :]: tensor for key, tensor in tensors.items()}, strict=True
        )
        head = model.get_output_embeddings()
        mtp.to(device=head.weight.device)
        for parameter in mtp.parameters():
            parameter.data = parameter.data.to(dtype)
        if mtp.training != model.training:
            mtp.train(model.training)
        # A top-level attribute, so the state dict -- and every saved checkpoint -- names these
        # tensors `mtp.*`, exactly as the base checkpoint and vLLM's loader do.
        model.mtp = mtp

    return LoadedModel(
        model=model,
        mtp=mtp,
        model_type=model_type,
        declared=declared,
        carried=carried,
        initialized_from=initialized_from,
    )
