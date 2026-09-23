"""The full-model component policy on a real (tiny, random) Qwen3.5 model.

Skipped without torch. Every test builds its model from a Qwen3.5 config shrunk to CPU size, and a
"base checkpoint" laid out exactly like the pinned one: language model, vision encoder and a native
``mtp.*`` layer in one ``model.safetensors``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5Config"):  # pragma: no cover - older transformers
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from safetensors.torch import load_file, save_file

from opengrad.training import mtp as M
from opengrad.training.model_components import (
    ModelComponentError,
    partition_keys,
    resolve_component_settings,
)

#: The `mtp.*` tensor names of Qwen/Qwen3.5-2B at revision 15852e8c…, read from its safetensors
#: header. The layer must load them with strict=True and save them back under the same names.
PINNED_MTP_TENSOR_NAMES = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]

VOCAB = 128


def _tiny_config():
    return transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "vocab_size": VOCAB,
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 4,
            "num_position_embeddings": 16,
        },
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("base")
    config = _tiny_config()
    transformers.Qwen3_5ForConditionalGeneration(config).save_pretrained(directory)
    state = load_file(directory / "model.safetensors")
    layer = M.Qwen35MultiTokenPredictor(config.text_config)
    for name, tensor in layer.state_dict().items():
        state[M.MTP_PREFIX + name] = torch.randn_like(tensor) * 0.02
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
    return directory


def _load(base_dir: Path, *, checkpoint: Path | None = None, trainer: dict | None = None):
    settings = resolve_component_settings(trainer or {}, algorithm="sft")
    return M.load_training_model(
        transformers,
        base_id=str(base_dir),
        base_revision=None,
        checkpoint=checkpoint,
        dtype=torch.float32,
        settings=settings,
        device_map=None,
    )


def _batch(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (2, 12), generator=generator)
    attention = torch.ones_like(ids)
    attention[1, 9:] = 0
    labels = ids.clone()
    labels[:, :5] = M.IGNORE_INDEX
    labels[attention == 0] = M.IGNORE_INDEX
    return ids, attention, labels


def test_mtp_layer_uses_the_pinned_checkpoint_tensor_names():
    layer = M.Qwen35MultiTokenPredictor(_tiny_config().text_config)
    names = sorted(M.MTP_PREFIX + name for name in layer.state_dict())
    assert names == PINNED_MTP_TENSOR_NAMES


def test_base_load_carries_every_component_with_the_base_weights(base_dir):
    loaded = _load(base_dir)
    assert loaded.carried == {"language_model": True, "vision": True, "mtp": True}
    on_disk = load_file(base_dir / "model.safetensors")
    for name, tensor in loaded.model.mtp.state_dict().items():
        assert torch.equal(tensor, on_disk[M.MTP_PREFIX + name]), name


def test_checkpoint_round_trip_keeps_vision_and_mtp(base_dir, tmp_path):
    loaded = _load(base_dir)
    loaded.model.save_pretrained(tmp_path / "ckpt")
    saved = partition_keys(load_file(tmp_path / "ckpt" / "model.safetensors"))
    base = partition_keys(load_file(base_dir / "model.safetensors"))
    assert saved == base


def test_carrying_components_leaves_the_main_logits_unchanged(base_dir):
    """The policy must not move the main model: same weights, same logits, same loss."""
    ids, attention, labels = _batch()
    full = _load(base_dir).model.eval()
    text_only = transformers.AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=torch.float32
    ).eval()
    with torch.no_grad():
        a = full(input_ids=ids, attention_mask=attention, labels=labels)
        b = text_only(input_ids=ids, attention_mask=attention, labels=labels)
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert torch.allclose(a.loss, b.loss, atol=1e-6)


def test_final_hidden_capture_is_the_input_of_the_output_head(base_dir):
    model = _load(base_dir).model.eval()
    ids, attention, _ = _batch()
    capture = M.FinalHiddenCapture(model)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attention).logits
    hidden = capture.take()
    with pytest.raises(RuntimeError):
        capture.take()  # taken once; a second read means a forward pass was skipped
    capture.remove()
    assert torch.allclose(model.get_output_embeddings()(hidden), logits, atol=1e-6)


def _mtp_outputs(model, ids, attention):
    capture = M.FinalHiddenCapture(model)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention)
        hidden = capture.take()
        positions = torch.arange(ids.shape[1] - 1).unsqueeze(0).expand(ids.shape[0], -1)
        embeddings = model.get_input_embeddings()(ids[:, 1:])
        outputs = model.mtp(embeddings, hidden[:, :-1], attention[:, 1:], positions)
    capture.remove()
    return outputs


def test_mtp_pairs_hidden_t_with_token_t_plus_1_to_predict_t_plus_2(base_dir):
    """vLLM's draft pairing: output t sees x[<=t+1] and nothing later."""
    model = _load(base_dir).model.eval()
    ids, attention, _ = _batch()
    attention = torch.ones_like(ids)
    k = 8
    reference = _mtp_outputs(model, ids, attention)

    later = ids.clone()
    later[:, k] = (later[:, k] + 1) % VOCAB  # the target of output k-2
    changed = _mtp_outputs(model, later, attention)
    assert torch.allclose(reference[:, : k - 1], changed[:, : k - 1], atol=1e-6)

    input_token = ids.clone()
    input_token[:, k - 1] = (input_token[:, k - 1] + 1) % VOCAB  # the embedding of output k-2
    moved = _mtp_outputs(model, input_token, attention)
    assert torch.allclose(reference[:, : k - 3], moved[:, : k - 3], atol=1e-6)
    assert not torch.allclose(reference[:, k - 2], moved[:, k - 2], atol=1e-6)


def test_mtp_loss_scores_output_t_against_label_t_plus_2(base_dir):
    model = _load(base_dir).model.eval()
    ids, attention, _ = _batch()
    attention = torch.ones_like(ids)
    k = 7
    labels = torch.full_like(ids, M.IGNORE_INDEX)
    labels[0, k] = ids[0, k]
    capture = M.FinalHiddenCapture(model)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention)
        loss, count = M.mtp_loss(
            model=model,
            mtp=model.mtp,
            final_hidden=capture.take(),
            input_ids=ids,
            attention_mask=attention,
            labels=labels,
            gradient_scope="joint",
        )
    outputs = _mtp_outputs(model, ids, attention)
    expected = torch.nn.functional.cross_entropy(
        model.get_output_embeddings()(outputs[0, k - 2]).float().unsqueeze(0), ids[0, k : k + 1]
    )
    assert count == 1
    assert torch.allclose(loss, expected, atol=1e-5)


def _backward_mtp_only(model, scope: str):
    ids, attention, labels = _batch()
    model.train()
    model.zero_grad(set_to_none=True)
    capture = M.FinalHiddenCapture(model)
    model(input_ids=ids, attention_mask=attention)
    loss, count = M.mtp_loss(
        model=model,
        mtp=model.mtp,
        final_hidden=capture.take(),
        input_ids=ids,
        attention_mask=attention,
        labels=labels,
        gradient_scope=scope,
    )
    assert count > 0
    loss.backward()
    capture.remove()
    return {name: parameter.grad for name, parameter in model.named_parameters()}


def test_head_only_mtp_gradient_reaches_only_the_mtp_layer(base_dir):
    grads = _backward_mtp_only(_load(base_dir).model, "head_only")
    _items = list(grads.items())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for name, grad in _items:
        if name.startswith(M.MTP_PREFIX):
            assert grad is not None and grad.abs().sum() > 0, name
        else:
            assert grad is None, name


def test_joint_mtp_gradient_reaches_the_shared_model(base_dir):
    grads = _backward_mtp_only(_load(base_dir).model, "joint")
    embedding = "model.language_model.embed_tokens.weight"
    assert grads[embedding] is not None and grads[embedding].abs().sum() > 0
    trunk = [name for name in grads if ".language_model.layers." in name]
    assert any(grads[name] is not None and grads[name].abs().sum() > 0 for name in trunk)


def test_text_batches_send_no_gradient_to_the_vision_encoder(base_dir):
    grads = _backward_mtp_only(_load(base_dir).model, "joint")
    vision = [name for name in grads if ".visual." in name]
    assert vision and all(grads[name] is None for name in vision)


def test_text_only_checkpoint_gets_vision_and_mtp_grafted_from_the_base(base_dir, tmp_path):
    text_only = transformers.AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=torch.float32
    )
    with torch.no_grad():
        text_only.get_input_embeddings().weight.add_(1.0)  # distinguishable from the base
    checkpoint = tmp_path / "text-only"
    text_only.save_pretrained(checkpoint)
    assert not partition_keys(load_file(checkpoint / "model.safetensors"))["mtp"]

    loaded = _load(base_dir, checkpoint=checkpoint)
    base_label = f"base:{base_dir}"
    assert loaded.initialized_from == {
        "language_model": str(checkpoint),
        "vision": base_label,
        "mtp": base_label,
    }
    assert torch.equal(
        loaded.model.get_input_embeddings().weight, text_only.get_input_embeddings().weight
    )
    base = load_file(base_dir / "model.safetensors")
    state = loaded.model.state_dict()
    for key in (k for k in base if ".visual." in k or k.startswith(M.MTP_PREFIX)):
        assert torch.equal(state[key], base[key]), key


def test_a_policy_checkpoint_resumes_every_component_from_itself(base_dir, tmp_path):
    first = _load(base_dir).model
    with torch.no_grad():
        first.mtp.fc.weight.add_(0.5)
    first.save_pretrained(tmp_path / "ckpt")
    loaded = _load(base_dir, checkpoint=tmp_path / "ckpt")
    assert set(loaded.initialized_from.values()) == {str(tmp_path / "ckpt")}
    assert torch.equal(loaded.model.mtp.fc.weight, first.mtp.fc.weight)


def test_excluding_vision_still_saves_mtp_under_its_own_names(base_dir, tmp_path):
    loaded = _load(base_dir, trainer={"model_components": {"vision": "exclude"}})
    assert loaded.carried["vision"] is False and loaded.initialized_from["vision"] is None
    loaded.model.save_pretrained(tmp_path / "ckpt")
    saved = partition_keys(load_file(tmp_path / "ckpt" / "model.safetensors"))
    assert saved["mtp"] == PINNED_MTP_TENSOR_NAMES
    assert not saved["vision"]


def test_excluding_everything_reproduces_the_pre_policy_text_only_model(base_dir):
    loaded = _load(base_dir, trainer={"model_components": {"vision": "exclude", "mtp": "exclude"}})
    assert loaded.mtp is None
    assert not hasattr(loaded.model, "mtp")
    assert all(component_is_text(name) for name, _ in loaded.model.named_parameters())


def component_is_text(name: str) -> bool:
    return ".visual." not in name and not name.startswith(M.MTP_PREFIX)


def test_a_checkpoint_of_another_language_model_is_refused(base_dir, tmp_path):
    config = _tiny_config()
    config.text_config.hidden_size = 32
    config.text_config.head_dim = 8
    config.text_config.linear_key_head_dim = 8
    config.text_config.linear_value_head_dim = 8
    other = transformers.Qwen3_5ForCausalLM(config.text_config)
    other.save_pretrained(tmp_path / "other")
    with pytest.raises(ModelComponentError, match="hidden_size"):
        _load(base_dir, checkpoint=tmp_path / "other")


def test_mtp_loss_refuses_batches_with_images(base_dir):
    model = _load(base_dir).model
    ids, attention, labels = _batch()
    with pytest.raises(ModelComponentError, match="multimodal position ids"):
        M.mtp_loss(
            model=model,
            mtp=model.mtp,
            final_hidden=torch.zeros(2, 12, 64),
            input_ids=ids,
            attention_mask=attention,
            labels=labels,
            gradient_scope="joint",
            has_images=True,
        )


def test_gradient_checkpointing_trains_with_the_mtp_layer_attached(base_dir):
    model = _load(base_dir).model
    model.gradient_checkpointing_enable()
    ids, attention, labels = _batch()
    model.train()
    capture = M.FinalHiddenCapture(model)
    outputs = model(input_ids=ids, attention_mask=attention, labels=labels)
    loss, _ = M.mtp_loss(
        model=model,
        mtp=model.mtp,
        final_hidden=capture.take(),
        input_ids=ids,
        attention_mask=attention,
        labels=labels,
        gradient_scope="joint",
    )
    (outputs.loss + 0.3 * loss).backward()
    assert model.mtp.fc.weight.grad is not None


def test_lora_target_modules_reach_the_mtp_layer(base_dir):
    peft = pytest.importorskip("peft")
    model = _load(base_dir).model
    wrapped = peft.get_peft_model(
        model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    )
    adapted = [name for name, _ in wrapped.named_parameters() if "lora_" in name]
    assert any(".mtp.layers.0.self_attn.q_proj." in name for name in adapted)
    assert not any(".visual." in name for name in adapted)


def test_sft_checkpoint_writes_every_component_and_its_lineage(base_dir, tmp_path):
    import json

    from opengrad.training.model_components import component_lineage
    from opengrad.training.sft import _save_checkpoint
    from opengrad.training.sft_runner import resolve_settings

    experiment = {
        "experiment_id": "unit-test",
        "model": {"model_id": str(base_dir), "model_revision": "local"},
        "reproducibility": {"seed": 1, "precision": "float32"},
    }
    settings = resolve_settings(experiment, {"tuning_method": "full", "micro_batch_tokens": 64})
    loaded = _load(base_dir)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in loaded.model.parameters() if parameter.requires_grad]
    )
    world = {
        "optimizer_step": 3,
        "examples_seen": 6,
        "supervised_tokens_seen": 60,
        "epoch": 0,
        "image_batches_seen": 0,
        "mtp_loss_steps": 3,
    }
    block = component_lineage(
        settings=settings.model_components,
        declared=loaded.declared,
        carried=loaded.carried,
        initialized_from=loaded.initialized_from,
        image_batches_seen=0,
        mtp_loss_steps=3,
    )

    class _Tokenizer:
        def save_pretrained(self, path):
            pass

    (tmp_path / "checkpoints").mkdir()
    path = _save_checkpoint(
        loaded.model,
        optimizer,
        _Tokenizer(),
        tmp_path,
        world,
        settings,
        experiment,
        ["corpus"],
        {"corpus": "b" * 64},
        "a" * 40,
        model_components=block,
    )
    saved = partition_keys(load_file(path / "model.safetensors"))
    assert saved["mtp"] == PINNED_MTP_TENSOR_NAMES
    assert saved["vision"]
    lineage = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert lineage["model_components"]["trained"] == {
        "language_model": True,
        "vision": False,
        "mtp": True,
    }
    state = torch.load(path / "training_state.pt", weights_only=False)
    assert (state["mtp_loss_steps"], state["image_batches_seen"]) == (3, 0)


def test_a_lora_adapter_directory_is_refused_as_a_checkpoint(base_dir, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ModelComponentError, match="LoRA adapter"):
        _load(base_dir, checkpoint=adapter)
