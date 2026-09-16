"""The full-model component policy, tested without torch.

Resolution defaults and refusals, what a checkpoint declares, parameter attribution, and the
lineage block every checkpoint carries. The torch side is in ``test_mtp_components.py``.
"""

from __future__ import annotations

import json

import pytest

from opengrad.training.dpo_runner import PreferenceDataError, resolve_dpo_settings
from opengrad.training.model_components import (
    MODEL_COMPONENT_POLICY_VERSION,
    ComponentSettings,
    ModelComponentError,
    carried_components,
    check_same_language_model,
    component_lineage,
    component_of,
    component_parameter_report,
    declared_components,
    partition_keys,
    resolve_component_settings,
)
from opengrad.training.sft_runner import TrainingConfigError, checkpoint_lineage, resolve_settings

REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

#: The shape of the pinned Qwen/Qwen3.5-2B config.json, reduced to what the policy reads.
PINNED_BASE_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    },
    "vision_config": {"depth": 24, "hidden_size": 1024, "out_hidden_size": 2048},
}

#: A pre-policy text-only SFT checkpoint's config: the text config at the top level.
TEXT_ONLY_CHECKPOINT_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    **PINNED_BASE_CONFIG["text_config"],
}


def _experiment() -> dict:
    return {
        "experiment_id": "unit-test",
        "model": {"model_id": "Qwen/Qwen3.5-2B", "model_revision": REVISION},
        "reproducibility": {"seed": 7, "precision": "bfloat16"},
        "checkpointing": {"save_steps": 10, "max_checkpoints": 2},
    }


# ── resolution ────────────────────────────────────────────────────────────────────────────────────


def test_absent_configuration_carries_everything_and_records_the_defaults():
    settings = resolve_component_settings({}, algorithm="sft")
    assert (settings.vision, settings.mtp) == ("include", "include")
    assert (settings.mtp_loss_weight, settings.mtp_gradient_scope) == (0.3, "joint")
    assert set(settings.defaults_used) == {
        "model_components.vision",
        "model_components.mtp",
        "mtp.loss_weight",
        "mtp.gradient_scope",
    }
    assert settings.to_dict()["policy_version"] == MODEL_COMPONENT_POLICY_VERSION


def test_dpo_defaults_keep_the_mtp_gradient_out_of_the_preference_objective():
    settings = resolve_component_settings({}, algorithm="dpo")
    assert settings.mtp_gradient_scope == "head_only"
    assert settings.mtp_loss_weight == 1.0


def test_explicit_values_are_not_reported_as_defaults():
    settings = resolve_component_settings(
        {
            "model_components": {"vision": "include", "mtp": "include"},
            "mtp": {"loss_weight": 0.1, "gradient_scope": "head_only"},
        },
        algorithm="sft",
    )
    assert settings.defaults_used == ()
    assert (settings.mtp_loss_weight, settings.mtp_gradient_scope) == (0.1, "head_only")


@pytest.mark.parametrize(
    ("trainer", "message"),
    [
        ({"model_components": {"audio": "include"}}, "unknown components"),
        ({"model_components": {"language_model": "exclude"}}, "unknown components"),
        ({"model_components": {"vision": "yes"}}, "must be one of"),
        ({"model_components": ["vision"]}, "must be a mapping"),
        ({"mtp": {"loss_weight": 0}}, "must be positive"),
        ({"mtp": {"gradient_scope": "trunk"}}, "gradient_scope"),
        (
            {"model_components": {"mtp": "exclude"}, "mtp": {"loss_weight": 0.3}},
            "one of them is a mistake",
        ),
    ],
)
def test_ambiguous_component_configuration_is_refused(trainer, message):
    with pytest.raises(ModelComponentError, match=message):
        resolve_component_settings(trainer, algorithm="sft")


def test_sft_settings_record_the_component_contract():
    settings = resolve_settings(
        _experiment(), {"tuning_method": "full", "micro_batch_tokens": 4096}
    )
    recorded = settings.to_dict()["model_components"]
    assert recorded["policy_version"] == MODEL_COMPONENT_POLICY_VERSION
    assert (recorded["vision"], recorded["mtp"]) == ("include", "include")


def test_sft_component_errors_surface_as_configuration_errors():
    with pytest.raises(TrainingConfigError, match="unknown components"):
        resolve_settings(
            _experiment(),
            {
                "tuning_method": "full",
                "micro_batch_tokens": 4096,
                "model_components": {"audio": "include"},
            },
        )


def test_dpo_settings_record_the_component_contract_and_refuse_bad_ones():
    settings = resolve_dpo_settings(_experiment(), {"reference": "initial_policy"})
    assert settings.to_dict()["model_components"]["mtp_gradient_scope"] == "head_only"
    with pytest.raises(PreferenceDataError, match="gradient_scope"):
        resolve_dpo_settings(
            _experiment(), {"reference": "initial_policy", "mtp": {"gradient_scope": "x"}}
        )


# ── what a checkpoint declares and what a run carries ───────────────────────────────────────────


def test_the_pinned_base_declares_vision_and_mtp():
    assert declared_components(PINNED_BASE_CONFIG) == {"vision": True, "mtp": True}


def test_a_text_only_config_declares_mtp_but_not_vision():
    assert declared_components(TEXT_ONLY_CHECKPOINT_CONFIG) == {"vision": False, "mtp": True}
    assert declared_components({"model_type": "llama"}) == {"vision": False, "mtp": False}


def test_exclusion_is_honoured_and_absence_is_not_invented():
    declared = {"vision": True, "mtp": False}
    settings = resolve_component_settings(
        {"model_components": {"vision": "exclude"}}, algorithm="sft"
    )
    assert carried_components(declared, settings, model_type="qwen3_5") == {
        "language_model": True,
        "vision": False,
        "mtp": False,
    }


def test_an_mtp_without_an_implementation_is_refused_not_dropped():
    settings = resolve_component_settings({}, algorithm="sft")
    with pytest.raises(ModelComponentError, match="no MTP implementation"):
        carried_components({"vision": False, "mtp": True}, settings, model_type="deepseek_v3")
    excluded = resolve_component_settings({"model_components": {"mtp": "exclude"}}, algorithm="sft")
    assert carried_components({"mtp": True}, excluded, model_type="deepseek_v3")["mtp"] is False


def test_a_text_only_checkpoint_of_the_base_language_model_is_accepted():
    check_same_language_model(PINNED_BASE_CONFIG, TEXT_ONLY_CHECKPOINT_CONFIG)
    check_same_language_model(PINNED_BASE_CONFIG, {**PINNED_BASE_CONFIG, "use_cache": False})


def test_a_checkpoint_of_a_different_language_model_is_refused():
    other = {**TEXT_ONLY_CHECKPOINT_CONFIG, "num_hidden_layers": 28}
    with pytest.raises(ModelComponentError, match="num_hidden_layers"):
        check_same_language_model(PINNED_BASE_CONFIG, other)


# ── attribution ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("model.language_model.layers.0.mlp.up_proj.weight", "language_model"),
        ("lm_head.weight", "language_model"),
        ("model.visual.blocks.0.attn.qkv.weight", "vision"),
        ("mtp.layers.0.self_attn.q_proj.weight", "mtp"),
        ("base_model.model.mtp.layers.0.self_attn.q_proj.lora_A.default.weight", "mtp"),
        ("base_model.model.model.visual.merger.linear_fc1.weight", "vision"),
        # A segment match, not a substring one: a name merely containing "mtp" is not the layer.
        ("model.language_model.layers.0.mtp_gate.weight", "language_model"),
    ],
)
def test_parameters_are_attributed_by_path_segment(name, component):
    assert component_of(name) == component


def test_partition_matches_the_pinned_checkpoint_counts():
    keys = (
        [f"model.language_model.t{i}" for i in range(320)]
        + [f"model.visual.v{i}" for i in range(297)]
        + [f"mtp.m{i}" for i in range(15)]
    )
    groups = partition_keys(keys)
    assert {name: len(values) for name, values in groups.items()} == {
        "language_model": 320,
        "vision": 297,
        "mtp": 15,
    }


def test_component_parameter_report_counts_total_and_trainable():
    class _Param:
        def __init__(self, n: int, requires: bool) -> None:
            self._n = n
            self.requires_grad = requires

        def numel(self) -> int:
            return self._n

    report = component_parameter_report(
        [
            ("model.language_model.embed_tokens.weight", _Param(100, True)),
            ("model.visual.patch_embed.proj.weight", _Param(40, True)),
            ("mtp.fc.weight", _Param(10, False)),
        ]
    )
    assert report == {
        "language_model": {"total": 100, "trainable": 100},
        "vision": {"total": 40, "trainable": 40},
        "mtp": {"total": 10, "trainable": 0},
    }


# ── lineage ─────────────────────────────────────────────────────────────────────────────────────


def _lineage(**overrides) -> dict:
    arguments = {
        "settings": resolve_component_settings({}, algorithm="sft"),
        "declared": {"vision": True, "mtp": True},
        "carried": {"language_model": True, "vision": True, "mtp": True},
        "initialized_from": {
            "language_model": "runs/m0/checkpoints/checkpoint-100",
            "vision": f"base:Qwen/Qwen3.5-2B@{REVISION}",
            "mtp": f"base:Qwen/Qwen3.5-2B@{REVISION}",
        },
        "image_batches_seen": 0,
        "mtp_loss_steps": 25,
    }
    arguments.update(overrides)
    return component_lineage(**arguments)


def test_lineage_never_claims_the_vision_encoder_trained_on_text():
    block = _lineage()
    assert block["carried"]["vision"] is True
    assert block["trained"] == {"language_model": True, "vision": False, "mtp": True}
    assert _lineage(image_batches_seen=3)["trained"]["vision"] is True


def test_lineage_of_an_excluded_component_has_no_source_and_no_mtp_block():
    settings = ComponentSettings(
        vision="exclude", mtp="exclude", mtp_loss_weight=0.3, mtp_gradient_scope="joint"
    )
    block = _lineage(
        settings=settings,
        carried={"language_model": True, "vision": False, "mtp": False},
        mtp_loss_steps=0,
    )
    assert block["initialized_from"]["vision"] is None
    assert block["initialized_from"]["mtp"] is None
    assert block["mtp"] is None
    assert block["trained"]["mtp"] is False


def test_checkpoint_lineage_carries_the_component_block_when_given():
    settings = resolve_settings(
        _experiment(), {"tuning_method": "full", "micro_batch_tokens": 4096}
    )
    common = {
        "step": 1,
        "tokens_seen": 1,
        "examples_seen": 1,
        "git_commit": "a" * 40,
        "dataset_manifest_ids": ["x"],
        "dataset_hashes": {"x": "b" * 64},
        "parent_checkpoint": None,
    }
    without = checkpoint_lineage(_experiment(), settings, **common)
    assert "model_components" not in without  # a pre-policy lineage stays readable as before
    block = _lineage()
    with_block = checkpoint_lineage(_experiment(), settings, model_components=block, **common)
    assert with_block["model_components"] == block
    assert json.dumps(with_block)
