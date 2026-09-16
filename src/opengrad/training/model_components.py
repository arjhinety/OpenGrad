"""Which parts of a checkpoint a training run carries, and which it trains.

The full-model component policy
-------------------------------
A pinned base checkpoint can hold more than a language model. Qwen3.5-2B holds three components:

* ``language_model`` -- the decoder stack, embeddings and tied output head;
* ``vision`` -- a vision encoder and merger (``model.visual.*``);
* ``mtp`` -- a native multi-token-prediction layer (``mtp.*``) used for speculative decoding.

Until this policy every trainer loaded ``AutoModelForCausalLM``, which silently drops ``vision``
and ``mtp`` at load time. The checkpoints those runs produced are text-only: they cannot take an
image and cannot draft tokens with the native MTP layer, so they are not usable as open weights
in the way the base model is.

From ``full-model-components-v1`` onward, **a trainer carries every component the checkpoint
declares and trains every component the data can reach**:

* a declared component is loaded, kept in the optimizer, and written back into every checkpoint;
* ``mtp`` is trained with its own next-next-token loss (see :mod:`opengrad.training.mtp`);
* ``vision`` receives gradients whenever a batch carries images. A text-only corpus reaches none,
  and the lineage records that rather than claiming the encoder was trained;
* a component can only be left out by declaring ``exclude`` in ``trainer.model_components``. That
  is how a pre-policy, text-only run is reproduced, and the lineage records the exclusion.

A checkpoint that predates the policy lacks ``vision`` and ``mtp``. Starting from one grafts the
missing components from the pinned base revision, and the lineage names where each component's
weights came from.

Runs before and after the policy are **not interchangeable**: with ``gradient_scope: joint`` the
MTP loss reaches the shared decoder, embeddings and output head, so the main model is trained on
a different objective. See ``docs/MODEL_COMPONENT_POLICY.md``.

This module is pure Python so the policy can be tested without torch. The torch side lives in
:mod:`opengrad.training.mtp`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Bump when what a trainer carries or trains changes. Written into every checkpoint's lineage.
MODEL_COMPONENT_POLICY_VERSION = "full-model-components-v1"

COMPONENTS = ("language_model", "vision", "mtp")
OPTIONAL_COMPONENTS = ("vision", "mtp")
COMPONENT_CHOICES = ("include", "exclude")

#: ``joint``: the MTP loss reaches the shared decoder, embeddings and output head, as in native MTP
#: pre-training. ``head_only``: it reaches the ``mtp.*`` parameters and nothing else, so the main
#: model's objective is unchanged.
MTP_GRADIENT_SCOPES = ("joint", "head_only")

#: Per-algorithm defaults, recorded in the resolved settings whenever they are used.
#: SFT trains MTP jointly: the MTP loss is the same kind of objective as SFT's own. DPO keeps the
#: preference objective exactly the preference objective and keeps the MTP layer aligned with the
#: moving policy through the head alone.
MTP_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "sft": {"loss_weight": 0.3, "gradient_scope": "joint"},
    "dpo": {"loss_weight": 1.0, "gradient_scope": "head_only"},
}

#: Model families whose MTP layer this repository implements. A checkpoint that declares MTP under
#: any other ``model_type`` is refused rather than trained text-only.
MTP_IMPLEMENTED_MODEL_TYPES = ("qwen3_5",)


class ModelComponentError(ValueError):
    """A component configuration that cannot be honoured without silently dropping weights."""


@dataclass(frozen=True)
class ComponentSettings:
    """What the run carries, and how MTP is trained."""

    vision: str
    mtp: str
    mtp_loss_weight: float
    mtp_gradient_scope: str
    defaults_used: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": MODEL_COMPONENT_POLICY_VERSION,
            "vision": self.vision,
            "mtp": self.mtp,
            "mtp_loss_weight": self.mtp_loss_weight,
            "mtp_gradient_scope": self.mtp_gradient_scope,
            "defaults_used": list(self.defaults_used),
        }


def resolve_component_settings(trainer: Mapping[str, Any], *, algorithm: str) -> ComponentSettings:
    """Resolve ``trainer.model_components`` and ``trainer.mtp``, recording every default used.

    Absent means the policy default: include every component. ``exclude`` is always explicit.
    """
    if algorithm not in MTP_DEFAULTS:
        raise ModelComponentError(f"no component defaults for algorithm {algorithm!r}")
    raw = trainer.get("model_components") or {}
    if not isinstance(raw, Mapping):
        raise ModelComponentError("trainer.model_components must be a mapping")
    unknown = sorted(set(raw) - set(OPTIONAL_COMPONENTS))
    if unknown:
        raise ModelComponentError(
            f"trainer.model_components names unknown components {unknown}; "
            f"known: {', '.join(OPTIONAL_COMPONENTS)} (the language model is always carried)"
        )
    defaults_used: list[str] = []
    choices: dict[str, str] = {}
    for component in OPTIONAL_COMPONENTS:
        if component not in raw:
            defaults_used.append(f"model_components.{component}")
        value = str(raw.get(component, "include"))
        if value not in COMPONENT_CHOICES:
            raise ModelComponentError(
                f"trainer.model_components.{component} must be one of "
                f"{', '.join(COMPONENT_CHOICES)}; got {value!r}"
            )
        choices[component] = value

    mtp_raw = trainer.get("mtp") or {}
    if not isinstance(mtp_raw, Mapping):
        raise ModelComponentError("trainer.mtp must be a mapping")
    defaults = MTP_DEFAULTS[algorithm]
    for key in ("loss_weight", "gradient_scope"):
        if key not in mtp_raw:
            defaults_used.append(f"mtp.{key}")
    loss_weight = float(mtp_raw.get("loss_weight", defaults["loss_weight"]))
    scope = str(mtp_raw.get("gradient_scope", defaults["gradient_scope"]))
    if loss_weight <= 0:
        raise ModelComponentError(
            "trainer.mtp.loss_weight must be positive; to leave MTP out, declare "
            "model_components.mtp: exclude so the exclusion is recorded"
        )
    if scope not in MTP_GRADIENT_SCOPES:
        raise ModelComponentError(
            f"trainer.mtp.gradient_scope must be one of {', '.join(MTP_GRADIENT_SCOPES)}; "
            f"got {scope!r}"
        )
    if choices["mtp"] == "exclude" and mtp_raw:
        raise ModelComponentError(
            "trainer.mtp is configured but model_components.mtp is exclude; one of them is a mistake"
        )
    return ComponentSettings(
        vision=choices["vision"],
        mtp=choices["mtp"],
        mtp_loss_weight=loss_weight,
        mtp_gradient_scope=scope,
        defaults_used=tuple(defaults_used),
    )


def declared_components(config: Mapping[str, Any]) -> dict[str, bool]:
    """The optional components a checkpoint's ``config.json`` declares.

    ``vision`` is declared by a ``vision_config``; ``mtp`` by ``mtp_num_hidden_layers > 0`` on the
    text config (nested under ``text_config`` for a composite model, top level for a text-only one).
    """
    text = config.get("text_config")
    text_config = text if isinstance(text, Mapping) else config
    return {
        "vision": isinstance(config.get("vision_config"), Mapping),
        "mtp": int(text_config.get("mtp_num_hidden_layers") or 0) > 0,
    }


def carried_components(
    declared: Mapping[str, bool], settings: ComponentSettings, *, model_type: str
) -> dict[str, bool]:
    """Which components the run carries. Refuses an MTP it would otherwise have to drop."""
    carried = {"language_model": True}
    for component in OPTIONAL_COMPONENTS:
        carried[component] = bool(declared.get(component)) and (
            getattr(settings, component) == "include"
        )
    if carried["mtp"] and model_type not in MTP_IMPLEMENTED_MODEL_TYPES:
        raise ModelComponentError(
            f"the checkpoint declares an MTP layer but no MTP implementation exists for "
            f"model_type {model_type!r}; training it text-only would silently drop the layer. "
            "Implement it in opengrad.training.mtp, or declare model_components.mtp: exclude."
        )
    return carried


#: Text-config fields that must agree for a checkpoint's language model to belong to the base.
LANGUAGE_MODEL_IDENTITY_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "vocab_size",
)


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def check_same_language_model(base: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    """Refuse to graft base components onto a checkpoint of a different language model.

    A composite checkpoint nests its text config; a text-only one keeps it at the top level. Only
    the fields that fix the tensor shapes are compared, so a checkpoint whose config merely differs
    in ``use_cache`` or ``dtype`` is still accepted.
    """
    base_text = _text_config(base)
    checkpoint_text = _text_config(checkpoint)
    problems = [
        f"{name}: base={base_text.get(name)!r} checkpoint={checkpoint_text.get(name)!r}"
        for name in LANGUAGE_MODEL_IDENTITY_FIELDS
        if name in base_text
        and name in checkpoint_text
        and base_text.get(name) != checkpoint_text.get(name)
    ]
    if problems:
        raise ModelComponentError(
            "the checkpoint's language model is not the base's: " + "; ".join(problems)
        )


def component_of(parameter_name: str) -> str:
    """The component a parameter belongs to, from its name.

    Robust to wrapper prefixes (PEFT's ``base_model.model.``) because it matches path segments,
    not string starts.
    """
    segments = parameter_name.split(".")
    if "mtp" in segments:
        return "mtp"
    if "visual" in segments:
        return "vision"
    return "language_model"


def partition_keys(keys: Iterable[str]) -> dict[str, list[str]]:
    """Checkpoint tensor names grouped by component, each group sorted."""
    groups: dict[str, list[str]] = {component: [] for component in COMPONENTS}
    for key in keys:
        groups[component_of(key)].append(key)
    return {component: sorted(names) for component, names in groups.items()}


def component_parameter_report(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Total and trainable parameter counts per component."""
    report = {component: {"total": 0, "trainable": 0} for component in COMPONENTS}
    for name, parameter in named_parameters:
        counts = report[component_of(name)]
        counts["total"] += int(parameter.numel())
        if parameter.requires_grad:
            counts["trainable"] += int(parameter.numel())
    return report


def component_lineage(
    *,
    settings: ComponentSettings,
    declared: Mapping[str, bool],
    carried: Mapping[str, bool],
    initialized_from: Mapping[str, str | None],
    image_batches_seen: int,
    mtp_loss_steps: int,
) -> dict[str, Any]:
    """The ``model_components`` block of a checkpoint's lineage.

    ``trained`` answers what the weights actually received, not what the configuration allowed:
    ``vision`` is only true once a batch carrying images has been seen, and ``mtp`` only once an
    MTP loss has been back-propagated.
    """
    return {
        "policy_version": MODEL_COMPONENT_POLICY_VERSION,
        "declared": {component: bool(declared.get(component)) for component in OPTIONAL_COMPONENTS},
        "carried": {component: bool(carried.get(component)) for component in COMPONENTS},
        "initialized_from": {
            component: (initialized_from.get(component) if carried.get(component) else None)
            for component in COMPONENTS
        },
        "trained": {
            "language_model": True,
            "vision": bool(carried.get("vision")) and image_batches_seen > 0,
            "mtp": bool(carried.get("mtp")) and mtp_loss_steps > 0,
        },
        "image_batches_seen": image_batches_seen,
        "mtp": (
            {
                "loss_weight": settings.mtp_loss_weight,
                "gradient_scope": settings.mtp_gradient_scope,
                "loss_steps": mtp_loss_steps,
            }
            if carried.get("mtp")
            else None
        ),
        "settings": settings.to_dict(),
    }
