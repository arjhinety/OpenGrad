"""Model cards carry the base-model licence and per-source terms; the registry agrees with the audit.

Found 2026-09-24 (`reports/ERRATA.md` §23): no model card linked the Apache-2.0 licence of the base model,
the evaluation-record card claimed `apache-2.0` over third-party prompts, and the registry still said
`NOT_ASSESSED` for five sources the redistribution audit had cleared on 2026-09-04.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CARDS = sorted((ROOT / "release/huggingface").glob("qwen35-2b-*/README.md"))
QWEN_LICENSE = (
    "https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE"
)


def _frontmatter(card: Path) -> dict:
    text = card.read_text(encoding="utf-8")
    assert text.startswith("---"), card
    return yaml.safe_load(text.split("---", 2)[1])


def test_every_model_card_states_a_composite_license_and_its_attribution() -> None:
    assert len(CARDS) == 6
    for card in CARDS:
        meta = _frontmatter(card)
        assert meta["license"] == "other", card
        assert meta["license_name"] == "composite-per-source", card
        assert "## License and attribution" in card.read_text(encoding="utf-8"), card


def test_weight_cards_link_the_base_model_license_at_its_pinned_revision() -> None:
    weights = [card for card in CARDS if "evaluation" not in card.parent.name]
    assert len(weights) == 5
    models = yaml.safe_load((ROOT / "registry/models.yaml").read_text(encoding="utf-8"))
    qwen = next(model for model in models["models"] if model["id"] == "qwen3.5-2b")
    assert qwen["license"]["source"] == QWEN_LICENSE
    for card in weights:
        assert _frontmatter(card)["license_link"] == QWEN_LICENSE, card


def test_the_gguf_card_is_what_its_generator_renders() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_gguf_card", ROOT / "scripts/build_gguf_card.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = module.OUT.read_text(encoding="utf-8")
    assert module.render() == committed


def test_released_sources_are_assessed_in_the_registry() -> None:
    datasets = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in datasets["datasets"]}
    # Every source any published release carried (v1 also carried BUTTON and LoopTool). APIGen-MT is
    # CC-BY-NC, used only as a contamination namespace and never released, so it may stay NOT_ASSESSED.
    released = (
        "xlam-function-calling-60k",
        "when2call",
        "toolace",
        "glaive-function-calling-v2",
        "button",
        "looptool-23k",
    )
    _items = list(released)
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for source in _items:
        assert by_id[source]["redistribution"] in {
            "PERMITTED_WITH_ATTRIBUTION",
            "REDISTRIBUTION_WITH_ATTRIBUTION",
        }, source
