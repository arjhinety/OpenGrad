import json
from pathlib import Path

import pytest

from opengrad.data.canonical import ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer

ROOT = Path(__file__).parents[1] / "fixtures" / "rendered"


def test_qwen35_pinned_answer_snapshot():
    pytest.importorskip("transformers")  # the renderer applies the pinned Qwen tokenizer
    example = ToolConversation(
        "answer",
        "fixture",
        [],
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        {"split": "fixture"},
    )
    rendered = Qwen35_2BRenderer(enable_thinking=False).render_sft(example)
    metadata = json.loads((ROOT / "qwen35_2b_metadata.json").read_text())
    assert rendered.text == (ROOT / "qwen35_2b_answer.txt").read_text()
    assert rendered.model_revision == metadata["model_revision"]
    assert rendered.chat_template_hash == metadata["chat_template_hash"]
    assert rendered.renderer == metadata["renderer"]


def test_qwen35_expanded_snapshot_metadata_and_semantics():
    for fixture_id in (
        "single_call",
        "multiple_calls",
        "observation_answer",
        "observation_second_call",
        "multi_turn",
        "clarify",
        "unsupported",
    ):
        metadata = json.loads((ROOT / f"qwen35_2b_{fixture_id}.json").read_text())
        text = (ROOT / f"qwen35_2b_{fixture_id}.txt").read_text()
        assert metadata["model"] == "Qwen/Qwen3.5-2B"
        assert metadata["model_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
        assert (
            metadata["template_hash"]
            == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
        )
        assert metadata["renderer"] == "qwen3_5_2b_v1"
        assert text


def test_qwen35_snapshot_semantic_fixture_contracts():
    assert "lookup" in (ROOT / "qwen35_2b_single_call.txt").read_text()
    assert (ROOT / "qwen35_2b_multiple_calls.txt").read_text().count("lookup") >= 2
    assert "value x" in (ROOT / "qwen35_2b_observation_answer.txt").read_text()
