import pytest

from opengrad.data.adapters import adapt
from opengrad.data.canonical import CanonicalSFTExample, ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.data.schema import SchemaValidationError, effective_schema, validate_arguments
from opengrad.data.semantic import validate_training_trajectory


def tool_schema() -> dict[str, object]:
    return {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "limit": {"type": "integer", "enum": [1, 2]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string", "nullable": True},
            },
            "required": ["q"],
            "additionalProperties": False,
        },
    }


def conversation(messages: list[dict[str, object]]) -> ToolConversation:
    return ToolConversation("id", "fixture", [tool_schema()], messages, {"split": "train"})


def test_adapter_embeds_top_level_required_without_schema_loss():
    result = adapt(
        {
            "id": "schema-loss",
            "tools": [
                {
                    "name": "lookup",
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                }
            ],
            "messages": [{"role": "user", "content": "find"}],
        },
        "fixture",
    )
    assert result.tools[0]["parameters"]["required"] == ["q"]
    assert result.tools[0]["parameters"]["additionalProperties"] is False


def test_conflicting_nested_and_top_level_schema_rejects_stably():
    with pytest.raises(SchemaValidationError, match="SCH_CONFLICT"):
        effective_schema(
            {
                "name": "lookup",
                "parameters": {"type": "object", "required": ["q"]},
                "required": ["other"],
            }
        )


def test_argument_subset_covers_nested_arrays_enum_nullable_and_extra_fields():
    schema = effective_schema(tool_schema())
    validate_arguments(schema, {"q": "x", "limit": 2, "tags": ["a"], "note": None})
    for arguments, code in [
        ({}, "ARG_REQUIRED"),
        ({"q": 3}, "ARG_TYPE"),
        ({"q": "x", "limit": 3}, "ARG_ENUM"),
        ({"q": "x", "tags": [1]}, "ARG_TYPE"),
        ({"q": "x", "extra": True}, "ARG_ADDITIONAL_PROPERTY"),
    ]:
        with pytest.raises(ValueError, match=code):
            validate_arguments(schema, arguments)


def test_training_boundary_rejects_semantic_trajectory_mutations():
    valid = conversation(
        [
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    valid.validate_training_semantics()
    mutants = [
        (
            "missing id",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
                }
            ],
            "SEM_CALL_ID_REQUIRED",
        ),
        (
            "undeclared",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "missing", "arguments": {}}],
                }
            ],
            "SEM_UNDECLARED_TOOL",
        ),
        (
            "orphan result",
            [{"role": "tool", "tool_call_id": "c1", "content": "x"}],
            "SEM_ORPHAN_RESULT",
        ),
        (
            "unresolved",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
                }
            ],
            "SEM_UNRESOLVED_CALL",
        ),
        (
            "bad argument",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {}}],
                }
            ],
            "SEM_ARGUMENT_INVALID",
        ),
    ]
    for _, messages, reason in mutants:
        with pytest.raises(ValueError, match=reason):
            conversation(messages).validate_training_semantics()


def test_canonical_sft_uses_strict_boundary():
    with pytest.raises(ValueError, match="SEM_UNRESOLVED_CALL"):
        CanonicalSFTExample(
            conversation(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
                    }
                ]
            )
        ).validate()


def test_qwen_renderer_receives_effective_schema(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeTokenizer:
        chat_template = "fixture-template"

        def apply_chat_template(self, messages, **kwargs):
            captured["tools"] = kwargs.get("tools")
            return "rendered"

    renderer = Qwen35_2BRenderer(enable_thinking=False)
    monkeypatch.setattr(renderer, "_load", lambda: FakeTokenizer())
    example = conversation(
        [{"role": "user", "content": "find"}, {"role": "assistant", "content": "done"}]
    )
    renderer.render_sft(example)
    assert captured["tools"] == [tool_schema()]


def test_shared_state_machine_rejects_reversed_results_and_interleaved_calls():
    messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a1", "name": "lookup", "arguments": {"q": "a"}},
                {"id": "b1", "name": "lookup", "arguments": {"q": "b"}},
            ],
        },
        {"role": "tool", "tool_call_id": "b1", "content": "b"},
        {"role": "tool", "tool_call_id": "a1", "content": "a"},
    ]
    example = conversation(messages)
    assert any(
        issue.code == "INVALID_MESSAGE_SEQUENCE" for issue in validate_training_trajectory(example)
    )
    with pytest.raises(ValueError, match="SEM_RESULT_ORDER"):
        example.validate_training_semantics()

    interleaved = conversation(
        [
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a1", "name": "lookup", "arguments": {"q": "a"}}],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "b1", "name": "lookup", "arguments": {"q": "b"}}],
            },
        ]
    )
    assert any(
        issue.code == "INVALID_MESSAGE_SEQUENCE"
        for issue in validate_training_trajectory(interleaved)
    )
    with pytest.raises(ValueError, match="SEM_RESULT_ORDER"):
        interleaved.validate_training_semantics()


def test_no_tool_answer_is_a_valid_training_target():
    example = conversation(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi!"}]
    )
    assert validate_training_trajectory(example) == []
    example.validate_training_semantics()


def test_sft_renderer_rejects_unresolved_tool_call(monkeypatch: pytest.MonkeyPatch):
    class FakeTokenizer:
        chat_template = "fixture-template"

        def apply_chat_template(self, messages, **kwargs):
            return "must not render"

    renderer = Qwen35_2BRenderer(enable_thinking=False)
    monkeypatch.setattr(renderer, "_load", lambda: FakeTokenizer())
    example = conversation(
        [
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
            },
        ]
    )
    with pytest.raises(ValueError, match="SEM_UNRESOLVED_CALL"):
        renderer.render_sft(example)
