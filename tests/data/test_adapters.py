import pytest

from opengrad.data.adapters import (
    adapt_button,
    adapt_glaive,
    adapt_looptool,
    adapt_toolace,
    adapt_when2call,
    adapt_xlam,
)


@pytest.mark.parametrize(
    "fn", [adapt_xlam, adapt_when2call, adapt_toolace, adapt_button, adapt_looptool, adapt_glaive]
)
def test_source_adapters_accept_documented_common_fixture(fn):
    result = fn(
        {
            "id": "fixture",
            "tools": [{"name": "lookup"}],
            "messages": [
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        }
    )
    assert result.metadata["source_fields"] == ["id", "messages", "tools"]


def test_adapter_rejects_missing_messages():
    with pytest.raises(TypeError):
        adapt_xlam({"id": "bad", "tools": []})


def test_adapter_preserves_unexpected_fields_in_provenance():
    result = adapt_glaive(
        {"id": "fixture", "messages": [{"role": "user", "content": "hi"}], "unexpected": True}
    )
    assert "unexpected" in result.metadata["source_fields"]


def test_toolace_safe_parser_handles_nested_and_punctuated_calls():
    result = adapt_toolace(
        {
            "id": "nested",
            "system": '[{"name":"Market, API"},{"name":"other.tool"}]',
            "conversations": [
                {"from": "user", "value": "lookup"},
                {
                    "from": "assistant",
                    "value": '[Market, API(from="US", payload={"xs":[1, true, null]})]',
                },
            ],
        }
    )
    call = next(message for message in result.messages if message.get("tool_calls"))["tool_calls"][
        0
    ]
    assert call["name"] == "Market, API"
    assert call["arguments"] == {"from": "US", "payload": {"xs": [1, True, None]}}


# --------------------------------------------------------------------------------------
# Glaive unterminated function-call blocks
# --------------------------------------------------------------------------------------


def _glaive_raw(chat: str) -> dict:
    return {
        "id": "fixture",
        "system": "SYSTEM: You are helpful. ",
        "chat": chat,
        "tools": [
            {
                "name": "get_news",
                "description": "news",
                "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
            }
        ],
    }


# The exact shape the released corpus contains: an opening marker, a complete JSON value, no
# closing tag, and an <|endoftext|> artifact.
UNTERMINATED = (
    "USER: what is the news? "
    'ASSISTANT: : <functioncall> {"name": "get_news", "arguments": "{\\"country\\": \\"US\\"}"} '
    "<|endoftext|> "
    'FUNCTION RESPONSE: {"headlines": []} '
    "ASSISTANT: Here you go."
)


def test_glaive_v1_cannot_parse_the_shape_the_source_actually_uses():
    """The regression this captures: zero calls parsed, marker left in the content."""
    from opengrad.data.adapters import adapt_glaive

    result = adapt_glaive(_glaive_raw(UNTERMINATED))
    assistant = [m for m in result.messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"] == []
    assert "<functioncall>" in str(assistant[0]["content"])


def test_glaive_v2_parses_an_unterminated_block():
    from opengrad.data.adapters import adapt_glaive_v2

    result = adapt_glaive_v2(_glaive_raw(UNTERMINATED))
    assistant = [m for m in result.messages if m["role"] == "assistant"]
    calls = assistant[0]["tool_calls"]
    assert [call["name"] for call in calls] == ["get_news"]
    assert calls[0]["arguments"] == {"country": "US"}
    # The raw marker must not survive into the text the model is trained to produce.
    assert "<functioncall>" not in str(assistant[0]["content"] or "")
    assert "<|endoftext|>" not in str(assistant[0]["content"] or "")


def test_glaive_v2_still_parses_a_terminated_block():
    from opengrad.data.adapters import adapt_glaive_v2

    terminated = UNTERMINATED.replace("<|endoftext|> ", "</functioncall> ")
    assistant = [
        m for m in adapt_glaive_v2(_glaive_raw(terminated)).messages if m["role"] == "assistant"
    ]
    assert [call["name"] for call in assistant[0]["tool_calls"]] == ["get_news"]
    assert "</functioncall>" not in str(assistant[0]["content"] or "")


def test_glaive_v2_parses_two_calls_in_one_turn():
    from opengrad.data.adapters import adapt_glaive_v2

    double = (
        "USER: compare. "
        'ASSISTANT: <functioncall> {"name": "get_news", "arguments": {"country": "US"}} '
        ' <functioncall> {"name": "get_news", "arguments": {"country": "FR"}} '
        "FUNCTION RESPONSE: {} "
        "ASSISTANT: done."
    )
    assistant = [
        m for m in adapt_glaive_v2(_glaive_raw(double)).messages if m["role"] == "assistant"
    ]
    assert [call["name"] for call in assistant[0]["tool_calls"]] == ["get_news", "get_news"]
    assert [call["arguments"] for call in assistant[0]["tool_calls"]] == [
        {"country": "US"},
        {"country": "FR"},
    ]


def test_glaive_v2_consumes_exactly_one_json_value():
    """Text after the JSON value must survive: the block ends at the JSON, not at a delimiter."""
    from opengrad.data.adapters import adapt_glaive_v2

    trailing = (
        "USER: q "
        'ASSISTANT: <functioncall> {"name": "get_news", "arguments": {}} and then some prose. '
        "FUNCTION RESPONSE: {} "
        "ASSISTANT: done."
    )
    assistant = [
        m for m in adapt_glaive_v2(_glaive_raw(trailing)).messages if m["role"] == "assistant"
    ]
    assert len(assistant[0]["tool_calls"]) == 1
    assert "and then some prose." in str(assistant[0]["content"])


def test_glaive_v2_leaves_the_stray_delimiter_colon_out_of_the_target():
    from opengrad.data.adapters import adapt_glaive_v2

    assistant = [
        m for m in adapt_glaive_v2(_glaive_raw(UNTERMINATED)).messages if m["role"] == "assistant"
    ]
    assert not str(assistant[0]["content"] or "").startswith(":")


def test_glaive_v2_is_registered_and_v1_is_still_the_default():
    from opengrad.data.adapters import ADAPTERS

    assert "glaive_v2" in ADAPTERS
    # The pinned release must remain reproducible from the code that produced it.
    assert ADAPTERS["glaive"].__name__ == "adapt_glaive"


def test_glaive_v2_rejects_a_malformed_block():
    from opengrad.data.adapters import adapt_glaive_v2

    broken = "USER: q ASSISTANT: <functioncall> {not json} FUNCTION RESPONSE: {} ASSISTANT: done."
    with pytest.raises(ValueError):
        adapt_glaive_v2(_glaive_raw(broken))


def test_glaive_v2_rejects_a_call_without_a_name():
    from opengrad.data.adapters import adapt_glaive_v2

    nameless = (
        'USER: q ASSISTANT: <functioncall> {"arguments": {}} FUNCTION RESPONSE: {} ASSISTANT: x.'
    )
    with pytest.raises(ValueError, match="malformed Glaive function call"):
        adapt_glaive_v2(_glaive_raw(nameless))


def test_glaive_v2_reads_the_single_quoted_arguments_shape():
    """`arguments` is a single-quoted JSON string, which is the shape the source actually uses."""
    from opengrad.data.adapters import adapt_glaive_v2

    assistant = [
        m for m in adapt_glaive_v2(_glaive_raw(UNTERMINATED)).messages if m["role"] == "assistant"
    ]
    assert assistant[0]["tool_calls"][0]["arguments"] == {"country": "US"}


def test_glaive_v2_reads_json_booleans_inside_the_single_quoted_value():
    from opengrad.data.adapters import adapt_glaive_v2

    call = (
        "USER: make one "
        'ASSISTANT: <functioncall> {"name": "get_news", "arguments": \'{"length": 12, '
        '"include_symbols": true, "nested": {"a": 1}}\'} <|endoftext|> '
        "FUNCTION RESPONSE: {} "
        "ASSISTANT: done."
    )
    assistant = [m for m in adapt_glaive_v2(_glaive_raw(call)).messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["arguments"] == {
        "length": 12,
        "include_symbols": True,
        "nested": {"a": 1},
    }


def test_glaive_balanced_body_is_not_ended_by_a_brace_inside_a_string():
    from opengrad.data.adapters import parse_glaive_body

    body = '{"name": "f", "arguments": \'{"pattern": "}{"}\'}'
    assert parse_glaive_body(body)["name"] == "f"


def test_glaive_v2_still_refuses_a_body_it_cannot_read():
    from opengrad.data.adapters import adapt_glaive_v2

    with pytest.raises(ValueError):
        adapt_glaive_v2(
            _glaive_raw(
                'USER: q ASSISTANT: <functioncall> {"name": "get_news", '
                "\"arguments\": 'not json at all'} FUNCTION RESPONSE: {} ASSISTANT: x."
            )
        )
