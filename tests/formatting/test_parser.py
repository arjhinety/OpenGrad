import pytest

from opengrad.formatting.parser import parse_calls, parse_qwen_native_output

# Verbatim output captured from Qwen/Qwen3.5-2B at the pinned revision for the prompt
# "Look up worker 12." with a `lookup` tool. The pinned chat template mandates this XML
# payload, and renders assistant tool calls the same way, so this is the native protocol.
REAL_QWEN_XML = (
    "<tool_call>\n<function=lookup>\n<parameter=q>\nworker 12\n</parameter>\n"
    "</function>\n</tool_call><|im_end|>\n<|endoftext|>"
)


def test_native_xml_tool_call_from_real_model_output_parses_as_call():
    parsed = parse_qwen_native_output(REAL_QWEN_XML)
    assert parsed.status == "RAW_VALID"
    assert parsed.decision == "CALL"
    assert parsed.errors == []
    assert [(call.name, call.arguments) for call in parsed.calls] == [
        ("lookup", {"q": "worker 12"})
    ]


def test_native_xml_parallel_calls_in_separate_tool_call_blocks():
    raw = (
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
        "<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_qwen_native_output(raw)
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert [(call.name, call.arguments) for call in parsed.calls] == [
        ("a", {"x": "1"}),
        ("b", {"y": "2"}),
    ]


def test_native_xml_preserves_multiline_and_comma_bearing_values():
    raw = (
        "<tool_call>\n<function=book>\n<parameter=city>\nSan Francisco, CA\n</parameter>\n"
        "<parameter=note>\nfirst line\nsecond line\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_qwen_native_output(raw)
    assert parsed.calls[0].arguments == {
        "city": "San Francisco, CA",
        "note": "first line\nsecond line",
    }


def test_native_xml_call_without_parameters_is_valid():
    parsed = parse_qwen_native_output("<tool_call>\n<function=get_time>\n</function>\n</tool_call>")
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert parsed.calls[0].name == "get_time" and parsed.calls[0].arguments == {}


def test_native_xml_without_function_block_is_a_format_error():
    parsed = parse_qwen_native_output("<tool_call>\nnot a function call\n</tool_call>")
    assert parsed.status == "FORMAT_ERROR"
    assert any("neither a JSON object" in error for error in parsed.errors)


def test_native_xml_unclosed_parameter_is_a_format_error():
    parsed = parse_qwen_native_output(
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</function>\n</tool_call>"
    )
    assert parsed.status == "FORMAT_ERROR"
    assert any("unclosed <parameter>" in error for error in parsed.errors)


def test_native_call_survives_special_tokens_and_surrounding_reasoning():
    raw = "I should call the tool.\n" + REAL_QWEN_XML.replace("<|im_end|>\n<|endoftext|>", "")
    parsed = parse_qwen_native_output(raw)
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert "I should call the tool." in parsed.content


def test_json_payload_remains_supported():
    parsed = parse_qwen_native_output(
        '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'
    )
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert parsed.calls[0].arguments == {"q": "x"}


def test_parser_handles_nested_unicode_and_multiple_calls():
    calls = parse_calls(
        [
            '{"name":"a","arguments":{"x":[1,{"u":"\u2603"}]},"id":"1"}',
            {"name": "b", "arguments": {}},
        ]
    )
    assert calls[0].arguments["x"][1]["u"] == "☃" and len(calls) == 2


@pytest.mark.parametrize(
    "value", ['{"name":"a","arguments":', {"arguments": {}}, {"name": "a", "arguments": []}]
)
def test_parser_marks_invalid_without_repair(value):
    with pytest.raises((ValueError, TypeError), match="INVALID"):
        parse_calls(value)
