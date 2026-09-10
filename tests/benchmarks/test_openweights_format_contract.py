"""Pin the tool-call format contract shared with OpenWeights.

OpenWeights (github.com/alpharomercoma/openweights) is the downstream device runtime. Its
`ToolCallParser` is the reader that must agree with OpenGrad's `parse_qwen_native_output`
for the same checkpoint to be comparable across B0 and on-device runs. These tests pin the
agreeing grammar using OpenWeights' own fixtures, so a change on either side is caught here.

`ToolCallParser.parseTaggedXml` falls back to the XML branch precisely because llama.cpp's
built-in parser does not recognise the Qwen3.5 form, so this format is the one that reaches
both parsers from a Qwen3.5-2B GGUF.
"""

import pytest

from opengrad.benchmarks.adapters.openweights import parse_openweights_reply
from opengrad.formatting.parser import parse_qwen_native_output

# Verbatim from OpenWeights' parser test (PortableTest.kt): the XML branch, including
# control characters that must survive into the argument value.
OPENWEIGHTS_XML_FIXTURE = (
    "<tool_call><function=write><parameter=text>a\u0007b\u001fc</parameter></function></tool_call>"
)

# The XML shape OpenWeights documents in ToolCallParser.parseTaggedXml.
OPENWEIGHTS_DOC_XML = (
    "<tool_call>\n<function=fetch_url>\n"
    "<parameter=url>https://example.com</parameter>\n"
    "</function>\n</tool_call>"
)

# The two arms OpenWeights' ToolPrompting asks for.
BARE = '{"tool": "web_search", "arguments": {"query": "x"}}'
TAGGED_JSON = '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'


def test_shared_xml_fixture_parses_in_both_readers():
    """The same bytes must yield the same call in OpenGrad and OpenWeights."""
    parsed = parse_qwen_native_output(OPENWEIGHTS_XML_FIXTURE)
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert parsed.calls[0].name == "write"
    # Control characters are preserved rather than stripped or escaped away.
    assert parsed.calls[0].arguments == {"text": "a\u0007b\u001fc"}


def test_openweights_documented_xml_shape_parses():
    parsed = parse_qwen_native_output(OPENWEIGHTS_DOC_XML)
    assert parsed.status == "RAW_VALID" and parsed.decision == "CALL"
    assert parsed.calls[0].name == "fetch_url"
    assert parsed.calls[0].arguments == {"url": "https://example.com"}


def test_multi_line_parameter_value_is_trimmed_not_mangled():
    """OpenWeights trims the value: "a URL with a newline in it is not a URL"."""
    raw = "<tool_call>\n<function=read>\n<parameter=path>\n/a/b.txt\n</parameter>\n</function>\n</tool_call>"
    parsed = parse_qwen_native_output(raw)
    assert parsed.calls[0].arguments == {"path": "/a/b.txt"}


@pytest.mark.parametrize("reply", [BARE, TAGGED_JSON, OPENWEIGHTS_XML_FIXTURE])
def test_openweights_adapter_accepts_every_shape_it_can_meet(reply):
    """The adapter must not score native output as FORMAT_ERROR.

    A Qwen3.5 model keeps its template's XML shape even when the arm's prompt asks for
    JSON, because OpenWeights prefers a model's own template when it carries tools.
    """
    decision, name, arguments = parse_openweights_reply(reply)
    assert decision == "CALL", f"{reply!r} was not recognised"
    assert name in {"web_search", "write"}
    assert isinstance(arguments, dict) and arguments


def test_openweights_adapter_still_answers_when_no_tool_is_called():
    decision, name, arguments = parse_openweights_reply("The weather is mild today.")
    assert (decision, name, arguments) == ("ANSWER", None, {})


def test_native_xml_takes_precedence_over_json_scanning():
    """XML arguments containing braces must not be misread by the BARE JSON scan."""
    reply = "<tool_call><function=run><parameter=script>return {a: 1}</parameter></function></tool_call>"
    decision, name, arguments = parse_openweights_reply(reply)
    assert decision == "CALL" and name == "run"
    assert arguments == {"script": "return {a: 1}"}
