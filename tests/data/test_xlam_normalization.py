"""Tests for xLAM parameter-schema normalization.

xLAM/APIGen stores ``tools[].parameters`` as a property-definition map rather than as JSON
Schema, and its type annotations are Python-flavoured strings.  These tests pin the grammar
that is actually observed in the pinned corpus, the requiredness rule, and the invariants
that keep the transformation from becoming silent schema inference.
"""

from __future__ import annotations

import pytest

from opengrad.data.adapters import adapt_xlam
from opengrad.data.schema import SchemaValidationError, effective_schema
from opengrad.data.xlam_types import (
    RULE_GENERIC,
    RULE_REQUIREDNESS_UNASSERTED,
    RULE_OPTIONAL,
    RULE_SCALAR_ALIAS,
    RULE_WRAPPER,
    XlamAnnotationError,
    is_canonical_object_schema,
    normalize_xlam_tools,
    parse_xlam_annotation,
)

# --- scalar and optional forms ------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("str", {"type": "string"}),
        ("text", {"type": "string"}),
        ("int", {"type": "integer"}),
        ("long", {"type": "integer"}),
        ("float", {"type": "number"}),
        ("double", {"type": "number"}),
        ("bool", {"type": "boolean"}),
        ("dict", {"type": "object"}),
        ("Dict", {"type": "object"}),
        ("object", {"type": "object"}),
        ("list", {"type": "array"}),
        ("List", {"type": "array"}),
    ],
)
def test_scalar_aliases_are_normalized(annotation: str, expected: dict) -> None:
    parsed = parse_xlam_annotation(annotation)
    assert parsed.schema == expected
    assert parsed.optional is False
    assert RULE_SCALAR_ALIAS in parsed.rules or expected["type"] == "array"


@pytest.mark.parametrize("annotation", ["str", "int", "float", "bool"])
def test_optional_suffix_sets_flag_without_changing_the_leaf(annotation: str) -> None:
    bare = parse_xlam_annotation(annotation)
    optional = parse_xlam_annotation(f"{annotation}, optional")
    assert optional.schema == bare.schema, "the suffix must not change the leaf type"
    assert optional.optional is True
    assert bare.optional is False
    assert RULE_OPTIONAL in optional.rules


@pytest.mark.parametrize("annotation", [" STR , OPTIONAL ", "Str, Optional", "str ,optional"])
def test_optional_marker_is_case_and_space_tolerant(annotation: str) -> None:
    parsed = parse_xlam_annotation(annotation)
    assert parsed.schema == {"type": "string"}
    assert parsed.optional is True


def test_default_modifier_is_recorded_but_does_not_imply_optional() -> None:
    parsed = parse_xlam_annotation("str, default='elonmusk'")
    assert parsed.schema == {"type": "string"}
    assert parsed.optional is False
    assert parsed.has_default is True


@pytest.mark.parametrize(
    "annotation",
    ["str, optional, default 'London'", "int, optional, default=100", "str, optional, default='fr-FR'"],
)
def test_combined_optional_and_default_modifiers(annotation: str) -> None:
    parsed = parse_xlam_annotation(annotation)
    assert parsed.optional is True
    assert parsed.has_default is True


def test_quoted_commas_do_not_split_modifiers() -> None:
    parsed = parse_xlam_annotation("str, default='a,b'")
    assert parsed.schema == {"type": "string"}
    assert parsed.has_default is True


# --- generics -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("List[int]", {"type": "array", "items": {"type": "integer"}}),
        ("List[float]", {"type": "array", "items": {"type": "number"}}),
        ("List[str]", {"type": "array", "items": {"type": "string"}}),
        (
            "List[List[int]]",
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        ),
        (
            "List[List[float]]",
            {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        ),
        (
            "List[List[str]]",
            {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        ),
    ],
)
def test_list_generics_expand_recursively(annotation: str, expected: dict) -> None:
    parsed = parse_xlam_annotation(annotation)
    assert parsed.schema == expected
    assert RULE_GENERIC in parsed.rules


@pytest.mark.parametrize(
    ("annotation", "count"),
    [("Tuple[float, float]", 2), ("Tuple[int, int]", 2), ("Tuple[str, str, str]", 3)],
)
def test_homogeneous_tuples_become_fixed_length_arrays(annotation: str, count: int) -> None:
    parsed = parse_xlam_annotation(annotation)
    assert parsed.schema["type"] == "array"
    assert parsed.schema["minItems"] == count
    assert parsed.schema["maxItems"] == count


def test_nested_tuple_inside_list_is_supported() -> None:
    parsed = parse_xlam_annotation("List[Tuple[int, int]]")
    assert parsed.schema == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
    }


# --- quarantines: never guess -------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "code"),
    [
        ("set", "XLAM_TYPE_UNSUPPORTED_SET"),
        ("frozenset", "XLAM_TYPE_UNSUPPORTED_SET"),
        ("List[Union[int, float]]", "XLAM_TYPE_UNSUPPORTED_UNION"),
        ("Union[str, int]", "XLAM_TYPE_UNSUPPORTED_UNION"),
        ("Optional[str]", "XLAM_TYPE_UNSUPPORTED_UNION"),
        ("Callable[[float], float]", "XLAM_TYPE_UNSUPPORTED_CALLABLE"),
        ("", "XLAM_TYPE_EMPTY"),
        ("   ", "XLAM_TYPE_EMPTY"),
        ("weird<type>", "XLAM_TYPE_UNSUPPORTED_NAME"),
        ("Dict[str, int]", "XLAM_TYPE_UNSUPPORTED_ARGS"),
        ("str[0]", "XLAM_TYPE_MALFORMED"),
        ("List[int", "XLAM_TYPE_MALFORMED"),
        ("List[]", "XLAM_TYPE_MALFORMED"),
        ("str, optional, whee", "XLAM_TYPE_UNKNOWN_MODIFIER"),
        ("Tuple[float, int]", "XLAM_TYPE_TUPLE_NOT_HOMOGENEOUS"),
    ],
)
def test_unsupported_annotations_are_quarantined_with_reason_codes(
    annotation: str, code: str
) -> None:
    with pytest.raises(XlamAnnotationError) as excinfo:
        parse_xlam_annotation(annotation)
    assert excinfo.value.reason_code == code


def test_non_string_annotation_is_rejected() -> None:
    with pytest.raises(XlamAnnotationError) as excinfo:
        parse_xlam_annotation(123)
    assert excinfo.value.reason_code == "XLAM_TYPE_EMPTY"


def test_quarantine_never_yields_a_valid_schema() -> None:
    """An unsupported annotation must abort, not degrade into a permissive leaf."""
    for annotation in ["set", "Callable[[float], float]", "List[Union[int, float]]"]:
        with pytest.raises(XlamAnnotationError):
            parsed = parse_xlam_annotation(annotation)
            assert parsed.schema


# --- parameter-map wrapping ---------------------------------------------------------


def test_bare_parameter_map_is_wrapped_as_an_object_schema() -> None:
    tools, report = normalize_xlam_tools(
        [{"name": "peers", "parameters": {"symbol": {"type": "str"}}}]
    )
    assert tools[0]["parameters"] == {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
    }
    assert report.bare_parameter_maps_seen == 1
    assert report.object_wrappers_added == 1
    assert report.rules[RULE_WRAPPER] == 1


def test_parameter_literally_named_type_is_a_property_not_a_schema_keyword() -> None:
    """The core ambiguity: a map whose only key is `type` must become an object schema."""
    tools, _ = normalize_xlam_tools(
        [{"name": "by_type", "parameters": {"type": {"type": "str", "description": "kind"}}}]
    )
    schema = tools[0]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"] == {"type": {"type": "string", "description": "kind"}}


@pytest.mark.parametrize("keyword_like", ["type", "format", "items", "properties", "required", "enum"])
def test_every_keyword_like_parameter_name_is_treated_as_a_property(keyword_like: str) -> None:
    tools, _ = normalize_xlam_tools(
        [{"name": "t", "parameters": {keyword_like: {"type": "str"}}}]
    )
    assert tools[0]["parameters"]["properties"] == {keyword_like: {"type": "string"}}


def test_is_canonical_object_schema_requires_both_markers() -> None:
    assert is_canonical_object_schema({"type": "object", "properties": {}}) is True
    # A property map with a parameter named `type` is not a schema.
    assert is_canonical_object_schema({"type": {"type": "str"}}) is False
    # `properties` alone is not enough.
    assert is_canonical_object_schema({"properties": {"a": {"type": "string"}}}) is False
    assert is_canonical_object_schema({"type": "object"}) is False


def test_empty_parameter_map_is_a_valid_empty_object_schema() -> None:
    tools, report = normalize_xlam_tools([{"name": "noargs", "parameters": {}}])
    assert tools[0]["parameters"] == {"type": "object", "properties": {}}
    assert "required" not in tools[0]["parameters"]
    assert report.bare_parameter_maps_seen == 1


def test_nested_object_leaf_is_preserved() -> None:
    tools, _ = normalize_xlam_tools(
        [
            {
                "name": "t",
                "parameters": {"cfg": {"type": "Dict", "description": "config"}},
            }
        ]
    )
    assert tools[0]["parameters"]["properties"]["cfg"] == {
        "type": "object",
        "description": "config",
    }


def test_already_canonical_schema_is_untouched() -> None:
    original = {
        "name": "t",
        "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
    }
    tools, report = normalize_xlam_tools([dict(original)])
    assert tools[0] == original
    assert report.already_canonical == 1
    assert report.object_wrappers_added == 0
    assert report.bare_parameter_maps_seen == 0


def test_tool_without_parameters_is_untouched() -> None:
    original = {"name": "t", "description": "d"}
    tools, report = normalize_xlam_tools([dict(original)])
    assert tools[0] == original
    assert report.bare_parameter_maps_seen == 0


def test_non_object_parameter_map_is_rejected() -> None:
    with pytest.raises(XlamAnnotationError) as excinfo:
        normalize_xlam_tools([{"name": "t", "parameters": ["not", "a", "map"]}])
    assert excinfo.value.reason_code == "XLAM_PARAMETER_MAP_NOT_OBJECT"


def test_descriptor_without_type_is_rejected() -> None:
    with pytest.raises(XlamAnnotationError) as excinfo:
        normalize_xlam_tools([{"name": "t", "parameters": {"a": {"description": "no type"}}}])
    assert excinfo.value.reason_code == "XLAM_PARAMETER_TYPE_MISSING"


def test_non_object_descriptor_is_rejected() -> None:
    with pytest.raises(XlamAnnotationError) as excinfo:
        normalize_xlam_tools([{"name": "t", "parameters": {"a": "just a string"}}])
    assert excinfo.value.reason_code == "XLAM_PARAMETER_DESCRIPTOR_NOT_OBJECT"


# --- required / optional semantics --------------------------------------------------


def test_optional_parameters_are_counted_and_requiredness_is_not_asserted() -> None:
    """The source's optional marker is recorded, but no obligation is invented.

    Marking unmarked parameters required made the schema reject xLAM's own gold arguments
    (3,050 records failed `ARG_REQUIRED`), so no `required` list is emitted at all.
    """
    tools, report = normalize_xlam_tools(
        [
            {
                "name": "t",
                "parameters": {
                    "needed": {"type": "str"},
                    "spare": {"type": "str, optional"},
                },
            }
        ]
    )
    schema = tools[0]["parameters"]
    assert "required" not in schema
    assert schema["properties"]["spare"] == {"type": "string"}
    assert report.optional_parameters == 1
    assert report.unmarked_parameters == 1
    assert report.optional_annotations_seen == 1


def test_gold_arguments_are_never_rejected_for_a_missing_parameter() -> None:
    """Whatever the source omits, the canonical schema must accept the source's own calls."""
    from opengrad.data.schema import validate_arguments

    tools, report = normalize_xlam_tools(
        [
            {
                "name": "t",
                "parameters": {
                    "a": {"type": "str"},
                    "b": {"type": "str"},  # unmarked, but the gold call below omits it
                    "c": {"type": "str, optional"},
                },
            }
        ]
    )
    validate_arguments(tools[0]["parameters"], {"a": "x"})
    assert report.rules[RULE_REQUIREDNESS_UNASSERTED] == 3


def test_default_is_preserved_but_not_treated_as_an_optionality_signal() -> None:
    """A quarter of observed defaults are empty-string placeholders, so a default is a value
    annotation only. It is preserved verbatim and does not affect requiredness."""
    tools, report = normalize_xlam_tools(
        [{"name": "t", "parameters": {"symbol": {"type": "str", "default": ""}}}]
    )
    schema = tools[0]["parameters"]
    assert schema["properties"]["symbol"] == {"type": "string", "default": ""}
    assert "required" not in schema
    assert report.defaults_preserved == 1


def test_all_optional_map_omits_required_entirely() -> None:
    tools, _ = normalize_xlam_tools(
        [{"name": "t", "parameters": {"a": {"type": "str, optional"}}}]
    )
    assert "required" not in tools[0]["parameters"]


# --- invariants ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        "str",
        "str, optional",
        "List[int]",
        "List[List[float]]",
        "Tuple[float, float]",
        "Dict",
        "str, default='x'",
    ],
)
def test_normalization_is_idempotent(annotation: str) -> None:
    parsed = parse_xlam_annotation(annotation)
    once = parse_xlam_annotation(annotation)
    assert parsed == once, "parsing must be a pure deterministic function"


def test_wrapping_is_idempotent() -> None:
    tools, _ = normalize_xlam_tools([{"name": "t", "parameters": {"a": {"type": "str"}}}])
    again, report = normalize_xlam_tools(tools)
    assert again == tools
    assert report.already_canonical == 1
    assert report.object_wrappers_added == 0


def test_normalization_is_deterministic_including_rule_order() -> None:
    tool = {"name": "t", "parameters": {"b": {"type": "List[int]"}, "a": {"type": "str, optional"}}}
    first, report_a = normalize_xlam_tools([tool])
    second, report_b = normalize_xlam_tools([tool])
    assert first == second
    assert report_a.as_dict() == report_b.as_dict()


def test_normalized_output_passes_the_canonical_schema_validator() -> None:
    tools, _ = normalize_xlam_tools(
        [
            {
                "name": "t",
                "description": "d",
                "parameters": {
                    "a": {"type": "str"},
                    "b": {"type": "List[int]"},
                    "c": {"type": "str, optional"},
                    "d": {"type": "Tuple[float, float]"},
                },
            }
        ]
    )
    schema = effective_schema(tools[0])
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"a", "b", "c", "d"}
    assert "required" not in schema


def test_canonical_validator_still_rejects_an_unwrapped_bare_map() -> None:
    """The generic validator must keep failing closed; only the source adapter wraps."""
    with pytest.raises(SchemaValidationError) as excinfo:
        effective_schema({"name": "t", "parameters": {"symbol": {"type": "string"}}})
    assert excinfo.value.reason_code in {"SCH_UNSUPPORTED_KEYWORD", "SCH_UNSUPPORTED_TYPE"}


# --- adapter-level provenance -------------------------------------------------------


def _xlam_record(tools: list[dict]) -> dict:
    return {
        "id": "x1",
        "query": "look something up",
        "tools": tools,
        "answers": [{"name": tools[0]["name"], "arguments": {"q": 1}}],
    }


def test_adapter_records_normalization_provenance() -> None:
    record = _xlam_record([{"name": "find", "parameters": {"q": {"type": "str"}}}])
    conversation = adapt_xlam(record)
    metadata = conversation.metadata
    assert metadata["adapter"] == "xlam_function_calling_60k_v2"
    normalization = metadata["source_features"]["xlam_schema_normalization"]
    assert normalization["xlam_bare_parameter_maps_seen"] == 1
    assert normalization["xlam_object_wrappers_added"] == 1
    assert normalization["xlam_rules"]["XLAM_PARAMETER_MAP_WRAPPED_AS_OBJECT"] == 1


def test_adapter_tool_schema_is_canonical_and_required_is_correct() -> None:
    record = _xlam_record(
        [{"name": "find", "parameters": {"q": {"type": "str"}, "limit": {"type": "int, optional"}}}]
    )
    conversation = adapt_xlam(record)
    schema = conversation.tools[0]["parameters"]
    assert "required" not in schema
    assert schema["properties"]["limit"] == {"type": "integer"}


def test_adapter_fails_closed_on_an_unsupported_annotation() -> None:
    record = _xlam_record([{"name": "cb", "parameters": {"fn": {"type": "Callable[[float], float]"}}}])
    with pytest.raises(XlamAnnotationError) as excinfo:
        adapt_xlam(record)
    assert excinfo.value.reason_code == "XLAM_TYPE_UNSUPPORTED_CALLABLE"


def test_adapter_handles_a_tool_whose_parameter_is_named_type() -> None:
    record = _xlam_record([{"name": "by_type", "parameters": {"type": {"type": "str"}}}])
    conversation = adapt_xlam(record)
    schema = conversation.tools[0]["parameters"]
    assert schema["properties"] == {"type": {"type": "string"}}
