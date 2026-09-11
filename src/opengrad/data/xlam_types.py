"""xLAM/APIGen parameter-schema normalization.

Why this module exists
----------------------
`Salesforce/xlam-function-calling-60k` does **not** store tool parameters as JSON Schema.
Its dataset card documents the source contract as::

    parameters (object): An object representing the parameters required by the tool.
      - Each parameter is represented as a key-value pair, where the key is the
        parameter name and the value is an object with the following properties:
          - type (string): The data type of the parameter (e.g., "int", "float", "list").
          - description (string): A brief description of the parameter.
          - required (boolean): Indicates whether the parameter is required or optional.

So `parameters` is a *property-definition map*: parameter name -> descriptor.  The
canonical OpenGrad contract requires a JSON Schema object.  Wrapping the map as
``{"type": "object", "properties": <map>, ...}`` is therefore a **source-contract
transformation**, not generic schema inference.

The distinction is not academic.  A bare map is structurally indistinguishable from a
schema to a naive heuristic, and the corpus contains tools whose parameters are literally
named ``type`` (2551), ``format`` (2179), ``items`` (670) and ``properties`` (19).  A global
rule like "any bare dict is a schema" would silently reinterpret ~5500 tools.  This module
keeps the transformation at the provenance-aware source boundary and fails closed on
anything it cannot express exactly.

Type annotations are Python-flavoured strings (``str``, ``List[int]``, ``str, optional``),
not JSON Schema types, so they are parsed by a small explicit grammar.  ``eval`` is never
used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- reason codes -------------------------------------------------------------------
XLAM_TYPE_EMPTY = "XLAM_TYPE_EMPTY"
XLAM_TYPE_MALFORMED = "XLAM_TYPE_MALFORMED"
XLAM_TYPE_UNSUPPORTED_NAME = "XLAM_TYPE_UNSUPPORTED_NAME"
XLAM_TYPE_UNSUPPORTED_UNION = "XLAM_TYPE_UNSUPPORTED_UNION"
XLAM_TYPE_UNSUPPORTED_CALLABLE = "XLAM_TYPE_UNSUPPORTED_CALLABLE"
XLAM_TYPE_UNSUPPORTED_SET = "XLAM_TYPE_UNSUPPORTED_SET"
XLAM_TYPE_UNSUPPORTED_ARGS = "XLAM_TYPE_UNSUPPORTED_ARGS"
XLAM_TYPE_TUPLE_NOT_HOMOGENEOUS = "XLAM_TYPE_TUPLE_NOT_HOMOGENEOUS"
XLAM_TYPE_UNKNOWN_MODIFIER = "XLAM_TYPE_UNKNOWN_MODIFIER"
XLAM_PARAMETER_TYPE_MISSING = "XLAM_PARAMETER_TYPE_MISSING"
XLAM_PARAMETER_DESCRIPTOR_NOT_OBJECT = "XLAM_PARAMETER_DESCRIPTOR_NOT_OBJECT"
XLAM_PARAMETER_MAP_NOT_OBJECT = "XLAM_PARAMETER_MAP_NOT_OBJECT"

# Rules recorded in per-record provenance.
RULE_WRAPPER = "XLAM_PARAMETER_MAP_WRAPPED_AS_OBJECT"
RULE_OPTIONAL = "XLAM_OPTIONAL_SUFFIX_STRIPPED_TO_REQUIREDNESS"
RULE_SCALAR_ALIAS = "XLAM_SCALAR_ALIAS_NORMALIZED"
RULE_GENERIC = "XLAM_GENERIC_ANNOTATION_EXPANDED"
RULE_BARE_COLLECTION = "XLAM_BARE_COLLECTION_NORMALIZED"
RULE_DEFAULT_PRESERVED = "XLAM_DEFAULT_PRESERVED"


class XlamAnnotationError(ValueError):
    """A machine-readable rejection of an xLAM annotation or parameter map."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{(': ' + detail) if detail else ''}")


# --- scalar vocabulary --------------------------------------------------------------
# Names are matched case-insensitively for the *lookup*, but only names observed in the
# pinned corpus (or their documented aliases) are accepted.  An unknown name is a
# quarantine, never a guess.
_SCALARS: dict[str, str] = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
}

# Bare collection names that are valid on their own but carry no element type.
_BARE_COLLECTIONS: dict[str, str] = {
    "list": "array",
    "array": "array",
    "tuple": "array",
}

# Named forms we deliberately refuse rather than approximate.
_UNSUPPORTED_NAMES: dict[str, str] = {
    "union": XLAM_TYPE_UNSUPPORTED_UNION,
    "optional": XLAM_TYPE_UNSUPPORTED_UNION,  # typing.Optional[T] is a union
    "callable": XLAM_TYPE_UNSUPPORTED_CALLABLE,
    "set": XLAM_TYPE_UNSUPPORTED_SET,
    "frozenset": XLAM_TYPE_UNSUPPORTED_SET,
}

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_MAX_DEPTH = 8


@dataclass(frozen=True)
class ParsedAnnotation:
    """The JSON Schema leaf for one xLAM annotation, plus its requiredness signal."""

    schema: dict[str, Any]
    optional: bool
    has_default: bool
    rules: tuple[str, ...]


class _Parser:
    """Recursive-descent parser for the observed xLAM annotation grammar.

    grammar := expr (',' modifier)*
    expr    := NAME ('[' expr (',' expr)* ']')?
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.depth = 0

    # -- primitives -----------------------------------------------------------------
    def _skip_space(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _at_end(self) -> bool:
        self._skip_space()
        return self.index >= len(self.text)

    def _peek(self) -> str:
        self._skip_space()
        return self.text[self.index] if self.index < len(self.text) else ""

    def _expect(self, char: str) -> None:
        self._skip_space()
        if self.index >= len(self.text) or self.text[self.index] != char:
            raise XlamAnnotationError(XLAM_TYPE_MALFORMED, self.text[self.index : self.index + 16])
        self.index += 1

    def _read_name(self) -> str:
        self._skip_space()
        match = _NAME_RE.match(self.text, self.index)
        if not match:
            raise XlamAnnotationError(XLAM_TYPE_MALFORMED, self.text[self.index : self.index + 16])
        self.index = match.end()
        return match.group(0)

    # -- grammar --------------------------------------------------------------------
    def parse_expr(self) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self.depth > _MAX_DEPTH:
            raise XlamAnnotationError(XLAM_TYPE_MALFORMED, f"nesting depth > {_MAX_DEPTH}")
        name = self._read_name()
        lowered = name.lower()
        if lowered in _UNSUPPORTED_NAMES:
            raise XlamAnnotationError(_UNSUPPORTED_NAMES[lowered], name)
        if self._peek() != "[":
            return self._leaf(lowered, name)
        self.index += 1
        self.depth += 1
        args = [self.parse_expr()]
        while self._peek() == ",":
            self.index += 1
            args.append(self.parse_expr())
        self._expect("]")
        self.depth -= 1
        return self._container(lowered, name, args)

    def _leaf(self, lowered: str, original: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        if lowered in _SCALARS:
            return {"type": _SCALARS[lowered]}, (RULE_SCALAR_ALIAS,)
        if lowered in _BARE_COLLECTIONS:
            return {"type": _BARE_COLLECTIONS[lowered]}, (RULE_BARE_COLLECTION,)
        raise XlamAnnotationError(XLAM_TYPE_UNSUPPORTED_NAME, original)

    def _container(
        self, lowered: str, original: str, args: list[tuple[dict[str, Any], tuple[str, ...]]]
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if lowered in {"list", "array", "sequence"}:
            if len(args) != 1:
                raise XlamAnnotationError(XLAM_TYPE_UNSUPPORTED_ARGS, original)
            schema, rules = args[0]
            return {"type": "array", "items": schema}, rules + (RULE_GENERIC,)
        if lowered in {"tuple"}:
            schemas = [schema for schema, _ in args]
            rules = tuple(r for _, rs in args for r in rs)
            if not schemas or any(item != schemas[0] for item in schemas):
                # A heterogeneous tuple cannot be expressed without inventing positional
                # semantics, so it is quarantined rather than approximated.
                raise XlamAnnotationError(XLAM_TYPE_TUPLE_NOT_HOMOGENEOUS, original)
            return (
                {
                    "type": "array",
                    "items": schemas[0],
                    "minItems": len(schemas),
                    "maxItems": len(schemas),
                },
                rules + (RULE_GENERIC,),
            )
        if lowered in _UNSUPPORTED_NAMES:
            raise XlamAnnotationError(_UNSUPPORTED_NAMES[lowered], original)
        # A scalar such as `dict`/`str` with arguments is not a form we model.
        raise XlamAnnotationError(XLAM_TYPE_UNSUPPORTED_ARGS, original)

    def parse_modifiers(self) -> tuple[bool, bool]:
        """Parse ``(, modifier)*``; returns (optional, has_default)."""
        optional = False
        has_default = False
        while not self._at_end():
            self._expect(",")
            token = self._read_modifier_token().strip()
            if not token:
                raise XlamAnnotationError(XLAM_TYPE_MALFORMED, "empty modifier")
            lowered = token.lower()
            if lowered == "optional":
                optional = True
            elif lowered.startswith("default"):
                has_default = True
            else:
                raise XlamAnnotationError(XLAM_TYPE_UNKNOWN_MODIFIER, token[:40])
        return optional, has_default

    def _read_modifier_token(self) -> str:
        """Read up to the next top-level comma, honouring quotes and brackets."""
        self._skip_space()
        start = self.index
        quote = ""
        depth = 0
        while self.index < len(self.text):
            char = self.text[self.index]
            if quote:
                if char == quote:
                    quote = ""
            elif char in "'\"":
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                break
            self.index += 1
        return self.text[start : self.index]


def parse_xlam_annotation(text: Any) -> ParsedAnnotation:
    """Parse one xLAM type annotation into a canonical JSON Schema leaf.

    The ``optional`` flag reflects the source's *only* optionality marker.  A ``default``
    is recorded but deliberately does **not** imply optionality: a quarter of observed
    defaults are empty-string placeholders, so treating them as an optionality signal
    would be inference rather than interpretation.
    """
    if not isinstance(text, str) or not text.strip():
        raise XlamAnnotationError(XLAM_TYPE_EMPTY, repr(text)[:40])
    parser = _Parser(text)
    schema, rules = parser.parse_expr()
    optional, has_default = parser.parse_modifiers()
    if optional:
        rules = rules + (RULE_OPTIONAL,)
    return ParsedAnnotation(schema=schema, optional=optional, has_default=has_default, rules=rules)


# --- parameter-map transformation ---------------------------------------------------


def is_canonical_object_schema(parameters: Any) -> bool:
    """True when ``parameters`` is already a JSON Schema object, not a property map.

    Requires *both* ``type == "object"`` (a string) and ``properties`` being a dict.  A bare
    property map cannot satisfy this: if it has a parameter named ``type``, the value under
    that key is a descriptor object, never the string ``"object"``.
    """
    return (
        isinstance(parameters, dict)
        and parameters.get("type") == "object"
        and isinstance(parameters.get("properties"), dict)
    )


@dataclass
class RepairReport:
    """Per-record (or aggregate) provenance of the applied normalization rules."""

    bare_parameter_maps_seen: int = 0
    object_wrappers_added: int = 0
    already_canonical: int = 0
    parameters_total: int = 0
    optional_annotations_seen: int = 0
    optional_parameters: int = 0
    required_parameters: int = 0
    generic_annotations_seen: int = 0
    defaults_preserved: int = 0
    rules: dict[str, int] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)

    def note(self, rule: str, count: int = 1) -> None:
        self.rules[rule] = self.rules.get(rule, 0) + count

    def merge(self, other: RepairReport) -> None:
        self.bare_parameter_maps_seen += other.bare_parameter_maps_seen
        self.object_wrappers_added += other.object_wrappers_added
        self.already_canonical += other.already_canonical
        self.parameters_total += other.parameters_total
        self.optional_annotations_seen += other.optional_annotations_seen
        self.optional_parameters += other.optional_parameters
        self.required_parameters += other.required_parameters
        self.generic_annotations_seen += other.generic_annotations_seen
        self.defaults_preserved += other.defaults_preserved
        for rule, count in other.rules.items():
            self.rules[rule] = self.rules.get(rule, 0) + count

    def as_dict(self) -> dict[str, Any]:
        return {
            "xlam_bare_parameter_maps_seen": self.bare_parameter_maps_seen,
            "xlam_object_wrappers_added": self.object_wrappers_added,
            "xlam_already_canonical_schemas": self.already_canonical,
            "xlam_parameters_total": self.parameters_total,
            "xlam_optional_annotations_seen": self.optional_annotations_seen,
            "xlam_optional_parameters": self.optional_parameters,
            "xlam_required_parameters": self.required_parameters,
            "xlam_generic_annotations_seen": self.generic_annotations_seen,
            "xlam_defaults_preserved": self.defaults_preserved,
            "xlam_rules": dict(sorted(self.rules.items())),
        }


def _convert_parameter_map(
    name: str, parameters: dict[str, Any], report: RepairReport
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter_name, descriptor in parameters.items():
        if not isinstance(parameter_name, str):
            raise XlamAnnotationError(XLAM_PARAMETER_MAP_NOT_OBJECT, name)
        if not isinstance(descriptor, dict):
            raise XlamAnnotationError(XLAM_PARAMETER_DESCRIPTOR_NOT_OBJECT, parameter_name)
        annotation = descriptor.get("type")
        if annotation is None:
            raise XlamAnnotationError(XLAM_PARAMETER_TYPE_MISSING, parameter_name)
        parsed = parse_xlam_annotation(annotation)
        report.parameters_total += 1
        if "[" in str(annotation):
            report.generic_annotations_seen += 1
        if parsed.optional:
            report.optional_annotations_seen += 1
        for rule in parsed.rules:
            report.note(rule)
        leaf = dict(parsed.schema)
        description = descriptor.get("description")
        if isinstance(description, str) and description:
            leaf["description"] = description
        if "default" in descriptor and _finite(descriptor["default"]):
            leaf["default"] = descriptor["default"]
            report.defaults_preserved += 1
            report.note(RULE_DEFAULT_PRESERVED)
        properties[parameter_name] = leaf
        if parsed.optional:
            report.optional_parameters += 1
        else:
            report.required_parameters += 1
            required.append(parameter_name)
    # Requiredness is a property of the parameter map as a whole, so it is assembled here
    # rather than inferred from any single descriptor.
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    report.object_wrappers_added += 1
    report.note(RULE_WRAPPER)
    return schema


def _finite(value: Any) -> bool:
    import math

    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def normalize_xlam_tool(tool: dict[str, Any], report: RepairReport) -> dict[str, Any]:
    """Return ``tool`` with ``parameters`` expressed as canonical JSON Schema."""
    if not isinstance(tool, dict):
        raise XlamAnnotationError(XLAM_PARAMETER_MAP_NOT_OBJECT, type(tool).__name__)
    parameters = tool.get("parameters")
    if parameters is None:
        return tool
    if is_canonical_object_schema(parameters):
        report.already_canonical += 1
        return tool
    if not isinstance(parameters, dict):
        raise XlamAnnotationError(XLAM_PARAMETER_MAP_NOT_OBJECT, str(tool.get("name"))[:40])
    report.bare_parameter_maps_seen += 1
    normalized = dict(tool)
    normalized["parameters"] = _convert_parameter_map(str(tool.get("name", "")), parameters, report)
    return normalized


def normalize_xlam_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], RepairReport]:
    """Normalize every tool in one record, returning the tools and a repair report.

    Fails closed: the first annotation that cannot be expressed exactly aborts the record
    so it is quarantined with a reason code rather than silently approximated.
    """
    report = RepairReport()
    return [normalize_xlam_tool(tool, report) for tool in tools], report
