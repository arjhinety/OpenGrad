"""Source-scoped translation of an upstream tool-schema type vocabulary into the canonical one.

The problem this solves
-----------------------
When2Call's tool schemas are Python/typing annotations, not JSON Schema: ``parameters.type`` is
``"dict"`` for every tool, and property types include ``"str, optional"``, ``"int, optional"``,
``"List[int]"``, ``"Dict"``, ``"Set[T]"`` and ``"Tuple[float, float]"``. The canonical validator's
vocabulary is ``SUPPORTED_TYPES`` (JSON Schema names), so records carrying those forms were quarantined
as ``SCH_UNSUPPORTED_TYPE`` -- measured at 8,445 of 15,000 rows (56.3%) for When2Call alone.

The fix deliberately does **not** widen ``opengrad.data.schema``. Strict validation is a property worth
keeping: it is the last line of defence before a wrong reading of a record becomes a training target.
Instead the *source adapter* translates its own vocabulary at the adapter boundary, records what it
translated, and the canonical validator then runs unchanged over the translated schema. A form whose
semantics cannot be represented safely is **quarantined with an explicit reason** rather than coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opengrad.data.versions import SCHEMA_NORMALIZATION_VERSION

#: Types the canonical validator accepts. A literal copy on purpose: this module must not import the
#: validator's set, because a future widening of the validator must not silently widen translation too.
CANONICAL_TYPES = frozenset({"object", "string", "number", "integer", "boolean", "array", "null"})

SCALAR_ALIASES: dict[str, str | None] = {
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
    "list": "array",
    "array": "array",
    "any": None,
    "none": None,
    "null": "null",
}

_SEQUENCE_GENERICS = frozenset({"list", "sequence", "tuple", "set", "frozenset", "iterable"})
_MAPPING_GENERICS = frozenset({"dict", "mapping", "ordereddict", "defaultdict"})

#: Quarantine reason codes, deliberately distinct from the canonical ``SCH_*`` codes so that "what did
#: translation reject" can never be confused with "what did validation reject".
SRC_HETEROGENEOUS_TUPLE = "SRC_SCHEMA_HETEROGENEOUS_TUPLE"
SRC_MULTI_TYPE_UNION = "SRC_SCHEMA_MULTI_TYPE_UNION"
SRC_UNREPRESENTABLE_TYPE = "SRC_SCHEMA_UNREPRESENTABLE_TYPE"
SRC_TYPE_NOT_STRING = "SRC_SCHEMA_TYPE_NOT_STRING"


@dataclass(frozen=True)
class TypeTranslation:
    """What one upstream annotation translated into, and what was lost on the way."""

    canonical_type: str | None
    nullable: bool = False
    items: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()
    quarantine_reason: str | None = None
    quarantine_detail: str = ""

    @property
    def quarantined(self) -> bool:
        return self.quarantine_reason is not None


def _split_top_level(value: str) -> list[str]:
    """Split on commas outside brackets, so ``List[List[int]]`` stays one argument."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in value:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _inner(annotation: str) -> tuple[str, str] | None:
    """Return ``(head, inner)`` for a well-formed subscript, else None.

    The final ``]`` closes the outer subscript, so the inner text is everything between the first
    ``[`` and that last character -- and it is the *inner* text that must be balanced. Checking the
    balance of the whole remainder instead rejects correctly nested forms such as ``List[List[int]]``.
    """
    if not annotation.endswith("]") or "[" not in annotation:
        return None
    head, _, rest = annotation.partition("[")
    inner = rest[:-1]
    depth = 0
    for char in inner:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
            if depth < 0:
                return None
    if depth != 0:
        return None
    return head.strip(), inner.strip()


def _from(inner: TypeTranslation, canonical_type: str | None, nullable: bool, notes: tuple[str, ...]) -> TypeTranslation:
    if inner.quarantined:
        return inner
    return TypeTranslation(
        canonical_type,
        nullable=nullable or inner.nullable,
        items=inner.items,
        notes=inner.notes + notes,
    )


def translate_type_annotation(annotation: Any) -> TypeTranslation:
    """Translate one upstream annotation into the canonical vocabulary.

    Deterministic and total: every input returns a verdict, and an input whose semantics cannot be
    represented returns a quarantine reason rather than a guess.
    """
    if annotation is None:
        return TypeTranslation(None, notes=("absent_type",))
    if isinstance(annotation, list):
        # JSON Schema allows a list of types ("type": ["string", "null"]). One non-null member plus
        # null is faithfully representable; more than one non-null member is not, and the canonical
        # validator's own alias pass resolves that case by dropping the constraint. Translation keeps
        # that behaviour *and records it*, so the coercion is visible rather than silent -- and so
        # that normalising never rejects a row the canonical layer already accepted.
        members = [m for m in annotation if isinstance(m, str)]
        non_null = [m for m in members if m.casefold() != "null"]
        if len(non_null) == 1 and len(members) != len(non_null):
            resolved = translate_type_annotation(non_null[0])
            return _from(resolved, resolved.canonical_type, True, ("type_list_nullable",))
        return TypeTranslation(
            None,
            nullable=True,
            notes=(f"multi_type_list_dropped:{'|'.join(non_null) or 'null_only'}",),
        )
    if isinstance(annotation, dict):
        # Some exports nest a complete schema under ``type``; recover only its declared type, which
        # is what the canonical validator's own alias pass does. Without this, a translation would
        # reject rows the canonical layer accepts -- a regression, not a normalisation.
        nested = annotation.get("type")
        if isinstance(nested, str):
            recovered = translate_type_annotation(nested)
            if recovered.quarantined:
                return recovered
            return TypeTranslation(
                recovered.canonical_type,
                nullable=recovered.nullable,
                items=recovered.items,
                notes=recovered.notes + ("nested_schema_under_type",),
            )
        # A dict without a string ``type`` is not a representable annotation. The canonical layer
        # resolves *any* non-string type by dropping the constraint, so translation does the same and
        # records it: normalising must never reject a row the canonical layer already accepted.
        return TypeTranslation(
            None,
            nullable=True,
            notes=(f"unparsed_type_dropped:{sorted(annotation)[:4]}",),
        )
    if not isinstance(annotation, str):
        return TypeTranslation(
            None,
            nullable=True,
            notes=(f"unparsed_type_dropped:{type(annotation).__name__}:{annotation!r}"[:120],),
        )
    text = annotation.strip()
    if not text:
        return TypeTranslation(None, notes=("empty_type",))

    nullable = False
    lowered = text.casefold()
    # "str, optional" / "int, optional" / "bool, optional" -- the dominant When2Call forms.
    for suffix in (", optional", ",optional", " optional"):
        if lowered.endswith(suffix):
            nullable = True
            text = text[: -len(suffix)].strip()
            break
    if text.casefold() == "optional":
        return TypeTranslation(None, nullable=True, notes=("optional_without_type",))
    # Recompute after stripping: the lookup below must see the type, not the suffix.
    lowered = text.casefold()

    subscript = _inner(text)
    if subscript is not None:
        head, inner = subscript
        folded = head.casefold()
        if folded == "optional":
            resolved = translate_type_annotation(inner)
            return _from(resolved, resolved.canonical_type, True, ())
        if folded == "union":
            args = _split_top_level(inner)
            non_none = [a for a in args if a.casefold() not in {"none", "nonetype"}]
            if len(non_none) == 1 and len(non_none) != len(args):
                resolved = translate_type_annotation(non_none[0])
                return _from(resolved, resolved.canonical_type, True, ())
            return TypeTranslation(
                None, quarantine_reason=SRC_MULTI_TYPE_UNION, quarantine_detail=text
            )
        if folded in _SEQUENCE_GENERICS:
            args = _split_top_level(inner)
            if folded in {"set", "frozenset"}:
                resolved = translate_type_annotation(args[0])
                if resolved.quarantined:
                    return resolved
                return _from(resolved, "array", nullable, ("uniqueness_not_enforced",))
            if folded == "tuple" and len(args) > 1:
                first = translate_type_annotation(args[0])
                if first.quarantined:
                    return first
                homogeneous = all(
                    translate_type_annotation(a).canonical_type == first.canonical_type for a in args
                )
                if not homogeneous:
                    return TypeTranslation(
                        None, quarantine_reason=SRC_HETEROGENEOUS_TUPLE, quarantine_detail=text
                    )
                return _from(first, "array", nullable, ("arity_not_enforced",))
            resolved = translate_type_annotation(args[0])
            note = ("arity_not_enforced",) if folded == "tuple" else ()
            return _from(resolved, "array", nullable, note)
        if folded in _MAPPING_GENERICS:
            args = _split_top_level(inner)
            notes: tuple[str, ...] = ("key_type_not_enforced",)
            if len(args) > 1:
                value = translate_type_annotation(args[1])
                if value.quarantined:
                    return value
                if value.canonical_type not in {"object", None}:
                    notes = notes + (f"value_type={value.canonical_type}",)
            return TypeTranslation("object", nullable=nullable, notes=notes)
        return TypeTranslation(
            None, quarantine_reason=SRC_UNREPRESENTABLE_TYPE, quarantine_detail=text
        )

    folded = lowered
    if folded in _SEQUENCE_GENERICS:
        return TypeTranslation("array", nullable=nullable, notes=(f"element_type_unknown:{text}",))
    if folded in _MAPPING_GENERICS:
        return TypeTranslation("object", nullable=nullable, notes=(f"value_type_unknown:{text}",))
    if folded in SCALAR_ALIASES:
        canonical = SCALAR_ALIASES[folded]
        if canonical is None:
            # ``any``/``none``: no constraint asserted, matching the canonical validator's reading.
            return TypeTranslation(None, nullable=True, notes=(f"unconstrained:{text}",))
        return TypeTranslation(canonical, nullable=nullable)
    if text in CANONICAL_TYPES:
        return TypeTranslation(text, nullable=nullable)
    return TypeTranslation(None, quarantine_reason=SRC_UNREPRESENTABLE_TYPE, quarantine_detail=text)


def _items_schema(translation: TypeTranslation) -> dict[str, Any] | None:
    if translation.canonical_type is None:
        return None
    schema: dict[str, Any] = {"type": translation.canonical_type}
    if translation.items is not None:
        schema["items"] = translation.items
    if translation.nullable:
        schema["nullable"] = True
    return schema


@dataclass(frozen=True)
class SchemaTranslation:
    """A translated tool, what changed, and -- if anything could not be represented -- why."""

    tool: dict[str, Any]
    records: list[dict[str, Any]]
    quarantine_reason: str | None = None
    quarantine_detail: str = ""
    version: str = SCHEMA_NORMALIZATION_VERSION

    @property
    def quarantined(self) -> bool:
        return self.quarantine_reason is not None


def _walk(node: Any, path: str, records: list[dict[str, Any]]) -> tuple[Any, str | None, str]:
    """Translate a schema node in place, recursing through ``properties`` and ``items``."""
    if not isinstance(node, dict):
        return node, None, ""
    result: dict[str, Any] = dict(node)
    declared = result.get("type")
    translated = translate_type_annotation(declared)
    if translated.quarantined:
        return None, translated.quarantine_reason, f"{path}: {translated.quarantine_detail}"
    if declared is not None:
        if translated.canonical_type is None:
            result.pop("type", None)
        else:
            result["type"] = translated.canonical_type
        if translated.nullable:
            result["nullable"] = True
        if translated.items is not None and "items" not in result:
            result["items"] = translated.items
        # A record is emitted only when something actually changed, which makes the function
        # idempotent: translating an already-canonical schema yields no records.
        if translated.canonical_type != declared or translated.notes:
            records.append(
                {
                    "path": path,
                    "from": declared,
                    "to": translated.canonical_type,
                    "nullable": translated.nullable,
                    "notes": list(translated.notes),
                }
            )
    properties = result.get("properties")
    if isinstance(properties, dict):
        rebuilt: dict[str, Any] = {}
        for name, value in properties.items():
            child, reason, detail = _walk(value, f"{path}.properties.{name}", records)
            if reason:
                return None, reason, detail
            rebuilt[name] = child
        result["properties"] = rebuilt
    if "items" in result:
        child, reason, detail = _walk(result["items"], f"{path}.items", records)
        if reason:
            return None, reason, detail
        result["items"] = child
    return result, None, ""


def translate_source_schema(tool: dict[str, Any]) -> SchemaTranslation:
    """Translate one source-shaped tool's parameter schema into the canonical vocabulary.

    Idempotent by construction: a schema already in canonical form produces an empty ``records`` list,
    so a non-empty ``records`` list means a translation actually happened.
    """
    if not isinstance(tool, dict):
        return SchemaTranslation(tool={}, records=[], quarantine_reason=SRC_TYPE_NOT_STRING)
    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    parameters = source.get("parameters")
    if not isinstance(parameters, dict):
        # No parameter schema to translate; the canonical validator will judge the tool as it stands.
        return SchemaTranslation(tool=dict(tool), records=[])
    records: list[dict[str, Any]] = []
    translated, reason, detail = _walk(parameters, "parameters", records)
    if reason:
        return SchemaTranslation(
            tool=dict(tool), records=records, quarantine_reason=reason, quarantine_detail=detail
        )
    rebuilt = dict(tool)
    if isinstance(tool.get("function"), dict):
        rebuilt["function"] = dict(tool["function"], parameters=translated)
    else:
        rebuilt["parameters"] = translated
    return SchemaTranslation(tool=rebuilt, records=records)


def translate_source_tools(tools: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, str]:
    """Translate a whole tool list, returning the tools, the records, and any quarantine.

    Source tool lists arrive as a list of JSON *strings* in the When2Call parquet, so string elements
    are parsed here and an element that does not parse is quarantined rather than skipped.
    """
    import json

    if tools is None:
        return [], [], None, ""
    if not isinstance(tools, list):
        return [], [], SRC_TYPE_NOT_STRING, f"tools is {type(tools).__name__}"
    translated_tools: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, element in enumerate(tools):
        tool = element
        if isinstance(element, str):
            try:
                tool = json.loads(element)
            except json.JSONDecodeError:
                return [], records, SRC_UNREPRESENTABLE_TYPE, f"tools[{index}]: unparsable JSON"
        if not isinstance(tool, dict):
            return [], records, SRC_TYPE_NOT_STRING, f"tools[{index}]: {type(tool).__name__}"
        translation = translate_source_schema(tool)
        if translation.quarantined:
            return (
                [],
                records + translation.records,
                translation.quarantine_reason,
                f"tools[{index}] {translation.quarantine_detail}",
            )
        translated_tools.append(translation.tool)
        for record in translation.records:
            records.append({"tool_index": index, **record})
    return translated_tools, records, None, ""