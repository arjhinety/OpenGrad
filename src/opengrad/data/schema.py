from __future__ import annotations

import math
from typing import Any

SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "additionalProperties",
    "nullable",
    # JSON-Schema annotations and common validation keywords used by source
    # catalogues.  They are retained for rendering even when the lightweight
    # argument checker does not need to enforce every constraint yet.
    "description",
    "default",
    "format",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
}
SUPPORTED_TYPES = {"object", "string", "number", "integer", "boolean", "array", "null"}


class SchemaValidationError(ValueError):
    """A stable, machine-readable effective-schema rejection."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{(': ' + detail) if detail else ''}")


class ArgumentValidationError(ValueError):
    """A stable, machine-readable tool-argument rejection."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}{(': ' + detail) if detail else ''}")


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _validate_schema(schema: Any, path: str = "schema") -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise SchemaValidationError("SCH_SCHEMA_NOT_OBJECT", path)
    if not _finite(schema):
        raise SchemaValidationError("SCH_NONFINITE", path)
    unsupported = sorted(set(schema) - SCHEMA_KEYS)
    if unsupported:
        raise SchemaValidationError("SCH_UNSUPPORTED_KEYWORD", ",".join(unsupported))
    result = dict(schema)
    schema_type = result.get("type")
    if schema_type is not None and (
        not isinstance(schema_type, str) or schema_type not in SUPPORTED_TYPES
    ):
        raise SchemaValidationError("SCH_UNSUPPORTED_TYPE", path)
    if "nullable" in result and not isinstance(result["nullable"], bool):
        raise SchemaValidationError("SCH_NULLABLE_NOT_BOOLEAN", path)
    if "required" in result:
        required = result["required"]
        if not isinstance(required, list) or any(
            not isinstance(item, str) or not item for item in required
        ):
            raise SchemaValidationError("SCH_REQUIRED_NOT_STRING_LIST", path)
        if len(set(required)) != len(required):
            raise SchemaValidationError("SCH_DUPLICATE_REQUIRED", path)
    if "properties" in result:
        properties = result["properties"]
        if not isinstance(properties, dict) or any(not isinstance(key, str) for key in properties):
            raise SchemaValidationError("SCH_PROPERTIES_NOT_OBJECT", path)
        result["properties"] = {
            key: _validate_schema(value, f"{path}.properties.{key}")
            for key, value in properties.items()
        }
    if "items" in result:
        result["items"] = _validate_schema(result["items"], f"{path}.items")
    if "enum" in result:
        values = result["enum"]
        if not isinstance(values, list) or not values or not _finite(values):
            raise SchemaValidationError("SCH_ENUM_INVALID", path)
    if "additionalProperties" in result and not isinstance(result["additionalProperties"], bool):
        raise SchemaValidationError("SCH_ADDITIONAL_PROPERTIES_NOT_BOOLEAN", path)
    if schema_type == "object" and "properties" not in result:
        result["properties"] = {}
    return result


def effective_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return one canonical parameter schema without dropping invocation constraints."""
    if not isinstance(tool, dict):
        raise SchemaValidationError("SCH_TOOL_NOT_OBJECT")
    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("name"), str)
        or not source["name"]
    ):
        raise SchemaValidationError("SCH_TOOL_NAME_REQUIRED")
    nested = source.get("parameters")
    # Tool descriptions/defaults belong to the tool object; do not accidentally
    # promote them into its parameter schema when the source uses the OpenAI
    # function shape.
    schema_shape_keys = SCHEMA_KEYS - {
        "description",
        "default",
        "format",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    }
    direct = {key: source[key] for key in schema_shape_keys if key in source}
    if nested is None:
        schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
        schema.update(direct)
    else:
        if not isinstance(nested, dict):
            raise SchemaValidationError("SCH_PARAMETERS_NOT_OBJECT", source["name"])
        schema = dict(nested)
        for key, value in direct.items():
            if key in schema and schema[key] != value:
                raise SchemaValidationError("SCH_CONFLICT", f"{source['name']}.{key}")
            schema[key] = value

    def aliases(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        type_value = result.get("type")
        if isinstance(type_value, dict):
            # A few evaluation exports accidentally nest a complete schema
            # under ``type``; recover only its declared type.
            type_value = type_value.get("type")
        if isinstance(type_value, str):
            result["type"] = {
                "dict": "object",
                "list": "array",
                "tuple": "array",
                "float": "number",
                "double": "number",
                "int": "integer",
                "long": "integer",
                "bool": "boolean",
                "str": "string",
                "text": "string",
                "any": None,
            }.get(type_value, type_value)
        elif "type" in result:
            result["type"] = None
        properties = result.get("properties")
        if isinstance(properties, dict):
            result["properties"] = {key: aliases(item) for key, item in properties.items()}
        if "items" in result:
            result["items"] = aliases(result["items"])
        return result

    return _validate_schema(aliases(schema), f"tool.{source['name']}.parameters")


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI/function and source-shaped tools to the effective contract."""
    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    assert isinstance(source, dict)
    normalized = {key: value for key, value in source.items() if key in {"name", "description"}}
    normalized["parameters"] = effective_schema(tool)
    return normalized


def validate_arguments(schema: dict[str, Any], arguments: Any) -> None:
    def fail(code: str, path: str) -> None:
        raise ArgumentValidationError(code, path)

    if not _finite(arguments):
        fail("ARG_NONFINITE", "arguments")

    def visit(rule: dict[str, Any], value: Any, path: str) -> None:
        if "enum" in rule and value not in rule["enum"]:
            fail("ARG_ENUM", path)
        if rule.get("nullable") and value is None:
            return
        expected = rule.get("type")
        valid = {
            "object": isinstance(value, dict),
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "null": value is None,
        }
        if expected is not None and not valid[expected]:
            fail("ARG_TYPE", path)
        if isinstance(value, dict):
            required = rule.get("required", [])
            missing = [key for key in required if key not in value]
            if missing:
                fail("ARG_REQUIRED", f"{path}.{missing[0]}")
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False:
                unknown = sorted(set(value) - set(properties))
                if unknown:
                    fail("ARG_ADDITIONAL_PROPERTY", f"{path}.{unknown[0]}")
            for key, item in value.items():
                if key in properties:
                    visit(properties[key], item, f"{path}.{key}")
        elif isinstance(value, list) and "items" in rule:
            for index, item in enumerate(value):
                visit(rule["items"], item, f"{path}[{index}]")

    visit(schema, arguments, "arguments")
