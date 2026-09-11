"""Derived experiment index.

``results/registry.jsonl`` is a materialized projection of authoritative run artifacts, not a
store. See :mod:`opengrad.results.registry`.
"""

from opengrad.results.registry import (
    REGISTRY_RELATIVE_PATH,
    SCHEMA_VERSION,
    RegistryFinding,
    RegistryRow,
    build_registry,
    load_registry,
    rebuild_registry,
    registry_exists,
    registry_path,
    render_findings,
    render_registry_table,
    validate_registry,
    write_registry,
)

__all__ = [
    "REGISTRY_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "RegistryFinding",
    "RegistryRow",
    "build_registry",
    "load_registry",
    "rebuild_registry",
    "registry_exists",
    "registry_path",
    "render_findings",
    "render_registry_table",
    "validate_registry",
    "write_registry",
]
