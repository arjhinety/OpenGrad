"""Export `registry/datasets.yaml` records as MLCommons Croissant 1.1 JSON-LD (`opengrad croissant`).

The mapping is not written here. Each property of `registry/dataset_record.schema.json` names its
Croissant 1.1 or Croissant RAI 1.0 counterpart in `x-croissant`, and this module reads that
annotation, so the schema is the one place the mapping is defined. The properties that need a
transformation (licence, distribution, lineage, URL, citation) are built explicitly, and
`tests/registry/test_croissant.py` checks that each one agrees with the annotation.

The export is metadata only. It carries dataset-level metadata, the pinned files and responsible-use
fields; it has no RecordSet, because the registry does not describe record fields. A Croissant-required
property the registry cannot supply honestly -- most often an upstream publication date -- is left
out and reported in `missing` rather than filled with a guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CROISSANT = "http://mlcommons.org/croissant/1.1"
CROISSANT_RAI = "http://mlcommons.org/croissant/RAI/1.0"
MEDIA_TYPE = f'application/ld+json; profile="{CROISSANT}"'

#: The standard Croissant 1.1 context, as mlcroissant 1.1.0 builds it (`rdf.make_context()`), plus the
#: PROV-O prefix the spec's own lineage example adds. A non-standard context draws a validator warning.
CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "prov": "http://www.w3.org/ns/prov#",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

#: Croissant 1.1's required dataset-level properties, beside @context and @type.
REQUIRED = (
    "conformsTo",
    "name",
    "description",
    "license",
    "url",
    "creator",
    "datePublished",
    "distribution",
)

#: Registry fields exported as a plain value of the property their annotation names.
SCALAR_FIELDS = ("display_name", "notes", "retrieval_date", "date_published", "source_revision")

COMPOSITE_LICENCES = {"COMPOSITE_PER_SOURCE", "DERIVED_FROM_COMPOSITE_TRAINING_SOURCES"}


def load_schema(root: Path) -> dict[str, Any]:
    path = root / "registry/dataset_record.schema.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def croissant_mapping(schema: dict[str, Any]) -> dict[str, str | None]:
    """`x-croissant` per top-level property and per `responsible_use` sub-property."""
    properties = schema["properties"]
    mapping = {name: spec.get("x-croissant") for name, spec in properties.items()}
    for name, spec in properties["responsible_use"]["properties"].items():
        mapping[f"responsible_use.{name}"] = spec.get("x-croissant")
    return mapping


def _key(annotation: str) -> str:
    """The JSON-LD key: schema.org terms are the default vocabulary, so `sc:` is dropped."""
    return annotation.removeprefix("sc:")


def _spdx_url(identifier: str) -> str:
    return f"https://spdx.org/licenses/{identifier}.html"


def _licences(record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    """SPDX identifiers of a record, resolving per-source composites through derived_from."""
    value = str((record.get("license") or {}).get("value", ""))
    if value not in COMPOSITE_LICENCES:
        return [value] if value else []
    found: list[str] = []
    for parent in record.get("derived_from") or []:
        upstream = str(parent.get("upstream_license", ""))
        if upstream in COMPOSITE_LICENCES and parent.get("id") in by_id:
            nested = _licences(by_id[parent["id"]], by_id)
        else:
            nested = [upstream] if upstream else []
        found += [item for item in nested if item not in found]
    return found


def _url(record: dict[str, Any]) -> str | None:
    location = str(record.get("source_repository") or "")
    if location.startswith("https://"):
        return location
    if record.get("hf_id"):
        return f"https://huggingface.co/datasets/{record['hf_id']}"
    return None


def _papers(root: Path) -> dict[str, dict[str, Any]]:
    import yaml

    value = yaml.safe_load((root / "docs/references/papers.yaml").read_text(encoding="utf-8"))
    papers = value.get("papers", []) if isinstance(value, dict) else []
    return {str(p["id"]): p for p in papers if isinstance(p, dict) and "id" in p}


def export(
    record: dict[str, Any],
    registry: list[dict[str, Any]],
    schema: dict[str, Any],
    papers: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """One record as Croissant JSON-LD, and the required Croissant properties it could not fill."""
    mapping = croissant_mapping(schema)
    by_id = {str(r["id"]): r for r in registry}
    document: dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": [CROISSANT],
    }
    for field in SCALAR_FIELDS:
        annotation = mapping.get(field)
        value = record.get(field)
        if isinstance(value, dict):
            value = value.get("value")
        if annotation and value not in (None, ""):
            document[_key(annotation)] = value
    document["identifier"] = record["id"]
    document["creator"] = {"@type": "sc:Organization", "name": record["organization"]}
    document["keywords"] = [record.get("category"), record.get("role"), record.get("lifecycle")]
    licences = _licences(record, by_id)
    if licences:
        document["license"] = [_spdx_url(item) for item in licences]
    url = _url(record)
    if url:
        document["url"] = url
    cited = [papers[p] for p in record.get("papers") or [] if p in papers]
    if cited:
        document["citeAs"] = str(cited[0].get("canonical_url") or cited[0].get("title"))
        document["citation"] = [str(p.get("canonical_url") or p.get("title")) for p in cited]
    distribution = [
        {
            "@type": "cr:FileObject",
            "@id": item["name"],
            "name": item["name"],
            "contentUrl": item["content_url"],
            "encodingFormat": item["encoding_format"],
            "sha256": item["sha256"],
            "contentSize": f"{item['size_bytes']} B",
        }
        for item in record.get("distribution") or []
    ]
    if distribution:
        document["distribution"] = distribution
    derived = [
        f"opengrad:{parent['id']}@{parent['source_revision']}"
        for parent in record.get("derived_from") or []
    ]
    if derived:
        document["prov:wasDerivedFrom"] = derived
    responsible = record.get("responsible_use") or {}
    for name, value in responsible.items():
        annotation = mapping.get(f"responsible_use.{name}")
        if annotation and value:
            document[annotation] = value
    if any(str(key).startswith("rai:") for key in document):
        document["conformsTo"].append(CROISSANT_RAI)
    missing = [prop for prop in REQUIRED if prop not in document]
    return document, missing


def export_all(root: Path) -> dict[str, tuple[dict[str, Any], list[str]]]:
    import yaml

    registry = yaml.safe_load((root / "registry/datasets.yaml").read_text(encoding="utf-8"))
    records = [r for r in registry.get("datasets", []) if isinstance(r, dict)]
    schema = load_schema(root)
    papers = _papers(root)
    return {str(r["id"]): export(r, records, schema, papers) for r in records}


def run(root: Path, dataset_id: str | None, out: Path | None, strict: bool) -> int:
    """`opengrad croissant`: print one record or all of them, or write each to `out/<id>.json`."""
    exported = export_all(root)
    if dataset_id is not None:
        if dataset_id not in exported:
            print(f"unknown dataset id: {dataset_id}")
            return 2
        exported = {dataset_id: exported[dataset_id]}
    if out is None:
        documents = {key: document for key, (document, _) in exported.items()}
        value = documents[dataset_id] if dataset_id is not None else documents
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        out.mkdir(parents=True, exist_ok=True)
        for key, (document, _) in exported.items():
            (out / f"{key}.json").write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(f"wrote {len(exported)} Croissant documents to {out}")
    incomplete = {key: missing for key, (_, missing) in exported.items() if missing}
    for key, missing in sorted(incomplete.items()):
        print(
            f"{key}: missing required Croissant properties: {', '.join(missing)}", file=sys.stderr
        )
    return 1 if strict and incomplete else 0
