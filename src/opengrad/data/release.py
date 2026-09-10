from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from opengrad.data.materialize import iter_materialized_rows
from opengrad.data.real_analysis import _restore_row

_SOURCE_REPOS = {
    "xlam-function-calling-60k": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
    "button": "https://github.com/PKU-Baichuan-MLSystemLab/BUTTON",
    "toolace": "https://huggingface.co/datasets/Team-ACE/ToolACE",
    "looptool-23k": "https://huggingface.co/datasets/zhangkangning/LoopTool-23k",
    "glaive-function-calling-v2": "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
    "when2call": "https://huggingface.co/datasets/nvidia/When2Call",
}

# The card a release is published under follows its config. v1 is the default so
# that releases which predate this field keep building byte-identically.
_DEFAULT_CARD_DIRECTORY = "release/huggingface/toolpolicy-canonical-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "UNKNOWN"


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("release config must be an object")
    return value


def _parquet_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    provenance = metadata.get("source", {}) if isinstance(metadata, dict) else {}
    behavior = metadata.get("behavior", {}) if isinstance(metadata, dict) else {}
    return {
        "opengrad_id": str(row["id"]),
        "source_dataset": source["id"],
        "source_repo": _SOURCE_REPOS.get(source["id"], ""),
        "source_record_id": str(provenance.get("upstream_id", row["id"])),
        "source_split": str(metadata.get("split", "")) if isinstance(metadata, dict) else "",
        "source_revision": source["source_revision"],
        "source_license": source["upstream_license"],
        "redistribution_status": source["redistribution_status"],
        "modification_status": source.get("modification_status", "NORMALIZED_DERIVATIVE"),
        "upstream_access_mode": source.get("upstream_access_mode", "public"),
        "downstream_access_requirement": source.get(
            "downstream_access_requirement", "public_allowed"
        ),
        "adapter": str(metadata.get("adapter", "")) if isinstance(metadata, dict) else "",
        "adapter_version": str(metadata.get("adapter_version", ""))
        if isinstance(metadata, dict)
        else "",
        "canonical_schema_version": str(row.get("schema_version", "tool_use_ir_v1")),
        "canonical_hash": str(row.get("canonical_hash", "")),
        "quality_status": str(metadata.get("parse_status", "VALID"))
        if isinstance(metadata, dict)
        else "VALID",
        "contamination_status": str(metadata.get("contamination_status", "UNASSESSED"))
        if isinstance(metadata, dict)
        else "UNASSESSED",
        "behavior_decision": str(behavior.get("decision", "UNKNOWN")),
        "behavior_confidence": str(behavior.get("confidence", "unknown")),
        "behavior_capabilities": _json(behavior.get("capabilities", [])),
        "tools": _json(row.get("tools", [])),
        "messages": _json(row.get("messages", [])),
        "metadata": _json(metadata),
    }


def build_release(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Build a deterministic local Hugging Face-compatible Parquet staging release."""
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    config = _load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    shard_size = int(config.get("shard_size", 1000))
    sources: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total = 0
    for source in config.get("included_sources", []):
        artifact = root / source["artifact"]
        manifest = artifact / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"source manifest missing: {manifest}")
        redistribution = source.get("redistribution_status")
        if not source.get("source_revision"):
            raise ValueError(f"source revision missing for {source['id']}")
        if not source.get("upstream_license"):
            raise ValueError(f"upstream license missing for {source['id']}")
        if redistribution not in {"PERMITTED_WITH_ATTRIBUTION", "REDISTRIBUTION_WITH_ATTRIBUTION"}:
            excluded.append(
                {
                    "source": source["id"],
                    "source_revision": source["source_revision"],
                    "reason": redistribution or "REDISTRIBUTION_STATUS_UNKNOWN",
                    "manifest": str(manifest.relative_to(root)).replace("\\", "/"),
                    "input_manifest_sha256": _sha256(manifest),
                }
            )
            continue
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        source_count = 0
        shard_names: list[str] = []
        source_slug = source["id"].replace("/", "-").replace(" ", "-")
        for row in iter_materialized_rows(artifact):
            rows.append(_parquet_row(_restore_row(row), source))
            source_count += 1
            total += 1
            if len(rows) >= shard_size:
                name = f"{source_slug}-{len(shard_names):06d}.parquet"
                pq.write_table(
                    pa.Table.from_pylist(rows),
                    output / name,
                    compression="zstd",
                    write_statistics=False,
                    version="2.6",
                )
                shard_names.append(name)
                rows = []
        if rows:
            name = f"{source_slug}-{len(shard_names):06d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(rows),
                output / name,
                compression="zstd",
                write_statistics=False,
                version="2.6",
            )
            shard_names.append(name)
        sources.append(
            {
                "source": source["id"],
                "source_revision": source["source_revision"],
                "upstream_license": source["upstream_license"],
                "redistribution_status": redistribution,
                "upstream_access_mode": source.get("upstream_access_mode", "public"),
                "downstream_access_requirement": source.get(
                    "downstream_access_requirement", "public_allowed"
                ),
                "attribution_required": source.get("attribution_required", True),
                "citation_required": source.get("citation_required", True),
                "citation_target": source.get("citation_target", "source"),
                "modifications_disclosed": source.get("modifications_disclosed", True),
                "modification_status": source.get("modification_status", "NORMALIZED_DERIVATIVE"),
                "input_manifest": str(manifest.relative_to(root)).replace("\\", "/"),
                "input_manifest_sha256": _sha256(manifest),
                "records": source_count,
                "adapter_version": manifest_data.get("config", {}).get(
                    "adapter_version", source.get("adapter_version", "see source manifest")
                ),
                "raw_count": manifest_data.get("counts", {}).get("source_rows", "unknown"),
                "output_shards": shard_names,
                "raw_manifest_counts": manifest_data.get("counts", {}),
            }
        )
    source_date = os.environ.get("SOURCE_DATE_EPOCH", "0")
    release_manifest = {
        "schema_version": 1,
        "release_name": config["release_name"],
        "release_version": config["release_version"],
        "release_class": config["release_class"],
        "hub_repository": config["hub_repository"],
        "opengrad_git_commit": _git_commit(root),
        "canonical_schema_version": config["canonical_schema_version"],
        "behavior_taxonomy_version": config["behavior_taxonomy_version"],
        "generation_timestamp": source_date,
        "sources": sources,
        "excluded_sources": excluded,
        "record_count": total,
        "release_filters": {
            "quality": "valid_only",
            "preference": "excluded",
            "evaluation": "excluded",
            "gated": "exclude_uncleared",
        },
        "dedup_policy": "upstream canonical manifests; no release-time semantic deduplication",
        "contamination_policy": "confirmed contamination and unassessed records are not silently relabeled",
        "evaluation_exclusion_policy": "When2Call preference, MCQ, and LLM-judge artifacts are excluded",
        "output_shards": [],
    }
    for path in sorted(output.glob("*.parquet")):
        release_manifest["output_shards"].append(
            {"file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    (output / "release-manifest.json").write_text(_json(release_manifest) + "\n", encoding="utf-8")
    card_dir = root / config.get("card_directory", _DEFAULT_CARD_DIRECTORY)
    card = (card_dir / "README.template.md").read_text(encoding="utf-8")
    source_rows: list[str] = []
    for item in sources:
        raw = item.get("raw_manifest_counts", {})
        source_cfg = next(x for x in config["included_sources"] if x["id"] == item["source"])
        source_rows.append(
            "| {source} | canonical SFT | {repo} | `{revision}` | {raw} | {retained} | {published} | {license} | {adapter} |".format(
                source=item["source"],
                repo=_SOURCE_REPOS.get(item["source"], ""),
                revision=item["source_revision"],
                raw=raw.get(
                    "source_rows",
                    raw.get("raw", raw.get("total", item.get("raw_count", "unknown"))),
                ),
                retained=raw.get("training_eligible", raw.get("valid", item["records"])),
                published=item["records"],
                license=f"{source_cfg['upstream_license']} ({source_cfg['redistribution_status']})",
                adapter=item.get(
                    "adapter_version", source_cfg.get("adapter_version", "see source manifest")
                ),
            )
        )
    card = card.replace("{{SOURCE_TABLE}}", "\n".join(source_rows))
    card = card.replace("{{RECORD_COUNT}}", str(total)).replace(
        "{{EXCLUDED_SOURCES}}", _json(excluded)
    )
    (output / "README.md").write_text(card, encoding="utf-8")
    for name in ("source-licenses.md", "CITATIONS.bib"):
        source = card_dir / name
        (output / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return release_manifest


def validate_release(staging: Path) -> list[str]:
    """Validate a local release staging directory and fail closed on integrity errors."""
    errors: list[str] = []
    manifest_path = staging / "release-manifest.json"
    if not manifest_path.exists():
        return ["release-manifest.json missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["release-manifest.json invalid JSON"]
    required = {
        "release_name",
        "release_version",
        "sources",
        "excluded_sources",
        "record_count",
        "output_shards",
    }
    errors.extend(f"manifest field missing: {x}" for x in sorted(required - manifest.keys()))
    if not (staging / "README.md").exists():
        errors.append("dataset card missing")
    if not (staging / "source-licenses.md").exists():
        errors.append("source-licenses.md missing")
    if not (staging / "CITATIONS.bib").exists():
        errors.append("CITATIONS.bib missing")
    seen = 0
    for source in manifest.get("sources", []):
        input_manifest = Path(source.get("input_manifest", ""))
        if not input_manifest.exists():
            errors.append(f"input manifest missing: {input_manifest}")
        elif _sha256(input_manifest) != source.get("input_manifest_sha256"):
            errors.append(f"input manifest checksum mismatch: {input_manifest}")
    for item in manifest.get("output_shards", []):
        path = staging / item["file"]
        if not path.exists():
            errors.append(f"output shard missing: {path.name}")
            continue
        if _sha256(path) != item["sha256"]:
            errors.append(f"output checksum mismatch: {path.name}")
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(path, columns=["source_dataset", "source_split"])
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"output parquet unreadable: {path.name}: {exc}")
            continue
        seen += table.num_rows
        forbidden = {"preference", "train_pref", "mcq", "llm_judge", "evaluation"}
        if any(value in forbidden for value in table.column("source_split").to_pylist()):
            errors.append(f"non-SFT split present: {path.name}")
        if any(
            value in {"when2call-preference", "when2call-mcq", "when2call-llm-judge"}
            for value in table.column("source_dataset").to_pylist()
        ):
            errors.append(f"excluded artifact present: {path.name}")
    if seen != manifest.get("record_count"):
        errors.append(
            f"record count mismatch: manifest={manifest.get('record_count')} physical={seen}"
        )
    for item in manifest.get("excluded_sources", []):
        if item.get("source") in {"when2call-preference", "when2call-mcq", "when2call-llm-judge"}:
            continue
    return errors
