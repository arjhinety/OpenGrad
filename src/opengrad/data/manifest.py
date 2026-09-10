"""Immutable dataset manifest schema and fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class DatasetManifest:
    dataset_name: str
    source: str
    upstream_revision: str
    adapter: str
    schema_version: str
    split: str
    raw_record_count: int
    valid_record_count: int
    rejected_record_count: int
    deduplicated_count: int
    final_count: int
    checksum: str
    deterministic_fingerprint: str
    tokenizer: str
    token_statistics: dict[str, Any] = field(default_factory=dict)
    contamination_status: str = "UNSCANNED"
    license_constraints: str = "Apache-2.0"
    generated_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    transformation_pipeline: list[str] = field(default_factory=list)
    filtering_operations: list[str] = field(default_factory=list)
    exclusion_rules: list[str] = field(default_factory=list)
    shards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source": self.source,
            "upstream_revision": self.upstream_revision,
            "adapter": self.adapter,
            "schema_version": self.schema_version,
            "split": self.split,
            "raw_record_count": self.raw_record_count,
            "valid_record_count": self.valid_record_count,
            "rejected_record_count": self.rejected_record_count,
            "deduplicated_count": self.deduplicated_count,
            "final_count": self.final_count,
            "checksum": self.checksum,
            "deterministic_fingerprint": self.deterministic_fingerprint,
            "tokenizer": self.tokenizer,
            "token_statistics": self.token_statistics,
            "contamination_status": self.contamination_status,
            "license_constraints": self.license_constraints,
            "generated_timestamp": self.generated_timestamp,
            "transformation_pipeline": self.transformation_pipeline,
            "filtering_operations": self.filtering_operations,
            "exclusion_rules": self.exclusion_rules,
            "shards": self.shards,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        return cls(
            dataset_name=str(data["dataset_name"]),
            source=str(data["source"]),
            upstream_revision=str(data.get("upstream_revision", "unknown")),
            adapter=str(data.get("adapter", "canonical")),
            schema_version=str(data.get("schema_version", "1.0")),
            split=str(data.get("split", "train")),
            raw_record_count=int(data.get("raw_record_count", 0)),
            valid_record_count=int(data.get("valid_record_count", 0)),
            rejected_record_count=int(data.get("rejected_record_count", 0)),
            deduplicated_count=int(data.get("deduplicated_count", 0)),
            final_count=int(data.get("final_count", 0)),
            checksum=str(data.get("checksum", "")),
            deterministic_fingerprint=str(data.get("deterministic_fingerprint", "")),
            tokenizer=str(data.get("tokenizer", "Qwen/Qwen3.5-2B")),
            token_statistics=dict(data.get("token_statistics") or {}),
            contamination_status=str(data.get("contamination_status", "UNSCANNED")),
            license_constraints=str(data.get("license_constraints", "Apache-2.0")),
            generated_timestamp=str(data.get("generated_timestamp", "")),
            transformation_pipeline=list(data.get("transformation_pipeline") or []),
            filtering_operations=list(data.get("filtering_operations") or []),
            exclusion_rules=list(data.get("exclusion_rules") or []),
            shards=list(data.get("shards") or []),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def compute_dataset_fingerprint(record_hashes: list[str]) -> str:
    """Compute a deterministic, order-independent dataset fingerprint."""
    sorted_hashes = sorted(record_hashes)
    return hashlib.sha256("\n".join(sorted_hashes).encode("utf-8")).hexdigest()
