"""Benchmark contamination registry managing contamination state, licenses, and scan history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.benchmarks.registry import BenchmarkRegistry


@dataclass
class ContaminationEntry:
    benchmark_id: str
    benchmark_name: str
    benchmark_version: str
    exact_source_revision: str
    license: str
    local_namespace: str
    prohibit_training: bool
    scan_status: str  # "UNSCANNED", "CLEAN", "CONTAMINATED", "SUSPICIOUS_CANDIDATES"
    scan_date: str = ""
    training_corpus_fingerprint: str = ""
    levels_completed: dict[str, str] = field(default_factory=dict)
    findings_count: int = 0
    audit_queue: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "benchmark_version": str(self.benchmark_version),
            "exact_source_revision": str(self.exact_source_revision),
            "license": self.license,
            "local_namespace": self.local_namespace,
            "prohibit_training": self.prohibit_training,
            "scan_status": self.scan_status,
            "scan_date": self.scan_date,
            "training_corpus_fingerprint": self.training_corpus_fingerprint,
            "levels_completed": self.levels_completed,
            "findings_count": self.findings_count,
            "audit_queue": self.audit_queue,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationEntry:
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            benchmark_name=str(data.get("benchmark_name", data["benchmark_id"])),
            benchmark_version=str(data.get("benchmark_version", "unknown")),
            exact_source_revision=str(data.get("exact_source_revision", "unknown")),
            license=str(data.get("license", "unknown")),
            local_namespace=str(data.get("local_namespace", "benchmarks")),
            prohibit_training=bool(data.get("prohibit_training", True)),
            scan_status=str(data.get("scan_status", "UNSCANNED")),
            scan_date=str(data.get("scan_date", "")),
            training_corpus_fingerprint=str(data.get("training_corpus_fingerprint", "")),
            levels_completed=dict(data.get("levels_completed") or {}),
            findings_count=int(data.get("findings_count", 0)),
            audit_queue=list(data.get("audit_queue") or []),
        )


class ContaminationRegistry:
    """Manages benchmark contamination states and scan provenance."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.storage_file = self.root / "reports" / "data" / "benchmark_contamination_registry.json"
        self._entries: dict[str, ContaminationEntry] = {}
        self._load()

    def _load(self) -> None:
        # Seed from registry/benchmarks.yaml
        bm_reg = BenchmarkRegistry(self.root)
        for bm in bm_reg.list_all():
            self._entries[bm.id] = ContaminationEntry(
                benchmark_id=bm.id,
                benchmark_name=bm.name,
                benchmark_version=bm.version or "v1.0",
                exact_source_revision=bm.commit_sha or "HEAD",
                license=bm.license or "PROPRIETARY_OR_UNSPECIFIED",
                local_namespace=f"benchmarks/{bm.id}",
                prohibit_training=bm.prohibit_training,
                scan_status="UNSCANNED",
                levels_completed={
                    "level_1_exact_canonical_hash": "UNSCANNED",
                    "level_2_normalized_prompt_answer_hash": "UNSCANNED",
                    "level_3_ngram_minhash_near_duplicate": "UNSCANNED",
                    "level_4_semantic_similarity_candidates": "UNSCANNED",
                    "level_5_manual_audit_queue": "UNSCANNED",
                },
            )

        # Merge persisted scan results if file exists
        if self.storage_file.exists():
            try:
                data = json.loads(self.storage_file.read_text(encoding="utf-8"))
                for entry_raw in data.get("entries", []):
                    entry = ContaminationEntry.from_dict(entry_raw)
                    self._entries[entry.benchmark_id] = entry
            except (OSError, json.JSONDecodeError, KeyError):
                pass

    def get(self, benchmark_id: str) -> ContaminationEntry:
        canonical = benchmark_id.replace("_", "-")
        if canonical in self._entries:
            return self._entries[canonical]
        if benchmark_id in self._entries:
            return self._entries[benchmark_id]
        raise KeyError(f"Unknown benchmark in contamination registry: {benchmark_id}")

    def list_all(self) -> list[ContaminationEntry]:
        return list(self._entries.values())

    def update_scan(
        self,
        benchmark_id: str,
        *,
        corpus_fingerprint: str,
        scan_status: str,
        levels_completed: dict[str, str],
        audit_queue: list[dict[str, Any]],
    ) -> None:
        entry = self.get(benchmark_id)
        updated = ContaminationEntry(
            benchmark_id=entry.benchmark_id,
            benchmark_name=entry.benchmark_name,
            benchmark_version=entry.benchmark_version,
            exact_source_revision=entry.exact_source_revision,
            license=entry.license,
            local_namespace=entry.local_namespace,
            prohibit_training=entry.prohibit_training,
            scan_status=scan_status,
            scan_date=datetime.now(UTC).isoformat(),
            training_corpus_fingerprint=corpus_fingerprint,
            levels_completed=levels_completed,
            findings_count=len(audit_queue),
            audit_queue=audit_queue,
        )
        self._entries[entry.benchmark_id] = updated
        self.save()

    def save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "last_updated": datetime.now(UTC).isoformat(),
            "total_benchmarks": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        self.storage_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
