"""Benchmark contamination registry: each benchmark's scan state, licence and provenance.

The registry records what a scan measured and refuses what a scan cannot establish. A recorded scan
must name the sha256 of both the benchmark data and the training corpus it compared, and its level 5
must be ``NOT_RUN``: level 5 is a human adjudication, and until 2026-09-24 this registry held a
``bfcl-v4`` entry marked ``CLEAN`` with every level ``COMPLETED`` after a scan of placeholder tasks
against two fixture prompts, under the fingerprint ``"sample-or-materialized-fingerprint"``
(`reports/ERRATA.md` §25).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.benchmarks.contamination.scanner import BenchmarkScanReport
from opengrad.benchmarks.registry import BenchmarkRegistry
from opengrad.contamination.levels import (
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LEVEL_4,
    LEVEL_5,
    NOT_RUN,
    STATUS_MEASURED,
    STATUS_REVIEW_REQUIRED,
)

UNSCANNED = "UNSCANNED"
SCAN_STATUSES = (UNSCANNED, STATUS_MEASURED, STATUS_REVIEW_REQUIRED)
LEVELS = (LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def unscanned_levels() -> dict[str, str]:
    return dict.fromkeys(LEVELS, NOT_RUN)


@dataclass
class ContaminationEntry:
    benchmark_id: str
    benchmark_name: str
    benchmark_version: str
    exact_source_revision: str
    license: str
    local_namespace: str
    prohibit_training: bool
    scan_status: str = UNSCANNED
    scan_date: str = ""
    benchmark_data_sha256: str = ""
    training_corpus_sha256: str = ""
    benchmark_samples: int = 0
    training_samples: int = 0
    levels: dict[str, str] = field(default_factory=unscanned_levels)
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
            "benchmark_data_sha256": self.benchmark_data_sha256,
            "training_corpus_sha256": self.training_corpus_sha256,
            "benchmark_samples": self.benchmark_samples,
            "training_samples": self.training_samples,
            "levels": self.levels,
            "findings_count": self.findings_count,
            "audit_queue": self.audit_queue,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationEntry:
        entry = cls(
            benchmark_id=str(data["benchmark_id"]),
            benchmark_name=str(data.get("benchmark_name", data["benchmark_id"])),
            benchmark_version=str(data.get("benchmark_version", "unknown")),
            exact_source_revision=str(data.get("exact_source_revision", "unknown")),
            license=str(data.get("license", "unknown")),
            local_namespace=str(data.get("local_namespace", "benchmarks")),
            prohibit_training=bool(data.get("prohibit_training", True)),
            scan_status=str(data.get("scan_status", UNSCANNED)),
            scan_date=str(data.get("scan_date", "")),
            benchmark_data_sha256=str(data.get("benchmark_data_sha256", "")),
            training_corpus_sha256=str(data.get("training_corpus_sha256", "")),
            benchmark_samples=int(data.get("benchmark_samples", 0)),
            training_samples=int(data.get("training_samples", 0)),
            levels=dict(data.get("levels") or unscanned_levels()),
            findings_count=int(data.get("findings_count", 0)),
            audit_queue=list(data.get("audit_queue") or []),
        )
        problems = entry.problems()
        if problems:
            raise ValueError(f"contamination registry entry {entry.benchmark_id}: {problems}")
        return entry

    def problems(self) -> list[str]:
        """Why this entry claims more than a scan can establish; empty when it does not."""
        problems: list[str] = []
        if self.scan_status not in SCAN_STATUSES:
            problems.append(f"scan_status {self.scan_status!r} is not one of {SCAN_STATUSES}")
        if set(self.levels) != set(LEVELS):
            problems.append(f"levels must be exactly {LEVELS}")
        if self.levels.get(LEVEL_5) != NOT_RUN:
            problems.append("level 5 is a human adjudication; a scan records it NOT_RUN")
        if self.scan_status != UNSCANNED:
            for name in ("benchmark_data_sha256", "training_corpus_sha256"):
                if not _SHA256.fullmatch(getattr(self, name)):
                    problems.append(f"{name} {getattr(self, name)!r} is not a sha256")
            if self.benchmark_samples <= 0 or self.training_samples <= 0:
                problems.append("a recorded scan compared at least one sample on each side")
        return problems


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
            )

        # Merge persisted scan results. A malformed file or entry is an error, not an empty
        # registry: silently dropping it would turn a recorded finding back into UNSCANNED.
        if self.storage_file.exists():
            data = json.loads(self.storage_file.read_text(encoding="utf-8"))
            for entry_raw in data.get("entries", []):
                entry = ContaminationEntry.from_dict(entry_raw)
                self._entries[entry.benchmark_id] = entry

    def get(self, benchmark_id: str) -> ContaminationEntry:
        canonical = benchmark_id.replace("_", "-")
        if canonical in self._entries:
            return self._entries[canonical]
        if benchmark_id in self._entries:
            return self._entries[benchmark_id]
        raise KeyError(f"Unknown benchmark in contamination registry: {benchmark_id}")

    def list_all(self) -> list[ContaminationEntry]:
        return list(self._entries.values())

    def record_scan(self, report: BenchmarkScanReport) -> ContaminationEntry:
        """Record a scan report; refuses one that claims more than the scan measured."""
        entry = self.get(report.benchmark_id)
        updated = ContaminationEntry(
            benchmark_id=entry.benchmark_id,
            benchmark_name=entry.benchmark_name,
            benchmark_version=entry.benchmark_version,
            exact_source_revision=entry.exact_source_revision,
            license=entry.license,
            local_namespace=entry.local_namespace,
            prohibit_training=entry.prohibit_training,
            scan_status=report.status,
            scan_date=datetime.now(UTC).isoformat(),
            benchmark_data_sha256=report.benchmark_data_sha256,
            training_corpus_sha256=report.training_corpus_sha256,
            benchmark_samples=report.benchmark_samples,
            training_samples=report.training_samples,
            levels=dict(report.levels),
            findings_count=len(report.audit_queue),
            audit_queue=list(report.audit_queue),
        )
        problems = updated.problems()
        if problems:
            raise ValueError(f"refusing to record the {entry.benchmark_id} scan: {problems}")
        self._entries[entry.benchmark_id] = updated
        self.save()
        return updated

    def save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "last_updated": datetime.now(UTC).isoformat(),
            "total_benchmarks": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        self.storage_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
