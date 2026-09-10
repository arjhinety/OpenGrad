"""Canonical training-corpus access for the trainer backends.

The SFT/DPO/OPD backends all train on the published canonical release
(``arrochi112/OpenGrad-ToolPolicy-Canonical-v1``). That corpus is *model-independent*: records
carry canonical `tools`/`messages`/`metadata`, not model-rendered text. Every backend must
therefore pass records through the model-family renderer rather than training on the stored
strings.

This module is the single place that:

* reads the release shards in a deterministic, reproducible order,
* reconstructs the canonical IR (`ToolConversation`) with its provenance,
* hands out record identities that are stable across runs,
* measures the rendered token-length distribution used to set and justify sequence policy.

It deliberately contains no model-specific logic: the renderer is passed in.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.data.canonical import ToolConversation

DEFAULT_RELEASE_DIR = Path(".release/hf/toolpolicy-canonical-v1")
RELEASE_MANIFEST = "release-manifest.json"


@dataclass(frozen=True)
class TrainingRecord:
    """One canonical training record plus the provenance needed to audit it."""

    conversation: ToolConversation
    canonical_hash: str
    source_dataset: str
    source_split: str
    behavior_decision: str
    shard: str

    def provenance(self) -> dict[str, Any]:
        return {
            "canonical_hash": self.canonical_hash,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "behavior_decision": self.behavior_decision,
            "shard": self.shard,
        }


def release_identity(root: Path, release_dir: Path | None = None) -> dict[str, Any]:
    """Identity of the assembled corpus, read from the release manifest.

    The manifest's sha256 is the pinned dataset hash the experiment config declares, so this
    is also the check that the corpus on disk is the one the config describes.
    """
    import hashlib

    base = (root / (release_dir or DEFAULT_RELEASE_DIR)).resolve()
    manifest_path = base / RELEASE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical release manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "release_dir": str(base.relative_to(root)) if base.is_relative_to(root) else str(base),
        "hub_repository": manifest.get("hub_repository"),
        "release_version": manifest.get("release_version"),
        "record_count": manifest.get("record_count"),
        "shards": len(manifest.get("output_shards") or []),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "build_commit": manifest.get("opengrad_git_commit"),
        "evaluation_exclusion_policy": manifest.get("evaluation_exclusion_policy"),
    }


def verify_release_hash(
    root: Path, expected: str, release_dir: Path | None = None
) -> dict[str, Any]:
    """Fail closed when the corpus on disk is not the one the experiment pins."""
    identity = release_identity(root, release_dir)
    if identity["manifest_sha256"] != expected:
        raise ValueError(
            "canonical corpus does not match the pinned dataset hash: "
            f"config={expected} on_disk={identity['manifest_sha256']}"
        )
    return identity


def _shard_files(base: Path) -> list[Path]:
    """Release shards in a deterministic order (sorted by name)."""
    shards = sorted(path for path in base.glob("*.parquet") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"no release shards under {base}")
    return shards


def _restore(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def iter_training_records(
    root: Path,
    *,
    release_dir: Path | None = None,
    limit: int | None = None,
) -> Iterator[TrainingRecord]:
    """Stream canonical training records in a stable order.

    Ordering is shard-name then row order, which is deterministic for a fixed corpus and does
    not depend on filesystem iteration order. The training loop is responsible for shuffling
    with a seeded generator so that the on-disk order stays a reproducible input.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    base = (root / (release_dir or DEFAULT_RELEASE_DIR)).resolve()
    seen: Counter[str] = Counter()
    emitted = 0
    for shard in _shard_files(base):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                digest = str(row.get("canonical_hash") or "")
                if not digest:
                    raise ValueError(f"training row without canonical_hash in {shard.name}")
                # `opengrad_id` is literally "unknown" for every LoopTool row, so identity
                # comes from the canonical hash, disambiguated for the few genuine duplicates.
                seen[digest] += 1
                suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
                record_id = f"{digest}{suffix}"
                metadata = _restore(row.get("metadata"))
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                metadata.setdefault("split", str(row.get("source_split") or "train"))
                conversation = ToolConversation(
                    id=record_id,
                    source=str(row.get("source_dataset") or "unknown"),
                    tools=_restore(row.get("tools")) or [],
                    messages=_restore(row.get("messages")) or [],
                    metadata=metadata,
                )
                yield TrainingRecord(
                    conversation=conversation,
                    canonical_hash=digest,
                    source_dataset=str(row.get("source_dataset") or "unknown"),
                    source_split=str(row.get("source_split") or "train"),
                    behavior_decision=str(row.get("behavior_decision") or ""),
                    shard=shard.name,
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def source_distribution(records: Iterator[TrainingRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.source_dataset for record in records).items()))


def behavior_distribution(records: Iterator[TrainingRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.behavior_decision for record in records).items()))


@dataclass
class LengthDistribution:
    """Rendered token-length statistics for the training corpus."""

    total: int = 0
    rendered: int = 0
    render_failures: dict[str, int] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)

    def percentile(self, fraction: float) -> int:
        if not self.lengths:
            return 0
        ordered = sorted(self.lengths)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.lengths)
        n = len(ordered)
        if not n:
            return {"records": 0}
        return {
            "records_measured": n,
            "records_total": self.total,
            "render_failures": dict(sorted(self.render_failures.items())),
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "max": ordered[-1],
            "min": ordered[0],
            "mean": round(sum(ordered) / n, 2),
            "fraction_le_2048": round(sum(1 for value in ordered if value <= 2048) / n, 6),
            "fraction_gt_2048": round(sum(1 for value in ordered if value > 2048) / n, 6),
            "fraction_gt_4096": round(sum(1 for value in ordered if value > 4096) / n, 6),
        }
