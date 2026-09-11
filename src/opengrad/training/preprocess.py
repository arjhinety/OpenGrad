"""Preprocess the canonical corpus into rendered, assistant-masked training samples.

Rendering is the expensive step (template rendering plus up to one tokenization per message
boundary), so it happens once and is cached. The cache records the exact identities it was
built from — corpus manifest hash, renderer, tokenizer revision, template hash, sequence
length — so a stale cache is detected rather than silently reused.

Parallelism is shard-level: each worker owns a shard and returns its samples. That keeps the
expensive part out of the assembler process and makes the pass scale with cores rather than
with the size of the corpus.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.training.dataset import (
    DEFAULT_RELEASE_DIR,
    TrainingRecord,
    release_identity,
)
from opengrad.training.sft_data import (
    TRAINABLE,
    SupervisedSample,
    build_sample,
    overflow_report,
)

CACHE_VERSION = 1
SAMPLES_FILE = "sft_samples.parquet"
CACHE_MANIFEST = "sft_cache.json"


def default_cache_dir(
    root: Path, model_id: str, max_seq_length: int, release_dir: Path | None = None
) -> Path:
    """Where a run keeps its rendered sample cache.

    Keyed by the rendering contract (model and sequence length) rather than by experiment, so a
    micro-run and the full run of the same model share one cache instead of each paying for a
    full corpus render. Correctness does not rest on the directory name: the identity recorded
    inside the cache covers the corpus hash, template hash, tokenizer revision, and window, and
    a mismatch forces a rebuild.

    Under ``data/processed/`` because it is a large, regenerable derived artifact.
    """
    slug = model_id.replace("/", "-")
    base = root / "data/processed" / f"sft-cache-{slug}-{max_seq_length}"
    # Two corpora rendered with the same model and window would otherwise share one cache
    # directory and each rebuild would evict the other's samples. The default release keeps the
    # original name so an existing cache still resolves.
    if release_dir is None or release_dir.resolve() == (root / DEFAULT_RELEASE_DIR).resolve():
        return base
    return base.with_name(f"{base.name}-{release_dir.resolve().name}")


@dataclass
class CacheIdentity:
    """Everything that must match for a cache to be reusable."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    renderer: str
    template_hash: str
    max_seq_length: int
    corpus_manifest_sha256: str
    # Which supervision kinds this sample set contains. Part of the identity because filtering
    # changes the sample stream: reusing an unfiltered cache for a filtered run would silently
    # train on records the config excluded.
    supervision_include: tuple[str, ...] = ()
    cache_version: int = CACHE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_version": self.cache_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "renderer": self.renderer,
            "template_hash": self.template_hash,
            "max_seq_length": self.max_seq_length,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "supervision_include": list(self.supervision_include),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheIdentity:
        include = data.get("supervision_include") or []
        return cls(
            model_id=str(data.get("model_id", "")),
            model_revision=str(data.get("model_revision", "")),
            tokenizer_revision=str(data.get("tokenizer_revision", "")),
            renderer=str(data.get("renderer", "")),
            template_hash=str(data.get("template_hash", "")),
            max_seq_length=int(data.get("max_seq_length", 0)),
            corpus_manifest_sha256=str(data.get("corpus_manifest_sha256", "")),
            # Restored, not dropped: an identity that forgets its filter would compare equal
            # against a differently-filtered cache and be reused.
            supervision_include=tuple(str(item) for item in include),
            cache_version=int(data.get("cache_version", 0)),
        )


_RENDERER: Any = None


def _worker_renderer() -> Any:
    """Per-process renderer, built from the environment the parent set.

    The thread caps are re-applied here because a spawned worker may import the tokenizer
    before the parent's environment is observed by every native library.
    """
    global _RENDERER
    if _RENDERER is None:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ[variable] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from opengrad.data.renderers import Qwen35_2BRenderer

        _RENDERER = Qwen35_2BRenderer(
            revision=os.environ["OPENGRAD_RENDER_REVISION"], enable_thinking=False
        )
        _RENDERER._load()
    return _RENDERER


def _process_shard(payload: tuple[str, int, tuple[str, ...]]) -> dict[str, Any]:
    """Render one shard. Runs in a worker process."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    shard_name, max_seq_length, supervision_include = payload
    shard_path = Path(shard_name)
    renderer = _worker_renderer()
    samples: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    # Records dropped because their declared supervision kind is not selected. Counted
    # separately from failures: an excluded record is not a broken one.
    supervision_excluded: Counter[str] = Counter()

    parquet = pq.ParquetFile(shard_path)
    seen: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            digest = str(row.get("canonical_hash") or "")
            if not digest:
                continue
            seen[digest] += 1
            suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
            record = TrainingRecord(
                conversation=_conversation(row, f"{digest}{suffix}"),
                canonical_hash=digest,
                source_dataset=str(row.get("source_dataset") or "unknown"),
                source_split=str(row.get("source_split") or "train"),
                behavior_decision=str(row.get("behavior_decision") or ""),
                shard=shard_path.name,
            )
            sample = build_sample(
                renderer,
                record.conversation,
                max_seq_length=max_seq_length,
                canonical_hash=record.canonical_hash,
                source_dataset=record.source_dataset,
                behavior_decision=record.behavior_decision,
            )
            dispositions[sample.status] += 1
            source_counts[record.source_dataset] += 1
            if not sample.trainable:
                reason = str(sample.detail.get("reason") or sample.status)
                reasons[reason[:160]] += 1
                continue
            if supervision_include and sample.supervision_kind not in supervision_include:
                supervision_excluded[sample.supervision_kind or "UNCLASSIFIED"] += 1
                continue
            samples.append(
                {
                    "record_id": sample.record_id,
                    "canonical_hash": sample.canonical_hash,
                    "source_dataset": sample.source_dataset,
                    "behavior_decision": sample.behavior_decision,
                    "status": sample.status,
                    "supervision_kind": sample.supervision_kind,
                    "tokens": sample.tokens,
                    "supervised": sample.supervised,
                    "template_hash": str(sample.detail.get("template_hash") or ""),
                }
            )
    return {
        "shard": shard_path.name,
        "samples": samples,
        "dispositions": dict(dispositions),
        "reasons": dict(reasons),
        "source_counts": dict(source_counts),
        "supervision_excluded": dict(supervision_excluded),
    }


def _conversation(row: dict[str, Any], record_id: str) -> Any:
    from opengrad.data.canonical import ToolConversation

    def restore(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    metadata = restore(row.get("metadata"))
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.setdefault("split", str(row.get("source_split") or "train"))
    return ToolConversation(
        id=record_id,
        source=str(row.get("source_dataset") or "unknown"),
        tools=restore(row.get("tools")) or [],
        messages=restore(row.get("messages")) or [],
        metadata=metadata,
    )


def cache_is_current(cache_dir: Path, expected: CacheIdentity) -> bool:
    manifest_path = cache_dir / CACHE_MANIFEST
    samples_path = cache_dir / SAMPLES_FILE
    if not manifest_path.is_file() or not samples_path.is_file():
        return False
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return CacheIdentity.from_dict(stored.get("identity") or {}) == expected


def preprocess_corpus(
    root: Path,
    *,
    cache_dir: Path,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    renderer_name: str,
    max_seq_length: int,
    release_dir: Path | None = None,
    workers: int | None = None,
    progress: Any = None,
    supervision_include: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Render the whole corpus into a cached sample set. Idempotent per identity.

    ``supervision_include`` restricts the sample stream to the named supervision kinds. It is
    part of the cache identity, so an ablation cannot silently reuse a differently-filtered
    sample set.
    """

    from opengrad.data.renderers import Qwen35_2BRenderer

    root = root.resolve()
    base = (root / (release_dir or DEFAULT_RELEASE_DIR)).resolve()
    identity_info = release_identity(root, release_dir)

    probe = Qwen35_2BRenderer(revision=model_revision, enable_thinking=False)
    template_hash = _template_hash(probe)
    expected = CacheIdentity(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        renderer=renderer_name,
        template_hash=template_hash,
        max_seq_length=max_seq_length,
        corpus_manifest_sha256=str(identity_info["manifest_sha256"]),
        supervision_include=tuple(sorted(supervision_include)),
    )
    if cache_is_current(cache_dir, expected):
        return {"status": "CACHE_HIT", "identity": expected.to_dict(), **identity_info}

    shards = sorted(path for path in base.glob("*.parquet") if path.is_file())
    if not shards:
        raise FileNotFoundError(f"no release shards under {base}")

    # Parallelism is process-level via explicit subprocesses, not a worker pool.
    #
    # Two properties of this workload make a pool the wrong tool. First, the Hugging Face
    # tokenizer builds a rayon pool sized to every core, so N in-process workers each request N
    # cores: measured on this corpus that oversubscription made each shard ~15x slower than it
    # is alone. Second, a forked worker inherits the parent's already-started native thread
    # pools, which can deadlock instead of failing. A fresh, thread-capped interpreter per
    # worker has neither problem, and each worker is then exactly as fast as running a shard by
    # hand.
    worker_count = workers or max(1, min(len(shards), (os.cpu_count() or 4) - 1))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cache_dir.mkdir(parents=True, exist_ok=True)
    part_dir = cache_dir / "parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True)
    staging = cache_dir / f"{SAMPLES_FILE}.tmp"

    # Interleaved assignment, not contiguous: the sources differ by an order of magnitude in
    # cost, so contiguous chunks would leave one worker with every expensive shard.
    chunks = [shards[index::worker_count] for index in range(worker_count)]
    environment = {
        **os.environ,
        "OPENGRAD_RENDER_REVISION": model_revision,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    processes = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        command = [
            sys.executable,
            "-m",
            "opengrad.training.preprocess",
            "--worker",
            str(index),
            "--out",
            str(part_dir),
            "--max-seq-length",
            str(max_seq_length),
            "--revision",
            model_revision,
            "--supervision-include",
            ",".join(sorted(supervision_include)),
            *[str(path) for path in chunk],
        ]
        processes.append(subprocess.Popen(command, env=environment, cwd=str(root)))

    total_shards = len(shards)
    while any(process.poll() is None for process in processes):
        if progress is not None:
            progress(_completed_shards(part_dir), total_shards, _part_sample_count(part_dir))
        time.sleep(2)
    if progress is not None:
        progress(_completed_shards(part_dir), total_shards, _part_sample_count(part_dir))

    failures = [process.returncode for process in processes if process.returncode != 0]
    if failures:
        raise RuntimeError(
            f"{len(failures)} preprocessing worker(s) failed with exit codes {failures}; "
            f"partial output is retained under {part_dir}"
        )

    merged = _merge_parts(part_dir, staging)
    dispositions = Counter(merged["dispositions"])
    reasons = Counter(merged["reasons"])
    source_counts = Counter(merged["source_counts"])
    source_trainable = Counter(merged["source_trainable"])
    behavior_trainable = Counter(merged["behavior_trainable"])
    lengths = merged["lengths"]
    supervised_total = int(merged["supervised_tokens"])
    token_total = int(merged["total_tokens"])
    written = int(merged["written"])

    if written == 0:
        staging.unlink(missing_ok=True)
        raise RuntimeError("no trainable samples were produced from the canonical corpus")

    staging.replace(cache_dir / SAMPLES_FILE)
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        if not ordered:
            return 0
        return int(ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))])

    considered = sum(dispositions.values())
    dropped = sum(count for status, count in dispositions.items() if status not in TRAINABLE)
    rendering_report = {
        "schema_version": 1,
        "identity": expected.to_dict(),
        "corpus": identity_info,
        "records_considered": considered,
        "trainable_records": written,
        "dropped_records": dropped,
        "dropped_fraction": round(dropped / considered, 6) if considered else 0.0,
        "dispositions": dict(sorted(dispositions.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_trainable": dict(sorted(source_trainable.items())),
        "behavior_trainable": dict(sorted(behavior_trainable.items())),
        "top_exclusion_reasons": dict(reasons.most_common(25)),
        "rendered_token_statistics": {
            "records": len(ordered),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "min": ordered[0] if ordered else 0,
            "max": ordered[-1] if ordered else 0,
            "mean": round(sum(ordered) / len(ordered), 2) if ordered else 0.0,
            "fraction_le_2048": round(sum(1 for v in ordered if v <= 2048) / len(ordered), 6)
            if ordered
            else 0.0,
            "fraction_gt_2048": round(sum(1 for v in ordered if v > 2048) / len(ordered), 6)
            if ordered
            else 0.0,
            "fraction_gt_4096": round(sum(1 for v in ordered if v > 4096) / len(ordered), 6)
            if ordered
            else 0.0,
        },
        "supervised_tokens": supervised_total,
        "total_tokens": token_total,
        "note": (
            "Records failing the canonical schema or training-trajectory contract are "
            "quarantined, never repaired: training on a tool schema the canonical contract "
            "rejects, or on a trajectory whose tool results do not resolve, would be "
            "corrupted supervision."
        ),
    }
    (cache_dir / "rendering_report.json").write_text(
        json.dumps(rendering_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (cache_dir / CACHE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": expected.to_dict(),
                "rendering_report": rendering_report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rendering_report


def _template_hash(renderer: Any) -> str:
    import hashlib

    tokenizer = renderer._load()
    return hashlib.sha256(str(getattr(tokenizer, "chat_template", "")).encode()).hexdigest()


def _completed_shards(part_dir: Path) -> int:
    total = 0
    for path in part_dir.glob("part-*.progress"):
        with path.open("rb") as handle:
            total += sum(1 for _ in handle)
    return total


def _part_sample_count(part_dir: Path) -> int:
    total = 0
    for path in part_dir.glob("part-*.progress"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += int(line.strip())
    return total


SAMPLE_COLUMNS = (
    "record_id",
    "canonical_hash",
    "source_dataset",
    "behavior_decision",
    "status",
    "template_hash",
)


def samples_to_table(rows: list[dict[str, Any]], pa: Any) -> Any:
    """Build the sample table with explicit types.

    Built column-wise rather than from `from_pylist` because the token and supervision columns
    are variable-length integer lists, and letting Arrow infer them from a mixture of Python
    lists and arrays is what raises its inference error.
    """
    columns: dict[str, Any] = {
        name: pa.array([row.get(name) for row in rows], type=pa.string()) for name in SAMPLE_COLUMNS
    }
    columns["tokens"] = pa.array([row["tokens"] for row in rows], type=pa.list_(pa.int32()))
    columns["supervised"] = pa.array([row["supervised"] for row in rows], type=pa.list_(pa.int32()))
    return pa.table(columns)


def run_worker(
    index: int,
    shard_names: list[str],
    out_dir: Path,
    max_seq_length: int,
    supervision_include: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Render a list of shards, appending samples and a per-shard progress line.

    Progress is written per shard rather than per worker so the parent can report real
    progress even though a worker only finishes once, at the end.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"part-{index}.parquet"
    progress_path = out_dir / f"part-{index}.progress"
    writer: Any = None
    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_trainable: Counter[str] = Counter()
    behavior_trainable: Counter[str] = Counter()
    supervision_trainable: Counter[str] = Counter()
    supervision_excluded: Counter[str] = Counter()
    lengths: list[int] = []
    supervised_tokens = 0
    total_tokens = 0
    written = 0

    for name in shard_names:
        payload = _process_shard((name, max_seq_length, supervision_include))
        for key, value in payload["dispositions"].items():
            dispositions[key] += value
        for key, value in payload["reasons"].items():
            reasons[key] += value
        for key, value in payload["source_counts"].items():
            source_counts[key] += value
        for key, value in payload["supervision_excluded"].items():
            supervision_excluded[key] += value
        rows = payload["samples"]
        for row in rows:
            source_trainable[row["source_dataset"]] += 1
            behavior_trainable[row["behavior_decision"]] += 1
            supervision_trainable[str(row.get("supervision_kind") or "UNCLASSIFIED")] += 1
            lengths.append(len(row["tokens"]))
            supervised_tokens += len(row["supervised"])
            total_tokens += len(row["tokens"])
        if rows:
            table = samples_to_table(rows, pa)
            if writer is None:
                writer = pq.ParquetWriter(samples_path, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(rows)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{len(rows)}\n")

    if writer is not None:
        writer.close()
    stats = {
        "worker": index,
        "shards": len(shard_names),
        "written": written,
        "dispositions": dict(dispositions),
        "reasons": dict(reasons),
        "source_counts": dict(source_counts),
        "source_trainable": dict(source_trainable),
        "behavior_trainable": dict(behavior_trainable),
        "supervision_trainable": dict(supervision_trainable),
        "supervision_excluded": dict(supervision_excluded),
        "supervision_include": list(supervision_include),
        "lengths": lengths,
        "supervised_tokens": supervised_tokens,
        "total_tokens": total_tokens,
    }
    (out_dir / f"part-{index}.json").write_text(
        json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def _merge_parts(part_dir: Path, staging: Path) -> dict[str, Any]:
    """Concatenate worker parts into the final sample file and combine their statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dispositions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_trainable: Counter[str] = Counter()
    behavior_trainable: Counter[str] = Counter()
    supervision_trainable: Counter[str] = Counter()
    supervision_excluded: Counter[str] = Counter()
    lengths: list[int] = []
    supervised_tokens = 0
    total_tokens = 0
    written = 0
    writer: Any = None
    try:
        for stats_path in sorted(part_dir.glob("part-*.json")):
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            dispositions.update(stats["dispositions"])
            reasons.update(stats["reasons"])
            source_counts.update(stats["source_counts"])
            source_trainable.update(stats["source_trainable"])
            behavior_trainable.update(stats["behavior_trainable"])
            supervision_trainable.update(stats.get("supervision_trainable") or {})
            supervision_excluded.update(stats.get("supervision_excluded") or {})
            lengths.extend(stats["lengths"])
            supervised_tokens += stats["supervised_tokens"]
            total_tokens += stats["total_tokens"]
            written += stats["written"]
            samples_path = part_dir / f"part-{stats['worker']}.parquet"
            if samples_path.is_file():
                parquet = pq.ParquetFile(samples_path)
                for batch in parquet.iter_batches(batch_size=1024):
                    table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(staging, table.schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {
        "dispositions": dict(dispositions),
        "reasons": dict(reasons),
        "source_counts": dict(source_counts),
        "source_trainable": dict(source_trainable),
        "behavior_trainable": dict(behavior_trainable),
        "supervision_trainable": dict(supervision_trainable),
        "supervision_excluded": dict(supervision_excluded),
        "lengths": lengths,
        "supervised_tokens": supervised_tokens,
        "total_tokens": total_tokens,
        "written": written,
    }


def _worker_main(argv: list[str]) -> int:
    """Entry point for a preprocessing subprocess."""
    import argparse

    parser = argparse.ArgumentParser(prog="opengrad.training.preprocess")
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--supervision-include",
        default="",
        help="comma-separated supervision kinds to keep; empty keeps every kind",
    )
    parser.add_argument("shards", nargs="+")
    arguments = parser.parse_args(argv)
    os.environ["OPENGRAD_RENDER_REVISION"] = arguments.revision
    run_worker(
        arguments.worker,
        arguments.shards,
        Path(arguments.out),
        arguments.max_seq_length,
        tuple(k for k in arguments.supervision_include.split(",") if k),
    )
    return 0


def load_cached_samples(cache_dir: Path) -> list[dict[str, Any]]:
    """Read back the cached samples as plain dicts."""
    import pyarrow.parquet as pq

    path = cache_dir / SAMPLES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"sample cache missing: {path}")
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        rows.extend(batch.to_pylist())
    return rows


def samples_to_supervised(rows: list[dict[str, Any]]) -> list[SupervisedSample]:
    """Adapt cached rows back to SupervisedSample for reporting helpers."""
    return [
        SupervisedSample(
            record_id=str(row["record_id"]),
            canonical_hash=str(row["canonical_hash"]),
            source_dataset=str(row["source_dataset"]),
            behavior_decision=str(row["behavior_decision"]),
            status=str(row["status"]),
            tokens=list(row["tokens"]),
            supervised=list(row["supervised"]),
        )
        for row in rows
    ]


def write_overflow_report(
    cache_dir: Path, rows: list[dict[str, Any]], max_seq_length: int
) -> dict[str, Any]:
    report = overflow_report(samples_to_supervised(rows), max_seq_length)
    (cache_dir / "overflow_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
