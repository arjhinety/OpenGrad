"""Build the pre-classifier ``normalization-v3`` artifact from the canonical-v3 source manifest.

What this is
------------
The representation the future prose decision classifier will receive, built **before** that classifier
exists. Each accepted row holds normalized messages, normalized tools, structured tool calls,
structural facts, and source/adapter/schema provenance. It holds **no behaviour label**:
``adapters._base`` stamps a message-shape ``metadata.behavior`` (``ANSWER`` whenever no call is present),
and that default is behavioural inference, so this builder drops it. Deciding behaviour is the classifier's
job, and the classifier does not exist yet.

It is not the canonical-v3 training corpus. No mixture, balancing or membership decision is made here.

How it is built
---------------
For each source in ``configs/releases/toolpolicy_canonical_v3_sources.yaml``:

1. the raw file's SHA-256 must equal the manifest's value;
2. the registered adapter named by the manifest runs unchanged, on the untouched raw row, except that a
   source with ``schema_translation.applied`` has its raw ``tools`` translated first
   (:func:`opengrad.data.source_schema.translate_source_tools`);
3. ``raw_record_hash`` and the upstream id are computed on the untouched raw row, with the same rule as
   ``_base``, so translation cannot change a record's identity;
4. metadata is rebuilt: ``behavior`` removed, every version taken from :mod:`opengrad.data.versions`
   (including the supervision block, which ``_base`` stamps with the legacy module constant), and a
   ``structure`` block of structural facts added;
5. rows are deduplicated by ``canonical_hash`` within the source (first wins), as in ``materialize``.

Output is deterministic: shards, per-source manifests, a disposition ledger of every row not accepted,
and a top-level manifest whose ``fingerprint`` identifies the whole artifact.

    python -m opengrad.data.normalization_v3 --build            # writes data/processed/normalization-v3/
    python -m opengrad.data.normalization_v3 --verify           # re-hashes shards against the manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from opengrad.data import versions
from opengrad.data.adapters import ADAPTERS
from opengrad.data.canonical import ToolConversation, canonical_dict, stable_json
from opengrad.data.semantic import validate_training_trajectory
from opengrad.data.source_schema import translate_source_tools

ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = Path("configs/releases/toolpolicy_canonical_v3_sources.yaml")
OUTPUT_DIR = Path("data/processed/normalization-v3")
TOP_MANIFEST = "manifest.json"
SHARD_SIZE = 10_000
BUILDER = "opengrad.data.normalization_v3"
ARTIFACT_KIND = "NORMALIZATION_V3_PRE_CLASSIFIER"

#: Modules whose code decides what a row contains. Their LF-normalized hashes go into the manifest so
#: the artifact can be traced to the exact code that produced it.
CODE_MODULES = (
    "src/opengrad/data/adapters.py",
    "src/opengrad/data/canonical.py",
    "src/opengrad/data/schema.py",
    "src/opengrad/data/semantic.py",
    "src/opengrad/data/source_schema.py",
    "src/opengrad/data/supervision.py",
    "src/opengrad/data/versions.py",
    "src/opengrad/data/xlam_types.py",
    "src/opengrad/data/normalization_v3.py",
)

#: Adapter exceptions that are prose rather than a reason code, mapped to stable codes. Anything not
#: listed falls back to ``ADAPTER_REJECTED_OTHER``; the verbatim message head is counted alongside.
ADAPTER_MESSAGE_CODES = (
    ("malformed Glaive function call", "ADAPTER_GLAIVE_MALFORMED_CALL"),
    ("Glaive arguments must be an object", "ADAPTER_GLAIVE_ARGUMENTS_NOT_OBJECT"),
    ("unknown tool", "ADAPTER_UNDECLARED_TOOL"),
    ("duplicate tool name", "ADAPTER_DUPLICATE_TOOL_NAME"),
    ("duplicate tool call id", "ADAPTER_DUPLICATE_CALL_ID"),
)

STATEMENT = (
    "Pre-classifier normalization-v3: normalized messages, tools, structured tool calls, structural facts "
    "and provenance. No behaviour label is present and no decision classifier has run. This is not the "
    "canonical-v3 training corpus and implies no mixture, balancing or C1 membership decision."
)


class NormalizationV3Error(ValueError):
    """The manifest, the code and the raw bytes disagree; nothing is built."""


# ── the source manifest ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    upstream_revision: str
    split: str
    raw_path: Path
    raw_sha256: str
    raw_rows: int
    adapter_key: str
    adapter_function: str
    row_label: str
    schema_translation: bool

    def adapter(self) -> Callable[[dict[str, Any], str], ToolConversation]:
        return ADAPTERS[self.adapter_key]


def lf_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF folded to LF, so a checkout's line endings cannot move it."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest(root: Path = ROOT, path: Path = SOURCE_MANIFEST) -> dict[str, Any]:
    value = yaml.safe_load((root / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
        raise NormalizationV3Error(f"source manifest is malformed: {path}")
    return value


def check_manifest_versions(manifest: dict[str, Any]) -> None:
    """The manifest's versions must be the authoritative ones; a mismatch is refused, not reconciled."""
    declared = dict(manifest.get("versions") or {})
    versions.check_artifact_matches_authoritative_versions(declared)
    expected = {
        "adapter_version": versions.ADAPTER_VERSION,
        "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
        "canonical_schema_version": versions.CANONICAL_SCHEMA_VERSION,
        "supervision_contract_version": versions.SUPERVISION_CONTRACT_VERSION,
    }
    missing = sorted(field for field, value in expected.items() if declared.get(field) != value)
    if missing:
        raise NormalizationV3Error(f"source manifest versions missing or wrong: {missing}")
    if manifest.get("normalization_version") != versions.NORMALIZATION_VERSION:
        raise NormalizationV3Error(
            "source manifest normalization_version is not the authoritative one"
        )
    if (
        manifest.get("decision_classifier") != "NOT_APPLIED"
        or manifest.get("behavior_labels") != "ABSENT"
    ):
        raise NormalizationV3Error(
            "normalization-v3 is pre-classifier: no classifier, no behaviour labels"
        )


def source_specs(manifest: dict[str, Any], root: Path = ROOT) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for entry in manifest["sources"]:
        adapter = entry["adapter"]
        key = str(adapter["key"])
        if key not in ADAPTERS:
            raise NormalizationV3Error(f"{entry['name']}: adapter {key!r} is not registered")
        if ADAPTERS[key].__name__ != adapter["function"]:
            raise NormalizationV3Error(
                f"{entry['name']}: adapter {key!r} is {ADAPTERS[key].__name__}, "
                f"manifest says {adapter['function']}"
            )
        raw = entry["raw_artifact"]
        specs.append(
            SourceSpec(
                name=str(entry["name"]),
                dataset_id=str(entry["dataset_id"]),
                upstream_revision=str(entry["upstream_revision"]),
                split=str(entry["split"]),
                raw_path=root / str(raw["path"]),
                raw_sha256=str(raw["sha256"]),
                raw_rows=int(raw["rows"]),
                adapter_key=key,
                adapter_function=str(adapter["function"]),
                row_label=str(adapter["row_label"]),
                schema_translation=bool((entry.get("schema_translation") or {}).get("applied")),
            )
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise NormalizationV3Error("duplicate source name in the manifest")
    return specs


# ── one row ────────────────────────────────────────────────────────────────────────────────────────


def raw_record_hash(raw: dict[str, Any]) -> str:
    """The rule ``adapters._base`` uses, applied to the untouched raw row."""
    return hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def upstream_id(raw: dict[str, Any], raw_hash: str) -> str:
    """The rule ``adapters._base`` uses for the record id, applied to the untouched raw row."""
    return str(raw.get("id", raw.get("example_id", raw.get("uid", "og_" + raw_hash[:16]))))


def reason_code(exc: BaseException) -> tuple[str, str]:
    """A stable reason code for an adapter rejection, plus the verbatim message head (digits masked)."""
    message = str(exc)
    head = message.split(":", 1)[0].strip()
    masked = re.sub(r"\d+", "N", head)
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", head):
        return head, masked
    for prefix, code in ADAPTER_MESSAGE_CODES:
        if head.startswith(prefix):
            return code, masked
    return "ADAPTER_REJECTED_OTHER", masked


def _roles(messages: list[dict[str, Any]]) -> list[str]:
    return [str(message.get("role")) for message in messages]


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def structure_of(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural facts about a conversation. Shape only: no text is read beyond emptiness."""
    roles = _roles(messages)
    body = roles[1:] if roles[:1] == ["system"] else roles
    final_index = len(messages) - 1
    final = messages[-1] if messages else {}
    assistant_indices = [index for index, role in enumerate(roles) if role == "assistant"]
    last_assistant = assistant_indices[-1] if assistant_indices else None
    calls = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    tool_results = [index for index, role in enumerate(roles) if role == "tool"]
    return {
        "message_roles": roles,
        "leading_system_message": roles[:1] == ["system"],
        "exchange_shape": "single_exchange" if body == ["user", "assistant"] else "multi_turn",
        "user_turns": roles.count("user"),
        "assistant_turns": len(assistant_indices),
        "tool_result_messages": len(tool_results),
        "structured_calls": sum(len(messages[index]["tool_calls"]) for index in calls),
        "assistant_turns_with_structured_call": len(calls),
        "tools_available": len(tools),
        "final_role": roles[-1] if roles else None,
        "final_assistant_structured_call": bool(
            final.get("role") == "assistant" and final.get("tool_calls")
        ),
        "final_assistant_prose": bool(
            final.get("role") == "assistant" and _has_text(final.get("content"))
        ),
        "tool_result_before_final": bool(tool_results and tool_results[0] < final_index),
        "structured_call_before_final_assistant": bool(
            last_assistant is not None and any(index < last_assistant for index in calls)
        ),
    }


def _supervision(block: dict[str, Any]) -> dict[str, Any]:
    """The adapter's supervision block, with its version replaced by the authoritative one."""
    return {**block, "adapter_version": versions.ADAPTER_VERSION}


def normalize_row(spec: SourceSpec, raw: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Adapt one raw row into a normalization-v3 row, or raise with a reason.

    Raises :class:`RowRejected` for anything the adapter, the schema translation or the IR validator
    refuses. Any other exception is a bug and propagates.
    """
    raw_hash = raw_record_hash(raw)
    record_id = upstream_id(raw, raw_hash)
    adapter_input = raw
    translation: dict[str, Any] = {"applied": False, "changes": []}
    if spec.schema_translation:
        tools, changes, quarantine, detail = translate_source_tools(raw.get("tools"))
        if quarantine:
            raise RowRejected(quarantine, quarantine, detail)
        translation = {"applied": True, "changes": changes}
        if changes:
            adapter_input = {**raw, "tools": tools}
    try:
        conversation = spec.adapter()(adapter_input, spec.split)
    except (TypeError, ValueError) as exc:
        code, head = reason_code(exc)
        raise RowRejected(code, head, str(exc)[:300]) from exc
    produced = conversation.metadata
    if produced.get("adapter") != spec.row_label:
        raise NormalizationV3Error(
            f"{spec.name}: adapter stamped {produced.get('adapter')!r}, manifest says {spec.row_label!r}"
        )
    if adapter_input is raw and (
        produced.get("raw_record_hash") != raw_hash or conversation.id != record_id
    ):
        # Untranslated rows must reproduce the adapter's own identity exactly; otherwise the identity rule
        # above has drifted from `_base` and every translated row's identity would be suspect.
        raise NormalizationV3Error(
            f"{spec.name}: identity rule disagrees with the adapter at row {row_index}"
        )
    metadata: dict[str, Any] = {
        "split": spec.split,
        "source": {
            "dataset_id": spec.dataset_id,
            "source_name": spec.name,
            "upstream_id": record_id,
            "upstream_revision": spec.upstream_revision,
            "original_split": spec.split,
            "raw_artifact_sha256": spec.raw_sha256,
            "raw_row_index": row_index,
        },
        "raw_record_hash": raw_hash,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "adapter": spec.row_label,
        "adapter_key": spec.adapter_key,
        "adapter_function": f"opengrad.data.adapters.{spec.adapter_function}",
        "adapter_version": versions.ADAPTER_VERSION,
        "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
        "schema_translation": translation,
        "parse_status": produced.get("parse_status"),
        "contamination_status": produced.get("contamination_status"),
        "source_fields": produced.get("source_fields"),
        "source_features": produced.get("source_features"),
        "tool_context": produced.get("tool_context"),
        "supervision": _supervision(produced["supervision"]),
    }
    for optional in ("system", "eligibility"):
        if optional in produced:
            metadata[optional] = produced[optional]
    rebuilt = ToolConversation(
        record_id, conversation.source, conversation.tools, conversation.messages, metadata
    )
    try:
        item = canonical_dict(rebuilt)
    except (TypeError, ValueError) as exc:
        code, head = reason_code(exc)
        raise RowRejected(code, head, str(exc)[:300]) from exc
    issues = sorted({issue.code for issue in validate_training_trajectory(rebuilt)})
    item["metadata"]["structure"] = {
        **structure_of(rebuilt.messages, rebuilt.tools),
        "trajectory_issue_codes": issues,
    }
    return item


class RowRejected(ValueError):
    def __init__(self, code: str, head: str, detail: str) -> None:
        self.code = code
        self.head = head
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# ── storage ────────────────────────────────────────────────────────────────────────────────────────


def storage_row(item: dict[str, Any]) -> dict[str, Any]:
    """JSON columns for nested values, so heterogeneous rows share one Parquet schema (as materialize)."""
    return {
        key: stable_json(value) if isinstance(value, (dict, list)) else value
        for key, value in item.items()
    }


def decode_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for key in ("tools", "messages", "metadata"):
        if isinstance(decoded.get(key), str):
            decoded[key] = json.loads(decoded[key])
    return decoded


def row_bytes(row: dict[str, Any]) -> bytes:
    return (stable_json(row) + "\n").encode("utf-8")


def _write_parquet(rows: list[dict[str, Any]], destination: Path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pa.Table.from_pylist(rows)
    checksum = hashlib.sha256(b"".join(row_bytes(row) for row in rows)).hexdigest()
    table = table.replace_schema_metadata(
        {
            b"opengrad_row_checksum": checksum.encode("ascii"),
            b"opengrad_row_count": str(len(rows)).encode("ascii"),
        }
    )
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=False,
            version="2.6",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def iter_raw(path: Path, batch_size: int = 512) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_rows(output_dir: Path, source: str) -> Iterator[dict[str, Any]]:
    """Decoded normalization-v3 rows of one source, in artifact order."""
    import pyarrow.parquet as pq

    manifest = json.loads((output_dir / source / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        for batch in pq.ParquetFile(output_dir / source / shard["file"]).iter_batches(
            batch_size=512
        ):
            for row in batch.to_pylist():
                yield decode_row(row)


# ── build ──────────────────────────────────────────────────────────────────────────────────────────


_WORKER_SPEC: SourceSpec | None = None


def _worker_init(spec: SourceSpec) -> None:
    global _WORKER_SPEC
    _WORKER_SPEC = spec


def _disposition(spec: SourceSpec, task: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
    """One row's outcome as plain data, so it crosses a process boundary unchanged."""
    index, raw = task
    try:
        return ("accepted", normalize_row(spec, raw, index))
    except RowRejected as rejection:
        return ("rejected", rejection.code, rejection.head, raw_record_hash(raw))


def _worker(task: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
    assert _WORKER_SPEC is not None
    return _disposition(_WORKER_SPEC, task)


def _dispositions(
    spec: SourceSpec, max_rows: int | None, workers: int
) -> Iterator[tuple[Any, ...]]:
    """Row outcomes in source order. Parallelism never reorders: ``imap`` yields in input order.

    Workers exist only for speed. Every adapter call re-reads the behaviour taxonomy YAML inside
    ``ToolConversation.validate`` (about 24 ms a row), and the adapters are not edited to avoid it.
    """
    import itertools

    tasks = itertools.islice(enumerate(iter_raw(spec.raw_path)), max_rows)
    if workers <= 1:
        for task in tasks:
            yield _disposition(spec, task)
        return
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(workers, _worker_init, (spec,)) as pool:
        yield from pool.imap(_worker, tasks, chunksize=64)


def build_source(
    spec: SourceSpec, output_dir: Path, *, max_rows: int | None = None, workers: int = 1
) -> dict[str, Any]:
    """Normalize one source into ``output_dir/<name>/``; returns its manifest."""
    from opengrad.data.materialize import _validate_sft_eligibility

    # The same training-split allowlist materialize enforces (registry/datasets.yaml).
    _validate_sft_eligibility(spec.name, spec.split)
    observed = file_sha256(spec.raw_path)
    if observed != spec.raw_sha256:
        raise NormalizationV3Error(
            f"{spec.name}: raw artifact sha256 {observed} != manifest {spec.raw_sha256}"
        )
    directory = output_dir / spec.name
    if directory.exists() and any(directory.iterdir()):
        raise NormalizationV3Error(f"refusing to build into a non-empty directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    rejected_heads: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    content = hashlib.sha256()

    def flush() -> None:
        name = f"shard-{len(shards):06d}.parquet"
        _write_parquet(rows, directory / name)
        shards.append({"file": name, "rows": len(rows), "sha256": file_sha256(directory / name)})
        rows.clear()

    for index, outcome in enumerate(_dispositions(spec, max_rows, workers)):
        counts["source_rows"] += 1
        if outcome[0] == "rejected":
            _, code, head, raw_hash = outcome
            counts["rejected"] += 1
            rejected[code] += 1
            rejected_heads[head] += 1
            ledger.append(
                {
                    "raw_row_index": index,
                    "raw_record_hash": raw_hash,
                    "disposition": "rejected",
                    "reason": code,
                }
            )
            continue
        item = outcome[1]
        canonical_hash = item["canonical_hash"]
        if canonical_hash in seen:
            counts["duplicates"] += 1
            ledger.append(
                {
                    "raw_row_index": index,
                    "raw_record_hash": item["metadata"]["raw_record_hash"],
                    "disposition": "duplicate",
                    "reason": "DUPLICATE_CANONICAL_HASH",
                    "duplicate_of_raw_row_index": seen[canonical_hash],
                }
            )
            continue
        seen[canonical_hash] = index
        counts["accepted"] += 1
        stored = storage_row(item)
        content.update(row_bytes(stored))
        rows.append(stored)
        if len(rows) >= SHARD_SIZE:
            flush()
    if rows or not shards:
        flush()

    if counts["source_rows"] != counts["accepted"] + counts["rejected"] + counts["duplicates"]:
        raise NormalizationV3Error(f"{spec.name}: ledger does not reconcile")
    if max_rows is None and counts["source_rows"] != spec.raw_rows:
        raise NormalizationV3Error(
            f"{spec.name}: read {counts['source_rows']} rows, manifest says {spec.raw_rows}"
        )
    ledger_text = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in ledger)
    _write_text(directory / "dispositions.jsonl", ledger_text)
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "builder": BUILDER,
        "source": {
            "name": spec.name,
            "dataset_id": spec.dataset_id,
            "upstream_revision": spec.upstream_revision,
            "split": spec.split,
            "raw_artifact_sha256": spec.raw_sha256,
            "raw_rows": spec.raw_rows,
        },
        "config": {
            "adapter_key": spec.adapter_key,
            "adapter_function": f"opengrad.data.adapters.{spec.adapter_function}",
            "adapter_row_label": spec.row_label,
            "adapter_version": versions.ADAPTER_VERSION,
            "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
            "schema_translation_applied": spec.schema_translation,
            "canonical_schema_version": versions.CANONICAL_SCHEMA_VERSION,
            "supervision_contract_version": versions.SUPERVISION_CONTRACT_VERSION,
            "normalization_version": versions.NORMALIZATION_VERSION,
            "max_rows": max_rows,
            "shard_size": SHARD_SIZE,
        },
        "counts": {
            "source_rows": counts["source_rows"],
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "duplicates": counts["duplicates"],
        },
        "rejected_by_reason": dict(sorted(rejected.items())),
        "rejected_by_adapter_message": dict(sorted(rejected_heads.items())),
        "shards": shards,
        "content_hash": content.hexdigest(),
        "dispositions": {
            "file": "dispositions.jsonl",
            "sha256": hashlib.sha256(ledger_text.encode()).hexdigest(),
        },
    }
    versions.check_artifact_matches_authoritative_versions(manifest["config"])
    _write_text(directory / "manifest.json", _json_text(manifest))
    return manifest


def code_fingerprint() -> dict[str, str]:
    """Hashes of the code that decides row content, always read from this package's own tree."""
    return {module: lf_sha256(ROOT / module) for module in CODE_MODULES}


def environment() -> dict[str, str]:
    """Informational: shard bytes depend on the Parquet writer; row content and fingerprints do not."""
    import platform

    import pyarrow

    return {"python": platform.python_version(), "pyarrow": str(pyarrow.__version__)}


def build(
    output_dir: Path,
    *,
    root: Path = ROOT,
    source_manifest: Path = SOURCE_MANIFEST,
    only: tuple[str, ...] = (),
    max_rows: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    manifest = load_source_manifest(root, source_manifest)
    check_manifest_versions(manifest)
    specs = [spec for spec in source_specs(manifest, root) if not only or spec.name in only]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / TOP_MANIFEST).exists():
        raise NormalizationV3Error(
            f"refusing to overwrite {output_dir / TOP_MANIFEST}; build into a new directory"
        )
    per_source: dict[str, Any] = {}
    for spec in specs:
        source_manifest_value = build_source(spec, output_dir, max_rows=max_rows, workers=workers)
        per_source[spec.name] = {
            "manifest_sha256": lf_sha256(output_dir / spec.name / "manifest.json"),
            "content_hash": source_manifest_value["content_hash"],
            "counts": source_manifest_value["counts"],
        }
    top: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "statement": STATEMENT,
        "builder": BUILDER,
        "normalization_version": versions.NORMALIZATION_VERSION,
        "versions": {
            "adapter_version": versions.ADAPTER_VERSION,
            "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
            "canonical_schema_version": versions.CANONICAL_SCHEMA_VERSION,
            "supervision_contract_version": versions.SUPERVISION_CONTRACT_VERSION,
            "provenance_invariant_version": versions.PROVENANCE_INVARIANT_VERSION,
            "normalization_version": versions.NORMALIZATION_VERSION,
        },
        "decision_classifier": "NOT_APPLIED",
        "behavior_labels": "ABSENT",
        "source_manifest": {
            "path": source_manifest.as_posix(),
            "sha256_lf": lf_sha256(root / source_manifest),
        },
        "code_sha256_lf": code_fingerprint(),
        "environment": environment(),
        "max_rows": max_rows,
        "sources": per_source,
    }
    top["fingerprint"] = hashlib.sha256(
        stable_json(
            {
                "normalization_version": top["normalization_version"],
                "versions": top["versions"],
                "source_manifest_sha256_lf": top["source_manifest"]["sha256_lf"],
                "content": {name: value["content_hash"] for name, value in per_source.items()},
            }
        ).encode("utf-8")
    ).hexdigest()
    versions.check_artifact_matches_authoritative_versions(top["versions"])
    _write_text(output_dir / TOP_MANIFEST, _json_text(top))
    return top


# ── verify ─────────────────────────────────────────────────────────────────────────────────────────


def verify(output_dir: Path, *, root: Path = ROOT) -> list[str]:
    """Re-hash every shard and row against the manifests and re-check every row's provenance.

    Returns the list of problems; empty means the artifact is exactly what its manifests say.
    """
    import pyarrow.parquet as pq

    problems: list[str] = []
    top = json.loads((output_dir / TOP_MANIFEST).read_text(encoding="utf-8"))
    if top["source_manifest"]["sha256_lf"] != lf_sha256(root / top["source_manifest"]["path"]):
        problems.append("source manifest changed since the build")
    for module, digest in top["code_sha256_lf"].items():
        if lf_sha256(ROOT / module) != digest:
            problems.append(f"code changed since the build: {module}")
    for name, summary in top["sources"].items():
        directory = output_dir / name
        if lf_sha256(directory / "manifest.json") != summary["manifest_sha256"]:
            problems.append(f"{name}: manifest hash mismatch")
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        content = hashlib.sha256()
        count = 0
        for shard in manifest["shards"]:
            path = directory / shard["file"]
            if not path.is_file() or file_sha256(path) != shard["sha256"]:
                # A changed shard may no longer parse; its rows are not read, so the content check
                # below also fails for this source.
                problems.append(f"{name}: {shard['file']} bytes changed")
                continue
            for row in pq.read_table(path).to_pylist():
                content.update(row_bytes(row))
                count += 1
                metadata = json.loads(row["metadata"])
                if "behavior" in metadata:
                    problems.append(f"{name}: behaviour label present on {row['id']}")
                try:
                    versions.check_version_agreement(metadata, manifest["config"])
                    versions.check_version_agreement(metadata["supervision"], manifest["config"])
                except versions.ProvenanceVersionMismatch as exc:
                    problems.append(f"{name}: {row['id']}: {exc}")
                if metadata.get("adapter_version") != versions.ADAPTER_VERSION:
                    problems.append(
                        f"{name}: {row['id']}: adapter_version {metadata.get('adapter_version')!r}"
                    )
        if (
            content.hexdigest() != manifest["content_hash"]
            or count != manifest["counts"]["accepted"]
        ):
            problems.append(f"{name}: content hash or row count mismatch")
        ledger = (directory / manifest["dispositions"]["file"]).read_bytes()
        if hashlib.sha256(ledger).hexdigest() != manifest["dispositions"]["sha256"]:
            problems.append(f"{name}: dispositions ledger changed")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_DIR)
    parser.add_argument(
        "--source", action="append", default=[], help="build only this source (repeatable)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="smoke builds only; never the artifact"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, help="speed only; output is identical"
    )
    args = parser.parse_args(argv)
    if args.build:
        top = build(
            args.output, only=tuple(args.source), max_rows=args.max_rows, workers=args.workers
        )
        for name, summary in top["sources"].items():
            print(name, summary["counts"], summary["content_hash"][:16])
        print("fingerprint", top["fingerprint"])
        return 0
    problems = verify(args.output)
    for problem in problems:
        print("FAIL", problem)
    print("PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
