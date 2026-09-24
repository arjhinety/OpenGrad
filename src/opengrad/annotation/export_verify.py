"""Verify an exported annotation package against its manifest: every file's hash, the recounted labels,
the history, the gold set and the exclusions. Split out of `export.py` on 2026-09-24 with no change in
behaviour; `export.verify_package` is re-exported.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.annotation.config import PRIMARY_KEY, find_repo_root
from opengrad.annotation.export_common import (
    COMPOSITE,
    GOLD_KIND,
    UNCOUNTED_TYPES,
    UNRESOLVED,
    _loads,
)
from opengrad.annotation.provenance import (
    ADJUDICATION_ENTRY_FIELDS,
    ANNOTATION_ENTRY_FIELDS,
    verify_chain,
    verify_definition_chain,
)
from opengrad.annotation.service import (
    annotator_kind,
)
from opengrad.hashing import sha256_bytes as _sha256
from opengrad.hashing import sha256_file
from opengrad.verification.accounting import FAIL, PASS, REQUIRED_NONEMPTY, ValidationResult


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _primary_of(manifest: dict[str, Any]) -> str:
    return PRIMARY_KEY.get(str(manifest.get("task_type")), "label")


def _recount(records: list[dict[str, Any]], label_key: str, task_type: Any) -> dict[str, int]:
    """Same unit as ``_count_labels``: one count per label value on a record, none for text/ranking."""
    if task_type in UNCOUNTED_TYPES:
        return {}
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(label_key)
        if isinstance(value, list):
            counts.update(str(item) for item in value)
        elif value is not None:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _renamer(renames: dict[str, str]) -> Any:
    def rename(value: dict[str, Any] | None) -> dict[str, Any] | None:
        return (
            None if value is None else {renames.get(key, key): item for key, item in value.items()}
        )

    return rename


def _annotation_matches(
    record: dict[str, Any], image: dict[str, Any], renames: dict[str, str]
) -> bool:
    """Does an exported pass record say exactly what the chain's final state says?"""
    if (
        record.get("status") != image.get("status")
        or record.get("flagged") != bool(image.get("flagged"))
        or record.get("note") != image.get("note")
        or record.get("revision") != image.get("revision")
        or record.get("created_at") != image.get("created_at")
        or record.get("timestamp") != image.get("updated_at")
    ):
        return False
    value = _renamer(renames)(_loads(image.get("value_json"))) or {}
    return all(record.get(key) == item for key, item in value.items())


def _adjudication_matches(
    record: dict[str, Any], image: dict[str, Any], renames: dict[str, str]
) -> bool:
    rename = _renamer(renames)
    return (
        record.get("adjudicated") == rename(_loads(image.get("adjudicated_value_json")))
        and record.get("value_a") == rename(_loads(image.get("value_a_json")))
        and record.get("value_b") == rename(_loads(image.get("value_b_json")))
        and record.get("rationale") == image.get("rationale")
        and record.get("adjudicator_id") == image.get("adjudicator_id")
        and record.get("revision") == image.get("revision")
        and record.get("kind") == image.get("kind")
    )


def _check_history(
    history: list[dict[str, Any]],
    records: list[dict[str, Any]],
    id_key: str,
    fields: tuple[str, ...],
    label: str,
    matches: Any,
    renames: dict[str, str],
) -> list[str]:
    """The chain must verify, and its last state per item must be exactly the exported record --
    by hash *and* by content, so a record cannot be edited while keeping its old state hash."""
    errors = verify_chain(history, fields, label)
    last: dict[str, dict[str, Any]] = {}
    for entry in history:
        last[str(entry["item_id"])] = entry
    expected = {
        item: entry["after_sha256"]
        for item, entry in last.items()
        if entry.get("after") is not None
    }
    actual = {str(record.get(id_key)): record.get("state_sha256") for record in records}
    if expected != actual:
        unmatched = sorted(set(expected) ^ set(actual))
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        errors.append(
            f"FAIL_STATE: {label}: final records disagree with the change log "
            f"({len(unmatched)} unmatched, {len(changed)} with a different state hash)"
        )
    edited = [
        str(record.get(id_key))
        for record in records
        if str(record.get(id_key)) in last
        and last[str(record.get(id_key))].get("after") is not None
        and not matches(record, last[str(record.get(id_key))]["after"], renames)
    ]
    if edited:
        errors.append(
            f"FAIL_STATE: {label}: {len(edited)} records differ in content from their final chained "
            f"state (first: {edited[0]})"
        )
    return errors


def _check_gold(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    parsed: dict[str, list[dict[str, Any]]],
    id_key: str,
    renames: dict[str, str],
) -> list[str]:
    """Every gold label must follow from the pass and adjudication records it cites."""
    keys = [
        renames.get(str(key), str(key))
        for key in (manifest.get("disagreements") or {}).get("keys", [])
    ]
    sessions = {
        str(s.get("session_id")): {str(r.get(id_key)): r for r in parsed.get(s.get("file"), [])}
        for s in manifest.get("sessions") or []
    }
    adjudication = manifest.get("adjudication") or {}
    decisions = {str(r.get(id_key)): r for r in parsed.get(str(adjudication.get("file")), [])}
    kinds = {
        str(s.get("session_id")): s.get("annotator_kind", "human")
        for s in manifest.get("sessions") or []
    }
    priority = [str(s) for s in (manifest.get("composite") or {}).get("priority") or []]
    wrong: list[str] = []
    for row in rows:
        item = str(row.get(id_key))
        top = {key: row.get(key) for key in keys}
        passes = row.get("passes") or []
        pass_records = [
            sessions.get(str(entry.get("session_id")), {}).get(item) for entry in passes
        ]
        if any(
            # A pass that never touched the item has no record; the gold row must then cite nothing.
            (entry.get("value") is not None or entry.get("state_sha256") is not None)
            if record is None
            else (
                entry.get("state_sha256") != record.get("state_sha256")
                or any(
                    record.get(key) != value for key, value in (entry.get("value") or {}).items()
                )
            )
            for entry, record in zip(passes, pass_records)
        ):
            wrong.append(item)
            continue
        source = row.get("gold_source")
        subsets = [
            {key: record.get(key) for key in keys}
            for record in pass_records
            if record and record.get("status") == "labeled"
        ]
        if source == COMPOSITE:
            labeled = {
                str(entry.get("session_id")): record
                for entry, record in zip(passes, pass_records)
                if record and record.get("status") == "labeled"
            }
            first = next((session for session in priority if session in labeled), None)
            ok = (
                first is not None
                and [str(entry.get("session_id")) for entry in passes] == priority
                and top == {key: labeled[first].get(key) for key in keys}
                and row.get("label_source_session") == first
                and row.get("label_source_kind") == kinds.get(first)
            )
        elif source in ("adjudication", "review"):
            decision = decisions.get(item)
            chosen = (decision or {}).get("adjudicated") or {}
            ok = decision is not None and top == {key: chosen.get(key) for key in keys}
        elif source == "agreement":
            ok = len(subsets) == 2 and subsets[0] == subsets[1] == top
        elif source == "single_annotator":
            ok = len(subsets) == 1 and subsets[0] == top
        elif source == UNRESOLVED:
            ok = (
                len(subsets) == 2
                and subsets[0] != subsets[1]
                and all(v is None for v in top.values())
            )
        else:
            ok = False
        if not ok:
            wrong.append(item)
    if wrong:
        return [
            (
                f"FAIL_GOLD: {len(wrong)} gold records do not follow from the pass and adjudication "
                f"records they cite (first: {wrong[0]})"
            )
        ]
    return []


def _check_exclusions(
    manifest: dict[str, Any], parsed: dict[str, list[dict[str, Any]]], id_key: str, label_key: str
) -> list[str]:
    """The exclusion section must add up, and every record must carry exactly its item's exclusions,
    so a label cannot be moved into (or out of) a metric by editing either side alone."""
    section = manifest.get("metric_exclusions")
    if section is None:
        return []  # written before metric exclusions existed
    errors: list[str] = []
    expected: dict[str, list[str]] = {}
    for group in section.get("groups") or []:
        ids = [str(item) for item in group.get("item_ids") or []]
        if len(set(ids)) != len(ids) or len(ids) != group.get("items"):
            errors.append(
                f"FAIL_EXCLUSION: group {group.get('status')} item count differs from its ids"
            )
        for item_id in ids:
            expected.setdefault(item_id, []).append(str(group.get("status")))
    expected = {item_id: sorted(statuses) for item_id, statuses in expected.items()}
    population = (manifest.get("source") or {}).get("items")
    if (
        section.get("population_items") != population
        or section.get("excluded_items") != len(expected)
        or section.get("metric_eligible_items") != (population or 0) - len(expected)
    ):
        errors.append(
            "FAIL_EXCLUSION: excluded and metric-eligible counts do not add up to the population"
        )

    def carried(name: str | None, *, required: bool) -> list[dict[str, Any]]:
        records = parsed.get(str(name), [])
        wrong = [
            r
            for r in records
            if ("metric_exclusions" in r or required)
            and r.get("metric_exclusions") != expected.get(str(r.get(id_key)), [])
        ]
        if wrong:
            errors.append(
                f"FAIL_EXCLUSION: {name}: {len(wrong)} records carry exclusions that differ from the "
                f"manifest (first: {wrong[0].get(id_key)})"
            )
        return records

    for summary in manifest.get("sessions") or []:
        records = carried(summary.get("file"), required=True)
        eligible = sum(
            1
            for r in records
            if r.get("status") == "labeled" and str(r.get(id_key)) not in expected
        )
        if "labeled_metric_eligible" in summary and eligible != summary["labeled_metric_eligible"]:
            errors.append(
                f"FAIL_COUNT: session {summary.get('session_id')} metric-eligible count differs"
            )
    disagreements = manifest.get("disagreements") or {}
    if disagreements.get("file"):
        records = carried(disagreements["file"], required=True)
        eligible = sum(1 for r in records if str(r.get(id_key)) not in expected)
        if len(records) != disagreements.get("count") or eligible != disagreements.get(
            "metric_eligible"
        ):
            errors.append(
                "FAIL_COUNT: disagreement counts recomputed from the file differ from the manifest"
            )
    carried((manifest.get("adjudication") or {}).get("file"), required=False)
    gold = manifest.get("gold") or {}
    if gold.get("file") in parsed:
        rows = carried(gold["file"], required=True)
        missing = sorted(set(expected) - {str(r.get(id_key)) for r in rows})
        if missing:
            errors.append(
                f"FAIL_EXCLUSION: excluded items missing from the gold file (first: {missing[0]})"
            )
        eligible_rows = [r for r in rows if str(r.get(id_key)) not in expected]
        recounted = _recount(eligible_rows, label_key, manifest.get("task_type"))
        if len(eligible_rows) != gold.get("metric_eligible_items") or recounted != gold.get(
            "metric_eligible_label_counts"
        ):
            errors.append("FAIL_COUNT: metric-eligible gold counts recomputed from the file differ")
    return errors


def _package_root(package_dir: Path, manifest: dict[str, Any]) -> Path:
    """The repository root, recovered from where the manifest says the package was written."""
    relative = manifest.get("package_dir")
    if isinstance(relative, str) and relative and not Path(relative).is_absolute():
        parts = Path(relative).parts
        here = package_dir.parts
        if len(here) > len(parts) and [os.path.normcase(p) for p in here[-len(parts) :]] == [
            os.path.normcase(p) for p in parts
        ]:
            return Path(*here[: -len(parts)])
    return find_repo_root(package_dir)


def verify_package(
    manifest_path: Path, root: Path | None = None, *, require_source: bool = False
) -> tuple[ValidationResult, dict[str, Any]]:
    """Recompute every hash, count and chain in a package and compare them with its manifest.

    Blocking checks: the manifest matches its ``.sha256`` sidecar; every listed output exists with the
    recorded hash, size and record count; each change log's chain verifies and ends in exactly the
    exported records -- by state hash and by content; per-session and gold label counts recomputed from
    the files equal the manifest's; every record names the manifest's task and source hash; a gold file
    covers each source item exactly once and every gold label follows from the records it cites; every
    record carries exactly its item's metric exclusions and the metric-eligible counts add up; and,
    when the source is reachable, it still hashes to the pinned value (``require_source`` makes an
    unreachable source a failure instead of a reported ``source_unreachable``).

    What this cannot prove: a package rewritten *consistently* -- every chain entry, hash and sidecar
    recomputed -- is indistinguishable from an honest one, because nothing is signed. Anchor a freeze by
    committing its manifest digest; a later rewrite then changes a committed hash.
    """
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    if not manifest_path.is_file():
        result = ValidationResult(
            name="annotation package",
            policy=REQUIRED_NONEMPTY,
            errors=[f"FAIL_MISSING: {manifest_path} does not exist"],
        )
        return result, {"status": FAIL, "errors": result.all_errors()}
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sidecar.is_file():
        errors.append("FAIL_HASH: manifest .sha256 sidecar is missing")
    elif sidecar.read_text(encoding="utf-8").strip() != _sha256(raw):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")

    package_dir = manifest_path.parent
    outputs: dict[str, dict[str, Any]] = manifest.get("outputs") or {}
    passed = failed = 0
    parsed: dict[str, list[dict[str, Any]]] = {}
    for name, expected in sorted(outputs.items()):
        path = package_dir / name
        problems: list[str] = []
        if not path.is_file():
            problems.append(f"FAIL_MISSING: output {name} is missing")
        else:
            data = path.read_bytes()
            if _sha256(data) != expected.get("sha256"):
                problems.append(f"FAIL_HASH: {name} sha256 differs from the manifest")
            if len(data) != expected.get("bytes"):
                problems.append(f"FAIL_HASH: {name} size differs from the manifest")
            try:
                parsed[name] = _read_jsonl(path)
            except json.JSONDecodeError as exc:
                problems.append(f"FAIL_PARSE: {name}: {exc.msg}")
            else:
                if len(parsed[name]) != expected.get("records"):
                    problems.append(f"FAIL_COUNT: {name} record count differs from the manifest")
        errors.extend(problems)
        if problems:
            failed += 1
        else:
            passed += 1

    task_id = manifest.get("task_id")
    source_sha = (manifest.get("source") or {}).get("sha256")
    keys = manifest.get("record_keys") or {}
    renames: dict[str, str] = manifest.get("record_key_map") or {}
    id_key = keys.get("item_id", "item_id")
    label_key = keys.get(_primary_of(manifest), _primary_of(manifest))
    for name, records in parsed.items():
        wrong = [r for r in records if r.get("task_id") != task_id]
        if wrong:
            errors.append(f"FAIL_PROVENANCE: {name}: {len(wrong)} records name another task")
        stamped = [r for r in records if "source_population_sha256" in r]
        if any(r["source_population_sha256"] != source_sha for r in stamped):
            errors.append(f"FAIL_PROVENANCE: {name}: records carry a different source hash")

    for summary in manifest.get("sessions") or []:
        if "annotator_kind" in summary and summary["annotator_kind"] != annotator_kind(
            str(summary.get("annotator_id"))
        ):
            errors.append(
                f"FAIL_PROVENANCE: session {summary.get('session_id')} misstates its annotator kind"
            )
        name, history_name = summary.get("file"), summary.get("history_file")
        if name not in parsed or history_name not in parsed:
            errors.append(
                f"FAIL_MISSING: session {summary.get('session_id')} files are not listed outputs"
            )
            continue
        history = parsed[history_name]
        errors.extend(
            _check_history(
                history,
                parsed[name],
                id_key,
                ANNOTATION_ENTRY_FIELDS,
                f"session {summary['session_id']}",
                _annotation_matches,
                renames,
            )
        )
        head = history[-1].get("entry_sha256") if history else None
        if head != summary.get("history_head_sha256") or len(history) != summary.get(
            "history_entries"
        ):
            errors.append(
                f"FAIL_CHAIN: session {summary['session_id']}: chain head differs from the manifest"
            )
        labeled = [r for r in parsed[name] if r.get("status") == "labeled"]
        counts = _recount(labeled, label_key, manifest.get("task_type"))
        if counts != summary.get("label_counts") or len(labeled) != summary.get("labeled"):
            errors.append(
                f"FAIL_COUNT: session {summary['session_id']} label counts recomputed from {name} "
                f"({counts}, {len(labeled)} labeled) differ from the manifest"
            )

    adjudication = manifest.get("adjudication")
    if adjudication:
        name, history_name = adjudication.get("file"), adjudication.get("history_file")
        if name in parsed and history_name in parsed:
            history = parsed[history_name]
            errors.extend(
                _check_history(
                    history,
                    parsed[name],
                    id_key,
                    ADJUDICATION_ENTRY_FIELDS,
                    "adjudication",
                    _adjudication_matches,
                    renames,
                )
            )
            if history and history[-1].get("entry_sha256") != adjudication.get(
                "history_head_sha256"
            ):
                errors.append("FAIL_CHAIN: adjudication chain head differs from the manifest")
        else:
            errors.append("FAIL_MISSING: adjudication files are not listed outputs")

    gold = manifest.get("gold")
    if manifest.get("artifact_kind") == GOLD_KIND:
        if not gold or gold.get("file") not in parsed:
            errors.append("FAIL_GOLD: gold freeze manifest does not list a readable gold file")
        else:
            rows = parsed[gold["file"]]
            ids = [str(r.get(id_key)) for r in rows]
            if len(set(ids)) != len(ids):
                errors.append("FAIL_GOLD: an item appears more than once in the gold file")
            if len(ids) != (manifest.get("source") or {}).get("items"):
                errors.append("FAIL_GOLD: gold file does not cover every source item")
            if ids != sorted(ids):
                errors.append("FAIL_GOLD: gold file is not sorted by item id")
            if any(r.get(label_key) is None and r.get("gold_source") != UNRESOLVED for r in rows):
                errors.append("FAIL_GOLD: a gold record has no label")
            if _recount(rows, label_key, manifest.get("task_type")) != gold.get("label_counts"):
                errors.append("FAIL_COUNT: gold label counts recomputed from the file differ")
            sources = dict(
                sorted(Counter(r.get("label_source_kind", "passes") for r in rows).items())
            )
            if "label_sources" in gold and sources != gold["label_sources"]:
                errors.append(
                    "FAIL_COUNT: gold label sources (human/model) recomputed from the file differ"
                )
            if any(r.get("label_source_kind") == "model" for r in rows) and not manifest.get(
                "model_annotation"
            ):
                errors.append(
                    "FAIL_PROVENANCE: model-sourced gold labels in a package that declares none"
                )
            errors.extend(_check_gold(rows, manifest, parsed, id_key, renames))
        if manifest.get("completion_state") != "COMPLETE":
            errors.append("FAIL_GOLD: a gold freeze must be COMPLETE")
    errors.extend(_check_exclusions(manifest, parsed, id_key, label_key))
    definition_history = manifest.get("definition_history") or []
    errors.extend(verify_definition_chain(definition_history, "manifest definition history"))
    if definition_history and definition_history[-1].get("new_sha256") != manifest.get(
        "task_definition_sha256"
    ):
        errors.append(
            "FAIL_PROVENANCE: the definition history does not end at the manifest's task definition"
        )

    source = manifest.get("source") or {}
    base = root or _package_root(package_dir, manifest)
    source_path = base / str(source.get("path", ""))
    source_checked = source_path.is_file()
    if source_checked and sha256_file(source_path) != source_sha:
        errors.append(f"FAIL_SOURCE: {source.get('path')} no longer hashes to the pinned sha256")
    if not source_checked and require_source:
        errors.append(f"FAIL_SOURCE: source population not found at {source_path}")

    result = ValidationResult(
        name="annotation package",
        policy=REQUIRED_NONEMPTY,
        discovered=len(outputs),
        checked=len(outputs),
        passed=passed,
        failed=failed,
        errors=errors,
        detail={"outputs": len(outputs), "source_rehashed": int(source_checked)},
    )
    status = PASS if result.status == PASS else FAIL
    return result, {
        "status": status,
        "artifact_kind": manifest.get("artifact_kind"),
        "task_id": task_id,
        "completion_state": manifest.get("completion_state"),
        "outputs": len(outputs),
        "source_rehashed": source_checked,
        # Never a silent skip: if the source could not be re-hashed, the summary says where it looked.
        "source_unreachable": None if source_checked else str(source_path),
        "errors": result.all_errors(),
    }
