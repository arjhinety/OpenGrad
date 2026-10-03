"""Study 002's flag set (`study_002_prereg_v14`, 46 §5): the records the corpus intervention targets.

The flag set `F` is every canonical-v2-final record labelled `ANSWER` whose first reply the frozen
`prose-decision-classifier-v2` predicts `UNSUPPORTED` (a decline). The first reply is read under the shape rules of
the input contract `prose-decision-input-v2` (36 §2): after a leading system message the record opens with one
user turn and one assistant turn, the reply carries no structured call, and its content is non-empty text. What
follows the first reply does not matter, so multi-turn records count.

Beside `F` the manifest reports, so the choice of flag set stays checkable:

* every `ANSWER`-labelled first reply by predicted label (`CLARIFY` and `ABSTAIN` are counted, not flagged);
* the prior of 03: `HEURISTIC_REGEX_v1` on single-exchange records labelled `ANSWER` (the 18,114 of Study 001's
  audit), recomputed here, and its overlap with `F`.

The release is read from the Hugging Face cache and every shard is checked against the repository's release
manifest, which M0's dataset manifest pins, before any record is read. The classifier is checked to be the frozen
source. Counts and identifiers only: nothing prints or writes item text. The members file lists, per flagged
record, its id, canonical hash, source and unit kind; the triage population built from it materialises text
from the same verified release.

    python -m opengrad.verification.flag_set --dry-run      # counts, writes nothing
    python -m opengrad.verification.flag_set --build        # writes reports/study-002/flag-set/
    python -m opengrad.verification.flag_set --verify
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opengrad.data.classifier_input import ClassifierFeatures
from opengrad.data.decision_classifier_v2 import CLASSIFIER_VERSION, UNSUPPORTED, classify
from opengrad.evaluation.capability import detect_refusal
from opengrad.hashing import sha256_bytes

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("reports/study-002/flag-set")
MANIFEST_NAME = "flag-set.manifest.json"
MEMBERS_NAME = "flag-set.members.jsonl"
RELEASE_MANIFEST = Path(".release/hf/toolpolicy-canonical-v2-final/release-manifest.json")
DATASET_MANIFEST = Path("runs/m0_sft_canonical_v2_final/dataset_manifest.json")
REPOSITORY = "arrochi112/OpenGrad-ToolPolicy-Canonical-v2"
REVISION = "df1a1f5135cc4b08e24b7ff20582084988932c61"
# The builder and everything that decides a flag: the first-reply features and classifier v2 (also checked
# against its frozen tag on build). The v1 regex lives in the shared evaluation module capability.py, so only the
# detector itself is pinned (owner, 2026-10-03): see detector_fingerprint.
CODE_MODULES = (
    "src/opengrad/verification/flag_set.py",
    "src/opengrad/data/classifier_input.py",
    "src/opengrad/data/decision_classifier_v2.py",
)
#: The module-level names `detect_refusal` reads; a test fails if it starts reading anything else, so the
#: fingerprint below cannot silently miss a dependency.
DETECTOR_GLOBALS = ("RefusalVerdict", "REFUSAL_WINDOW_CHARS", "_COMPILED")
COLUMNS = [
    "opengrad_id",
    "source_dataset",
    "canonical_hash",
    "behavior_decision",
    "tools",
    "messages",
]

UNIT_SINGLE_EXCHANGE = "single_exchange"
UNIT_HAS_CONTINUATION = "has_continuation"
# Why a record has no eligible first reply (the shape rules of contract v2).
NOT_USER_THEN_ASSISTANT = "first_turn_not_one_user_then_assistant"
STRUCTURAL_CALL = "first_reply_is_structural_call"
EMPTY_REPLY = "first_reply_empty"


class FlagSetError(RuntimeError):
    pass


def _lf_sha256(root: Path, path: str) -> str:
    return sha256_bytes((root / path).read_bytes().replace(b"\r\n", b"\n"))


def detector_fingerprint() -> str:
    """sha256 of the v1 regex as it decides: its function, its verdict type, its window and each compiled pattern."""
    from opengrad.evaluation import capability

    payload = {
        "detect_refusal": inspect.getsource(capability.detect_refusal).replace("\r\n", "\n"),
        "RefusalVerdict": inspect.getsource(capability.RefusalVerdict).replace("\r\n", "\n"),
        "REFUSAL_WINDOW_CHARS": capability.REFUSAL_WINDOW_CHARS,
        "_COMPILED": [[name, rx.pattern, int(rx.flags)] for name, rx in capability._COMPILED],
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _input_missing(exc: BaseException) -> bool:
    """True only when the release is not in the local cache, or pyarrow or huggingface_hub is not installed."""
    if isinstance(exc, ModuleNotFoundError):
        return exc.name in ("pyarrow", "huggingface_hub")
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False
    return isinstance(exc, LocalEntryNotFoundError)


def load_release(root: Path) -> tuple[Path, dict[str, Any]]:
    """The verified canonical-v2-final release: every shard's size and sha256 against the pinned manifest."""
    from huggingface_hub import snapshot_download

    manifest_bytes = (root / RELEASE_MANIFEST).read_bytes()
    pinned = json.loads((root / DATASET_MANIFEST).read_text(encoding="utf-8"))["corpus"][
        "manifest_sha256"
    ]
    if sha256_bytes(manifest_bytes) != pinned:
        raise FlagSetError(
            "the repository's release manifest is not the one M0's dataset manifest pins"
        )
    manifest = json.loads(manifest_bytes)
    path = Path(
        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["*.parquet"],
            local_files_only=True,
        )
    )
    for shard in manifest["output_shards"]:
        data = (path / shard["file"]).read_bytes()
        if sha256_bytes(data) != shard["sha256"] or len(data) != shard["bytes"]:
            raise FlagSetError(f"{shard['file']} does not match the release manifest")
    return path, {
        "repository": REPOSITORY,
        "revision": REVISION,
        "release_manifest_sha256": sha256_bytes(manifest_bytes),
        "shards_verified": len(manifest["output_shards"]),
        "record_count": manifest["record_count"],
    }


def iter_records(root: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path, _ = load_release(root)
    manifest = json.loads((root / RELEASE_MANIFEST).read_text(encoding="utf-8"))
    for shard in manifest["output_shards"]:
        yield from pq.read_table(path / shard["file"], columns=COLUMNS).to_pylist()


def first_reply(messages: list[Any]) -> tuple[str | None, dict[str, Any] | None, str | None, str]:
    """(user message, reply, ineligibility reason or None, unit kind) under contract v2's shape rules."""
    body = messages[1:] if messages and messages[0].get("role") == "system" else messages
    roles = [m.get("role") for m in body[:2]]
    unit = UNIT_SINGLE_EXCHANGE if len(body) == 2 else UNIT_HAS_CONTINUATION
    if roles != ["user", "assistant"]:
        return None, None, NOT_USER_THEN_ASSISTANT, unit
    user, reply = body[0], body[1]
    if reply.get("tool_calls"):
        return user.get("content"), reply, STRUCTURAL_CALL, unit
    content = reply.get("content")
    if not (isinstance(content, str) and content.strip()) or not isinstance(
        user.get("content"), str
    ):
        return user.get("content"), reply, EMPTY_REPLY, unit
    return user["content"], reply, None, unit


def evaluate(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one record's first reply. Returns what the counts and the members file need, never text."""
    messages = json.loads(record["messages"])
    tools = json.loads(record["tools"]) or []
    user, reply, reason, unit = first_reply(messages)
    out: dict[str, Any] = {
        "decision": record["behavior_decision"],
        "unit_kind": unit,
        "ineligible": reason,
    }
    if reason is not None or user is None or reply is None:
        out["label"] = None
        out["regex_v1"] = False
        return out
    decision = classify(
        ClassifierFeatures(
            user_message=user,
            assistant_response=reply["content"],
            tools=tuple(tools),
            structured_call_present=False,
        )
    )
    out["label"] = decision.label
    out["step"] = decision.step
    # 03's prior: the v1 regex on single exchanges (Study 001's audit scope).
    out["regex_v1"] = unit == UNIT_SINGLE_EXCHANGE and detect_refusal(reply["content"]).is_refusal
    return out


def build(root: Path = ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from opengrad.verification import prose_classifier_v2_oneshot as oneshot

    oneshot.check_frozen(root)
    _, release = load_release(root)
    members: list[dict[str, Any]] = []
    answer_labels: Counter[str] = Counter()
    ineligible: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_unit: Counter[str] = Counter()
    regex_v1: Counter[str] = Counter()
    overlap: Counter[str] = Counter()
    answer_labels_by_source: Counter[tuple[str, str]] = Counter()
    records = 0
    for record in iter_records(root):
        records += 1
        result = evaluate(record)
        if result["decision"] != "ANSWER":
            continue
        source = str(record["source_dataset"])
        if result["ineligible"] is not None:
            ineligible[result["ineligible"]] += 1
            continue
        answer_labels[result["label"]] += 1
        answer_labels_by_source[(source, result["label"])] += 1
        flagged = result["label"] == UNSUPPORTED
        if result["regex_v1"]:
            regex_v1[source] += 1
            overlap[source] += flagged
        if not flagged:
            continue
        by_source[source] += 1
        by_unit[result["unit_kind"]] += 1
        members.append(
            {
                "opengrad_id": str(record["opengrad_id"]),
                "canonical_hash": str(record["canonical_hash"]),
                "source_dataset": source,
                "unit_kind": result["unit_kind"],
                "classifier_step": result["step"],
            }
        )
    if records != release["record_count"]:
        raise FlagSetError(
            f"read {records} records, the release manifest says {release['record_count']}"
        )
    members.sort(key=lambda m: m["opengrad_id"])
    members_bytes = _members_bytes(members)
    manifest = {
        "artifact_kind": "STUDY_002_FLAG_SET",
        "preregistration": {
            "adoption_amendment": "study_002_prereg_v14",
            "document": "docs/research/study-002/46-READINESS-DESIGN-AMENDMENT-DRAFT.md",
            "section": "§5",
        },
        "definition": (
            "canonical-v2-final records labelled ANSWER whose first reply (contract prose-decision-input-v2 shape) "
            "prose-decision-classifier-v2 predicts UNSUPPORTED"
        ),
        "classifier_version": CLASSIFIER_VERSION,
        "corpus": release,
        "records_read": records,
        "flag_set": {
            "n": len(members),
            "by_source": dict(sorted(by_source.items())),
            "by_unit_kind": dict(sorted(by_unit.items())),
        },
        "answer_labelled_first_replies": {
            "eligible_by_predicted_label": dict(sorted(answer_labels.items())),
            "ineligible_by_reason": dict(sorted(ineligible.items())),
        },
        "prior_heuristic_regex_v1": {
            "scope": "single-exchange records labelled ANSWER whose reply HEURISTIC_REGEX_v1 calls a refusal",
            "n": sum(regex_v1.values()),
            "by_source": dict(sorted(regex_v1.items())),
            "also_in_flag_set": sum(overlap.values()),
            "also_in_flag_set_by_source": dict(sorted(overlap.items())),
        },
        # The triage reports flag precision for each source as well as pooled, because a pooled figure can hide
        # a source the classifier reads badly (its one test set held no When2Call item, and the v1 regex and
        # classifier v2 disagree most on ToolACE).
        "precision_reporting": {
            "unit": "per source and pooled",
            "sources": sorted(by_source),
            "flags_by_source": dict(sorted(by_source.items())),
            "eligible_answer_first_replies_by_source_and_label": {
                source: {
                    label: count
                    for (s2, label), count in sorted(answer_labels_by_source.items())
                    if s2 == source
                }
                for source in sorted({s2 for s2, _ in answer_labels_by_source})
            },
        },
        "members": {
            "file": MEMBERS_NAME,
            "sha256": sha256_bytes(members_bytes),
            "rows": len(members),
        },
        "code_sha256_lf": {path: _lf_sha256(root, path) for path in CODE_MODULES},
        "detector_sha256": detector_fingerprint(),
        "statement": (
            "Counts and identifiers only. A flag is a classifier prediction, not a label: the triage of 46 §5 "
            "decides each record's disposition, and measures the flag precision stop rule 1 needs."
        ),
    }
    return manifest, members


def _members_bytes(members: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(m, sort_keys=True) + "\n" for m in members).encode("utf-8")


def write(root: Path, manifest: dict[str, Any], members: list[dict[str, Any]]) -> None:
    out = root / OUTPUT_DIR
    if (out / MANIFEST_NAME).exists():
        raise FlagSetError(
            f"{OUTPUT_DIR / MANIFEST_NAME} exists; a built flag set is never overwritten"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / MEMBERS_NAME).write_bytes(_members_bytes(members))
    (out / MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def verify(root: Path = ROOT, *, rebuild: bool = True) -> dict[str, Any]:
    """Re-hash the committed members file and code; rebuild from the release when it is in the local cache."""
    manifest = json.loads((root / OUTPUT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    errors = []
    members_bytes = (root / OUTPUT_DIR / MEMBERS_NAME).read_bytes()
    if sha256_bytes(members_bytes) != manifest["members"]["sha256"]:
        errors.append("the members file does not hash to the manifest's sha256")
    for path, digest in manifest["code_sha256_lf"].items():
        if _lf_sha256(root, path) != digest:
            errors.append(f"{path} has changed since the flag set was built")
    if detector_fingerprint() != manifest["detector_sha256"]:
        errors.append("the v1 regex (detect_refusal) has changed since the flag set was built")
    if not rebuild:
        return {
            "status": "FAIL" if errors else "PASS",
            "errors": errors,
            "n": manifest["flag_set"]["n"],
        }
    try:
        rebuilt, members = build(root)
    except Exception as exc:  # noqa: BLE001 -- classified below; only a cache miss is a missing input
        # The release not in the local cache (CI), or pyarrow or huggingface_hub not installed: hashes only.
        # Anything else (a missing pin, a shard or pin mismatch, an unfrozen classifier, a broken module) fails.
        if _input_missing(exc):
            return {
                "status": "BLOCKED_INPUT_MISSING" if not errors else "FAIL",
                "errors": errors,
                "reason": str(exc)[:200],
                "n": manifest["flag_set"]["n"],
            }
        return {
            "status": "FAIL",
            "errors": [*errors, f"{type(exc).__name__}: {str(exc)[:200]}"],
            "n": manifest["flag_set"]["n"],
        }
    if _members_bytes(members) != members_bytes:
        errors.append("rebuilding from the release gives a different members file")
    for key in (
        "flag_set",
        "answer_labelled_first_replies",
        "prior_heuristic_regex_v1",
        "precision_reporting",
    ):
        if rebuilt[key] != manifest[key]:
            errors.append(f"rebuilding gives a different {key}")
    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "n": manifest["flag_set"]["n"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify(ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("PASS", "BLOCKED_INPUT_MISSING") else 1
    manifest, members = build(ROOT)
    if args.build:
        write(ROOT, manifest, members)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("flag_set", "answer_labelled_first_replies", "prior_heuristic_regex_v1")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
