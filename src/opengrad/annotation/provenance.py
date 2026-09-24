"""Hash-chained change log for annotation state.

Labels may change while annotation is in progress -- that is ordinary work, not an error -- but every
change is recorded, and the record is tamper-evident:

* each state of an annotation (or adjudication) is serialised canonically and hashed (``state_sha256``);
* each change is a log entry carrying the state before and after, both hashes, who made it, when, an
  optional reason, and ``prev_entry_sha256`` -- the hash of the previous entry in the same chain;
* an entry's own ``entry_sha256`` covers all of that, so editing, deleting or reordering any entry breaks
  every later link.

Undo is a new entry that names the entry it reverts; the log is never updated in place. The same checks run
over the live store and over an exported package, so a frozen package proves its own history.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from opengrad.hashing import sha256_text

#: Fields covered by an annotation-log entry hash.
ANNOTATION_ENTRY_FIELDS = (
    "task_id",
    "session_id",
    "item_id",
    "action",
    "actor_id",
    "reason",
    "before_sha256",
    "after_sha256",
    "reverts_entry_sha256",
    "prev_entry_sha256",
    "recorded_at",
)

#: Fields covered by an adjudication-log entry hash.
ADJUDICATION_ENTRY_FIELDS = (
    "task_id",
    "sessions_key",
    "item_id",
    "action",
    "actor_id",
    "before_sha256",
    "after_sha256",
    "prev_entry_sha256",
    "recorded_at",
)


#: Fields covered by a definition-history entry hash. The definitions themselves are covered through
#: their hashes, which are re-derived from the stored JSON on verification.
DEFINITION_ENTRY_FIELDS = (
    "task_id",
    "previous_sha256",
    "new_sha256",
    "document",
    "annotations_at_change",
    "prev_entry_sha256",
    "recorded_at",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def state_sha256(image: dict[str, Any] | None) -> str | None:
    if image is None:
        return None
    return sha256_text(canonical(image))


def entry_sha256(entry: dict[str, Any], fields: tuple[str, ...]) -> str:
    return sha256_text(canonical({key: entry.get(key) for key in fields}))


def definition_sha(definition: Any) -> str:
    """Same digest as :meth:`TaskConfig.definition_sha256`, from a parsed definition."""
    return hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_definition_chain(
    entries: list[dict[str, Any]], label: str = "definition history"
) -> list[str]:
    """Each entry must hash to itself, link to the one before, start where the previous one ended, and
    carry definitions whose hashes are the ones it names."""
    errors: list[str] = []
    previous_entry: str | None = None
    previous_definition: str | None = None
    for position, entry in enumerate(entries):
        where = f"{label} entry {position}"
        if entry_sha256(entry, DEFINITION_ENTRY_FIELDS) != entry.get("entry_sha256"):
            errors.append(f"FAIL_CHAIN: {where}: entry_sha256 does not match its contents")
        if entry.get("prev_entry_sha256") != previous_entry:
            errors.append(
                f"FAIL_CHAIN: {where}: prev_entry_sha256 does not link to the previous entry"
            )
        if previous_definition is not None and entry.get("previous_sha256") != previous_definition:
            errors.append(
                f"FAIL_CHAIN: {where}: starts from a definition the previous entry did not end at"
            )
        for side in ("previous", "new"):
            if definition_sha(entry.get(f"{side}_definition")) != entry.get(f"{side}_sha256"):
                errors.append(
                    f"FAIL_CHAIN: {where}: the {side} definition does not hash to {side}_sha256"
                )
        previous_entry = entry.get("entry_sha256")
        previous_definition = entry.get("new_sha256")
    return errors


def verify_chain(entries: list[dict[str, Any]], fields: tuple[str, ...], label: str) -> list[str]:
    """Re-derive every state hash, entry hash and link. ``entries`` must be in log order.

    Each entry needs ``before``/``after`` as parsed images (or None) alongside the hashed fields.
    """
    errors: list[str] = []
    previous: str | None = None
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        where = f"{label} entry {position}"
        if state_sha256(entry.get("before")) != entry.get("before_sha256"):
            errors.append(f"FAIL_CHAIN: {where}: before-state does not match before_sha256")
        if state_sha256(entry.get("after")) != entry.get("after_sha256"):
            errors.append(f"FAIL_CHAIN: {where}: after-state does not match after_sha256")
        if entry.get("prev_entry_sha256") != previous:
            errors.append(
                f"FAIL_CHAIN: {where}: prev_entry_sha256 does not link to the previous entry"
            )
        if entry_sha256(entry, fields) != entry.get("entry_sha256"):
            errors.append(f"FAIL_CHAIN: {where}: entry_sha256 does not match its contents")
        reverts = entry.get("reverts_entry_sha256")
        if reverts is not None and reverts not in seen:
            errors.append(f"FAIL_CHAIN: {where}: reverts an entry that is not earlier in the chain")
        previous = entry.get("entry_sha256")
        if previous:
            seen.add(previous)
    return errors


def final_states(entries: list[dict[str, Any]]) -> dict[str, str | None]:
    """item_id -> the after-state hash of its last entry: what the current record must equal."""
    states: dict[str, str | None] = {}
    for entry in entries:
        states[str(entry["item_id"])] = entry.get("after_sha256")
    return states
