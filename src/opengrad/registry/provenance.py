"""Provenance rules for registry claims.

OpenGrad requires immutable provenance, not hashes for the sake of hashes. A
``verified: true`` claim means nothing on its own: it is an assertion that the
evidence behind a value was checked. This module decides whether the *anchor*
that evidence resolves to is one that cannot change while the record stays the
same.

A verified claim resolves through one of two routes::

    verified claim
        |
        +-- mutable/local source
        |       -> content digest required (source_sha256)
        |
        +-- externally immutable source
                -> immutable revision/pin required

A repository path is mutable: the same path can hold different bytes tomorrow.
Pointing a claim at ``reports/foo.json`` therefore pins nothing, and the claim
must carry ``source_sha256`` instead. An external source pinned to an exact
commit, revision or immutable release is already anchored, and is *not* asked
for a redundant digest -- uniformity is not the requirement, immutability is.

Derived artifacts are held to the composition of the two::

    immutable source/input identity
                |
                v
         deterministic transform
                |
                v
           derived identity

This module is deliberately dependency-free and offline. It reads already-parsed
registry structures and hashes files on disk. Reaching the network to confirm an
external pin still exists upstream is the publication verifier's job, not this
one's; keeping the rules and the remote check separate is what lets the rules run
deterministically in CI.

Historical artifacts that predate this model are supported through a narrow,
named allowlist rather than a general escape hatch. See ``LEGACY_ARTIFACTS``.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Anchor vocabularies ──────────────────────────────────────────────────────

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FULL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHORT_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Categories a sound claim can fall into. These are the four the migration
# report counts; CORRECTED is not a runtime category because a corrected claim
# is simply reported under whichever category it now satisfies.
MUTABLE_HASHED = "mutable+hashed"
IMMUTABLE_PINNED = "immutable+pinned"
DOCUMENTED_LEGACY = "documented legacy"

# Tokens that make a URL a moving target. A URL is only an anchor if resolving
# it returns the same bytes forever, and every one of these says "the default
# branch" or "whatever is newest now".
MUTABLE_REF_TOKENS = (
    "/blob/main",
    "/blob/master",
    "/blob/head",
    "/tree/main",
    "/tree/master",
    "/resolve/main",
    "/resolve/master",
    "/raw/main",
    "/raw/master",
    "/releases/latest",
    "refs/heads/",
    "?ref=main",
    "?ref=master",
)
# A path segment that is a commit-ish, e.g. /blob/<40 hex>/LICENSE or
# /commit/<40 hex>. This is what makes GitHub and Hugging Face permalinks
# immutable.
_IMMUTABLE_PATH_RE = re.compile(r"/(?:blob|tree|resolve|raw|commit)/[0-9a-f]{40}(?:/|$|\\?)")

# ── Legacy exceptions ────────────────────────────────────────────────────────

LEGACY_PROVENANCE_VERSION = "legacy_single_digest_v1"

# The only artifacts permitted to carry LEGACY_PROVENANCE_VERSION.
#
# OpenGrad ToolPolicy Canonical v2-minus-xLAM was published under a single
# derived-digest model. Its card was later corrected for interpretation only
# (the removal drops the corpus's entire CALL_PREDICTION channel, not just a
# source), which moved the derived fingerprint from f8ba687e to 5fc73904 while
# the 118 parquet shards stayed byte-identical to the frozen parent. Deriving a
# source_digest for it now would mean inventing an input identity that was never
# recorded, so the artifact is preserved as it was and named here instead.
#
# This is not a general escape hatch. A record carrying the marker without being
# in this map is a validation failure, and the map is asserted to stay small.
LEGACY_ARTIFACTS: dict[str, str] = {
    "OpenGrad-ToolPolicy-Canonical-v2-minus-xlam": (
        "Published under the single-digest model before source_digest/derived_digest "
        "were separated. The interpretation-only card correction moved the derived "
        "fingerprint while the parquet shards stayed byte-identical to the parent; "
        "the superseded and current fingerprints are both preserved in "
        "reports/releases/hf-publication-2026-09-11-minus-xlam-ablation.json."
    ),
}

# A legacy list that grows without bound is not an exception, it is a policy.
MAX_LEGACY_ARTIFACTS = 3


@dataclass
class ClaimAudit:
    """What the audit concluded about one verified claim."""

    record_id: str
    claim: str
    category: str
    anchor: str
    errors: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        return not self.errors


@dataclass
class AuditResult:
    claims: list[ClaimAudit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        result = {MUTABLE_HASHED: 0, IMMUTABLE_PINNED: 0, DOCUMENTED_LEGACY: 0}
        for claim in self.claims:
            if claim.sound and claim.category in result:
                result[claim.category] += 1
        return result


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def is_immutable_revision(value: Any) -> bool:
    """True when ``value`` pins an external source to one immutable state.

    A full 40-hex commit is the only form accepted here. Short revisions are
    rejected on purpose: abbreviation is ambiguous, and a seven-character prefix
    can collide, so it does not pin anything.
    """
    return isinstance(value, str) and bool(FULL_REVISION_RE.match(value))


def mutable_ref(url: str) -> str | None:
    """Return the moving-target token in ``url``, or None if it is pinned."""
    lowered = url.lower()
    for token in MUTABLE_REF_TOKENS:
        if token in lowered:
            return token
    return None


def is_external(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )


def is_work_tree(root: Path) -> bool:
    """True when ``root`` is inside a Git work tree."""
    try:
        result = _git(root, "rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == b"true"


def read_evidence_bytes(root: Path, relative: str) -> tuple[bytes | None, str | None]:
    """The bytes a local provenance claim must hash, and why they are unavailable.

    In a Git work tree these are the **committed** bytes. That matters more than
    it looks: a working-tree file is not a stable artifact. This repository sets
    `core.autocrlf=true`, so a `.jsonl` checked out on Windows has CRLF where the
    committed blob has LF, and hashing the file on disk would make a
    `source_sha256` a claim about the verifier's operating system rather than
    about the artifact. Hashing the committed blob makes the digest
    platform-independent, and requiring the path to be tracked means a
    `verified: true` claim can only rest on a file that is actually part of the
    research record -- which is exactly what broke when two licence claims
    pointed into `.release/`, a gitignored build directory.

    Outside a work tree (unit-test fixtures) the working-tree bytes are used, and
    callers can tell the difference.
    """
    if not is_work_tree(root):
        path = root / relative
        if not path.exists():
            return None, f"evidence file is missing: {relative}"
        return path.read_bytes(), None
    listed = _git(root, "ls-files", "--error-unmatch", "--", relative)
    if listed.returncode != 0:
        return None, (
            f"evidence path is not tracked by Git, so it is not part of the record: {relative}"
        )
    blob = _git(root, "cat-file", "blob", f"HEAD:{relative}")
    if blob.returncode != 0:
        return None, f"evidence path is tracked but not committed at HEAD: {relative}"
    return blob.stdout, None


# ── Claim checks ─────────────────────────────────────────────────────────────


def _check_local_anchor(
    record_id: str, claim: str, source: str, digest: Any, root: Path
) -> ClaimAudit:
    audit = ClaimAudit(record_id=record_id, claim=claim, category=MUTABLE_HASHED, anchor=source)
    if not is_sha256(digest):
        audit.errors.append(
            f"{record_id}: {claim} verifies against the mutable local path '{source}' "
            f"without source_sha256; a path alone pins nothing"
        )
        return audit
    payload, problem = read_evidence_bytes(root, source)
    if payload is None:
        audit.errors.append(f"{record_id}: {claim} {problem}")
        return audit
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        audit.errors.append(
            f"{record_id}: {claim} source_sha256 does not match {source} "
            f"(recorded {str(digest)[:12]}…, found {actual[:12]}…)"
        )
    return audit


def _check_external_anchor(
    record_id: str,
    claim: str,
    source: str,
    pinned_revision: Any,
    revision_source: str,
) -> ClaimAudit:
    audit = ClaimAudit(record_id=record_id, claim=claim, category=IMMUTABLE_PINNED, anchor=source)
    token = mutable_ref(source)
    if token is not None:
        audit.errors.append(
            f"{record_id}: {claim} evidence URL '{source}' points at the mutable ref "
            f"'{token}'; use a commit or revision permalink"
        )
        return audit
    if _IMMUTABLE_PATH_RE.search(source):
        return audit
    # A bare repository URL carries no revision of its own. It is still an anchor
    # if the record pins the revision alongside it, which is how every upstream
    # dataset entry here is written.
    if is_immutable_revision(pinned_revision):
        audit.anchor = f"{source} @ {pinned_revision}"
        return audit
    audit.errors.append(
        f"{record_id}: {claim} verifies against external source '{source}' that is "
        f"neither hashed nor immutably pinned ({revision_source})"
    )
    return audit


def check_license_claim(record: dict[str, Any], root: Path, record_id: str) -> ClaimAudit | None:
    """Audit a record's ``license`` block, if it claims verification."""
    block = record.get("license")
    if not isinstance(block, dict) or block.get("verified") is not True:
        return None
    source = str(block.get("source", "") or "")
    if not source:
        return ClaimAudit(
            record_id=record_id,
            claim="license",
            category=MUTABLE_HASHED,
            anchor="",
            errors=[f"{record_id}: license is marked verified but names no source"],
        )
    if is_external(source):
        revision = _record_revision(record)
        return _check_external_anchor(
            record_id,
            "license",
            source,
            revision,
            "no source_revision or exact_revision on the record",
        )
    return _check_local_anchor(record_id, "license", source, block.get("source_sha256"), root)


def check_revision_claim(record: dict[str, Any], record_id: str) -> ClaimAudit | None:
    """Audit a record's ``source_revision`` block, if it claims verification.

    Here the claim's own ``value`` is the anchor, so the test is simply whether
    it is an immutable revision and not a branch name.
    """
    block = record.get("source_revision")
    if not isinstance(block, dict) or block.get("verified") is not True:
        return None
    value = block.get("value")
    audit = ClaimAudit(
        record_id=record_id,
        claim="source_revision",
        category=IMMUTABLE_PINNED,
        anchor=str(value),
    )
    if not is_immutable_revision(value):
        audit.errors.append(
            f"{record_id}: source_revision is marked verified but its value "
            f"{value!r} is not an immutable revision"
        )
    return audit


def _record_revision(record: dict[str, Any]) -> Any:
    """The revision a record pins, from whichever field it records it in."""
    exact = record.get("exact_revision")
    if is_immutable_revision(exact):
        return exact
    block = record.get("source_revision")
    if isinstance(block, dict) and is_immutable_revision(block.get("value")):
        return block["value"]
    return None


def check_legacy_marker(record: dict[str, Any], record_id: str) -> list[str]:
    """The legacy marker is only valid for a named historical artifact."""
    version = record.get("provenance_version")
    if version is None:
        return []
    if version != LEGACY_PROVENANCE_VERSION:
        return [
            (
                f"{record_id}: unknown provenance_version {version!r}; the only accepted "
                f"legacy marker is {LEGACY_PROVENANCE_VERSION!r}"
            )
        ]
    # A publication record identifies an artifact by repository name; a registry
    # entry identifies it by id. Match on the final path segment so both forms
    # resolve to the same allowlist entry.
    candidates = {record_id, record_id.rpartition("/")[2]}
    if not candidates & set(LEGACY_ARTIFACTS):
        return [
            (
                f"{record_id}: provenance_version {LEGACY_PROVENANCE_VERSION!r} is not "
                f"available to this artifact; the legacy exception is a named allowlist, "
                f"not an escape hatch"
            )
        ]
    return []


def check_two_digest(record: dict[str, Any], record_id: str) -> list[str]:
    """A derived artifact should record both the input and the output identity.

    The repository's existing names for these are `source_revision` (or
    `parent_fingerprint`) for the input and `processed_dataset_hash` /
    `checksum` / `derived_fingerprint` for the output. The check looks for the
    repository's own fields rather than inventing new vocabulary.
    """
    if record.get("provenance_version") == LEGACY_PROVENANCE_VERSION:
        return []
    derived = record.get("processed_dataset_hash")
    if not isinstance(derived, dict):
        return []
    value = derived.get("value")
    # Non-canonical or not-yet-built sources legitimately have no derived identity.
    if value in (None, ""):
        return []
    errors: list[str] = []
    if not is_sha256(value) and not str(value).startswith("see "):
        errors.append(
            f"{record_id}: processed_dataset_hash value {str(value)[:40]!r} is neither a "
            f"sha256 nor a pointer to a manifest"
        )
    checksum = record.get("checksum")
    # Only a genuine contradiction is an error: two different digests both
    # claiming to be the derived identity. A digest alongside a manifest pointer
    # is complementary, not contradictory -- the two fields are not documented
    # as synonyms, and deciding which one is authoritative is a research
    # judgment this validator is not entitled to make.
    if is_sha256(checksum) and is_sha256(value) and checksum != value:
        errors.append(
            f"{record_id}: checksum and processed_dataset_hash disagree, so the derived "
            f"identity of the artifact is ambiguous"
        )
    if _record_revision(record) is None and not record.get("derived_from"):
        errors.append(
            f"{record_id}: a derived artifact records no immutable input identity "
            f"(source_revision, exact_revision or derived_from)"
        )
    return errors


def audit_verified_claims(root: Path, records: list[dict[str, Any]], kind: str) -> AuditResult:
    """Audit every ``verified: true`` claim in a set of registry records."""
    result = AuditResult()
    if len(LEGACY_ARTIFACTS) > MAX_LEGACY_ARTIFACTS:
        result.errors.append(
            f"the legacy provenance allowlist names {len(LEGACY_ARTIFACTS)} artifacts, "
            f"above the {MAX_LEGACY_ARTIFACTS} this exception is scoped to"
        )
    for record in records:
        if not isinstance(record, dict) or "id" not in record:
            continue
        record_id = str(record["id"])
        for audit in (
            check_license_claim(record, root, record_id),
            check_revision_claim(record, record_id),
        ):
            if audit is not None:
                result.claims.append(audit)
                result.errors.extend(audit.errors)
        if kind == "datasets":
            result.errors.extend(check_legacy_marker(record, record_id))
            result.errors.extend(check_two_digest(record, record_id))
    return result
