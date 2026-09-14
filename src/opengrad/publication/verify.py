"""Publication verification for OpenGrad.

OpenGrad has a release discipline but historically had no publication discipline.
Every rule governed the payload -- hashed manifests, hashed shards, a validator
that fails closed, a fingerprint that survives a rebuild -- while the act of
publishing, which is when an artifact becomes citable, had no gate at all. A
licence file could be published describing three sources while the payload
carried four, and nothing in the repository could notice.

This module is that gate. It is the single implementation behind both
``scripts/verify_publication.py`` (the interface used before declaring a release
complete) and the ``network``-marked pytest coverage, so the rules are not
written twice.

Two properties are load-bearing, and the first was missing until contract v2:

1. **A gate must prove it executed.** Every result reports what it discovered,
   checked, skipped and blocked, and a required population that discovers nothing
   is a failure rather than a pass. Without this, ``opengrad.registry.validate``
   could exit 0 having run no checks at all, and freeze validation could skip
   every canonical corpus while reporting PASS.
2. **A blocked check is never success.** "We could not check" and "we checked and
   it is fine" are different claims, and collapsing them is how a provenance
   defect ships.

The overall status is FAIL if anything failed, otherwise a blocked status if
anything is blocked, otherwise PASS. Only PASS is safe to declare.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any

from opengrad.verification import (
    BLOCKED_NETWORK,
    BLOCKED_OPTIONAL_DEPENDENCY,
    CONDITIONALLY_REQUIRED,
    FAIL,
    OPTIONAL,
    PASS,
    REQUIRED_NONEMPTY,
    VERIFIER_CONTRACT,
    Skip,
    ValidationResult,
    VerificationReport,
)

USER_AGENT = "opengrad-publication-verifier/1.0"

# URLs that are paginated or rate limited are reported as blocked rather than
# failed: the pin may be perfectly good and simply unverifiable right now.
BLOCKING_HTTP_CODES = {401, 403, 429, 500, 502, 503, 504}


def _result_from_errors(
    name: str,
    errors: list[str],
    ids: list[str],
    *,
    policy: str = OPTIONAL,
    precondition: str | None = None,
    detail: dict[str, int] | None = None,
    skipped: list[Skip] | None = None,
) -> ValidationResult:
    from opengrad.registry.validate import result_from

    return result_from(
        name, policy, ids, errors, precondition=precondition, detail=detail, skipped=skipped
    )


# ── Remote pin resolution ────────────────────────────────────────────────────


def _huggingface_target(url: str) -> tuple[str, str] | None:
    """(repo_type, repo_id) for a Hugging Face URL, or None."""
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        return None
    parts = url[len(prefix) :].split("?")[0].strip("/").split("/")
    if len(parts) >= 3 and parts[0] in {"datasets", "spaces"}:
        return ("datasets" if parts[0] == "datasets" else "spaces", "/".join(parts[1:3]))
    if len(parts) >= 2 and parts[0] not in {"api", "blog", "docs"}:
        return ("models", "/".join(parts[:2]))
    return None


def _github_repo(url: str) -> tuple[str, str] | None:
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        return None
    parts = url[len(prefix) :].split("?")[0].strip("/").split("/")
    if len(parts) < 2:
        return None
    return (parts[0], parts[1])


def _http_status(url: str, timeout: float) -> int | str:
    """Return an HTTP status code, or the string 'network' if unreachable."""
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return "network"


def _verify_huggingface(
    repo_type: str, repo_id: str, revision: str, timeout: float
) -> tuple[bool, str]:
    subject = f"{repo_type}/{repo_id} @ {revision[:12]}…"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        status = _http_status(
            f"https://huggingface.co/api/{repo_type}/{repo_id}/revision/{revision}", timeout
        )
        if status == 200:
            return True, ""
        if status == "network":
            return False, f"BLOCKED_NETWORK {subject}: could not reach huggingface.co"
        if status in BLOCKING_HTTP_CODES:
            return (
                False,
                (
                    f"BLOCKED_OPTIONAL_DEPENDENCY {subject}: unauthenticated lookup "
                    f"returned HTTP {status}; needs the optional `huggingface_hub` dependency"
                ),
            )
        return False, f"{subject}: revision not found upstream (HTTP {status})"
    try:
        HfApi().repo_info(
            repo_id,
            repo_type="dataset" if repo_type == "datasets" else "model",
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001 - the library raises several unrelated types
        message = str(exc)
        if any(token in message for token in ("Connection", "timed out", "Max retries", "offline")):
            return False, f"BLOCKED_NETWORK {subject}: {message[:160]}"
        if "401" in message or "403" in message or "gated" in message.lower():
            return False, f"BLOCKED_OPTIONAL_DEPENDENCY {subject}: {message[:160]}"
        return False, f"{subject}: {message[:160]}"
    return True, ""


def _verify_github(owner: str, repo: str, revision: str, timeout: float) -> tuple[bool, str]:
    subject = f"github/{owner}/{repo} @ {revision[:12]}…"
    status = _http_status(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{revision}", timeout
    )
    if status == 200:
        return True, ""
    if status == "network":
        return False, f"BLOCKED_NETWORK {subject}: could not reach api.github.com"
    if status in BLOCKING_HTTP_CODES:
        return False, f"BLOCKED_NETWORK {subject}: api.github.com returned HTTP {status}"
    return False, f"{subject}: commit not found upstream (HTTP {status})"


def collect_remote_pins(root: Path) -> list[tuple[str, str, str, str]]:
    """Every external pin the registry makes, as (url, unused, revision, described_by).

    A pin is only collected when the record says what kind of revision it is.
    ``source_revision.value`` is not always an upstream revision: for the
    canonical corpora it is the OpenGrad build commit, and pairing that with the
    release's Hugging Face URL would be an inference the record does not support
    and would produce a false failure. Those records are anchored internally --
    by the release manifest and the publication record -- and are verified by the
    freeze and reference checks instead of by the network.
    """
    from opengrad.registry import provenance

    pins: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(url: str, revision: str, described_by: str) -> None:
        key = (url, revision)
        if revision and url.startswith("http") and key not in seen:
            seen.add(key)
            pins.append((url, "", revision, described_by))

    for filename, key in (("datasets.yaml", "datasets"), ("models.yaml", "models")):
        path = root / "registry" / filename
        if not path.exists():
            continue
        try:
            import yaml
        except ImportError:
            continue
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            continue
        for record in registry.get(key, []) or []:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("id", "<missing id>"))
            block = record.get("source_revision")
            kind, revision = _revision_kind(record, block)
            license_block = record.get("license")
            license_source = ""
            if isinstance(license_block, dict) and license_block.get("verified") is True:
                license_source = str(license_block.get("source", "") or "")
                embedded = provenance._IMMUTABLE_PATH_RE.search(license_source)
                if embedded is not None:
                    add(
                        license_source,
                        embedded.group(0).strip("/").split("/")[1],
                        f"{record_id}.license",
                    )
            if kind == "huggingface":
                repository = str(record.get("hf_id") or "")
                if not repository:
                    target = _huggingface_target(license_source or "")
                    repository = target[1] if target else ""
                if repository:
                    add(
                        f"https://huggingface.co/datasets/{repository}",
                        revision,
                        f"{record_id}.source_revision",
                    )
            elif kind == "github":
                source = str(block.get("source", "")) if isinstance(block, dict) else ""
                if _github_repo(source) is not None:
                    add(source, revision, f"{record_id}.source_revision")
    return pins


def _revision_kind(record: dict[str, Any], block: Any) -> tuple[str, str]:
    """(kind, revision) for a record's declared source revision.

    ``kind`` is one of ``huggingface``, ``github``, ``internal`` or ``none``.
    """
    from opengrad.registry import provenance

    if not isinstance(block, dict) or block.get("verified") is not True:
        return ("none", "")
    value = block.get("value")
    if not provenance.is_immutable_revision(value):
        return ("none", "")
    described = str(block.get("source", "") or "")
    if "opengrad_git_commit" in described:
        # The build commit of the release itself. Its anchor is the manifest the
        # commit produced, which is a repository artifact, not an upstream repo.
        return ("internal", str(value))
    if described.startswith("https://github.com/") or _github_repo(described):
        return ("github", str(value))
    if "hugging face" in described.lower() or record.get("hf_id"):
        return ("huggingface", str(value))
    return ("none", str(value))


def _pin_units(pins: list[tuple[str, str, str, str]]) -> dict[tuple[str, ...], str]:
    """Deduplicate pins into the units actually resolved, keeping their subjects.

    The census is built from these units rather than from the raw claim list, so
    a claim collapsed into another claim's lookup cannot make ``discovered``
    exceed what was checked.
    """
    units: dict[tuple[str, ...], str] = {}
    for url, _, revision, described_by in pins:
        target = _huggingface_target(url)
        if target is not None:
            key: tuple[str, ...] = ("hf", target[0], target[1], revision)
        else:
            github = _github_repo(url)
            if github is None:
                continue
            key = ("gh", github[0], github[1], revision)
        units.setdefault(key, f"{described_by} -> {revision[:12]}…")
    return units


def check_remote_pins(root: Path, timeout: float, network: bool) -> ValidationResult:
    """Resolve every external pin, or account for every one it could not.

    The population is never allowed to collapse: an unreachable network blocks
    the pins it could not resolve and still reports how many were required. A
    verifier that quietly attempts zero pins and reports success is the failure
    mode this gate exists to prevent.
    """
    pins = collect_remote_pins(root)
    units = _pin_units(pins)
    ids = sorted(units.values())
    if not units:
        return ValidationResult(
            name="remote immutable pins",
            policy=OPTIONAL,
            # The skip is itself the discovered item: a gate cannot skip something
            # it never saw, and a skip recorded without discovery is an accounting
            # contradiction rather than a pass.
            discovered=1,
            checked=0,
            blocked=0,
            skipped=[Skip(id="<no externally pinned claims>", reason="nothing to resolve")],
            detail={"registry_claims": len(pins), "distinct_pins": 0},
        )
    if not network:
        return ValidationResult(
            name="remote immutable pins",
            policy=REQUIRED_NONEMPTY,
            discovered=len(ids),
            blocked=len(ids),
            blocked_status=BLOCKED_NETWORK,
            detail={
                "registry_claims": len(pins),
                "distinct_pins": len(units),
                "required": len(ids),
                "attempted": 0,
                "resolved": 0,
            },
        )
    resolved: list[tuple[tuple[str, ...], str]] = []
    rejected: list[tuple[tuple[str, ...], str, str]] = []
    blocked: list[tuple[tuple[str, ...], str, str]] = []
    for key, subject in units.items():
        if key[0] == "hf":
            ok, message = _verify_huggingface(key[1], key[2], key[3], timeout)
        else:
            ok, message = _verify_github(key[1], key[2], key[3], timeout)
        if ok:
            resolved.append((key, subject))
        elif message.startswith("BLOCKED_"):
            blocked.append((key, subject, message))
        else:
            rejected.append((key, subject, message))
    blocked_status: str | None = None
    if blocked:
        blocked_status = (
            BLOCKED_OPTIONAL_DEPENDENCY
            if any(entry[2].startswith("BLOCKED_OPTIONAL_DEPENDENCY") for entry in blocked)
            else BLOCKED_NETWORK
        )
    blocked_reasons = [entry[2].split(" ", 1)[1] for entry in blocked]
    errors = [entry[2] for entry in rejected]
    return ValidationResult(
        name="remote immutable pins",
        policy=REQUIRED_NONEMPTY,
        discovered=len(ids),
        checked=len(resolved) + len(rejected),
        passed=len(resolved),
        failed=len(rejected),
        blocked=len(blocked),
        errors=errors,
        blocked_reasons=blocked_reasons,
        blocked_status=blocked_status,
        detail={
            "registry_claims": len(pins),
            "distinct_pins": len(units),
            "required": len(ids),
            "attempted": len(resolved) + len(rejected) + len(blocked),
            "resolved": len(resolved),
        },
    )


def check_remote_freeze_identities(root: Path, timeout: float, network: bool) -> ValidationResult:
    """Reproduce a frozen corpus identity from its immutable published artifact.

    Some corpora were built on the training host and their release manifest is not
    in this repository, so the digest cannot be re-derived from committed bytes.
    Such a record may name the published artifact and revision its identity
    belongs to. That claim is only worth anything if something actually fetches
    it: this gate downloads the named file at the named revision and hashes it,
    so a remotely anchored identity is reproduced rather than asserted.

    Offline, the population is blocked rather than dropped -- an unavailable
    network must never reduce a required population to zero.
    """
    from opengrad.registry.validate import _load_datasets

    records, _ = _load_datasets(root)
    declared: list[tuple[str, dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        derived = record.get("processed_dataset_hash")
        if not isinstance(derived, dict):
            continue
        remote = derived.get("remote_identity")
        if isinstance(remote, dict) and remote.get("hub_revision"):
            declared.append((str(record.get("id")), {**remote, "value": derived.get("value")}))
    ids = [record_id for record_id, _ in declared]
    if not declared:
        return ValidationResult(
            name="remote freeze identities",
            policy=OPTIONAL,
            # The skip is itself the discovered item: a gate cannot skip something
            # it never saw, and a skip recorded without discovery is an accounting
            # contradiction rather than a pass.
            discovered=1,
            checked=0,
            blocked=0,
            skipped=[Skip(id="<none>", reason="no identity is anchored remotely")],
        )
    if not network:
        return ValidationResult(
            name="remote freeze identities",
            policy=REQUIRED_NONEMPTY,
            discovered=len(ids),
            blocked=len(ids),
            blocked_status=BLOCKED_NETWORK,
            detail={"required": len(ids), "attempted": 0, "reproduced": 0},
        )
    reproduced: list[str] = []
    rejected: list[str] = []
    blocked: list[str] = []
    for record_id, remote in declared:
        repository = str(remote.get("hub_repository", ""))
        revision = str(remote.get("hub_revision", ""))
        path = str(remote.get("path", ""))
        expected = str(remote.get("value", ""))
        if not (repository and revision and path and expected):
            rejected.append(
                f"{record_id}: remote_identity is incomplete "
                f"(needs hub_repository, hub_revision, path)"
            )
            continue
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            blocked.append(f"{record_id}: needs the optional `huggingface_hub` dependency")
            continue
        try:
            local = hf_hub_download(repository, path, revision=revision, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if any(
                token in message
                for token in ("Connection", "timed out", "Max retries", "offline", "Network")
            ):
                blocked.append(f"{record_id}: could not reach {repository}: {message[:120]}")
            else:
                rejected.append(
                    f"{record_id}: {repository}@{revision[:12]}…/{path} could not be "
                    f"retrieved: {message[:120]}"
                )
            continue
        actual = sha256(Path(local).read_bytes()).hexdigest()
        if actual == expected:
            reproduced.append(record_id)
        else:
            rejected.append(
                f"{record_id}: published artifact does not reproduce the recorded identity "
                f"(expected {expected[:12]}…, {path} at {revision[:12]}… is {actual[:12]}…)"
            )
    blocked_status = BLOCKED_NETWORK if blocked else None
    return ValidationResult(
        name="remote freeze identities",
        policy=REQUIRED_NONEMPTY,
        discovered=len(ids),
        checked=len(reproduced) + len(rejected),
        passed=len(reproduced),
        failed=len(rejected),
        blocked=len(blocked),
        errors=rejected,
        blocked_reasons=blocked,
        blocked_status=blocked_status,
        detail={
            "required": len(ids),
            "attempted": len(reproduced) + len(rejected) + len(blocked),
            "reproduced": len(reproduced),
        },
    )


def _check_legacy_markers(root: Path) -> ValidationResult:
    """Legacy provenance markers are valid only for the named artifacts."""
    from opengrad.registry import provenance

    errors: list[str] = []
    candidates: list[str] = []
    for path in sorted((root / "reports/releases").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        for _, value in _walk(record):
            if value.get("provenance_version") != provenance.LEGACY_PROVENANCE_VERSION:
                continue
            identifier = str(value.get("repository") or value.get("id") or "<unnamed artifact>")
            candidates.append(identifier)
            errors.extend(provenance.check_legacy_marker(value, identifier))
    return _result_from_errors(
        "legacy provenance markers",
        errors,
        sorted(set(candidates)),
        policy=OPTIONAL,
        detail={"documented_legacy_artifacts": len(set(candidates))},
        skipped=[]
        if candidates
        else [Skip(id="<no legacy markers>", reason="no artifact declares a legacy version")],
    )


def _walk(value: Any) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        found.append(("", value))
        for item in value.values():
            found.extend(_walk(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk(item))
    return found


def verify_publication(
    root: Path, *, network: bool = True, timeout: float = 15.0
) -> VerificationReport:
    """Run every publication-readiness check and return a census-bearing report."""
    from opengrad.registry import validate as registry_validate

    report = VerificationReport(contract=VERIFIER_CONTRACT)
    audit = registry_validate.audit(root)
    for gate in (
        "structure",
        "references",
        "provenance",
        "freeze",
        "publication_records",
        "reconstructed_events",
        "semantic",
    ):
        report.add(audit[gate])
    report.add(_check_legacy_markers(root))
    report.add(check_remote_freeze_identities(root, timeout, network))
    report.add(check_remote_pins(root, timeout, network))
    return report


__all__ = [
    "BLOCKED_NETWORK",
    "BLOCKED_OPTIONAL_DEPENDENCY",
    "CONDITIONALLY_REQUIRED",
    "FAIL",
    "PASS",
    "VERIFIER_CONTRACT",
    "ValidationResult",
    "VerificationReport",
    "check_remote_freeze_identities",
    "check_remote_pins",
    "collect_remote_pins",
    "verify_publication",
]
