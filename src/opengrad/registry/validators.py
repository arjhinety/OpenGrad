"""The registry validators `opengrad-validate` runs: structure, references, publication records,
reconstructed events, freezes and semantic consistency. Split out of `validate.py` on 2026-09-24 with no
change in behaviour; `validate` re-exports them.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the dev extra for YAML validation") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _digest_shaped(value: object) -> bool:
    """True for a bare hex string, which is the only thing we assert a length for."""
    return (
        isinstance(value, str)
        and bool(value)
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def validate_structure(root: Path) -> list[str]:
    """Schema-level checks: identity, required fields, enums, digest formats.

    These are the checks that need no research judgment. Anything requiring a
    decision about which of two recorded values is authoritative belongs in the
    registry, not here.
    """
    errors: list[str] = []
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"datasets.yaml: {exc}"]
    records = registry.get("datasets", []) if isinstance(registry, dict) else []

    # Presence and non-emptiness are different requirements. An empty
    # `forbidden_splits` is a meaningful value -- it says nothing is forbidden --
    # so only the scalar identity fields must be non-empty.
    required_present = (
        "id",
        "display_name",
        "organization",
        "license",
        "category",
        "allowed_splits",
        "forbidden_splits",
        "contamination_status",
    )
    required_nonempty = ("id", "display_name", "organization", "category", "contamination_status")
    seen: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            errors.append("datasets.yaml: record is not a mapping")
            continue
        record_id = str(record.get("id", "<missing id>"))
        seen[record_id] = seen.get(record_id, 0) + 1
        for key in required_present:
            if key not in record:
                errors.append(f"{record_id}: required field '{key}' is absent")
        for key in required_nonempty:
            if record.get(key) in (None, "", [], {}):
                errors.append(f"{record_id}: required field '{key}' is empty")
        for key in ("allowed_splits", "forbidden_splits"):
            if key in record and not isinstance(record[key], list):
                errors.append(f"{record_id}: '{key}' must be a list")
        allowed = set(record.get("allowed_splits") or [])
        forbidden = set(record.get("forbidden_splits") or [])
        if allowed & forbidden:
            errors.append(
                f"{record_id}: splits are both allowed and forbidden: {sorted(allowed & forbidden)}"
            )
        percent = record.get("planned_initial_mixture_percent")
        if percent is not None and not (isinstance(percent, (int, float)) and 0 <= percent <= 100):
            errors.append(f"{record_id}: planned_initial_mixture_percent out of range: {percent!r}")
        date = record.get("retrieval_date")
        if date is not None and not (
            isinstance(date, str) and len(date) == 10 and date[4] == "-" and date[7] == "-"
        ):
            errors.append(f"{record_id}: retrieval_date is not ISO 8601: {date!r}")
        license_block = record.get("license")
        if isinstance(license_block, dict):
            digest = license_block.get("source_sha256")
            if digest is not None and not (
                isinstance(digest, str) and len(digest) == 64 and _digest_shaped(digest)
            ):
                errors.append(f"{record_id}: license.source_sha256 is not a sha256 digest")
            source = license_block.get("source")
            if isinstance(source, str) and source.startswith("/"):
                errors.append(
                    f"{record_id}: license.source is an absolute path ({source!r}); "
                    f"registry paths must be repository-relative"
                )
        for field_name in ("checksum", "exact_revision"):
            value = record.get(field_name)
            if isinstance(value, str) and _digest_shaped(value) and len(value) not in (40, 64):
                errors.append(
                    f"{record_id}: {field_name} looks like a digest but is {len(value)} "
                    f"characters, not 40 or 64"
                )
    for record_id, count in seen.items():
        if count > 1:
            errors.append(f"datasets.yaml: duplicate dataset id '{record_id}' ({count} entries)")
    if isinstance(registry, dict) and _schema_version(registry) >= 2:
        errors += _record_schema_errors(root, registry, "datasets.yaml", "datasets")
    return errors


DATASET_RECORD_SCHEMA = "registry/dataset_record.schema.json"


def _schema_version(registry: dict[str, Any]) -> int:
    value = registry.get("schema_version")
    return value if isinstance(value, int) else 0


def _record_schema_errors(
    root: Path, registry: dict[str, Any], name: str, key: str, expected: str = DATASET_RECORD_SCHEMA
) -> list[str]:
    """Validate every record against the JSON Schema the file declares.

    From schema_version 2 a registry names its record schema, and the schema -- not a list of
    field names in this module -- is the definition of a valid record. A file that stops naming
    it, or names another schema, is an error rather than a way to skip the check.
    """
    declared = registry.get("record_schema")
    if declared != expected:
        return [f"{name}: schema_version 2 requires record_schema: {expected} (found {declared!r})"]
    try:
        import jsonschema
    except ImportError:
        return [f"{name}: install the dev extra (jsonschema) to validate records"]
    try:
        schema = json.loads((root / expected).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
    except (OSError, ValueError, jsonschema.SchemaError) as exc:
        return [f"{expected}: {exc}"]
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for record in registry.get(key) or []:
        record_id = str(record.get("id", "<missing id>")) if isinstance(record, dict) else "?"
        for error in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path)):
            where = ".".join(str(part) for part in error.absolute_path) or "(record)"
            errors.append(f"{record_id}: {where}: {error.message} ({name} record schema)")
    return errors


SOURCE_SCREENING = "registry/source_screening.yaml"
SOURCE_SCREENING_SCHEMA = "registry/source_screening.schema.json"


def validate_source_screening(root: Path) -> tuple[list[str], list[str]]:
    """Check the screening log; return (candidate ids as `<screening>/<candidate>`, errors).

    Beyond the schema, the decisions must follow from the verdicts: every candidate is judged on
    every criterion, an exclusion names criteria that actually failed, a shortlisted candidate fails
    none, and an adopted candidate is registered in datasets.yaml.
    """
    path = root / SOURCE_SCREENING
    if not path.is_file():
        return [], []
    try:
        registry = load_yaml(path)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [], [f"source_screening.yaml: {exc}"]
    if not isinstance(registry, dict):
        return [], ["source_screening.yaml: not a mapping"]
    ids: list[str] = []
    errors = _record_schema_errors(
        root, registry, "source_screening.yaml", "screenings", expected=SOURCE_SCREENING_SCHEMA
    )
    try:
        datasets = load_yaml(root / "registry/datasets.yaml") or {}
    except (ImportError, OSError, TypeError, ValueError):
        datasets = {}
    registered = {str(r.get("id")) for r in datasets.get("datasets", []) if isinstance(r, dict)}
    for screening in registry.get("screenings") or []:
        if not isinstance(screening, dict):
            continue
        sid = str(screening.get("id"))
        criteria = [
            str(c.get("id")) for c in screening.get("criteria") or [] if isinstance(c, dict)
        ]
        search = screening.get("search") or {}
        local = [
            screening.get("requirement_source"),
            search.get("report"),
            *(search.get("artifacts") or []),
        ]
        for reference in local:
            if isinstance(reference, str) and not (root / reference).exists():
                errors.append(f"{sid}: path does not exist: {reference}")
        candidates = [c for c in screening.get("candidates") or [] if isinstance(c, dict)]
        seen: set[str] = set()
        for candidate in candidates:
            cid = f"{sid}/{candidate.get('id')}"
            ids.append(cid)
            if cid in seen:
                errors.append(f"{cid}: duplicate candidate id")
            seen.add(cid)
            errors += _candidate_errors(root, cid, candidate, criteria)
        decision = screening.get("owner_decision") or {}
        shortlisted = {str(c.get("id")) for c in candidates if c.get("decision") == "SHORTLISTED"}
        for adopted in decision.get("adopted") or []:
            if adopted not in shortlisted:
                errors.append(f"{sid}: adopted candidate '{adopted}' is not SHORTLISTED")
            if adopted not in registered:
                errors.append(
                    f"{sid}: adopted candidate '{adopted}' is not in registry/datasets.yaml"
                )
    return ids, errors


def _candidate_errors(
    root: Path, cid: str, candidate: dict[str, Any], criteria: list[str]
) -> list[str]:
    errors = []
    verdicts = candidate.get("verdicts") or {}
    if sorted(verdicts) != sorted(criteria):
        errors.append(
            f"{cid}: verdicts {sorted(verdicts)} do not match the criteria {sorted(criteria)}"
        )
    failing = {key for key, value in verdicts.items() if (value or {}).get("result") == "FAIL"}
    decision = candidate.get("decision")
    if decision == "SHORTLISTED" and failing:
        errors.append(f"{cid}: SHORTLISTED but fails {sorted(failing)}")
    for criterion in candidate.get("decisive_criteria") or []:
        if criterion not in failing:
            errors.append(f"{cid}: decisive criterion {criterion} is not a FAIL verdict")
    references = list(candidate.get("evidence") or [])
    artifact = (candidate.get("answer_items") or {}).get("estimate_artifact")
    if artifact:
        references.append(artifact)
    for reference in references:
        text = str(reference)
        if text.startswith("https://"):
            continue
        if text.startswith("http://") or not (root / text).exists():
            errors.append(f"{cid}: evidence is neither an https URL nor an existing path: {text}")
    return errors


def _release_config_revisions(root: Path) -> dict[tuple[str, str], str]:
    """Source revisions as the release configs record them.

    The v2-final card documents that Glaive and ToolACE carry the SHA-256 of the
    local input file the adapter read rather than a Hub revision. That value is
    recorded in the release config, so a citation using it is traceable -- to the
    config rather than to the dataset record. Both are legitimate; the check is
    for traceability, not for one specific file to be the source of truth.
    """
    revisions: dict[tuple[str, str], str] = {}
    for path in sorted((root / "configs/releases").glob("*.yaml")):
        try:
            config = load_yaml(path)
        except (ImportError, OSError, TypeError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        for source in config.get("included_sources", []) or []:
            if isinstance(source, dict) and source.get("id") and source.get("source_revision"):
                revisions[(path.name, str(source["id"]))] = str(source["source_revision"])
    return revisions


def validate_references(root: Path) -> list[str]:
    """Every reference a record makes must resolve to something that exists."""
    errors: list[str] = []
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"datasets.yaml: {exc}"]
    records = registry.get("datasets", []) if isinstance(registry, dict) else []
    by_id = {str(r.get("id")): r for r in records if isinstance(r, dict)}
    paper_ids = _paper_ids(root)

    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id", "<missing id>"))
        license_block = record.get("license")
        source = license_block.get("source") if isinstance(license_block, dict) else None
        if (
            isinstance(source, str)
            and source
            and not source.startswith(("http://", "https://"))
            and not (root / source).exists()
        ):
            errors.append(f"{record_id}: license.source does not exist: {source}")
        findings = record.get("findings")
        if isinstance(findings, dict):
            for key, value in findings.items():
                if (
                    isinstance(value, str)
                    and "/" in value
                    and not value.startswith("http")
                    and not (root / value).exists()
                    and key.endswith(("report", "config", "path"))
                ):
                    errors.append(f"{record_id}: findings.{key} does not exist: {value}")
        # `verified_from` is where a recorded count was checked against. A path
        # that does not resolve -- or that resolves only on the machine which
        # built the release, because it lives under .gitignore -- is not an
        # anchor anyone else can use.
        for key in ("sample_count", "retained_after_filtering"):
            block = record.get(key)
            if not isinstance(block, dict):
                continue
            reference = block.get("verified_from")
            if (
                isinstance(reference, str)
                and reference
                and not reference.startswith(("http", "/"))
                and not (root / reference).exists()
            ):
                errors.append(f"{record_id}: {key}.verified_from does not resolve: {reference}")
        for citation in record.get("derived_from") or []:
            if not isinstance(citation, dict):
                errors.append(f"{record_id}: derived_from entry is not a mapping")
                continue
            target_id = str(citation.get("id", ""))
            target = by_id.get(target_id)
            if target is None:
                errors.append(f"{record_id}: derived_from cites unknown dataset '{target_id}'")
                continue
            cited = str(citation.get("source_revision", ""))
            if not cited:
                errors.append(f"{record_id}: derived_from cites {target_id} without a revision")
                continue
            traceable = {
                str(target.get("exact_revision") or ""),
                str((target.get("source_revision") or {}).get("value") or "")
                if isinstance(target.get("source_revision"), dict)
                else "",
                str(target.get("checksum") or ""),
                str((target.get("processed_dataset_hash") or {}).get("value") or "")
                if isinstance(target.get("processed_dataset_hash"), dict)
                else "",
            }
            traceable |= {
                revision
                for (config_name, source_id), revision in _release_config_revisions(root).items()
                if source_id == target_id
            }
            if cited not in {value for value in traceable if value}:
                errors.append(
                    f"{record_id}: derived_from cites {target_id} at {cited[:16]}…, which "
                    f"matches neither its recorded revision nor its derived digest nor any "
                    f"release config"
                )
        errors += _evidence_reference_errors(root, record_id, record, paper_ids)
    return errors


def _paper_ids(root: Path) -> set[str] | None:
    """Ids in docs/references/papers.yaml, or None when the repository has no papers file."""
    path = root / "docs/references/papers.yaml"
    if not path.is_file():
        return None
    value = load_yaml(path)
    papers = value.get("papers", value) if isinstance(value, dict) else value
    return {str(p["id"]) for p in papers or [] if isinstance(p, dict) and "id" in p}


def _leading_path(value: str) -> str:
    """The repository path a prose reference starts with (`reports/ERRATA.md §23 (…)`)."""
    return value.split(" ", 1)[0]


def _evidence_reference_errors(
    root: Path, record_id: str, record: dict[str, Any], paper_ids: set[str] | None
) -> list[str]:
    """Schema-version-2 references: papers, overlap evidence, redistribution basis, RAI evidence."""
    errors = []
    if paper_ids is not None:
        for paper in record.get("papers") or []:
            if str(paper) not in paper_ids:
                errors.append(f"{record_id}: papers cites unknown id '{paper}'")
    local = list(record.get("overlap_evidence") or [])
    basis = record.get("redistribution_basis")
    if isinstance(basis, str) and basis:
        local.append(_leading_path(basis))
    responsible = record.get("responsible_use")
    if isinstance(responsible, dict):
        local += [e for e in responsible.get("evidence") or [] if not str(e).startswith("http")]
    for reference in local:
        if not (root / str(reference)).exists():
            errors.append(f"{record_id}: evidence path does not exist: {reference}")
    # A file digest describes one snapshot. A distribution URL that does not carry the record's
    # pinned revision could describe a different one, so the digest would pin nothing.
    revision = record.get("source_revision")
    pinned = str(revision.get("value", "")) if isinstance(revision, dict) else ""
    for item in record.get("distribution") or []:
        url = str(item.get("content_url", "")) if isinstance(item, dict) else ""
        if not pinned or pinned not in url:
            errors.append(
                f"{record_id}: distribution '{url}' does not name the pinned revision {pinned!r}"
            )
    return errors


# The Hugging Face namespace was renamed from `arrochi112` to `arjhinety` on
# 2026-09-13. Old links redirect and committed records keep the name they were
# written with, so both names appear in publication history. The revision chain
# has to be validated across the rename: keyed separately, the split would
# silently hide whether a supersession actually resolves.
NAMESPACE_ALIASES = {"arrochi112": "arjhinety"}


def _canonical_repository(name: str) -> str:
    if "/" not in name:
        return name
    namespace, _, remainder = name.partition("/")
    return f"{NAMESPACE_ALIASES.get(namespace, namespace)}/{remainder}"


def validate_publication_records(root: Path) -> list[str]:
    """Publication records are append-only history and must stay internally sound.

    The check is deliberately order-independent. Records carry publication dates,
    not sequence numbers, and several describe the same repository on the same
    day, so filename sort order is not chronology and must not be treated as it.
    What can be decided without an ordering is whether the set of revisions a
    repository has been published at resolves to exactly one current revision
    once every explicitly superseded revision is removed.
    """
    errors: list[str] = []
    named: dict[str, set[str]] = {}
    superseded: dict[str, set[str]] = {}
    for path in sorted((root / "reports/releases").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for key, value in _walk_dicts(record):
            revision = value.get("hub_revision")
            prior = (
                value.get("superseded_hub_revision")
                or value.get("previous_hub_revision")
                or value.get("corrected_hub_revision")
            )
            repository = value.get("repository") or value.get("hub_repository")
            if isinstance(revision, str) and revision and not _digest_shaped(revision):
                errors.append(f"{path.name}: {key}.hub_revision is not a hex revision")
            if isinstance(prior, str) and prior and not _digest_shaped(prior):
                errors.append(f"{path.name}: {key} names a superseded revision that is not hex")
            if isinstance(prior, str) and prior and prior == revision:
                errors.append(
                    f"{path.name}: {key} supersedes its own current revision {revision[:12]}…"
                )
            if not isinstance(repository, str) or not repository:
                continue
            repository = _canonical_repository(repository)
            if isinstance(revision, str) and revision:
                named.setdefault(repository, set()).add(revision)
            if isinstance(prior, str) and prior:
                superseded.setdefault(repository, set()).add(prior)
    for repository, revisions in sorted(named.items()):
        current = revisions - superseded.get(repository, set())
        if len(current) > 1:
            errors.append(
                f"{repository}: {len(current)} revisions are recorded without any of them "
                f"being named as superseded ({', '.join(sorted(r[:12] for r in current))}), "
                f"so the repository's current revision is ambiguous"
            )
    return errors


def _walk_dicts(value: object, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        found.append((path or "<root>", value))
        for key, item in value.items():
            found.extend(_walk_dicts(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_dicts(item, f"{path}[{index}]"))
    return found


RECONSTRUCTION_MARKER = "recorded_retroactively"


def _is_iso_date(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
        and value[:4].isdigit()
        and value[5:7].isdigit()
        and value[8:].isdigit()
    )


def validate_reconstructed_events(root: Path) -> list[str]:
    """A reconstructed publication event must not read as a contemporaneous one.

    OpenGrad reconstructs publication events from immutable Git and Hub history
    when a publication was made without a record. That is legitimate and worth
    doing, but it creates a specific hazard: an entry written today about an
    event three days ago is indistinguishable from an entry written at the time
    unless it says which it is. A later reader asking "what did we record then?"
    would be answered with something we recorded afterwards, and would have no
    way to tell.

    So every event carries three separate facts -- when the event happened, when
    the entry was written, and what immutable evidence supports the
    reconstruction -- and an entry claiming contemporaneity while its own dates
    disagree is a validation failure rather than a formatting preference.
    """
    errors: list[str] = []
    for path in sorted((root / "reports/releases").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        published = record.get("published")
        for key, value in _walk_dicts(record):
            if "hub_revision" not in value:
                continue
            retroactive = value.get(RECONSTRUCTION_MARKER)
            occurred = value.get("event_occurred_at")
            recorded = value.get("entry_recorded_at")
            if retroactive is True:
                if not (_is_iso_date(occurred) and _is_iso_date(recorded)):
                    errors.append(
                        f"{path.name}: {key} is marked {RECONSTRUCTION_MARKER} but does not give "
                        f"both event_occurred_at and entry_recorded_at as ISO dates"
                    )
                    continue
                if str(recorded) <= str(occurred):
                    errors.append(
                        f"{path.name}: {key} claims to be reconstructed but entry_recorded_at "
                        f"({recorded}) is not after event_occurred_at ({occurred})"
                    )
                evidence = value.get("reconstruction_evidence")
                if not isinstance(evidence, dict) or not evidence.get("kind"):
                    errors.append(
                        f"{path.name}: {key} is reconstructed without reconstruction_evidence "
                        f"naming the immutable record it was reconstructed from"
                    )
                elif not (evidence.get("immutable_identifier") or evidence.get("observed_from")):
                    errors.append(
                        f"{path.name}: {key} reconstruction_evidence names no immutable "
                        f"identifier or source it was observed from"
                    )
            elif not (_is_iso_date(occurred) and _is_iso_date(recorded)):
                # No explicit dates: the entry reads as contemporaneous, which is
                # the historical default and is left alone.
                continue
            elif str(occurred) != str(recorded):
                errors.append(
                    f"{path.name}: {key} does not declare {RECONSTRUCTION_MARKER} yet "
                    f"event_occurred_at ({occurred}) differs from entry_recorded_at "
                    f"({recorded}), so an entry written later would read as contemporaneous"
                )
            if (
                _is_iso_date(recorded)
                and _is_iso_date(published)
                and str(recorded) != str(published)
            ):
                errors.append(
                    f"{path.name}: {key} says it was recorded on {recorded} but the record is "
                    f"dated {published}"
                )
    return errors


def validate_freeze(root: Path) -> list[str]:
    """A frozen artifact must still match the identity recorded for it.

    A record asserting a validated derived identity has to be able to show what
    that identity is the identity *of*. Where the artifact is not in the record,
    the record must say so rather than leave the assertion unreproducible and
    unremarked.

    This rule exists because its absence made the check decorative. Before it,
    `_local_manifest_for` returned None for all three canonical corpora, so the
    strongest freeze assertion in the system was never executed and the check
    reported PASS by doing nothing.
    """
    errors: list[str] = []
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"datasets.yaml: {exc}"]
    records = registry.get("datasets", []) if isinstance(registry, dict) else []
    from opengrad.registry import provenance

    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id", "<missing id>"))
        derived = record.get("processed_dataset_hash")
        if not isinstance(derived, dict):
            continue
        value = derived.get("value")
        if not (isinstance(value, str) and len(value) == 64 and _digest_shaped(value)):
            continue
        manifest = _local_manifest_for(record)
        declaration = derived.get("identity_artifact_unavailable")
        remote = derived.get("remote_identity")
        if manifest is None and isinstance(remote, dict) and remote.get("hub_revision"):
            # The identity is anchored to an immutable published artifact rather
            # than to committed bytes. That is a counted, named deferral, not a
            # silent skip: the remote freeze-identity gate must reproduce the
            # digest over the network, and reports BLOCKED if it cannot.
            continue
        payload, problem = (
            provenance.read_evidence_bytes(root, manifest)
            if manifest
            else (None, "no artifact named")
        )
        if payload is not None:
            actual = hashlib.sha256(payload).hexdigest()
            if actual != value:
                errors.append(
                    f"{record_id}: frozen artifact identity moved: recorded {value[:12]}…, "
                    f"{manifest} hashes to {actual[:12]}…"
                )
            continue
        # The identity could not be re-derived. That is allowed only when the
        # record says so explicitly and names a tracked artifact that
        # independently corroborates the same digest. An unbacked declaration
        # would be an escape hatch; a declaration whose corroborating record
        # disagrees is the most important case of all, because two records then
        # disagree about the identity of the same published corpus.
        declaration = derived.get("identity_artifact_unavailable")
        if not isinstance(declaration, dict):
            errors.append(
                f"{record_id}: records a validated derived identity {value[:12]}… that no "
                f"committed artifact reproduces ({problem}), and does not declare "
                f"processed_dataset_hash.identity_artifact_unavailable"
            )
            continue
        corroborating = str(declaration.get("corroborated_by", "") or "")
        field_name = str(declaration.get("corroborating_field", "") or "")
        if not corroborating or not field_name:
            errors.append(
                f"{record_id}: identity_artifact_unavailable must name corroborated_by and "
                f"corroborating_field, so the unavailability is backed by something checkable"
            )
            continue
        try:
            corroborating_record = json.loads((root / corroborating).read_bytes())
        except (OSError, TypeError, ValueError) as exc:
            errors.append(
                f"{record_id}: identity_artifact_unavailable cites an unreadable corroborating "
                f"record {corroborating}: {exc}"
            )
            continue
        recorded = corroborating_record.get(field_name)
        if recorded != value:
            errors.append(
                f"{record_id}: identity {value[:12]}… is not corroborated: {corroborating} "
                f"records {field_name} = {str(recorded)[:12]}…, so two records disagree about "
                f"the identity of the same corpus"
            )
    return errors


def _local_manifest_for(record: dict[str, Any]) -> str | None:
    """The local file a record's derived digest is the identity of, if it says.

    `source_repository` is included because it is where the canonical releases
    name the manifest their digest was computed from. Leaving it out is what made
    the freeze check vacuous for those records.
    """
    derived = record.get("processed_dataset_hash")
    if isinstance(derived, dict) and isinstance(derived.get("source"), str):
        return str(derived["source"])
    findings = record.get("findings")
    if isinstance(findings, dict):
        candidate = findings.get("release_manifest")
        if isinstance(candidate, str):
            return candidate
    repository = record.get("source_repository")
    if isinstance(repository, str) and repository and not repository.startswith(("http", "/")):
        return repository
    return None


def validate_semantic_consistency(root: Path) -> list[str]:
    """Relationships the registry asserts between records must hold.

    Only mechanically decidable relations are checked. Nothing here decides
    whether a result is scientifically sound -- the registry records that
    judgment, and this only verifies the judgment is well-formed.
    """
    errors: list[str] = []
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"datasets.yaml: {exc}"]
    records = registry.get("datasets", []) if isinstance(registry, dict) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id", "<missing id>"))
        findings = record.get("findings")
        if not isinstance(findings, dict):
            continue
        experiment = findings.get("experiment_id")
        if isinstance(experiment, str) and experiment:
            status = root / "docs/EXPERIMENT_STATUS.md"
            if status.exists() and experiment not in status.read_text(encoding="utf-8"):
                errors.append(
                    f"{record_id}: findings.experiment_id '{experiment}' does not appear in "
                    f"docs/EXPERIMENT_STATUS.md"
                )
        for key, value in findings.items():
            if (
                key.endswith("_split")
                and isinstance(value, str)
                and value in (record.get("forbidden_splits") or [])
            ):
                errors.append(
                    f"{record_id}: findings.{key} names '{value}', which this dataset forbids"
                )
        plan = record.get("planned_initial_mixture_percent")
        if record.get("contamination_status") == "EXCLUDED_FROM_CLEAN_DEFAULT" and plan not in (
            0,
            None,
        ):
            errors.append(
                f"{record_id}: excluded from the clean default mixture yet planned at {plan}%"
            )
    return errors
