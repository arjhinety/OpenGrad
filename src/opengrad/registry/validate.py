import hashlib
import json
from pathlib import Path
from typing import Any

from opengrad.verification import (
    CONDITIONALLY_REQUIRED,
    OPTIONAL,
    REQUIRED_NONEMPTY,
    Skip,
    ValidationResult,
)


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the dev extra for YAML validation") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate(root: Path) -> list[str]:
    errors = []
    dataset_ids: set[str] = set()
    for name in (
        "datasets.yaml",
        "benchmarks.yaml",
        "models.yaml",
        "runtimes.yaml",
        "hardware.yaml",
        "workload_profiles.yaml",
    ):
        try:
            value = load_yaml(root / "registry" / name)
            if not isinstance(value, dict) or "schema_version" not in value:
                errors.append(name + ": missing schema_version")
            if name == "datasets.yaml" and isinstance(value, dict):
                dataset_ids = {
                    str(item["id"])
                    for item in value.get("datasets", [])
                    if isinstance(item, dict) and "id" in item
                }
        except (ImportError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    try:
        taxonomy = load_yaml(root / "registry/tool_behaviors.yaml")
        if (
            not isinstance(taxonomy, dict)
            or not taxonomy.get("decisions")
            or not taxonomy.get("capabilities")
        ):
            errors.append("tool_behaviors.yaml: decisions and capabilities are required")
        elif any(
            not isinstance(value, str) or not value for value in taxonomy["capabilities"].values()
        ):
            errors.append("tool_behaviors.yaml: every capability needs a description")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        errors.append(f"tool_behaviors.yaml: {exc}")
    try:
        from opengrad.data.mixture import load_mixture

        for path in sorted((root / "configs/data/tool_calling").glob("*.yaml")):
            try:
                # The legacy contamination config is intentionally not a mixture.
                if path.name != "contamination.yaml":
                    value = load_yaml(path)
                    load_mixture(path)
                    sources = (
                        set(value.get("source_manifests", [])) if isinstance(value, dict) else set()
                    )
                    if sources - dataset_ids:
                        errors.append(f"{path.name}: unknown source manifest")
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        errors.append(f"mixture configs: {exc}")
    try:
        schema = json.loads((root / "registry/experiments.schema.json").read_text())
        if schema.get("$schema") is None:
            errors.append("experiments.schema.json: missing $schema")
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"experiments.schema.json: {exc}")
    for name in (
        "evaluation_manifest.schema.json",
        "gpu_preflight.schema.json",
        "runtime_components.schema.json",
    ):
        try:
            schema = json.loads((root / "registry" / name).read_text(encoding="utf-8"))
            if schema.get("$schema") is None:
                errors.append(f"{name}: missing $schema")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    for name in ("training_recipes.yaml", "runtime_components.yaml"):
        try:
            value = load_yaml(root / "registry" / name)
            if not isinstance(value, dict) or "schema_version" not in value:
                errors.append(name + ": missing schema_version")
        except (ImportError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    # The baseline is the one pre-result source of truth.  Keep its mirrored
    # experiment record honest so a GPU run cannot start with stale null pins.
    try:
        evaluation_path = root / "configs/evaluation/tool_calling/qwen35_2b_baseline.yaml"
        experiment_path = root / "configs/experiments/tool_calling/qwen35_2b_baseline.yaml"
        evaluation = load_yaml(evaluation_path)
        experiment = load_yaml(experiment_path)
        required = ("model_revision", "template_hash", "generation", "evaluations", "runtime")
        if not isinstance(evaluation, dict) or any(key not in evaluation for key in required):
            errors.append("baseline evaluation: incomplete frozen contract")
        elif evaluation.get("status") != "FROZEN_PRE_GPU":
            errors.append("baseline evaluation: status must be FROZEN_PRE_GPU")
        else:
            manifest_ref = evaluation["evaluations"].get("behavioral_manifest")
            manifest_path = root / str(manifest_ref)
            if not manifest_path.exists():
                errors.append("baseline evaluation: behavioral manifest missing")
            if any(
                "TO_BE_RECORDED" in str(value) or "PINNED_REGISTRY_REVISION_REQUIRED" in str(value)
                for key, value in evaluation.items()
                if key not in {"hardware"}
            ):
                errors.append("baseline evaluation: pre-result placeholder remains")
        if isinstance(evaluation, dict) and isinstance(experiment, dict):
            manifest_ref = evaluation["evaluations"]["behavioral_manifest"]
            manifest_path = root / str(manifest_ref)
            if manifest_path.exists():
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if experiment.get("dataset_hash") != digest:
                    errors.append("baseline experiment: dataset_hash does not match manifest")
            if experiment.get("dataset_manifest") != manifest_ref:
                errors.append("baseline experiment: dataset manifest disagrees with evaluation")
            if experiment.get("generation_config") != evaluation.get("generation"):
                errors.append("baseline experiment: generation config disagrees with evaluation")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        errors.append(f"baseline contract: {exc}")
    # The registry is human-authored research state. Everything below verifies
    # that the judgment recorded in it is internally and evidentially
    # well-formed; none of it decides whether a result is scientifically sound.
    # Each gate also reports what it executed, so a gate that ran nothing cannot
    # be read as a gate that passed.
    report = audit(root)
    for gate in (
        "structure",
        "references",
        "publication_records",
        "reconstructed_events",
        "provenance",
        "freeze",
        "semantic",
    ):
        errors.extend(report[gate].all_errors())
    return errors


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
        for key in ("sample_count", "retained_after_filtering", "original_sample_count"):
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


def provenance_report(root: Path) -> tuple[list[str], dict[str, int]]:
    """Audit every verified claim and return (errors, category counts)."""
    from opengrad.registry import provenance

    datasets: list[dict[str, Any]] = []
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
        if isinstance(registry, dict):
            datasets = [r for r in registry.get("datasets", []) if isinstance(r, dict)]
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"datasets.yaml: {exc}"], {}
    models: list[dict[str, Any]] = []
    try:
        registry = load_yaml(root / "registry/models.yaml")
        if isinstance(registry, dict):
            models = [r for r in registry.get("models", []) if isinstance(r, dict)]
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [f"models.yaml: {exc}"], {}
    result = provenance.audit_verified_claims(root, datasets, "datasets")
    result.claims.extend(
        audit
        for audit in (
            provenance.check_license_claim(record, root, str(record.get("id"))) for record in models
        )
        if audit is not None
    )
    result.errors = [error for claim in result.claims for error in claim.errors]
    return result.errors, result.counts()


def _candidate_ids(records: list[object]) -> list[str]:
    return [str(r.get("id")) for r in records if isinstance(r, dict) and r.get("id")]


def result_from(
    name: str,
    policy: str,
    ids: list[str],
    errors: list[str],
    *,
    skipped: list[Skip] | None = None,
    blocked_ids: list[str] | None = None,
    blocked_status: str | None = None,
    detail: dict[str, int] | None = None,
    precondition: str | None = None,
) -> ValidationResult:
    """Build a census from a discovered population and a flat error list.

    Errors are attributed to a candidate by the ``<id>: `` convention every check
    in this module already uses. The attribution is a lookup against the ids
    actually discovered rather than open-ended string parsing, and an error that
    matches no candidate is not discarded: it is attributed to a gate-level
    sentinel that counts as a failed candidate, so an unattributable error can
    never reduce the census.
    """
    skipped = list(skipped or [])
    blocked_ids = list(blocked_ids or [])
    skipped_set = {item.id for item in skipped}
    blocked_set = set(blocked_ids)
    # A skipped or blocked candidate is still a discovered candidate. Folding any
    # that were not listed into the population keeps the census self-consistent
    # rather than letting a caller's bookkeeping become an accounting failure.
    ids = [*ids, *(i for i in [*skipped_set, *blocked_set] if i not in ids)]
    checked_ids = [i for i in ids if i not in skipped_set and i not in blocked_set]

    attributed: dict[str, list[str]] = {i: [] for i in checked_ids}
    unattributed: list[str] = []
    for message in errors:
        for candidate in checked_ids:
            if message.startswith(f"{candidate}:"):
                attributed[candidate].append(message)
                break
        else:
            unattributed.append(message)

    if unattributed:
        sentinel = f"<{name}>"
        ids = [*ids, sentinel]
        checked_ids = [*checked_ids, sentinel]
        attributed[sentinel] = unattributed

    passed = sum(1 for i in checked_ids if not attributed.get(i))
    failed = len(checked_ids) - passed
    return ValidationResult(
        name=name,
        policy=policy,
        discovered=len(ids),
        checked=len(checked_ids),
        passed=passed,
        failed=failed,
        blocked=len(blocked_ids),
        skipped=skipped,
        errors=errors,
        detail=detail or {},
        blocked_status=blocked_status,
        precondition=precondition,
    )


def _load_datasets(root: Path) -> tuple[list[object], str | None]:
    try:
        registry = load_yaml(root / "registry/datasets.yaml")
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return [], f"datasets.yaml: {exc}"
    if not isinstance(registry, dict):
        return [], "datasets.yaml: not a mapping"
    return list(registry.get("datasets", []) or []), None


def check_structure(root: Path) -> ValidationResult:
    records, load_error = _load_datasets(root)
    ids = _candidate_ids(records)
    errors = [load_error] if load_error else validate_structure(root)
    return result_from("registry structure", REQUIRED_NONEMPTY, ids, errors)


def check_references(root: Path) -> ValidationResult:
    records, load_error = _load_datasets(root)
    ids = _candidate_ids(records)
    errors = [load_error] if load_error else validate_references(root)
    return result_from("registry references", REQUIRED_NONEMPTY, ids, errors)


def check_publication_records(root: Path) -> ValidationResult:
    paths = sorted((root / "reports/releases").glob("*.json"))
    ids = [str(path.relative_to(root)) for path in paths]
    errors = validate_publication_records(root)
    return result_from(
        "publication records",
        CONDITIONALLY_REQUIRED,
        ids,
        errors,
        precondition="reports/releases/*.json present" if paths else None,
    )


def check_reconstructed_events(root: Path) -> ValidationResult:
    candidates: list[str] = []
    for path in sorted((root / "reports/releases").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        for key, value in _walk_dicts(record):
            if "hub_revision" in value:
                candidates.append(f"{path.name}:{key}")
    errors = validate_reconstructed_events(root)
    return result_from(
        "reconstructed events",
        CONDITIONALLY_REQUIRED,
        candidates,
        errors,
        precondition="publication records carry revisions" if candidates else None,
    )


def check_semantic_consistency(root: Path) -> ValidationResult:
    records, load_error = _load_datasets(root)
    with_findings: list[object] = [
        r for r in records if isinstance(r, dict) and isinstance(r.get("findings"), dict)
    ]
    ids = _candidate_ids(with_findings)
    errors = [load_error] if load_error else validate_semantic_consistency(root)
    skipped = (
        [] if ids else [Skip(id="<no records declare findings>", reason="optional cross-checks")]
    )
    return result_from(
        "semantic consistency",
        OPTIONAL,
        ids,
        errors,
        skipped=skipped,
    )


def _declares_corpora(records: list[object]) -> bool:
    """Whether the registry declares any derived corpus at all.

    Used to decide whether the freeze gate's non-emptiness requirement binds. The
    requirement is real for a repository that has corpora -- discovering none then
    means discovery is broken -- and vacuous-but-honest for one that has none.
    """
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("id", "")).startswith("canonical"):
            return True
        if any(key in record for key in ("processed_dataset_hash", "checksum", "exact_revision")):
            return True
    return False


def check_freeze(root: Path) -> ValidationResult:
    """Canonical corpora exist in this repository, so discovering none is a failure.

    This gate was the vacuous one: it read `processed_dataset_hash.source` and
    `findings.release_manifest`, neither of which the canonical records use, and
    therefore skipped every corpus while reporting PASS.
    """
    from opengrad.registry import provenance

    records, load_error = _load_datasets(root)
    candidates: list[tuple[str, str, str | None]] = []
    all_ids: list[str] = []
    deferred: list[Skip] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        derived = record.get("processed_dataset_hash")
        if not isinstance(derived, dict):
            continue
        value = derived.get("value")
        if not (isinstance(value, str) and len(value) == 64 and _digest_shaped(value)):
            continue
        record_id = str(record.get("id"))
        # The census is the whole population; a deferred candidate is skipped, not
        # removed, so that discovered == checked + blocked + skipped holds.
        all_ids.append(record_id)
        remote = derived.get("remote_identity")
        if isinstance(remote, dict) and remote.get("hub_revision"):
            deferred.append(
                Skip(
                    id=record_id,
                    reason=(
                        f"identity anchored to {remote.get('hub_repository')}@"
                        f"{str(remote.get('hub_revision'))[:12]}…; verified by the remote "
                        f"freeze-identity gate"
                    ),
                )
            )
            continue
        candidates.append((record_id, value, _local_manifest_for(record)))
    errors = [load_error] if load_error else validate_freeze(root)

    reproduced = corroborated = unresolved = 0
    for record_id, _value, manifest in candidates:
        if f"{record_id}:" not in " ".join(errors) and manifest is not None:
            payload, _ = provenance.read_evidence_bytes(root, manifest)
            if payload is not None:
                reproduced += 1
                continue
        if any(error.startswith(f"{record_id}:") for error in errors):
            unresolved += 1
        else:
            corroborated += 1

    return result_from(
        "canonical freezes",
        CONDITIONALLY_REQUIRED,
        all_ids,
        errors,
        skipped=deferred,
        # The requirement binds because this repository declares corpora. A root
        # that declares none is not silently passed -- it is skipped with a reason.
        precondition="registry declares corpora with a validated identity"
        if _declares_corpora(records)
        else None,
        detail={
            "reproduced_from_committed_artifact": reproduced,
            "corroborated_by_a_tracked_record": corroborated,
            "unresolved": unresolved,
            "deferred_to_remote_identity": len(deferred),
        },
    )


def check_verified_claims(root: Path) -> ValidationResult:
    errors, counts = provenance_report(root)
    records, load_error = _load_datasets(root)
    claim_ids = [
        f"{record.get('id')!s}.license"
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("license"), dict)
        and record["license"].get("verified") is True
    ]
    claim_ids += [
        f"{record.get('id')!s}.source_revision"
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("source_revision"), dict)
        and record["source_revision"].get("verified") is True
    ]
    try:
        models = load_yaml(root / "registry/models.yaml")
        for record in (models or {}).get("models", []) or []:
            if (
                isinstance(record, dict)
                and isinstance(record.get("license"), dict)
                and record["license"].get("verified") is True
            ):
                claim_ids.append(f"{record.get('id')!s}.license")
    except (ImportError, OSError, TypeError, ValueError, AttributeError):
        pass
    result = result_from(
        "verified claims",
        REQUIRED_NONEMPTY,
        claim_ids,
        errors,
        detail={key: value for key, value in counts.items()},
    )
    # Claim errors name the record rather than the claim, so attribute them at
    # record granularity as well rather than losing them.
    if load_error:
        result.errors.append(load_error)
    return result


def audit(root: Path) -> dict[str, ValidationResult]:
    """The whole registry audit as one census-bearing result per gate."""
    return {
        "structure": check_structure(root),
        "references": check_references(root),
        "publication_records": check_publication_records(root),
        "reconstructed_events": check_reconstructed_events(root),
        "provenance": check_verified_claims(root),
        "freeze": check_freeze(root),
        "semantic": check_semantic_consistency(root),
    }


def audit_errors(root: Path) -> dict[str, list[str]]:
    """The flat error lists, for callers that predate the census."""
    report = audit(root)
    return {name: list(result.all_errors()) for name, result in report.items()}


def main() -> int:
    errors = validate(Path.cwd())
    print("registry validation: OK" if not errors else "\n".join(errors))
    return int(bool(errors))


if __name__ == "__main__":
    # Without this guard, `python -m opengrad.registry.validate` imports the module,
    # runs nothing and exits 0 -- which is indistinguishable from a passing
    # validation. Only the `opengrad-validate` console script actually ran the
    # check, so any report of this validation passing via `-m` was vacuous.
    raise SystemExit(main())
