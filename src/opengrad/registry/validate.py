import hashlib
import json
from pathlib import Path
from typing import Any

from opengrad.registry.validators import (
    _digest_shaped,
    _local_manifest_for,
    _walk_dicts,
    load_yaml,
    validate_freeze,
    validate_publication_records,
    validate_reconstructed_events,
    validate_references,
    validate_semantic_consistency,
    validate_source_screening,
    validate_structure,
)
from opengrad.verification import (
    CONDITIONALLY_REQUIRED,
    OPTIONAL,
    REQUIRED_NONEMPTY,
    Skip,
    ValidationResult,
)

# The names other modules and the tests import from here, including those now defined in
# opengrad.registry.validators.
__all__ = [
    "_load_datasets",
    "audit",
    "check_freeze",
    "check_source_screening",
    "result_from",
    "validate",
    "validate_freeze",
    "validate_publication_records",
    "validate_reconstructed_events",
    "validate_references",
    "validate_source_screening",
    "validate_structure",
]


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
        "screening",
    ):
        errors.extend(report[gate].all_errors())
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
    detail: dict[str, Any] | None = None,
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


def check_source_screening(root: Path) -> ValidationResult:
    """Every screened candidate is a census member.

    A repository without a screening log has nothing to screen (OPTIONAL); one with a log must
    discover candidates in it, so a log that parses to nothing cannot pass.
    """
    ids, errors = validate_source_screening(root)
    present = (root / "registry/source_screening.yaml").is_file()
    return result_from("source screening", REQUIRED_NONEMPTY if present else OPTIONAL, ids, errors)


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
        "screening": check_source_screening(root),
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
