import hashlib
import json
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
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
    return errors


def main() -> int:
    errors = validate(Path.cwd())
    print("registry validation: OK" if not errors else "\n".join(errors))
    return int(bool(errors))
