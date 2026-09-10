"""Authoritative preflight validation gate for training and evaluation experiments."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.env_capture import tracked_tree_provenance
from opengrad.experiments.schema import ExperimentConfig, TrainingAlgorithm


@dataclass
class PreflightCheckItem:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    error_code: str | None = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error_code": self.error_code,
            "details": self.details,
        }


@dataclass
class PreflightResult:
    experiment_id: str
    overall_status: str  # "PASS", "WARN", "FAIL"
    checks: list[PreflightCheckItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "overall_status": self.overall_status,
            "checks": [c.to_dict() for c in self.checks],
        }

    def render_summary(self) -> str:
        lines = [
            f"Preflight Verification for Experiment: {self.experiment_id}",
            f"Overall Status: {self.overall_status}\n",
            f"{'Check':<32} {'Status':<8} {'Details'}",
            "-" * 70,
        ]
        for c in self.checks:
            lines.append(f"{c.name:<32} {c.status:<8} {c.details}")
        return "\n".join(lines)


def run_experiment_preflight(
    config: ExperimentConfig | Path | str, root: Path | None = None
) -> PreflightResult:
    """Run all preflight checks on an experiment configuration."""
    root_dir = root or Path.cwd()
    checks: list[PreflightCheckItem] = []

    # 1. Config loading & schema
    if isinstance(config, (str, Path)):
        try:
            exp_config = ExperimentConfig.from_file(config)
            checks.append(
                PreflightCheckItem("Config Schema", "PASS", details="Valid experiment schema")
            )
        except (OSError, ValueError, TypeError) as exc:
            checks.append(PreflightCheckItem("Config Schema", "FAIL", "CONFIG_INVALID", str(exc)))
            return PreflightResult("unknown", "FAIL", checks)
    else:
        exp_config = config
        checks.append(
            PreflightCheckItem("Config Schema", "PASS", details="Valid experiment schema")
        )

    # 2. Algorithm compatibility
    trainer_type = exp_config.trainer.get("type", "sft")
    valid_types = {t.value for t in TrainingAlgorithm}
    if trainer_type not in valid_types:
        checks.append(
            PreflightCheckItem(
                "Training Algorithm",
                "FAIL",
                "ALGORITHM_UNSUPPORTED",
                f"Unsupported algorithm '{trainer_type}'. Must be one of {valid_types}",
            )
        )
    else:
        checks.append(
            PreflightCheckItem(
                "Training Algorithm", "PASS", details=f"Algorithm '{trainer_type}' supported"
            )
        )

    # 3. Model & Tokenizer specification
    model_id = exp_config.model.get("model_id")
    model_rev = exp_config.model.get("model_revision")
    if not model_id or not model_rev:
        checks.append(
            PreflightCheckItem(
                "Model Specification",
                "FAIL",
                "MODEL_INVALID",
                "model_id and model_revision must be explicitly specified",
            )
        )
    else:
        checks.append(
            PreflightCheckItem(
                "Model Specification",
                "PASS",
                details=f"{model_id} (rev: {str(model_rev)[:10]})",
            )
        )

    # 4. Dataset manifests check
    manifest_paths = exp_config.datasets.get("manifest_paths", [])
    if not manifest_paths and not exp_config.datasets.get("source_manifests"):
        checks.append(
            PreflightCheckItem(
                "Dataset Manifests",
                "WARN",
                "DATASET_EMPTY",
                "No explicit dataset manifests defined in config",
            )
        )
    else:
        missing_paths = [p for p in manifest_paths if not (root_dir / str(p)).exists()]
        if missing_paths:
            checks.append(
                PreflightCheckItem(
                    "Dataset Manifests",
                    "FAIL",
                    "CHECKSUM_MISMATCH",
                    f"Missing manifest file(s): {', '.join(missing_paths)}",
                )
            )
        else:
            checks.append(
                PreflightCheckItem("Dataset Manifests", "PASS", details="Manifest files verified")
            )

    # 5. Output directory & disk capacity
    out_dir = root_dir / "runs" / exp_config.experiment_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _total, _used, free = shutil.disk_usage(out_dir)
        free_gb = free // (2**30)
        if free_gb < 2:
            checks.append(
                PreflightCheckItem(
                    "Disk Capacity", "WARN", "DISK_LOW", f"Only {free_gb} GB free disk remaining"
                )
            )
        else:
            checks.append(
                PreflightCheckItem(
                    "Disk Capacity", "PASS", details=f"{free_gb} GB available disk space"
                )
            )
    except OSError as exc:
        checks.append(PreflightCheckItem("Disk Capacity", "FAIL", "PATH_NOT_WRITABLE", str(exc)))

    # 6. Git state capture
    # Tracked-tree provenance, not `capture()`'s whole-tree flag: a run's own untracked
    # outputs would otherwise make every run report as dirty. `capture()` also returns flat
    # `git_sha`/`git_dirty` keys, so the previous nested lookup always printed "unknown".
    provenance = tracked_tree_provenance(root_dir)
    git_sha = provenance["sha"] or "unknown"
    git_dirty = provenance["dirty"]
    if git_dirty:
        checks.append(
            PreflightCheckItem(
                "Git State",
                "WARN",
                details=f"SHA: {git_sha[:10]} (uncommitted changes to tracked files)",
            )
        )
    else:
        checks.append(
            PreflightCheckItem("Git State", "PASS", details=f"Clean tree at SHA {git_sha[:10]}")
        )

    # 7. Environment & seeds
    seed = exp_config.reproducibility.get("seed")
    if seed is None:
        checks.append(
            PreflightCheckItem(
                "Seed Configuration", "FAIL", "SEED_MISSING", "Random seed must be explicit"
            )
        )
    else:
        checks.append(PreflightCheckItem("Seed Configuration", "PASS", details=f"Seed {seed}"))

    # Overall verdict
    has_fail = any(c.status == "FAIL" for c in checks)
    has_warn = any(c.status == "WARN" for c in checks)
    overall = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    return PreflightResult(exp_config.experiment_id, overall, checks)
