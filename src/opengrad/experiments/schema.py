"""Canonical experiment identity, lifecycle status, and immutable configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from opengrad.data.supervision import declared_kinds


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    REVIEW = "REVIEW"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    # Preserved as evidence, but not a valid training result: a run that produced no real
    # model artifact (for example a mock-provider infrastructure pass). A record must not
    # carry INVALID together with an effective TRAINED/PROMOTED claim in its metadata.
    INVALID = "INVALID"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


class TrainingAlgorithm(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    ON_POLICY_DISTILLATION = "on_policy_distillation"
    # Future extension boundary (not implemented)
    RL = "rl"


@dataclass
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    training_algorithm: str
    training_config: dict[str, Any]
    dataset_manifest_ids: list[str] = field(default_factory=list)
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    parent_experiment_id: str | None = None
    git_commit: str = "unknown"
    git_dirty: bool = False
    environment: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    hardware_info: dict[str, Any] = field(default_factory=dict)
    launch_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completion_timestamp: str | None = None
    checkpoints: list[str] = field(default_factory=list)
    evaluations: list[str] = field(default_factory=list)
    status: str = ExperimentStatus.CREATED.value
    promotion_decision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "parent_experiment_id": self.parent_experiment_id,
            "hypothesis": self.hypothesis,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "training_algorithm": self.training_algorithm,
            "training_config": self.training_config,
            "dataset_manifest_ids": self.dataset_manifest_ids,
            "dataset_hashes": self.dataset_hashes,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "environment": self.environment,
            "random_seed": self.random_seed,
            "hardware_info": self.hardware_info,
            "launch_timestamp": self.launch_timestamp,
            "completion_timestamp": self.completion_timestamp,
            "checkpoints": self.checkpoints,
            "evaluations": self.evaluations,
            "status": self.status,
            "promotion_decision": self.promotion_decision,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        return cls(
            experiment_id=str(data["experiment_id"]),
            parent_experiment_id=data.get("parent_experiment_id"),
            hypothesis=str(data.get("hypothesis", "")),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            tokenizer_revision=str(data.get("tokenizer_revision", data["model_revision"])),
            training_algorithm=str(data.get("training_algorithm", "sft")),
            training_config=dict(data.get("training_config") or {}),
            dataset_manifest_ids=list(data.get("dataset_manifest_ids") or []),
            dataset_hashes=dict(data.get("dataset_hashes") or {}),
            git_commit=str(data.get("git_commit", "unknown")),
            git_dirty=bool(data.get("git_dirty", False)),
            environment=dict(data.get("environment") or {}),
            random_seed=int(data.get("random_seed", 42)),
            hardware_info=dict(data.get("hardware_info") or {}),
            launch_timestamp=str(data.get("launch_timestamp", "")),
            completion_timestamp=data.get("completion_timestamp"),
            checkpoints=list(data.get("checkpoints") or []),
            evaluations=list(data.get("evaluations") or []),
            status=str(data.get("status", ExperimentStatus.CREATED.value)),
            promotion_decision=data.get("promotion_decision"),
            metadata=dict(data.get("metadata") or {}),
        )


def _validated_datasets(value: dict[str, Any]) -> dict[str, Any]:
    """Run the dataset-section validators on the way in, so a bad filter fails at config load."""
    value["exclude_sources"] = validate_source_exclusion(value.get("exclude_sources"))
    return value


def validate_source_exclusion(value: Any) -> list[str]:
    """Validate `datasets.exclude_sources`, the source-ablation filter.

    Names are checked for shape only. Whether a named source exists in the corpus is checked at
    preprocessing time, where the release manifest is available -- a name that matches nothing
    must fail rather than silently exclude nothing.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("datasets.exclude_sources must be a list of source ids")
    cleaned = [str(item).strip() for item in value]
    if any(not item for item in cleaned):
        raise ValueError("datasets.exclude_sources entries must be non-empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("datasets.exclude_sources contains duplicates")
    return sorted(cleaned)


def validate_supervision_selection(value: Any) -> dict[str, Any]:
    """Validate the optional `supervision` block of an experiment config.

    Declared kinds are checked against the supervision contract registry, so a typo fails at
    config load rather than silently selecting nothing (or everything) at training time.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("supervision must be a mapping")
    unknown = sorted(set(value) - {"include", "sampling_weights"})
    if unknown:
        raise ValueError(f"unknown supervision keys: {unknown}")
    if "sampling_weights" in value:
        weights = value["sampling_weights"]
        if not isinstance(weights, dict):
            raise TypeError("supervision.sampling_weights must be a mapping")
        bad_keys = sorted(str(key) for key in weights if str(key) not in declared_kinds())
        if bad_keys:
            raise ValueError(
                f"supervision.sampling_weights has unknown kinds: {bad_keys}; "
                f"declared kinds are: {sorted(declared_kinds())}"
            )
        raise ValueError(
            "supervision.sampling_weights is not implemented; weighting would be silently "
            "ignored. Use supervision.include to select kinds for an ablation, or remove the "
            "block to train on the whole corpus under natural sampling."
        )
    if "include" not in value:
        raise ValueError(
            "a declared supervision block must contain supervision.include; remove the block "
            "to train on the whole corpus under natural sampling"
        )
    include = value["include"]
    if not isinstance(include, list) or not include:
        raise ValueError("supervision.include must be a non-empty list of supervision kinds")
    bad = sorted(str(item) for item in include if str(item) not in declared_kinds())
    if bad:
        raise ValueError(
            f"supervision.include has unknown kinds: {bad}; "
            f"declared kinds are: {sorted(declared_kinds())}"
        )
    if len({str(item) for item in include}) != len(include):
        raise ValueError("supervision.include contains duplicate supervision kinds")
    return {"include": [str(item) for item in include]}


@dataclass(frozen=True)
class ExperimentConfig:
    """Canonical declarative experiment configuration."""

    experiment_id: str
    hypothesis: str
    model: dict[str, Any]
    datasets: dict[str, Any]
    trainer: dict[str, Any]
    evaluation: dict[str, Any]
    generation: dict[str, Any]
    checkpointing: dict[str, Any]
    promotion: dict[str, Any]
    reproducibility: dict[str, Any]
    speculative_decoding: dict[str, Any] = field(default_factory=dict)
    # Optional supervision selectivity: which supervision kinds to train on, and how to weight
    # them. Absent means every kind in the corpus is trained under natural sampling.
    supervision: dict[str, Any] = field(default_factory=dict)
    parent_experiment_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        required = [
            "experiment_id",
            "hypothesis",
            "model",
            "datasets",
            "trainer",
            "evaluation",
            "generation",
            "checkpointing",
            "promotion",
            "reproducibility",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Experiment configuration missing required field(s): {', '.join(missing)}"
            )

        return cls(
            experiment_id=str(data["experiment_id"]),
            hypothesis=str(data["hypothesis"]),
            model=dict(data["model"]),
            datasets=_validated_datasets(dict(data["datasets"])),
            trainer=dict(data["trainer"]),
            evaluation=dict(data["evaluation"]),
            generation=dict(data["generation"]),
            checkpointing=dict(data["checkpointing"]),
            promotion=dict(data["promotion"]),
            reproducibility=dict(data["reproducibility"]),
            speculative_decoding=dict(data.get("speculative_decoding") or {}),
            supervision=validate_supervision_selection(data.get("supervision")),
            parent_experiment_id=data.get("parent_experiment_id"),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> ExperimentConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Experiment config must be a YAML object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "parent_experiment_id": self.parent_experiment_id,
            "hypothesis": self.hypothesis,
            "model": self.model,
            "datasets": self.datasets,
            "trainer": self.trainer,
            "evaluation": self.evaluation,
            "generation": self.generation,
            "checkpointing": self.checkpointing,
            "promotion": self.promotion,
            "reproducibility": self.reproducibility,
            "speculative_decoding": self.speculative_decoding,
        }
        # An absent filter is serialized by omitting the block, never as `supervision: {}`: a
        # present-but-empty block is rejected on load, so emitting one would make a valid no-filter
        # config fail its own round-trip.
        if self.supervision:
            payload["supervision"] = self.supervision
        return payload

    def write_resolved(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
