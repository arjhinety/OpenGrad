"""ExperimentStore managing the canonical artifact layout and experiment metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.env_capture import capture
from opengrad.experiments.ledger import ExperimentLedger, LedgerEventType
from opengrad.experiments.schema import (
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStatus,
)


class ExperimentStore:
    """Manages the canonical runs/<experiment-id>/ directory layout and immutable metadata."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.central_ledger = ExperimentLedger(self.runs_dir / "central_ledger.jsonl")

    def run_dir(self, experiment_id: str) -> Path:
        return self.runs_dir / experiment_id

    def create_experiment(
        self,
        config: ExperimentConfig,
        *,
        env: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        r_dir = self.run_dir(config.experiment_id)
        if (r_dir / "experiment.json").exists():
            raise FileExistsError(f"Experiment already exists: {config.experiment_id}")

        # Create canonical directory layout (Section 46)
        r_dir.mkdir(parents=True, exist_ok=True)
        for sub in [
            "dataset_manifests",
            "logs",
            "metrics",
            "checkpoints",
            "eval",
            "failures",
            "regression",
            "promotion",
        ]:
            (r_dir / sub).mkdir(parents=True, exist_ok=True)

        environment = env or capture(self.root)
        config.write_resolved(r_dir / "resolved_config.yaml")

        (r_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        record = ExperimentRecord(
            experiment_id=config.experiment_id,
            parent_experiment_id=config.parent_experiment_id,
            hypothesis=config.hypothesis,
            model_id=str(config.model.get("model_id", "")),
            model_revision=str(config.model.get("model_revision", "")),
            tokenizer_revision=str(
                config.model.get("tokenizer_revision", config.model.get("model_revision", ""))
            ),
            training_algorithm=str(config.trainer.get("type", "sft")),
            training_config=config.trainer,
            dataset_manifest_ids=list(config.datasets.get("manifest_ids", [])),
            dataset_hashes=dict(config.datasets.get("hashes", {})),
            git_commit=str(environment.get("git", {}).get("sha", "unknown")),
            git_dirty=bool(environment.get("git_dirty", False)),
            environment=environment,
            random_seed=int(config.reproducibility.get("seed", 42)),
            hardware_info=dict(environment.get("gpu", {}) or {}),
            launch_timestamp=datetime.now(UTC).isoformat(),
            status=ExperimentStatus.CREATED.value,
        )

        self._save_record(record)

        # Record events
        local_ledger = ExperimentLedger(r_dir / "ledger.jsonl")
        local_ledger.record(LedgerEventType.EXPERIMENT_CREATED, config.experiment_id)
        self.central_ledger.record(LedgerEventType.EXPERIMENT_CREATED, config.experiment_id)

        return record

    def register_record(self, record: ExperimentRecord) -> ExperimentRecord:
        """Persist a non-training lifecycle record without inventing a config schema."""
        r_dir = self.run_dir(record.experiment_id)
        target = r_dir / "experiment.json"
        if target.exists():
            raise FileExistsError(f"Experiment already exists: {record.experiment_id}")
        r_dir.mkdir(parents=True, exist_ok=True)
        for sub in [
            "dataset_manifests",
            "logs",
            "metrics",
            "checkpoints",
            "eval",
            "failures",
            "regression",
            "promotion",
        ]:
            (r_dir / sub).mkdir(parents=True, exist_ok=True)
        self._save_record(record)
        local_ledger = ExperimentLedger(r_dir / "ledger.jsonl")
        local_ledger.record(LedgerEventType.EXPERIMENT_CREATED, record.experiment_id)
        self.central_ledger.record(LedgerEventType.EXPERIMENT_CREATED, record.experiment_id)
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        exp_file = self.run_dir(experiment_id) / "experiment.json"
        if not exp_file.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found at {exp_file}")
        data = json.loads(exp_file.read_text(encoding="utf-8"))
        return ExperimentRecord.from_dict(data)

    def list_experiments(self) -> list[ExperimentRecord]:
        results: list[ExperimentRecord] = []
        # Experiment identifiers may contain slashes (for example the baseline
        # namespace). Walk only canonical record files so nested run IDs are not
        # silently omitted from readiness and status projections.
        for exp_file in sorted(self.runs_dir.rglob("experiment.json")):
            if not exp_file.is_file():
                continue
            try:
                data = json.loads(exp_file.read_text(encoding="utf-8"))
                results.append(ExperimentRecord.from_dict(data))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return results

    def update_status(
        self,
        experiment_id: str,
        new_status: ExperimentStatus | str,
        details: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        record = self.get_experiment(experiment_id)
        status_str = (
            new_status.value if isinstance(new_status, ExperimentStatus) else str(new_status)
        )
        record.status = status_str
        if status_str in {
            ExperimentStatus.EVALUATED.value,
            ExperimentStatus.PROMOTED.value,
            ExperimentStatus.REJECTED.value,
        }:
            record.completion_timestamp = datetime.now(UTC).isoformat()

        self._save_record(record)

        r_dir = self.run_dir(experiment_id)
        local_ledger = ExperimentLedger(r_dir / "ledger.jsonl")
        local_ledger.record(status_str, experiment_id, details)
        self.central_ledger.record(status_str, experiment_id, details)

        return record

    def _save_record(self, record: ExperimentRecord) -> None:
        r_dir = self.run_dir(record.experiment_id)
        target = r_dir / "experiment.json"
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.replace(target)
