"""Multi-variable experiment diffing and causal attribution warning engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opengrad.experiments.schema import ExperimentRecord


@dataclass
class ExperimentDiff:
    exp_a_id: str
    exp_b_id: str
    differences: dict[str, tuple[Any, Any]]
    variables_changed_count: int
    is_single_variable: bool
    warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_a_id": self.exp_a_id,
            "exp_b_id": self.exp_b_id,
            "variables_changed_count": self.variables_changed_count,
            "is_single_variable": self.is_single_variable,
            "warning": self.warning,
            "differences": {k: {"baseline": v[0], "candidate": v[1]} for k, v in self.differences.items()},
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Experiment Diff: `{self.exp_a_id}` vs `{self.exp_b_id}`",
            "",
            f"- **Variables Changed:** {self.variables_changed_count}",
            f"- **Controlled Experiment Status:** {'Controlled Single-Variable (High Causal Validity)' if self.is_single_variable else 'Multi-Variable Change (Confounded Causal Attribution)'}",
            "",
        ]
        if self.warning:
            lines.append(f"> ⚠️ **Methodological Warning:** {self.warning}\n")

        lines.append("| Dimension | Baseline (`" + self.exp_a_id + "`) | Candidate (`" + self.exp_b_id + "`) |")
        lines.append("| :--- | :--- | :--- |")

        for key, (val_a, val_b) in sorted(self.differences.items()):
            lines.append(f"| `{key}` | `{val_a}` | `{val_b}` |")

        if not self.differences:
            lines.append("| *No differences* | - | - |")

        return "\n".join(lines)


def diff_experiments(
    exp_a: ExperimentRecord | dict[str, Any],
    exp_b: ExperimentRecord | dict[str, Any],
) -> ExperimentDiff:
    """Compare two experiments across training configuration, datasets, models, and seeds."""
    data_a = exp_a.to_dict() if hasattr(exp_a, "to_dict") else dict(exp_a)
    data_b = exp_b.to_dict() if hasattr(exp_b, "to_dict") else dict(exp_b)

    id_a = str(data_a.get("experiment_id", "exp_a"))
    id_b = str(data_b.get("experiment_id", "exp_b"))

    model_a = data_a.get("model", {}) if isinstance(data_a.get("model"), dict) else {}
    model_b = data_b.get("model", {}) if isinstance(data_b.get("model"), dict) else {}
    trainer_a = data_a.get("trainer", {}) if isinstance(data_a.get("trainer"), dict) else {}
    trainer_b = data_b.get("trainer", {}) if isinstance(data_b.get("trainer"), dict) else {}
    datasets_a = data_a.get("datasets", {}) if isinstance(data_a.get("datasets"), dict) else {}
    datasets_b = data_b.get("datasets", {}) if isinstance(data_b.get("datasets"), dict) else {}
    repro_a = data_a.get("reproducibility", {}) if isinstance(data_a.get("reproducibility"), dict) else {}
    repro_b = data_b.get("reproducibility", {}) if isinstance(data_b.get("reproducibility"), dict) else {}

    norm_a = {
        "model_id": data_a.get("model_id") or model_a.get("model_id"),
        "model_revision": data_a.get("model_revision") or model_a.get("model_revision"),
        "tokenizer_revision": data_a.get("tokenizer_revision") or model_a.get("tokenizer_revision"),
        "training_algorithm": data_a.get("training_algorithm") or trainer_a.get("type"),
        "dataset_manifest_ids": data_a.get("dataset_manifest_ids") or datasets_a.get("manifest_ids"),
        "random_seed": data_a.get("random_seed") or repro_a.get("seed"),
        "git_commit": data_a.get("git_commit"),
        "training_config": data_a.get("training_config") or trainer_a,
    }

    norm_b = {
        "model_id": data_b.get("model_id") or model_b.get("model_id"),
        "model_revision": data_b.get("model_revision") or model_b.get("model_revision"),
        "tokenizer_revision": data_b.get("tokenizer_revision") or model_b.get("tokenizer_revision"),
        "training_algorithm": data_b.get("training_algorithm") or trainer_b.get("type"),
        "dataset_manifest_ids": data_b.get("dataset_manifest_ids") or datasets_b.get("manifest_ids"),
        "random_seed": data_b.get("random_seed") or repro_b.get("seed"),
        "git_commit": data_b.get("git_commit"),
        "training_config": data_b.get("training_config") or trainer_b,
    }

    diffs: dict[str, tuple[Any, Any]] = {}

    keys_to_compare = [
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "training_algorithm",
        "dataset_manifest_ids",
        "random_seed",
        "git_commit",
    ]

    for key in keys_to_compare:
        val_a = norm_a.get(key)
        val_b = norm_b.get(key)
        if val_a != val_b:
            diffs[key] = (val_a, val_b)

    cfg_a = norm_a.get("training_config", {})
    cfg_b = norm_b.get("training_config", {})
    if isinstance(cfg_a, dict) and isinstance(cfg_b, dict):
        all_cfg_keys = sorted(set(cfg_a.keys()) | set(cfg_b.keys()))
        for c_key in all_cfg_keys:
            v_a = cfg_a.get(c_key)
            v_b = cfg_b.get(c_key)
            if v_a != v_b:
                diffs[f"training_config.{c_key}"] = (v_a, v_b)

    num_vars = len(diffs)
    is_single = num_vars == 1
    warning = None
    if num_vars > 1:
        warning = (
            f"Candidate changed {num_vars} variables simultaneously from baseline. "
            "Causal attribution of performance deltas to a single hyperparameter/dataset is weak."
        )

    return ExperimentDiff(
        exp_a_id=id_a,
        exp_b_id=id_b,
        differences=diffs,
        variables_changed_count=num_vars,
        is_single_variable=is_single,
        warning=warning,
    )
