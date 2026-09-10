"""Normalized benchmark result and task schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opengrad.benchmarks.taxonomy import normalize_failure_code


@dataclass
class NormalizedTaskResult:
    task_id: str
    input: str | dict[str, Any]
    raw_output: str
    parsed_output: dict[str, Any] | list[Any] | str | None
    expected: Any
    score: float
    success: bool
    failure_category: str | None = None
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input": self.input,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "expected": self.expected,
            "score": round(self.score, 4),
            "success": self.success,
            "failure_category": (
                normalize_failure_code(self.failure_category) if self.failure_category else None
            ),
            "latency": round(self.latency, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedTaskResult:
        return cls(
            task_id=str(data["task_id"]),
            input=data["input"],
            raw_output=str(data["raw_output"]),
            parsed_output=data.get("parsed_output"),
            expected=data.get("expected"),
            score=float(data.get("score", 0.0)),
            success=bool(data.get("success", False)),
            failure_category=data.get("failure_category"),
            latency=float(data.get("latency", 0.0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            tool_calls=list(data.get("tool_calls") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class NormalizedRunResult:
    run_id: str
    experiment_id: str
    model_id: str
    model_revision: str
    benchmark: str
    benchmark_revision: str
    evaluator_revision: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str = "unknown"
    environment: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    prompt_template_fingerprint: str = ""
    dataset_fingerprint: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        required = [
            "run_id",
            "experiment_id",
            "model_id",
            "model_revision",
            "benchmark",
            "benchmark_revision",
            "evaluator_revision",
            "timestamp",
            "git_commit",
            "environment",
            "generation_config",
            "prompt_template_fingerprint",
            "dataset_fingerprint",
            "result",
        ]
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            raise ValueError(f"NormalizedRunResult missing required field(s): {', '.join(missing)}")
        if not isinstance(self.result, dict):
            raise TypeError("result field must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "benchmark": self.benchmark,
            "benchmark_revision": self.benchmark_revision,
            "evaluator_revision": self.evaluator_revision,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "environment": self.environment,
            "generation_config": self.generation_config,
            "prompt_template_fingerprint": self.prompt_template_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "result": self.result,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedRunResult:
        return cls(
            run_id=str(data["run_id"]),
            experiment_id=str(data["experiment_id"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            benchmark=str(data["benchmark"]),
            benchmark_revision=str(data["benchmark_revision"]),
            evaluator_revision=str(data["evaluator_revision"]),
            timestamp=str(data.get("timestamp", "")),
            git_commit=str(data.get("git_commit", "unknown")),
            environment=dict(data.get("environment") or {}),
            generation_config=dict(data.get("generation_config") or {}),
            prompt_template_fingerprint=str(data.get("prompt_template_fingerprint", "")),
            dataset_fingerprint=str(data.get("dataset_fingerprint", "")),
            result=dict(data.get("result") or {}),
            artifacts=dict(data.get("artifacts") or {}),
        )

    def write_artifacts(
        self, output_dir: Path, tasks: list[NormalizedTaskResult], raw_benchmark_artifacts: dict[str, Any] | None = None
    ) -> None:
        """Write all machine-readable artifacts into reports/benchmarks/<run-id>/."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts["manifest"] = str(output_dir / "manifest.json")
        self.artifacts["environment"] = str(output_dir / "environment.json")
        self.artifacts["metrics"] = str(output_dir / "metrics.json")
        self.artifacts["failures"] = str(output_dir / "failures.json")
        self.artifacts["predictions"] = str(output_dir / "predictions.jsonl")

        # 1. manifest.json
        manifest_data = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "benchmark": self.benchmark,
            "benchmark_revision": self.benchmark_revision,
            "evaluator_revision": self.evaluator_revision,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "prompt_template_fingerprint": self.prompt_template_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "generation_config": self.generation_config,
            "artifacts": self.artifacts,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # 2. environment.json
        (output_dir / "environment.json").write_text(
            json.dumps(self.environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # 3. metrics.json
        (output_dir / "metrics.json").write_text(
            json.dumps(self.result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # 4. failures.json
        failure_records = [
            t.to_dict() for t in tasks if not t.success and t.failure_category
        ]
        failure_summary = {
            "total_tasks": len(tasks),
            "failed_tasks": len([t for t in tasks if not t.success]),
            "failure_breakdown": self.result.get("failure_breakdown", {}),
            "failures": failure_records,
        }
        (output_dir / "failures.json").write_text(
            json.dumps(failure_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # 5. predictions.jsonl
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")

        # 6. raw benchmark-native artifacts if present
        if raw_benchmark_artifacts:
            native_dir = output_dir / "benchmark-native"
            native_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts["benchmark_native"] = str(native_dir)
            for k, val in raw_benchmark_artifacts.items():
                target = native_dir / f"{k}.json"
                target.write_text(
                    json.dumps(val, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

        # 7. speculative.json if speculative metrics are present
        if "speculative" in self.result:
            self.artifacts["speculative"] = str(output_dir / "speculative.json")
            (output_dir / "speculative.json").write_text(
                json.dumps(self.result["speculative"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
