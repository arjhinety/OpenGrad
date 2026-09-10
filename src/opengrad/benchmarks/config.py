"""Declarative benchmark and benchmark suite configuration parser and validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int
    top_k: int | None = None
    do_sample: bool = False
    stop_sequences: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationParameters:
        required = ["temperature", "top_p", "max_output_tokens", "seed"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Missing research-critical generation parameter(s): {', '.join(missing)}"
            )
        return cls(
            temperature=float(data["temperature"]),
            top_p=float(data["top_p"]),
            max_output_tokens=int(data["max_output_tokens"]),
            seed=int(data["seed"]),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            do_sample=bool(data.get("do_sample", False)),
            stop_sequences=list(data.get("stop_sequences") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "stop_sequences": self.stop_sequences,
        }
        if self.top_k is not None:
            result["top_k"] = self.top_k
        return result


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_id: str
    benchmark_revision: str
    evaluator_revision: str
    task_subset: str
    split: str
    model_id: str
    model_revision: str
    tokenizer: str
    prompt_template: str
    generation: GenerationParameters
    max_context: int
    tool_schema_rendering_policy: str
    parser: str
    inference_backend: str
    batch_size: int
    concurrency: int
    precision: str
    device_policy: str = "cpu_or_accelerator"
    runtime_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        required = [
            "benchmark_id",
            "benchmark_revision",
            "evaluator_revision",
            "task_subset",
            "split",
            "model_id",
            "model_revision",
            "tokenizer",
            "prompt_template",
            "generation",
            "max_context",
            "tool_schema_rendering_policy",
            "parser",
            "inference_backend",
            "batch_size",
            "concurrency",
            "precision",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Benchmark configuration is missing research-critical parameter(s): {', '.join(missing)}"
            )

        gen_raw = data["generation"]
        if not isinstance(gen_raw, dict):
            raise TypeError("generation parameters must be a dictionary")
        gen_params = GenerationParameters.from_dict(gen_raw)

        return cls(
            benchmark_id=str(data["benchmark_id"]),
            benchmark_revision=str(data["benchmark_revision"]),
            evaluator_revision=str(data["evaluator_revision"]),
            task_subset=str(data["task_subset"]),
            split=str(data["split"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            tokenizer=str(data["tokenizer"]),
            prompt_template=str(data["prompt_template"]),
            generation=gen_params,
            max_context=int(data["max_context"]),
            tool_schema_rendering_policy=str(data["tool_schema_rendering_policy"]),
            parser=str(data["parser"]),
            inference_backend=str(data["inference_backend"]),
            batch_size=int(data["batch_size"]),
            concurrency=int(data["concurrency"]),
            precision=str(data["precision"]),
            device_policy=str(data.get("device_policy", "cpu_or_accelerator")),
            runtime_details=dict(data.get("runtime_details") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> BenchmarkConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Benchmark config file must contain a YAML object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_revision": self.benchmark_revision,
            "evaluator_revision": self.evaluator_revision,
            "task_subset": self.task_subset,
            "split": self.split,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer": self.tokenizer,
            "prompt_template": self.prompt_template,
            "generation": self.generation.to_dict(),
            "max_context": self.max_context,
            "tool_schema_rendering_policy": self.tool_schema_rendering_policy,
            "parser": self.parser,
            "inference_backend": self.inference_backend,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "precision": self.precision,
            "device_policy": self.device_policy,
            "runtime_details": self.runtime_details,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    suite_id: str
    description: str
    benchmarks: list[str]  # config names or paths
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSuiteConfig:
        if "suite_id" not in data or "benchmarks" not in data:
            raise ValueError("Benchmark suite config requires 'suite_id' and 'benchmarks'")
        benchmarks = data["benchmarks"]
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ValueError("'benchmarks' must be a non-empty list of benchmark config names")
        return cls(
            suite_id=str(data["suite_id"]),
            description=str(data.get("description", "")),
            benchmarks=[str(b) for b in benchmarks],
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> BenchmarkSuiteConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Suite config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Suite config file must contain a YAML object: {path}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "description": self.description,
            "benchmarks": self.benchmarks,
            "metadata": self.metadata,
        }


def check_run_compatibility(
    baseline_config: BenchmarkConfig, candidate_config: BenchmarkConfig
) -> list[str]:
    """Verify that two runs share identical research-critical evaluation parameters.
    
    Returns a list of incompatibility reasons. If empty, the runs are strictly comparable.
    """
    incompatibilities: list[str] = []

    if baseline_config.benchmark_id != candidate_config.benchmark_id:
        incompatibilities.append(
            f"Benchmark ID mismatch: '{baseline_config.benchmark_id}' vs '{candidate_config.benchmark_id}'"
        )
    if baseline_config.benchmark_revision != candidate_config.benchmark_revision:
        incompatibilities.append(
            f"Benchmark revision mismatch: '{baseline_config.benchmark_revision}' vs '{candidate_config.benchmark_revision}'"
        )
    if baseline_config.evaluator_revision != candidate_config.evaluator_revision:
        incompatibilities.append(
            f"Evaluator revision mismatch: '{baseline_config.evaluator_revision}' vs '{candidate_config.evaluator_revision}'"
        )
    if baseline_config.task_subset != candidate_config.task_subset:
        incompatibilities.append(
            f"Task subset mismatch: '{baseline_config.task_subset}' vs '{candidate_config.task_subset}'"
        )
    if baseline_config.split != candidate_config.split:
        incompatibilities.append(
            f"Split mismatch: '{baseline_config.split}' vs '{candidate_config.split}'"
        )
    if baseline_config.prompt_template != candidate_config.prompt_template:
        incompatibilities.append(
            f"Prompt template mismatch: '{baseline_config.prompt_template}' vs '{candidate_config.prompt_template}'"
        )
    if baseline_config.tool_schema_rendering_policy != candidate_config.tool_schema_rendering_policy:
        incompatibilities.append(
            f"Tool schema rendering policy mismatch: '{baseline_config.tool_schema_rendering_policy}' vs '{candidate_config.tool_schema_rendering_policy}'"
        )
    if baseline_config.parser != candidate_config.parser:
        incompatibilities.append(
            f"Parser mismatch: '{baseline_config.parser}' vs '{candidate_config.parser}'"
        )

    # Check generation sampling
    bg, cg = baseline_config.generation, candidate_config.generation
    if bg.temperature != cg.temperature:
        incompatibilities.append(f"Temperature mismatch: {bg.temperature} vs {cg.temperature}")
    if bg.top_p != cg.top_p:
        incompatibilities.append(f"Top-p mismatch: {bg.top_p} vs {cg.top_p}")
    if bg.max_output_tokens != cg.max_output_tokens:
        incompatibilities.append(
            f"Max output tokens mismatch: {bg.max_output_tokens} vs {cg.max_output_tokens}"
        )
    if bg.seed != cg.seed:
        incompatibilities.append(f"Seed mismatch: {bg.seed} vs {cg.seed}")
    if bg.stop_sequences != cg.stop_sequences:
        incompatibilities.append(
            f"Stop sequences mismatch: {bg.stop_sequences} vs {cg.stop_sequences}"
        )

    return incompatibilities
