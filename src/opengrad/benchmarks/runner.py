"""Unified Benchmark Runner for single benchmarks and full suites."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from opengrad.benchmarks.adapters import get_adapter
from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.benchmarks.backends.protocol import InferenceBackend
from opengrad.benchmarks.config import BenchmarkConfig, BenchmarkSuiteConfig
from opengrad.benchmarks.registry import BenchmarkRegistry
from opengrad.benchmarks.schema import NormalizedRunResult, NormalizedTaskResult
from opengrad.env_capture import capture


class BenchmarkRunner:
    """Orchestrates benchmark execution across adapters and inference backends."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.registry = BenchmarkRegistry(self.root)

    def run_benchmark(
        self,
        config: BenchmarkConfig | Path | str,
        *,
        backend: InferenceBackend | None = None,
        dry_run: bool = False,
        limit: int | None = None,
        output_dir: Path | None = None,
    ) -> NormalizedRunResult:
        if isinstance(config, (str, Path)):
            config = BenchmarkConfig.from_file(config)

        # Get adapter
        adapter = get_adapter(config.benchmark_id, root=self.root)

        # Select backend
        if backend is not None:
            active_backend = backend
        elif dry_run or config.inference_backend == "mock":
            active_backend = DeterministicFakeBackend(
                mode="mtp" if "mtp" in config.inference_backend else "ar"
            )
        else:
            from opengrad.benchmarks.backends.transformers_backend import (
                TransformersInferenceBackend,
            )

            active_backend = TransformersInferenceBackend(
                model_id=config.model_id,
                revision=config.model_revision,
                precision=config.precision,
            )

        # Load tasks
        tasks = adapter.load_tasks(split=config.split, limit=limit)
        task_results: list[NormalizedTaskResult] = []

        start_time = time.monotonic()
        for task in tasks:
            prompt_text = adapter.render_prompt(task, renderer_name=config.prompt_template)
            meta = {
                "task_id": task.task_id,
                "category": task.category,
                "expected_decision": task.expected_decision,
                **task.metadata,
            }
            gen_res = active_backend.generate(
                prompt_text,
                generation_config=config.generation.to_dict(),
                tools=task.tools,
                metadata=meta,
            )
            task_res = adapter.evaluate_task(task, gen_res)
            task_results.append(task_res)

        elapsed = time.monotonic() - start_time
        metrics = adapter.aggregate_metrics(task_results)
        metrics["elapsed_seconds"] = round(elapsed, 4)

        # Extract speculative metrics if present
        spec_entries: list[dict[str, Any]] = [
            t.metadata["speculative"]
            for t in task_results
            if isinstance(t.metadata.get("speculative"), dict)
        ]
        if spec_entries:
            acceptance_sum = sum(float(s.get("acceptance_rate", 0.0)) for s in spec_entries)
            tokens_step_sum = sum(float(s.get("accepted_tokens_per_step", 1.0)) for s in spec_entries)
            mean_acceptance = round(acceptance_sum / len(spec_entries), 4)
            mean_tokens_step = round(tokens_step_sum / len(spec_entries), 2)
            metrics["speculative"] = {
                "samples_evaluated": len(spec_entries),
                "mean_acceptance_rate": mean_acceptance,
                "mean_accepted_tokens_per_step": mean_tokens_step,
                "latest_sample_telemetry": spec_entries[-1],
            }

        # Environment capture
        env = capture(self.root)
        env["backend"] = getattr(active_backend, "name", "unknown")
        env["dry_run"] = dry_run

        run_id = f"{config.benchmark_id}/{config.model_id.replace('/', '_')}_{int(time.time())}"
        experiment_id = f"exp_{config.benchmark_id}"
        prompt_fp = hashlib.sha256(config.prompt_template.encode()).hexdigest()[:16]
        dataset_fp = hashlib.sha256(f"{config.benchmark_id}_{config.split}".encode()).hexdigest()[:16]

        run_result = NormalizedRunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            model_id=config.model_id,
            model_revision=config.model_revision,
            benchmark=config.benchmark_id,
            benchmark_revision=config.benchmark_revision,
            evaluator_revision=config.evaluator_revision,
            git_commit=str(env.get("git", {}).get("sha", "unknown")),
            environment=env,
            generation_config=config.generation.to_dict(),
            prompt_template_fingerprint=prompt_fp,
            dataset_fingerprint=dataset_fp,
            result=metrics,
        )

        out_path = output_dir or (self.root / "reports" / "benchmarks" / run_id)
        run_result.write_artifacts(out_path, task_results)
        return run_result

    def run_suite(
        self,
        suite_config: BenchmarkSuiteConfig | Path | str,
        *,
        backend: InferenceBackend | None = None,
        dry_run: bool = False,
        limit: int | None = None,
        output_base_dir: Path | None = None,
    ) -> dict[str, Any]:
        if isinstance(suite_config, (str, Path)):
            suite_config = BenchmarkSuiteConfig.from_file(suite_config)

        results: dict[str, Any] = {}
        for benchmark_name in suite_config.benchmarks:
            # Resolve config path
            cfg_file = self.root / "configs" / "benchmarks" / f"{benchmark_name}.yaml"
            if not cfg_file.exists():
                cfg_file = self.root / "configs" / "benchmarks" / f"{benchmark_name.replace('-', '_')}.yaml"
            if not cfg_file.exists():
                cfg_file = Path(benchmark_name)
            if not cfg_file.exists():
                raise FileNotFoundError(
                    f"Could not locate benchmark config for '{benchmark_name}'"
                )

            cfg = BenchmarkConfig.from_file(cfg_file)
            run_res = self.run_benchmark(
                cfg,
                backend=backend,
                dry_run=dry_run,
                limit=limit,
                output_dir=(
                    output_base_dir / cfg.benchmark_id
                    if output_base_dir
                    else None
                ),
            )
            results[cfg.benchmark_id] = run_res.to_dict()

        return {
            "suite_id": suite_config.suite_id,
            "description": suite_config.description,
            "dry_run": dry_run,
            "benchmarks_run": len(results),
            "results": results,
        }
