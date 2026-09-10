# How to Add a New Benchmark to OpenGrad

This guide explains how to add an evaluation benchmark while adhering to OpenGrad's reproducibility, typing, and schema standards.

---

## Step 1: Register in `registry/benchmarks.yaml`

Add an entry with pinned source information, tier classification, and licensing:

```yaml
- id: my-new-benchmark
  name: My New Benchmark
  tier: TIER_A  # TIER_A, TIER_B, TIER_C, TIER_D, or TIER_E
  canonical_repository: https://github.com/org/repo
  reference: arxiv:XXXX.XXXXX
  version: v1.0
  license: Apache-2.0
  commit_sha: abcdef0123456789abcdef0123456789abcdef01
  evaluator_version: opengrad-mybenchmark-eval-v1
  splits:
    - default
    - hard
  metrics:
    - accuracy
    - tool_selection_score
  contamination_sensitivity: HIGH
  prohibit_training: true
  parser_requirements: model-native-or-adapter
  notes: Brief description of what this benchmark evaluates.
```

Verify registry validity:
```bash
opengrad-validate
```

---

## Step 2: Implement the Benchmark Adapter

Create `src/opengrad/benchmarks/adapters/my_benchmark.py`:

```python
from __future__ import annotations

from typing import Any, ClassVar
from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output

class MyBenchmarkAdapter(BenchmarkAdapter):
    benchmark_id = "my-new-benchmark"
    name = "My New Benchmark"

    SPLITS: ClassVar[list[str]] = ["default", "hard"]

    def load_tasks(self, split: str = "default", limit: int | None = None) -> list[BenchmarkTask]:
        # Return BenchmarkTask objects (or load from local cache/disk)
        tasks = [
            BenchmarkTask(
                task_id="my_task_001",
                category="default",
                prompt="User request here...",
                tools=[...],
                expected={"tool": "target_tool"},
                expected_decision="CALL",
            )
        ]
        return tasks[:limit] if limit else tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        parsed = parse_qwen_native_output(generation.text)
        success = (parsed.decision == task.expected_decision)
        failure_cat = None if success else ToolFailureCategory.MISSED_TOOL.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"decision": parsed.decision},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_cat,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )
```

Register the adapter in `src/opengrad/benchmarks/adapters/__init__.py` in `ADAPTERS_MAP`.

---

## Step 3: Create Declarative Config

Create `configs/benchmarks/my_new_benchmark.yaml`:

```yaml
schema_version: 1
benchmark_id: my-new-benchmark
benchmark_revision: abcdef0123456789abcdef0123456789abcdef01
evaluator_revision: opengrad-mybenchmark-eval-v1
task_subset: all
split: default
model_id: Qwen/Qwen3.5-2B
model_revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
tokenizer: Qwen/Qwen3.5-2B
prompt_template: qwen3_5_2b_v1
generation:
  temperature: 0.0
  top_p: 1.0
  max_output_tokens: 512
  seed: 42
  do_sample: false
  stop_sequences: []
max_context: 4096
tool_schema_rendering_policy: qwen_native
parser: opengrad.formatting.parser.parse_qwen_native_output
inference_backend: mock
batch_size: 1
concurrency: 1
precision: bfloat16
device_policy: cpu_or_accelerator
runtime_details: {}
metadata:
  tier: TIER_A
```

Validate all configs:
```bash
opengrad benchmark validate
```

---

## Step 4: Add Unit Tests & Run Dry-Run

Add a unit test in `tests/benchmarks/` verifying:
- Task loading
- Evaluation logic
- Failure categorization

Execute a dry-run:
```bash
opengrad benchmark run --benchmark my_new_benchmark --dry-run --limit 2
```
