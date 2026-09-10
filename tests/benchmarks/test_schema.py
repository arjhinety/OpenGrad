import json
from pathlib import Path

from opengrad.benchmarks.schema import NormalizedRunResult, NormalizedTaskResult


def test_normalized_task_result_roundtrip() -> None:
    task = NormalizedTaskResult(
        task_id="test_01",
        input="Find the temperature in Paris.",
        raw_output='<tool_call>{"name": "get_temp", "arguments": {"city": "Paris"}}</tool_call>',
        parsed_output={"name": "get_temp", "arguments": {"city": "Paris"}},
        expected={"name": "get_temp"},
        score=1.0,
        success=True,
        failure_category=None,
        latency=0.125,
        prompt_tokens=32,
        completion_tokens=18,
        tool_calls=[{"name": "get_temp", "arguments": {"city": "Paris"}}],
        metadata={"category": "simple"},
    )
    d = task.to_dict()
    assert d["score"] == 1.0
    assert d["success"] is True
    reconstructed = NormalizedTaskResult.from_dict(d)
    assert reconstructed.task_id == "test_01"
    assert reconstructed.score == 1.0


def test_normalized_run_result_validation_and_artifacts(tmp_path: Path) -> None:
    run_res = NormalizedRunResult(
        run_id="bfcl_v4_test_run",
        experiment_id="exp_01",
        model_id="Qwen/Qwen3.5-2B",
        model_revision="pinned_rev",
        benchmark="bfcl-v4",
        benchmark_revision="rev_v4",
        evaluator_revision="eval_rev",
        git_commit="abcdef",
        environment={"python": "3.12"},
        generation_config={"temperature": 0.0},
        prompt_template_fingerprint="fp123",
        dataset_fingerprint="dfp123",
        result={"overall_accuracy": 92.5, "failure_breakdown": {"missed_tool": 2}},
    )

    task = NormalizedTaskResult(
        task_id="t1",
        input="q",
        raw_output="out",
        parsed_output={},
        expected={},
        score=0.0,
        success=False,
        failure_category="missed_tool",
    )

    run_res.write_artifacts(tmp_path, [task])
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "environment.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "failures.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["overall_accuracy"] == 92.5
