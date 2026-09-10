import json
from pathlib import Path

import yaml

from opengrad.data.renderers import RenderedTrainingExample
from opengrad.evaluation.runner import FakeDeterministicBackend, run_baseline
from opengrad.formatting.parser import parse_qwen_native_output


def test_native_parser_covers_golden_boundary_cases():
    valid = parse_qwen_native_output(
        '<think>hidden</think><tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'
    )
    assert valid.status == "RAW_VALID" and valid.decision == "CALL"
    assert valid.calls[0].arguments == {"q": "x"}
    assert parse_qwen_native_output("Could you clarify the account?").decision == "CLARIFY"
    assert (
        parse_qwen_native_output("I cannot help with that unsupported request.").decision
        == "UNSUPPORTED"
    )
    malformed = parse_qwen_native_output('<tool_call>{"name":"lookup","arguments":</tool_call>')
    assert malformed.status == "FORMAT_ERROR"
    assert (
        parse_qwen_native_output(
            '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>'
        )
        .calls[1]
        .name
        == "b"
    )


def test_dry_run_executes_manifest_to_artifacts(monkeypatch, tmp_path: Path):
    from opengrad.data.canonical import CanonicalEvaluationExample
    from opengrad.evaluation import runner

    def render(self, example):
        return RenderedTrainingExample(
            example.example_id,
            "fixture prompt",
            "Qwen/Qwen3.5-2B",
            self.model_revision,
            self.renderer_version,
            self.model_revision,
            "0" * 64,
            False,
        )

    def mock_examples(root, manifest_path):
        return [
            CanonicalEvaluationExample(
                f"fixture-{i}",
                {"dataset_id": "when2call-mcq"},
                "test question",
                [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                "CALL" if i % 2 == 0 else "ANSWER",
                [],
                {"source": "fixture"},
            )
            for i in range(4)
        ]

    monkeypatch.setattr(runner.Qwen35_2BRenderer, "render_evaluation", render)
    monkeypatch.setattr(runner, "load_evaluation_examples", mock_examples)
    config = yaml.safe_load(
        Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml").read_text()
    )
    config["outputs"] = {
        key: str(tmp_path / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = run_baseline(
        config_path,
        root=Path.cwd(),
        backend=FakeDeterministicBackend(),
        limit=4,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN" and result["records"] == 4
    paths = result["artifacts"]
    assert all(Path(path).exists() for path in paths.values())
    first = json.loads(Path(paths["predictions"]).read_text().splitlines()[0])
    assert {"raw_output", "prediction", "parser"} <= first.keys()
