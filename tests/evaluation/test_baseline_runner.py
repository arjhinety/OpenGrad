import json
from pathlib import Path

import pytest
import yaml

from opengrad.data.renderers import RenderedTrainingExample
from opengrad.evaluation.runner import FakeDeterministicBackend, run_baseline
from opengrad.formatting.parser import parse_qwen_native_output

BASELINE_CONFIG = Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml")
BASELINE_EXPERIMENT_ID = "tool_calling/qwen35_2b/baseline"


def _patch_manifest_and_renderer(monkeypatch, root: Path):
    from opengrad.data.canonical import CanonicalEvaluationExample
    from opengrad.evaluation import runner

    # `run_baseline` hashes the frozen manifest bytes even when the split
    # loader is stubbed, so the pinned manifest must exist under the test root.
    manifest = root / "reports/evaluation/behavioral-heldout-v2.manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(
        Path("reports/evaluation/behavioral-heldout-v2.manifest.json").read_bytes()
    )

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


def test_real_baseline_registers_canonical_record_and_refuses_overwrite(
    monkeypatch, tmp_path: Path
):
    from opengrad.experiments.store import ExperimentStore

    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    result = run_baseline(
        BASELINE_CONFIG.resolve(),
        root=tmp_path,
        backend=FakeDeterministicBackend(),
        limit=3,
        dry_run=False,
    )
    assert result["status"] == "EXECUTED"
    record = ExperimentStore(tmp_path).get_experiment(BASELINE_EXPERIMENT_ID)
    assert record.status == "EVALUATED"
    assert record.model_id == "Qwen/Qwen3.5-2B"
    assert record.model_revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    # A second real run must refuse to overwrite the recorded evidence.
    with pytest.raises(FileExistsError):
        run_baseline(
            BASELINE_CONFIG.resolve(),
            root=tmp_path,
            backend=FakeDeterministicBackend(),
            limit=3,
            dry_run=False,
        )


def test_dry_run_baseline_creates_no_experiment_record(monkeypatch, tmp_path: Path):
    from opengrad.experiments.store import ExperimentStore

    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["outputs"] = {
        key: str(tmp_path / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = run_baseline(
        config_path, root=tmp_path, backend=FakeDeterministicBackend(), limit=2, dry_run=True
    )
    assert result["status"] == "DRY_RUN"
    assert ExperimentStore(tmp_path).list_experiments() == []


def test_dry_run_baseline_never_writes_canonical_evidence_paths(monkeypatch, tmp_path: Path):
    """A dry run must not create files the real run then refuses to overwrite.

    The default config uses repository-relative output paths. Before this guard, a
    routine `baseline --dry-run` wrote deterministic-mock artifacts into
    reports/baselines/... and permanently blocked the real B0 behind the
    "evidence already exists" overwrite check.
    """
    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    result = run_baseline(
        BASELINE_CONFIG.resolve(),
        root=tmp_path,
        backend=FakeDeterministicBackend(),
        limit=2,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    for relative in (
        "reports/baselines/qwen35_2b_baseline/metrics.json",
        "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
        "reports/baselines/qwen35_2b_baseline/environment.json",
        "reports/failures/qwen35_2b_baseline/residual-profile.json",
    ):
        assert not (tmp_path / relative).exists(), f"dry run polluted {relative}"
    assert (tmp_path / "runs/.dry-run/qwen35_2b_baseline/metrics.json").is_file()

    # The canonical paths must still be free for the real, evidence-producing run.
    real = run_baseline(
        BASELINE_CONFIG.resolve(),
        root=tmp_path,
        backend=FakeDeterministicBackend(),
        limit=2,
        dry_run=False,
    )
    assert real["status"] == "EXECUTED"
    assert (tmp_path / "reports/baselines/qwen35_2b_baseline/metrics.json").is_file()


def test_baseline_config_validation_rejects_revision_and_output_drift():
    from opengrad.evaluation import runner

    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    runner._validate_baseline_config(config)
    drifted_revision = dict(config, model_revision="0" * 40)
    with pytest.raises(ValueError):
        runner._validate_baseline_config(drifted_revision)
    drifted_outputs = dict(
        config, outputs={k: v for k, v in config["outputs"].items() if k != "metrics"}
    )
    with pytest.raises(ValueError):
        runner._validate_baseline_config(drifted_outputs)


def test_baseline_paths_cannot_escape_the_project(tmp_path: Path):
    from opengrad.evaluation import runner

    with pytest.raises(ValueError):
        runner._project_path(tmp_path, "../escape.json", "output")
    with pytest.raises(ValueError):
        runner._project_path(tmp_path, str(tmp_path.parent / "abs.json"), "output")


def test_baseline_plumbing_runs_over_a_real_materialized_split(monkeypatch, tmp_path: Path):
    """End-to-end B0 plumbing over real Parquet, with no mocked manifest loader.

    Only the tokenizer-dependent renderer is stubbed, because it needs the pinned
    Qwen tokenizer. Everything else is real: manifest contract load, Parquet
    row/count/ID reads, content-hash verification, canonical example
    construction, prediction, and artifact writing. This is what pins the
    materializer and the evaluator to the same content-hash definition -- a
    one-character drift between them would otherwise only surface on GPU day.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from opengrad.data.materialize import materialize_parquet
    from opengrad.evaluation import runner

    source = tmp_path / "mcq-source.parquet"
    pa_rows = [
        {
            "uuid": f"e{i}",
            "question": f"Question {i}?",
            "correct_answer": "direct",
            "answers": {"direct": "yes", "tool": "no"},
            "tools": [
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
        }
        for i in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(pa_rows), source)

    split_dir = tmp_path / "data/processed/normalization-v1/when2call-mcq"
    materialized = materialize_parquet(
        source, split_dir, dataset="when2call", split="mcq", mode="evaluation"
    )
    assert materialized["manifest"]["counts"]["written"] == 3

    manifest_path = tmp_path / "reports/evaluation/behavioral-heldout-v2.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "behavioral-heldout-v2",
                "frozen": True,
                "status": "MATERIALIZED",
                "freeze_revision": runner.PINNED_EVALUATOR_REVISION,
                "model_renderer_contract": {
                    "model_revision": runner.PINNED_MODEL_REVISION,
                    "renderer": "qwen3_5_2b_v1",
                    "template_hash": runner.PINNED_TEMPLATE_HASH,
                },
                "contamination_policy": {
                    "derived_prompts_excluded": True,
                    "training_manifests_excluded": [],
                },
                "splits": [
                    {
                        "id": "when2call-mcq",
                        "items": 3,
                        "content_hash": materialized["manifest"]["content_hash"],
                        "source": "data/processed/normalization-v1/when2call-mcq/manifest.json",
                        "purpose": "fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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

    monkeypatch.setattr(runner.Qwen35_2BRenderer, "render_evaluation", render)
    monkeypatch.setattr(
        runner.Qwen35_2BRenderer, "text_token_length", lambda self, text: len(text.split())
    )

    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["outputs"] = {
        key: str(tmp_path / "out" / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_baseline(
        config_path,
        root=tmp_path,
        backend=FakeDeterministicBackend(),
        limit=3,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert result["records"] == 3

    # Artifact paths are root-relative by contract; resolve them for assertion.
    def artifact(name: str) -> Path:
        path = Path(result["artifacts"][name])
        return path if path.is_absolute() else tmp_path / path

    predictions = [json.loads(line) for line in artifact("predictions").read_text().splitlines()]
    assert {row["example_id"] for row in predictions} == {"e0", "e1", "e2"}
    assert all(row["parser"]["status"] == "RAW_VALID" for row in predictions)
    residual = json.loads(artifact("residual_profile").read_text())
    assert residual["sample_count"] == 3
    assert residual["manifest_sha256"] == result["manifest_sha256"]
