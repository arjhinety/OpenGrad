import json
from argparse import Namespace
from pathlib import Path

import pytest

from opengrad.cli import main

M0_SFT_CONFIG = Path("configs/experiments/m0_sft.yaml")


class _WarnPreflight:
    overall_status = "WARN"

    def to_dict(self) -> dict:
        return {"overall_status": "WARN"}

    def render_summary(self) -> str:
        return "Preflight Verification: WARN"


def test_cli_doctor(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "doctor"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "OpenGrad System Doctor" in captured.out
    assert "android_studio" in captured.out


def test_cli_preflight_experiment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "preflight", "configs/experiments/m0_sft.yaml"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Preflight Verification" in captured.out


def test_cli_inspect_template(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "inspect-template", "--max-tokens", "10"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Template & Loss Mask Inspection" in captured.out


def test_cli_experiment_and_checkpoint_subcommands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "checkpoint", "list"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Registered Checkpoints" in captured.out

    monkeypatch.setattr("sys.argv", ["opengrad", "experiment", "list"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Registered Experiments" in captured.out


def test_dry_run_sft_is_non_evidence_and_does_not_collide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from opengrad import agent_cli
    from opengrad.experiments.schema import ExperimentConfig
    from opengrad.experiments.store import ExperimentStore

    config = tmp_path / "sft.yaml"
    config.write_text(M0_SFT_CONFIG.read_text(), encoding="utf-8")
    monkeypatch.setattr(
        agent_cli, "run_experiment_preflight", lambda *args, **kwargs: _WarnPreflight()
    )

    # A pre-existing record with the same ID must not block an explicitly
    # non-evidence CPU dry-run, and the dry-run must not mutate it.
    store = ExperimentStore(tmp_path)
    store.create_experiment(ExperimentConfig.from_file(config))
    before = json.dumps(store.get_experiment("qwen35_2b_m0_sft").to_dict(), sort_keys=True)

    args = Namespace(config=str(config), dry_run=True, json=True)
    assert agent_cli.handle_train(args, tmp_path) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN"
    assert payload["evidence"] is False
    after = json.dumps(store.get_experiment("qwen35_2b_m0_sft").to_dict(), sort_keys=True)
    assert after == before


def test_real_sft_is_refused_when_preflight_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from opengrad import agent_cli
    from opengrad.experiments.store import ExperimentStore

    config = tmp_path / "sft.yaml"
    config.write_text(M0_SFT_CONFIG.read_text(), encoding="utf-8")
    monkeypatch.setattr(
        agent_cli, "run_experiment_preflight", lambda *args, **kwargs: _WarnPreflight()
    )

    args = Namespace(config=str(config), dry_run=False, json=True)
    assert agent_cli.handle_train(args, tmp_path) == 1
    assert ExperimentStore(tmp_path).list_experiments() == []


def test_cli_evaluate_rejects_a_non_allowlisted_suite(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["opengrad", "evaluate", "runs/example", "--suite", "../../etc/passwd", "--json"],
    )
    assert main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "SUITE_NOT_FOUND"


def test_cli_baseline_dry_run_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A completed dry run is a success; non-zero made bridges report a failure."""
    from opengrad import cli

    monkeypatch.setattr(
        cli,
        "run_baseline",
        lambda *a, **k: {"status": "DRY_RUN", "records": 1, "artifacts": {}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "opengrad",
            "baseline",
            "--config",
            "configs/evaluation/tool_calling/qwen35_2b_baseline.yaml",
            "--dry-run",
            "--json",
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DRY_RUN"
