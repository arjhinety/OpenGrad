
import pytest

from opengrad.cli import main


def test_cli_doctor(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "doctor"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "OpenGrad System Doctor" in captured.out
    assert "android_studio" in captured.out


def test_cli_preflight_experiment(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "preflight", "configs/experiments/m0_sft.yaml"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Preflight Verification" in captured.out


def test_cli_inspect_template(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "inspect-template", "--max-tokens", "10"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Template & Loss Mask Inspection" in captured.out


def test_cli_experiment_and_checkpoint_subcommands(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["opengrad", "checkpoint", "list"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Registered Checkpoints" in captured.out

    monkeypatch.setattr("sys.argv", ["opengrad", "experiment", "list"])
    assert main() == 0
    captured = capsys.readouterr()
    assert "Registered Experiments" in captured.out
