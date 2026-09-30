"""scripts/run_external_annotation.py: how each external CLI is launched (no CLI is run here)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_external_annotation", ROOT / "scripts/run_external_annotation.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agy_gets_its_isolated_directory_as_workspace() -> None:
    # agy 1.2.10 ignores its starting directory; --add-dir {workdir} is what lets it read input.md.
    argv = _runner().ANNOTATORS["model.gemini-3.8-flash-high"]["argv"]
    assert argv[argv.index("--add-dir") + 1] == "{workdir}"


def test_a_missing_executable_is_an_attempt_that_failed_not_a_crash(tmp_path: Path) -> None:
    runner = _runner()
    spec = {**runner.ANNOTATORS["model.gemini-3.8-flash-high"]}
    run = runner.run_once(
        spec, str(tmp_path / "no-such-cli.exe"), b"prompt", tmp_path, "batch-01.attempt-1", 5
    )
    assert run["exit_code"] == "executable_missing"
    assert run["files_created_in_isolated_dir"] == []
    assert (tmp_path / "batch-01.attempt-1.stdout.txt").read_bytes() == b""
    # The isolated directory is substituted and recorded without the author's absolute path.
    assert not any(str(tmp_path) in arg for arg in run["argv"])


def test_on_windows_the_executable_is_a_pathext_file_never_the_extensionless_shim(
    tmp_path: Path, monkeypatch
) -> None:
    # npm installs `cline` (a POSIX shell script) beside `cline.cmd`; Python 3.12.0's shutil.which returned
    # the script, which Windows cannot start (WinError 193).
    runner = _runner()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "cline").write_text("#!/bin/sh")
    (first / "cline.CMD").write_text("@echo off")
    (second / "cline.EXE").write_bytes(b"MZ")
    monkeypatch.setattr(runner.sys, "platform", "win32")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("PATH", runner.os.pathsep.join([str(first), str(second)]))
    # PATH order wins over PATHEXT order, as in cmd.exe: the first directory's .cmd, not the later .exe.
    assert runner.resolve_executable("cline") == str(first / "cline.CMD")
    (first / "cline.CMD").unlink()
    assert runner.resolve_executable("cline") == str(second / "cline.EXE")
    (second / "cline.EXE").unlink()
    assert runner.resolve_executable("cline") is None


def test_cline_never_updates_itself_during_a_run() -> None:
    # A self-update mid-run removed cline's launcher on 2026-09-30; the harness stays at one version.
    spec = _runner().ANNOTATORS["model.deepseek-v4.1-flash"]
    assert spec["env"] == {"CLINE_NO_AUTO_UPDATE": "1"}


def test_an_attempt_records_its_environment_overrides(tmp_path: Path) -> None:
    runner = _runner()
    spec = {**runner.ANNOTATORS["model.deepseek-v4.1-flash"]}
    run = runner.run_once(
        spec, str(tmp_path / "no-such-cli.exe"), b"prompt", tmp_path, "batch-01.attempt-1", 5
    )
    assert run["env_overrides"] == {"CLINE_NO_AUTO_UPDATE": "1"}


def test_the_version_check_carries_the_same_overrides_as_a_run(monkeypatch) -> None:
    # A bare `cline --version` may update cline; it once moved a pass from 3.0.65 to 3.0.66 mid-task.
    runner = _runner()
    seen = {}

    class Done:
        stdout = b"3.0.66"

    def fake_run(argv, **kwargs):
        seen.update(kwargs["env"])
        return Done()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    spec = runner.ANNOTATORS["model.deepseek-v4.1-flash"]
    assert runner.cli_version("cline", spec["version_argv"], spec["env"]) == "3.0.66"
    assert seen["CLINE_NO_AUTO_UPDATE"] == "1"
