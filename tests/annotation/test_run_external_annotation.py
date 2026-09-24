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
