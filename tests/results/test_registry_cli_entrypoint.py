"""The registry CLI must actually run when invoked as a module.

``python -m opengrad.results.cli`` used to do nothing and exit 0, because the module had no
``__main__`` entry point. A status command that silently does nothing is worse than none.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_registry_cli_runs_as_a_module_and_lists_its_commands() -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-I", "-m", "opengrad.results.cli", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "rebuild-registry" in done.stdout
    assert "validate-registry" in done.stdout
