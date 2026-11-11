"""Suite-wide fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _run_records_outside_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """`opengrad train` and `opengrad evaluate` record every call (opengrad.runrecords). Without this, a test that
    calls them from the repository root would leave a run record in the repository's `.run-records/`."""
    target = tmp_path_factory.mktemp("run-records")
    monkeypatch.setenv("OPENGRAD_RUN_RECORDS_DIR", str(target))
    return target
