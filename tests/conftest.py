"""Suite-wide fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

_RECORDS = Path(__file__).resolve().parents[1] / ".run-records"
#: Run records already in the repository when the suite starts: real runs of converted scripts (stage 3 records
#: every entry point), not the suite's doing.
_RECORDS_AT_START = frozenset(p.name for p in _RECORDS.iterdir()) if _RECORDS.is_dir() else frozenset()


@pytest.fixture
def run_records_at_start() -> frozenset[str]:
    """The names in the repository's ``.run-records/`` when the suite started."""
    return _RECORDS_AT_START


@pytest.fixture(autouse=True)
def _run_records_outside_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """`opengrad train` and `opengrad evaluate` record every call (opengrad.runrecords). Without this, a test that
    calls them from the repository root would leave a run record in the repository's `.run-records/`."""
    target = tmp_path_factory.mktemp("run-records")
    monkeypatch.setenv("OPENGRAD_RUN_RECORDS_DIR", str(target))
    return target
