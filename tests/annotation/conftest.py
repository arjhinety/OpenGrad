"""Fixtures for the annotation tests. Everything is built in ``tmp_path``; no real artifact is touched."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from opengrad.annotation.config import TaskConfig, parse_task_config
from opengrad.annotation.service import Workspace
from tests.annotation.helpers import BASE_CONFIG, rows, write_source


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., TaskConfig]:
    def build(
        data: dict[str, Any] | None = None, source_rows: list[dict[str, Any]] | None = None
    ) -> TaskConfig:
        config = copy.deepcopy(BASE_CONFIG if data is None else data)
        source = tmp_path / config["source"]["path"]
        if source_rows is not None or not source.exists():
            write_source(tmp_path, source_rows if source_rows is not None else rows(), config["source"]["path"])
        return parse_task_config(config, root=tmp_path, config_path=tmp_path / "task.yaml")

    return build


@pytest.fixture
def open_workspace(tmp_path: Path) -> Iterator[Callable[..., Workspace]]:
    opened: list[Workspace] = []

    def opener(config: TaskConfig, state: str = "state/annotation.sqlite3") -> Workspace:
        workspace = Workspace.open(config, state_db=tmp_path / state)
        opened.append(workspace)
        return workspace

    yield opener
    for workspace in opened:
        workspace.close()  # closing an already-closed sqlite connection is a no-op


@pytest.fixture
def workspace(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Workspace:
    ws = open_workspace(make_config())
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    return ws
