"""The run-log dataset is never a publish target (``opengrad.publication.private_repos``)."""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from opengrad.publication.private_repos import (
    PRIVATE_ONLY_REPOS,
    PrivateRepoError,
    is_private_only,
    refuse_private_only,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_LOGS = "arjhinety/opengrad-run-logs"
#: A call that writes to the Hub, or creates or changes a repository there.
HUB_WRITE = re.compile(
    r"\b(?:upload_file|upload_folder|upload_large_folder|create_repo|create_commit|update_repo_settings"
    r"|update_repo_visibility|push_to_hub|delete_file|delete_folder|delete_repo|move_repo|create_branch"
    r"|create_tag|super_squash_history|create_pull_request|metadata_update|add_collection_item"
    r"|CommitScheduler)\s*\("
    r"|[\"'](?:hf|huggingface-cli)[\"']\s*,\s*[\"']upload[\"']|\b(?:hf|huggingface-cli) upload\b"
)
#: The one writer allowed to target the run-log dataset.
INTENDED_WRITER = "src/opengrad/runrecords/cli.py"


def _load(relative: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_run_log_dataset_is_private_only() -> None:
    assert RUN_LOGS in PRIVATE_ONLY_REPOS


@pytest.mark.parametrize(
    "repo", [RUN_LOGS, "Arjhinety/OpenGrad-Run-Logs", f" {RUN_LOGS}/ ", "arrochi112/opengrad-run-logs"]
)
def test_any_spelling_of_the_run_log_dataset_is_refused(repo: str) -> None:
    assert is_private_only(repo)
    with pytest.raises(PrivateRepoError):
        refuse_private_only(repo)


@pytest.mark.parametrize("repo", ["arjhinety/opengrad-run-logs-public", "experimentalmachines/QwenGrad-x", ""])
def test_other_repositories_pass(repo: str) -> None:
    assert not is_private_only(repo)
    refuse_private_only(repo)


def _tracked_python() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\n") if p.startswith(("src/", "scripts/"))]


def test_every_script_that_writes_to_the_hub_refuses_the_run_log_dataset() -> None:
    writers = [
        path
        for path in _tracked_python()
        if HUB_WRITE.search((ROOT / path).read_text(encoding="utf-8"))
    ]
    # A check that examined nothing must not pass: these three write to the Hub today.
    assert {
        "scripts/publish_card_corrections.py",
        "scripts/publish_executorch_artifact.py",
        "scripts/modal/upload_artifact.py",
    } <= set(writers), writers
    missing = [
        path
        for path in writers
        if path != INTENDED_WRITER
        and not re.search(r"\b(?:refuse_private_only|is_private_only)\(", (ROOT / path).read_text(encoding="utf-8"))
    ]
    assert missing == [], f"these write to the Hub without the private-repository check: {missing}"


def test_card_corrections_refuse_a_card_targeting_the_run_log_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load("scripts/publish_card_corrections.py", "publish_card_corrections_private_repo_test")

    class NoHub:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"the Hub was called ({name}) for a private-only repository")

    item = {"repository": RUN_LOGS, "repo_type": "dataset", "changed": True, "unrecorded": [], "card": b"x"}
    status, entries = module.upload(NoHub(), [item])
    assert (status, entries) == (module.BLOCKED, [])
    assert "must never be a publish target" in capsys.readouterr().out


def test_the_executorch_publisher_refuses_before_reading_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("scripts/publish_executorch_artifact.py", "publish_executorch_private_repo_test")
    monkeypatch.setattr(sys, "argv", ["publish_executorch_artifact.py", "--target", "cpu", "--repo", RUN_LOGS])
    with pytest.raises(PrivateRepoError):
        module.main()


def test_the_modal_container_copy_of_the_list_matches() -> None:
    text = (ROOT / "scripts/modal/upload_artifact.py").read_text(encoding="utf-8")
    match = re.search(r"^PRIVATE_ONLY_REPOS = frozenset\((\{[^}]*\})\)$", text, re.MULTILINE)
    assert match, "the container copy of PRIVATE_ONLY_REPOS is missing"
    assert ast.literal_eval(match.group(1)) == set(PRIVATE_ONLY_REPOS)


@pytest.mark.parametrize(
    "line",
    ['api.delete_file(path, repo_id=r)', 'add_collection_item(slug, item_id=r)', 'subprocess.run(["hf", "upload", r])',
     "huggingface-cli upload org/name ./dir", "CommitScheduler(repo_id=r, folder_path=p)"],
)
def test_the_hub_write_pattern_covers_other_write_calls(line: str) -> None:
    assert HUB_WRITE.search(line), line

