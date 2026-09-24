"""`scripts/publish_card_corrections.py`: card-only re-publication, exercised against a fake Hub.

Nothing here reaches the network. The fake serves each repository at the revision the publication
records name as current, so the tests check the rules that make an upload safe: a dry run writes
nothing, a Hub head the records do not know blocks, and a card that does not come back byte for byte
fails before any record is written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from opengrad.hashing import sha256_bytes
from opengrad.registry.validators import validate_publication_records

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "publish_card_corrections", ROOT / "scripts/publish_card_corrections.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pcc = _script()


class FakeHub:
    """A linear Hub history per repository: the recorded revision, then `extra` unrecorded commits."""

    def __init__(
        self, tmp: Path, *, stale: bool = True, corrupt: bool = False, extra: int = 0
    ) -> None:
        self.tmp = tmp
        self.history: dict[str, list[str]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        recorded = pcc.current_revisions(ROOT)
        for directory, (repository, _repo_type) in pcc.CARDS.items():
            card = pcc.committed_card(directory, ROOT)
            chain = [recorded[repository]] + [
                sha256_bytes(f"{repository}{i}".encode())[:40] for i in range(extra)
            ]
            self.history[repository] = chain
            for revision in chain:
                self.files[(repository, revision)] = b"old card\n" if stale else card
        self.uploads: list[dict] = []
        self.corrupt = corrupt

    def repo_info(self, repository: str, *, repo_type: str) -> SimpleNamespace:
        assert repo_type == dict(pcc.CARDS.values())[repository]
        return SimpleNamespace(sha=self.history[repository][-1])

    def list_repo_commits(self, repository: str, *, repo_type: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(commit_id=c, created_at=datetime(2026, 9, 13, tzinfo=UTC), title="edit")
            for c in reversed(self.history[repository])
        ]

    def hf_hub_download(self, *, repo_id: str, filename: str, revision: str, repo_type: str) -> str:
        assert filename == "README.md"
        path = self.tmp / f"{len(list(self.tmp.iterdir()))}.md"
        path.write_bytes(self.files[(repo_id, revision)])
        return str(path)

    def upload_file(self, **kwargs) -> SimpleNamespace:
        self.uploads.append(kwargs)
        repository = kwargs["repo_id"]
        assert kwargs["parent_commit"] == self.history[repository][-1]
        new = f"{len(self.uploads):040x}"
        body = kwargs["path_or_fileobj"]
        self.files[(repository, new)] = body + b"x" if self.corrupt else body
        self.history[repository].append(new)
        return SimpleNamespace(oid=new)


def test_every_card_directory_exists_and_every_repository_has_a_current_revision() -> None:
    current = pcc.current_revisions(ROOT)
    assert len(pcc.CARDS) == 6
    for directory, (repository, _repo_type) in pcc.CARDS.items():
        assert (ROOT / pcc.card_path(directory)).is_file(), directory
        assert repository in current, repository


def test_the_gguf_card_is_checked_against_its_generator() -> None:
    assert pcc.gguf_card_matches_generator(ROOT)


def test_a_dry_run_diffs_every_card_and_writes_nothing(tmp_path: Path, capsys) -> None:
    hub = FakeHub(tmp_path)
    status, items = pcc.plan(hub, ROOT)
    assert status == pcc.PASS
    assert len(items) == 6 and all(item["changed"] for item in items)
    assert hub.uploads == []
    out = capsys.readouterr().out
    assert "6 of 6 cards examined, 6 changed, 0 unrecorded Hub commits found" in out
    assert "+## License and attribution" in out


def test_a_recorded_revision_missing_from_the_hub_blocks(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    repository, _repo_type = pcc.CARDS["qwen35-2b-m0-sft-canonicalv2-final"]
    hub.history[repository] = ["f" * 40]  # the recorded revision is not in the Hub's history
    status, _items = pcc.plan(hub, ROOT)
    assert status == pcc.BLOCKED


def test_an_upload_is_recorded_only_after_the_card_comes_back(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path)
    _status, items = pcc.plan(hub, ROOT)
    status, entries = pcc.upload(hub, items, ROOT)
    assert status == pcc.PASS and len(entries) == 6
    entry = entries[0]
    assert entry["files_replaced"] == ["README.md"]
    assert entry["superseded_hub_revision"] == items[0]["recorded_revision"]
    assert entry["card_sha256"] == sha256_bytes(items[0]["card"])
    assert entry["recorded_retroactively"] is False


def test_a_card_that_does_not_come_back_fails(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, corrupt=True)
    _status, items = pcc.plan(hub, ROOT)
    status, entries = pcc.upload(hub, items, ROOT)
    assert status == pcc.FAIL and entries == []


def test_unchanged_cards_are_not_uploaded(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path, stale=False)
    _status, items = pcc.plan(hub, ROOT)
    status, entries = pcc.upload(hub, items, ROOT)
    assert (status, entries, hub.uploads) == (pcc.PASS, [], [])


def test_the_record_supersedes_each_revision_and_validates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "reports/releases", root / "reports/releases")
    hub = FakeHub(tmp_path / "hub")
    (tmp_path / "hub").mkdir()
    _status, items = pcc.plan(hub, ROOT)
    _status, entries = pcc.upload(hub, items, ROOT)
    path = pcc.write_record(entries, root)
    assert json.loads(path.read_text(encoding="utf-8"))["cards"] == entries
    assert validate_publication_records(root) == []
    current = pcc.current_revisions(root)
    for entry in entries:
        assert current[entry["repository"]] == entry["hub_revision"]


def test_unrecorded_hub_commits_are_recorded_retroactively_and_the_chain_validates(
    tmp_path: Path, capsys
) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "reports/releases", root / "reports/releases")
    (tmp_path / "hub").mkdir()
    hub = FakeHub(tmp_path / "hub", extra=2)
    status, items = pcc.plan(hub, ROOT)
    assert status == pcc.PASS
    assert "12 unrecorded Hub commits found" in capsys.readouterr().out
    assert all(len(item["unrecorded"]) == 2 for item in items)
    # The diff and the upload are against the live head, not the stale recorded revision.
    assert all(item["head"] == hub.history[item["repository"]][2] for item in items)
    _status, entries = pcc.upload(hub, items, ROOT)
    chain = entries[0]["revision_chain_since_previous_record"]
    assert [link["recorded_retroactively"] for link in chain] == [True, True]
    assert chain[0]["superseded_hub_revision"] == items[0]["recorded_revision"]
    assert entries[0]["superseded_hub_revision"] == chain[-1]["hub_revision"]
    pcc.write_record(entries, root)
    assert validate_publication_records(root) == []
    assert pcc.current_revisions(root)[entries[0]["repository"]] == entries[0]["hub_revision"]
