"""Metric exclusions: items annotated normally but kept out of named metrics, recorded in every export."""

from __future__ import annotations

import copy
import hashlib
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.annotation.cli import main
from opengrad.annotation.config import TaskConfig, TaskConfigError, parse_task_config
from opengrad.annotation.export import export_snapshot, freeze_gold, verify_package
from opengrad.annotation.server import Api, ServerContext
from opengrad.annotation.service import Workspace, WorkspaceError
from tests.annotation.helpers import BASE_CONFIG, adjudicated, label_all, rows, write_source
from tests.annotation.test_annotation_export import STAMP, _reseal, read_jsonl, two_complete_passes

EXPOSED = "EXPOSED_WORKED_EXAMPLE"
EXCLUSION: dict[str, Any] = {
    "status": EXPOSED,
    "reason": "shown with a label in the instructions",
    "excluded_from": ["classifier_validation_metrics", "annotator_agreement_statistics"],
    "item_ids": ["item-003", "item-001"],
}
EXCLUDED = {"item-001", "item-003"}


def with_exclusion(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE_CONFIG)
    data["metric_exclusions"] = [{**copy.deepcopy(EXCLUSION), **changes}]
    return data


# ── configuration ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"status": EXPOSED}, "must be a list"),
        ([{**EXCLUSION, "status": "exposed"}], "UPPER_SNAKE_CASE"),
        ([{**EXCLUSION, "item_ids": []}], "at least one item"),
        ([{**EXCLUSION, "item_ids": ["item-001", "item-001"]}], "repeat an item"),
        ([{**EXCLUSION, "excluded_from": []}], "excluded_from"),
        ([{key: v for key, v in EXCLUSION.items() if key != "reason"}], "reason"),
        ([EXCLUSION, EXCLUSION], "declared twice"),
    ],
)
def test_invalid_exclusions_are_rejected(tmp_path: Path, value: Any, message: str) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["metric_exclusions"] = value
    with pytest.raises(TaskConfigError, match=message):
        parse_task_config(data, root=tmp_path, config_path=tmp_path / "task.yaml")


def test_exclusions_change_neither_meaning_nor_display(tmp_path: Path) -> None:
    base = parse_task_config(copy.deepcopy(BASE_CONFIG), root=tmp_path, config_path=tmp_path / "t.yaml")
    excluded = parse_task_config(with_exclusion(), root=tmp_path, config_path=tmp_path / "t.yaml")
    assert excluded.excluded_items() == {"item-001": [EXPOSED], "item-003": [EXPOSED]}
    # Recording an exclusion mid-annotation must not lock the store out of the labels already given.
    assert base.definition_sha256() == excluded.definition_sha256()
    public = json.dumps(excluded.public())
    assert EXPOSED not in public and "item-001" not in public and "metric_exclusions" not in public


def test_exclusions_must_name_source_items_and_the_document_must_record_them(
    tmp_path: Path, make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    with pytest.raises(WorkspaceError, match="not source items"):
        open_workspace(make_config(with_exclusion(item_ids=["item-001", "item-999"])))
    with pytest.raises(WorkspaceError, match="not found"):
        open_workspace(make_config(with_exclusion(document="addendum.md")))
    (tmp_path / "addendum.md").write_text("Records item-001 only.\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="does not record 1 of its ids"):
        open_workspace(make_config(with_exclusion(document="addendum.md")))
    assert not (tmp_path / "state").exists()  # refused before any state was created
    (tmp_path / "addendum.md").write_text("Records item-001 and item-003.\n", encoding="utf-8")
    assert open_workspace(make_config(with_exclusion(document="addendum.md"))).item_count == 5


def test_excluded_items_are_annotated_normally_and_never_marked_on_screen(
    make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]
) -> None:
    ws = open_workspace(make_config(with_exclusion()))
    ws.open_session("pass-a", "alice")
    ws.annotate("pass-a", "item-001", {"label": "DIRECT", "ambiguity_status": "NONE", "rationale_text": "r"})
    api = Api(ServerContext(ws, "annotate", session_id="pass-a"))
    screen = json.dumps([api.state(), api.get("/api/items", {"status": "all"}), api.get("/api/items/item-001", {})])
    assert EXPOSED not in screen and "metric_exclusion" not in screen and "excluded_from" not in screen
    assert ws.progress("pass-a")["completed"] == 1


# ── exports ─────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def excluded_ws(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace]) -> Workspace:
    ws = open_workspace(make_config(with_exclusion()))
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    return ws


def test_every_record_and_the_manifest_carry_the_exclusions(excluded_ws: Workspace, tmp_path: Path) -> None:
    # Worked example (test_annotation_export.two_complete_passes): the one disagreement is item-001,
    # which is excluded; the metric-eligible gold items are 000 DIRECT, 002 DIRECT, 004 CALL.
    two_complete_passes(excluded_ws)
    excluded_ws.adjudicate(["pass-a", "pass-b"], "item-001", adjudicated("CLARIFY"), rationale="r", adjudicator_id="carol")
    manifest = freeze_gold(excluded_ws, ["pass-a", "pass-b"], tmp_path / "gold", exported_at=STAMP)
    assert manifest["metric_exclusions"] == {
        "groups": [
            {
                "status": EXPOSED,
                "reason": "shown with a label in the instructions",
                "document": None,
                "excluded_from": ["classifier_validation_metrics", "annotator_agreement_statistics"],
                "item_ids": ["item-001", "item-003"],
                "items": 2,
            }
        ],
        "population_items": 5,
        "excluded_items": 2,
        "metric_eligible_items": 3,
    }
    assert [s["labeled_metric_eligible"] for s in manifest["sessions"]] == [3, 3]
    assert (manifest["disagreements"]["count"], manifest["disagreements"]["metric_eligible"]) == (1, 0)
    assert manifest["gold"]["metric_eligible_items"] == 3
    assert manifest["gold"]["metric_eligible_label_counts"] == {"CALL": 1, "DIRECT": 2}
    assert manifest["gold"]["label_counts"] == {"CALL": 1, "CLARIFY": 1, "DIRECT": 2, "UNKNOWN": 1}
    checked = 0
    for name in manifest["outputs"]:
        if name.startswith("audit/"):
            continue  # change logs hold chain entries, not per-item records
        for record in read_jsonl(tmp_path / "gold" / name):
            assert record["metric_exclusions"] == ([EXPOSED] if record["pdet_id"] in EXCLUDED else []), name
            checked += 1
    assert checked == 5 + 5 + 1 + 1 + 5  # two passes, disagreements, adjudication, gold
    assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


def test_a_snapshot_records_the_exclusions_too(excluded_ws: Workspace, tmp_path: Path) -> None:
    label_all(excluded_ws, "pass-a")
    manifest = export_snapshot(excluded_ws, ["pass-a"], tmp_path / "wip", exported_at=STAMP)
    assert manifest["metric_exclusions"]["metric_eligible_items"] == 3
    assert manifest["sessions"][0]["labeled_metric_eligible"] == 3
    assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


@pytest.fixture
def excluded_package(excluded_ws: Workspace, tmp_path: Path) -> Path:
    two_complete_passes(excluded_ws)
    excluded_ws.adjudicate(["pass-a", "pass-b"], "item-001", adjudicated("CLARIFY"), rationale="r", adjudicator_id="carol")
    manifest = freeze_gold(excluded_ws, ["pass-a", "pass-b"], tmp_path / "pkg", exported_at=STAMP)
    return Path(manifest["manifest_path"])


def _errors(path: Path) -> list[str]:
    _, summary = verify_package(path)
    assert summary["status"] == "FAIL"
    return summary["errors"]


def _reseal_manifest(package: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    manifest = json.loads(package.read_text(encoding="utf-8"))
    edit(manifest)
    stamped = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    package.write_bytes(stamped)
    package.with_name(package.name + ".sha256").write_text(hashlib.sha256(stamped).hexdigest() + "\n")


def test_moving_an_excluded_gold_label_into_the_metrics_is_caught(excluded_package: Path) -> None:
    name = "gold/toy-v1.gold.jsonl"
    rows_ = read_jsonl(excluded_package.parent / name)
    rows_[1]["metric_exclusions"] = []  # item-001

    def recount(manifest: dict[str, Any]) -> None:
        manifest["gold"]["metric_eligible_items"] = 4
        manifest["gold"]["metric_eligible_label_counts"] = {"CALL": 1, "CLARIFY": 1, "DIRECT": 2}

    _reseal(excluded_package, name, rows_, recount)
    errors = _errors(excluded_package)
    assert not any(e.startswith("FAIL_HASH") for e in errors), errors  # the forger's hashes hold...
    # ...but the record no longer carries what the manifest's exclusion section says, and the
    # metric-eligible counts no longer follow from that section.
    assert any(e.startswith("FAIL_EXCLUSION") and "carry exclusions that differ" in e for e in errors), errors
    assert any("metric-eligible gold counts" in e for e in errors), errors


def test_dropping_an_item_from_the_manifest_exclusions_is_caught(excluded_package: Path) -> None:
    def drop(manifest: dict[str, Any]) -> None:
        group = manifest["metric_exclusions"]["groups"][0]
        group["item_ids"], group["items"] = ["item-003"], 1
        manifest["metric_exclusions"].update(excluded_items=1, metric_eligible_items=4)

    _reseal_manifest(excluded_package, drop)
    assert any("carry exclusions that differ" in e for e in _errors(excluded_package))


def test_exclusion_counts_that_do_not_add_up_are_caught(excluded_package: Path) -> None:
    _reseal_manifest(excluded_package, lambda m: m["metric_exclusions"].update(metric_eligible_items=5))
    assert any("do not add up" in e for e in _errors(excluded_package))


# ── start --dry-run ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dry_run(tmp_path: Path) -> list[str]:
    write_source(tmp_path, rows())
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(with_exclusion(), sort_keys=False), encoding="utf-8")
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html>", encoding="utf-8")
    return ["start", str(tmp_path / "task.yaml"), "--root", str(tmp_path), "--state-db", str(tmp_path / "state.sqlite3"),
            "--ui-dir", str(ui), "--dry-run"]


def test_start_dry_run_checks_everything_and_creates_nothing(dry_run: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*dry_run, "--annotator", "alice", "--session", "pass-a", "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "(does not exist; start would create it)" in out and "DRY RUN   PASS" in out
    assert not (tmp_path / "state.sqlite3").exists()
    assert main([*dry_run, "--annotator", "not an id", "--session", "pass-a", "--port", "0"]) == 1
    assert "annotator id 'not an id'" in capsys.readouterr().out
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = str(busy.getsockname()[1])
        assert main([*dry_run, "--annotator", "alice", "--session", "pass-a", "--port", port]) == 1
    assert "is in use" in capsys.readouterr().out
    assert not (tmp_path / "state.sqlite3").exists()


def test_start_dry_run_reads_existing_state_without_changing_it(dry_run: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = parse_task_config(with_exclusion(), root=tmp_path, config_path=tmp_path / "task.yaml")
    ws = Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    ws.open_session("pass-a", "alice")
    label_all(ws, "pass-a")
    ws.close()
    before = (tmp_path / "state.sqlite3").read_bytes()
    assert main([*dry_run, "--annotator", "alice", "--session", "pass-a", "--port", "0"]) == 0
    assert "pass-a exists for alice (5 labeled); start resumes it" in capsys.readouterr().out
    assert main([*dry_run, "--annotator", "bob", "--session", "pass-a", "--port", "0"]) == 1
    assert "belongs to annotator 'alice'" in capsys.readouterr().out
    assert (tmp_path / "state.sqlite3").read_bytes() == before
