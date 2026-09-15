"""Edits are allowed while annotating, every one is chained, tampering is caught, freezes are final."""

from __future__ import annotations

import pytest

from opengrad.annotation.export import freeze_gold
from opengrad.annotation.provenance import ANNOTATION_ENTRY_FIELDS, state_sha256, verify_chain
from opengrad.annotation.service import FrozenError, Workspace, WorkspaceError
from opengrad.annotation.store import annotation_image
from tests.annotation.helpers import adjudicated, label_all, value


def test_relabel_records_time_previous_hash_actor_and_reason(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    first = workspace.annotate("pass-a", item, value("DIRECT"))["annotation"]
    second = workspace.annotate(
        "pass-a", item, value("CLARIFY"), replace=True, reason="it asks for a required id"
    )["annotation"]
    assert second["previous_state_sha256"] == first["state_sha256"]
    assert second["revision"] == 2
    assert second["created_at"] == first["created_at"]
    changes = workspace.item_history("pass-a", item)
    assert [c["action"] for c in changes] == ["label", "relabel"]
    relabel = changes[1]
    assert (relabel["from"], relabel["to"]) == ("DIRECT", "CLARIFY")
    assert relabel["reason"] == "it asks for a required id"
    assert relabel["actor_id"] == "alice"
    assert relabel["before_sha256"] == first["state_sha256"]
    assert relabel["after_sha256"] == second["state_sha256"]
    assert relabel["recorded_at"] >= changes[0]["recorded_at"]


def test_chain_links_every_entry_and_matches_current_state(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    workspace.set_flag("pass-a", item, True)
    workspace.annotate("pass-a", item, value("CALL"), replace=True)
    entries = workspace.store.history(workspace.task_id, "pass-a")
    assert verify_chain(entries, ANNOTATION_ENTRY_FIELDS, "pass-a") == []
    assert entries[0]["prev_entry_sha256"] is None
    assert [e["prev_entry_sha256"] for e in entries[1:]] == [e["entry_sha256"] for e in entries[:-1]]
    current = workspace.store.annotation(workspace.task_id, "pass-a", item)
    assert entries[-1]["after_sha256"] == current["state_sha256"] == state_sha256(annotation_image(current))
    assert workspace.audit() == []


def test_undo_walks_back_and_is_itself_recorded(workspace: Workspace) -> None:
    first, second = workspace.item_ids[:2]
    workspace.annotate("pass-a", first, value("DIRECT"))
    workspace.annotate("pass-a", first, value("CALL"), replace=True)
    workspace.annotate("pass-a", second, value("CLARIFY"))
    assert workspace.undo("pass-a")["item_id"] == second
    assert workspace.store.annotation(workspace.task_id, "pass-a", second) is None
    assert workspace.undo("pass-a")["item_id"] == first
    assert workspace.item_view("pass-a", first)["annotation"]["value"]["label"] == "DIRECT"
    assert workspace.undo("pass-a")["item_id"] == first
    assert workspace.store.annotation(workspace.task_id, "pass-a", first) is None
    with pytest.raises(WorkspaceError, match="nothing to undo"):
        workspace.undo("pass-a")
    entries = workspace.store.history(workspace.task_id, "pass-a")
    assert [e["action"] for e in entries] == ["label", "relabel", "label", "undo", "undo", "undo"]
    reverted = [e["reverts_entry_sha256"] for e in entries if e["action"] == "undo"]
    assert reverted == [entries[2]["entry_sha256"], entries[1]["entry_sha256"], entries[0]["entry_sha256"]]
    assert workspace.audit() == []


def test_audit_catches_a_label_edited_directly_in_sqlite(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    with workspace.store.transaction() as conn:
        conn.execute(
            "UPDATE annotations SET value_json = ? WHERE item_id = ?",
            ('{"ambiguity_status":"NONE","label":"CALL","rationale_text":"because"}', item),
        )
    assert any("does not hash to state_sha256" in e or "last state" in e for e in workspace.audit())


def test_audit_catches_a_rewritten_history_entry(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    workspace.annotate("pass-a", item, value("CALL"), replace=True, reason="original reason")
    with workspace.store.transaction() as conn:
        conn.execute("UPDATE history SET reason = 'a nicer reason' WHERE action = 'relabel'")
    assert any("entry_sha256 does not match" in e for e in workspace.audit())


def test_audit_catches_a_deleted_history_entry(workspace: Workspace) -> None:
    item = workspace.item_ids[0]
    workspace.annotate("pass-a", item, value("DIRECT"))
    workspace.annotate("pass-a", item, value("CALL"), replace=True)
    workspace.annotate("pass-a", item, value("CLARIFY"), replace=True)
    with workspace.store.transaction() as conn:
        conn.execute("DELETE FROM history WHERE action = 'relabel' AND id = (SELECT MIN(id) FROM history WHERE action = 'relabel')")
    assert any("prev_entry_sha256" in e for e in workspace.audit())


def test_freeze_locks_every_write_path(workspace: Workspace) -> None:
    label_all(workspace, "pass-a")
    label_all(workspace, "pass-b")
    freeze_gold(workspace, ["pass-a", "pass-b"])
    item = workspace.item_ids[0]
    for attempt in (
        lambda: workspace.annotate("pass-a", item, value("CALL"), replace=True),
        lambda: workspace.skip("pass-b", item),
        lambda: workspace.set_flag("pass-a", item, True),
        lambda: workspace.undo("pass-a"),
        lambda: workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("CALL"), rationale="r", adjudicator_id="carol"),
    ):
        with pytest.raises(FrozenError, match="frozen"):
            attempt()
    assert workspace.session_state("pass-a")["frozen"] is True
    # Reading stays possible, and a new session on the same task is not locked.
    assert workspace.item_view("pass-a", item)["annotation"]["value"]["label"] == "DIRECT"
    workspace.open_session("pass-c", "dana")
    workspace.annotate("pass-c", item, value("CALL"))


def test_adjudication_never_touches_the_passes(workspace: Workspace) -> None:
    label_all(workspace, "pass-a", {1: "CLARIFY"})
    label_all(workspace, "pass-b", {1: "UNSUPPORTED"})
    item = workspace.item_ids[1]
    before = {s: workspace.store.annotation(workspace.task_id, s, item) for s in ("pass-a", "pass-b")}
    workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("CLARIFY", step="2"), rationale="asks for the id", adjudicator_id="carol")
    workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("UNSUPPORTED", step="3"), rationale="changed my mind", adjudicator_id="carol")
    after = {s: workspace.store.annotation(workspace.task_id, s, item) for s in ("pass-a", "pass-b")}
    assert before == after
    view = workspace.adjudication_view(["pass-a", "pass-b"], item)
    assert [p["annotation"]["value"]["label"] for p in view["passes"]] == ["CLARIFY", "UNSUPPORTED"]
    assert view["disagreement"] is True
    assert view["adjudication"]["value"]["label"] == "UNSUPPORTED"
    assert view["adjudication"]["revision"] == 2
    entries = workspace.store.adjudication_history(workspace.task_id, "pass-a|pass-b")
    assert [e["action"] for e in entries] == ["adjudicate", "readjudicate"]
    assert workspace.audit() == []


def test_adjudication_flags(workspace: Workspace) -> None:
    label_all(workspace, "pass-a")
    label_all(workspace, "pass-b")
    item = workspace.item_ids[0]
    row = workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("DIRECT"), rationale="checking", adjudicator_id="carol")
    assert row["adjudication"]["flag"] == "ADJUDICATION_WITHOUT_DISAGREEMENT"
    with pytest.raises(Exception, match="rationale is required"):
        workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("DIRECT"), rationale="  ", adjudicator_id="carol")
    with pytest.raises(Exception, match="unexpected value keys"):
        workspace.adjudicate(["pass-a", "pass-b"], item, {**adjudicated("DIRECT"), "rationale_text": "x"}, rationale="r", adjudicator_id="carol")
