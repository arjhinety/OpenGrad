"""Export packages, gold freezes, manifests, verification and tamper detection."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from opengrad.annotation.config import TaskConfig
from opengrad.annotation.export import (
    ExportError,
    IncompleteGoldError,
    export_snapshot,
    freeze_gold,
    verify_package,
)
from opengrad.annotation.service import Workspace
from tests.annotation.helpers import BASE_CONFIG, adjudicated, label_all, value

STAMP = "2026-09-15T00:00:00Z"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def two_complete_passes(ws: Workspace) -> None:
    # Worked example over 5 items (index: pass-a / pass-b):
    #   0: DIRECT / DIRECT        agreement
    #   1: CLARIFY / UNSUPPORTED  disagreement on label
    #   2: DIRECT / DIRECT        agreement
    #   3: UNKNOWN / UNKNOWN      agreement (both NON_SUBSTANTIVE)
    #   4: CALL / CALL            agreement
    label_all(ws, "pass-a", {1: "CLARIFY", 3: "UNKNOWN", 4: "CALL"})
    label_all(ws, "pass-b", {1: "UNSUPPORTED", 3: "UNKNOWN", 4: "CALL"})


def test_snapshot_manifest_counts_match_the_worked_example(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    manifest = export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "wip", exported_at=STAMP)
    a, b = manifest["sessions"]
    assert a["label_counts"] == {"CALL": 1, "CLARIFY": 1, "DIRECT": 2, "UNKNOWN": 1}
    assert b["label_counts"] == {"CALL": 1, "DIRECT": 2, "UNKNOWN": 1, "UNSUPPORTED": 1}
    assert (a["labeled"], a["unlabeled"], a["unknown"]) == (5, 0, 1)
    assert manifest["disagreements"] == {
        "file": "disagreements.jsonl", "count": 1, "metric_eligible": 1, "adjudicated": 0, "unresolved": 1,
        "keys": ["label", "ambiguity_status"],
    }
    assert manifest["metric_exclusions"] == {"groups": [], "population_items": 5, "excluded_items": 0, "metric_eligible_items": 5}
    assert manifest["artifact_kind"] == "ANNOTATION_SNAPSHOT"
    assert manifest["completion_state"] == "INCOMPLETE"
    assert manifest["incomplete_reasons"] == ["1 disagreements not adjudicated (first: item-001)"]
    assert manifest["gold"] is None
    assert manifest["design"] == "two_pass" and manifest["independent_annotators"] is True
    assert manifest["annotator_ids"] == ["alice", "bob"]
    assert manifest["source"]["items"] == 5
    assert manifest["model_assistance"] is False
    assert not list((tmp_path / "wip").rglob("*gold*"))
    for key in ("task_id", "task_version", "annotation_schema_version", "app_version", "exported_at", "outputs"):
        assert manifest[key]
    rows = read_jsonl(tmp_path / "wip" / a["file"])
    assert rows[0]["pdet_id"] == "item-000" and rows[0]["gold_policy_label"] == "DIRECT"
    assert {"annotator_id", "session_id", "timestamp", "source_population_sha256", "state_sha256"} <= set(rows[0])
    disagreements = read_jsonl(tmp_path / "wip" / "disagreements.jsonl")
    assert [(r["pdet_id"], r["value_a"]["gold_policy_label"], r["value_b"]["gold_policy_label"]) for r in disagreements] == [
        ("item-001", "CLARIFY", "UNSUPPORTED")
    ]
    assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


def test_disagreement_on_a_secondary_key_counts(workspace: Workspace) -> None:
    config = workspace.config
    assert config.disagreement_keys == ("label", "ambiguity_status")
    label_all(workspace, "pass-a")
    label_all(workspace, "pass-b")
    # Same label, different ambiguity status: only possible for UNKNOWN under the constraints.
    workspace.annotate("pass-a", workspace.item_ids[0], value("UNKNOWN", "NON_SUBSTANTIVE"), replace=True)
    workspace.annotate("pass-b", workspace.item_ids[0], value("UNKNOWN", "NON_SUBSTANTIVE"), replace=True)
    assert workspace.adjudication_queue(["pass-a", "pass-b"]) == []


def test_gold_freeze_records_sources_and_full_trail(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    workspace.adjudicate(["pass-a", "pass-b"], workspace.item_ids[1], adjudicated("CLARIFY", step="3"), rationale="asks for the id", adjudicator_id="carol")
    manifest = freeze_gold(workspace, ["pass-a", "pass-b"], tmp_path / "gold", exported_at=STAMP)
    assert manifest["artifact_kind"] == "ANNOTATION_GOLD_FREEZE" and manifest["completion_state"] == "COMPLETE"
    assert manifest["gold"]["label_counts"] == {"CALL": 1, "CLARIFY": 1, "DIRECT": 2, "UNKNOWN": 1}
    assert manifest["gold"]["unknown"] == 1
    assert manifest["gold"]["sources"] == {"adjudication": 1, "agreement": 4}
    assert manifest["adjudicator_ids"] == ["carol"]
    gold = read_jsonl(tmp_path / "gold" / manifest["gold"]["file"])
    assert [row["pdet_id"] for row in gold] == sorted(workspace.item_ids)
    decided = gold[1]
    assert decided["gold_policy_label"] == "CLARIFY" and decided["gold_source"] == "adjudication"
    assert [p["value"]["gold_policy_label"] for p in decided["passes"]] == ["CLARIFY", "UNSUPPORTED"]
    assert decided["adjudication"]["value"]["decision_step"] == "3"
    assert decided["adjudication"]["rationale"] == "asks for the id"
    adjudication = read_jsonl(tmp_path / "gold" / "adjudication.jsonl")
    assert adjudication[0]["value_a"]["gold_policy_label"] == "CLARIFY"
    assert adjudication[0]["value_b"]["gold_policy_label"] == "UNSUPPORTED"
    assert adjudication[0]["adjudicated"]["gold_policy_label"] == "CLARIFY"
    assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


def _snapshot_bytes(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_exports_are_deterministic(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    first = export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "pkg", exported_at=STAMP)
    before = _snapshot_bytes(tmp_path / "pkg")
    second = export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "pkg", exported_at=STAMP)
    assert _snapshot_bytes(tmp_path / "pkg") == before  # same state, same place: identical bytes
    # Elsewhere, every data file is identical; only the manifest's own location differs.
    other = export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "elsewhere", exported_at=STAMP)
    assert first["outputs"] == second["outputs"] == other["outputs"]
    elsewhere = _snapshot_bytes(tmp_path / "elsewhere")
    assert {n: b for n, b in elsewhere.items() if "manifest" not in n} == {n: b for n, b in before.items() if "manifest" not in n}


def test_manifest_hashes_are_the_hashes_of_the_written_bytes(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    manifest = export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "pkg", exported_at=STAMP)
    for name, output in manifest["outputs"].items():
        data = (tmp_path / "pkg" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == output["sha256"]
        assert len(data) == output["bytes"]
        assert len(data.splitlines()) == output["records"]
    sidecar = (tmp_path / "pkg" / "manifest.json.sha256").read_text(encoding="utf-8").strip()
    assert sidecar == hashlib.sha256((tmp_path / "pkg" / "manifest.json").read_bytes()).hexdigest()


# ── tamper detection ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def package(workspace: Workspace, tmp_path: Path) -> Path:
    two_complete_passes(workspace)
    workspace.annotate("pass-a", workspace.item_ids[0], value("CALL"), replace=True, reason="second look")
    workspace.annotate("pass-a", workspace.item_ids[0], value("DIRECT"), replace=True, reason="third look")
    workspace.adjudicate(["pass-a", "pass-b"], workspace.item_ids[1], adjudicated("CLARIFY"), rationale="r", adjudicator_id="carol")
    manifest = freeze_gold(workspace, ["pass-a", "pass-b"], tmp_path / "pkg", exported_at=STAMP)
    path = Path(manifest["manifest_path"])
    assert verify_package(path)[1]["status"] == "PASS"
    return path


def _errors(path: Path) -> list[str]:
    _, summary = verify_package(path)
    assert summary["status"] == "FAIL"
    return summary["errors"]


def test_editing_a_label_in_a_pass_file_is_detected(package: Path) -> None:
    target = package.parent / "annotations" / "pass-a.jsonl"
    target.write_text(target.read_text(encoding="utf-8").replace('"DIRECT"', '"CALL"', 1), encoding="utf-8")
    assert any(e.startswith("FAIL_HASH") for e in _errors(package))


def test_editing_the_gold_file_is_detected(package: Path) -> None:
    target = package.parent / "gold" / "toy-v1.gold.jsonl"
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    errors = _errors(package)
    assert any("FAIL_HASH" in e for e in errors) and any("FAIL_COUNT" in e for e in errors)


def test_editing_the_manifest_is_detected(package: Path) -> None:
    data = json.loads(package.read_text(encoding="utf-8"))
    data["gold"]["label_counts"]["DIRECT"] += 1
    package.write_text(json.dumps(data), encoding="utf-8")
    assert any("sidecar" in e for e in _errors(package))


def test_missing_output_is_detected(package: Path) -> None:
    (package.parent / "adjudication.jsonl").unlink()
    assert any("FAIL_MISSING" in e for e in _errors(package))


def _reseal(package: Path, name: str, rows: list[dict], edit_manifest: Callable[[dict], None]) -> None:
    """A forger's best effort: rewrite a file, then every hash and count that covers it."""
    data = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode() for r in rows)
    (package.parent / name).write_bytes(data)
    manifest = json.loads(package.read_text(encoding="utf-8"))
    manifest["outputs"][name].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), records=len(rows))
    edit_manifest(manifest)
    stamped = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    package.write_bytes(stamped)
    package.with_name(package.name + ".sha256").write_text(hashlib.sha256(stamped).hexdigest() + "\n")


def test_forged_pass_label_with_resealed_hashes_is_still_caught(package: Path) -> None:
    rows = read_jsonl(package.parent / "annotations" / "pass-a.jsonl")
    assert rows[2]["gold_policy_label"] == "DIRECT"
    rows[2]["gold_policy_label"] = "UNSUPPORTED"

    def recount(manifest: dict) -> None:
        manifest["sessions"][0]["label_counts"] = {"CALL": 1, "CLARIFY": 1, "DIRECT": 1, "UNKNOWN": 1, "UNSUPPORTED": 1}

    _reseal(package, "annotations/pass-a.jsonl", rows, recount)
    errors = _errors(package)
    assert not any(e.startswith("FAIL_HASH") for e in errors)  # the forger's hashes are consistent...
    assert any("differ in content from their final chained state" in e for e in errors)  # ...the chain is not
    assert any("do not follow from the pass" in e for e in errors)


def test_forged_gold_label_with_resealed_hashes_is_still_caught(package: Path) -> None:
    name = "gold/toy-v1.gold.jsonl"
    rows = read_jsonl(package.parent / name)
    assert rows[4]["gold_policy_label"] == "CALL" and rows[4]["gold_source"] == "agreement"
    rows[4]["gold_policy_label"] = "DIRECT"

    def recount(manifest: dict) -> None:
        manifest["gold"]["label_counts"] = {"CLARIFY": 1, "DIRECT": 3, "UNKNOWN": 1}
        manifest["gold"]["metric_eligible_label_counts"] = {"CLARIFY": 1, "DIRECT": 3, "UNKNOWN": 1}

    _reseal(package, name, rows, recount)
    errors = _errors(package)
    assert not any(e.startswith(("FAIL_HASH", "FAIL_COUNT")) for e in errors)
    assert any("do not follow from the pass" in e for e in errors)


def test_history_file_carries_every_change_with_reasons(package: Path) -> None:
    history = read_jsonl(package.parent / "audit" / "pass-a.history.jsonl")
    relabels = [entry for entry in history if entry["action"] == "relabel"]
    assert [entry["reason"] for entry in relabels] == ["second look", "third look"]
    assert relabels[1]["before_sha256"] == relabels[0]["after_sha256"]
    manifest = json.loads(package.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["relabels"] == 2
    assert manifest["sessions"][0]["history_entries"] == 7


def test_verify_rehashes_the_source_found_from_the_package(package: Path) -> None:
    summary = verify_package(package)[1]
    assert summary["source_rehashed"] is True and summary["source_unreachable"] is None


def test_changed_source_fails_verification(package: Path, tmp_path: Path) -> None:
    source = tmp_path / "data" / "source.jsonl"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any("FAIL_SOURCE" in e for e in _errors(package))


def test_unreachable_source_is_reported_and_can_be_required(package: Path, tmp_path: Path) -> None:
    (tmp_path / "data" / "source.jsonl").rename(tmp_path / "data" / "moved.jsonl")
    summary = verify_package(package)[1]
    assert summary["status"] == "PASS" and summary["source_rehashed"] is False
    assert summary["source_unreachable"].endswith("source.jsonl")
    _, strict = verify_package(package, require_source=True)
    assert strict["status"] == "FAIL" and any("not found" in e for e in strict["errors"])


# ── refusals ────────────────────────────────────────────────────────────────────────────────────


def test_gold_freeze_refuses_incomplete_passes(workspace: Workspace, tmp_path: Path) -> None:
    label_all(workspace, "pass-a")
    for item in workspace.item_ids[:3]:
        workspace.annotate("pass-b", item, value("DIRECT"))
    workspace.skip("pass-b", workspace.item_ids[3])
    with pytest.raises(IncompleteGoldError) as refused:
        freeze_gold(workspace, ["pass-a", "pass-b"], tmp_path / "gold")
    assert refused.value.reasons == ["session 'pass-b': 2 of 5 items not labeled (1 skipped)"]
    assert not (tmp_path / "gold").exists()
    assert workspace.freeze_of("pass-a") is None


def test_gold_freeze_refuses_unresolved_disagreements(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    with pytest.raises(IncompleteGoldError, match="1 disagreements not adjudicated"):
        freeze_gold(workspace, ["pass-a", "pass-b"], tmp_path / "gold")


def test_gold_freeze_refuses_a_stale_adjudication(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    item = workspace.item_ids[1]
    workspace.adjudicate(["pass-a", "pass-b"], item, adjudicated("CLARIFY"), rationale="r", adjudicator_id="carol")
    workspace.annotate("pass-b", item, value("CALL"), replace=True, reason="changed after adjudication")
    with pytest.raises(IncompleteGoldError, match="predate a later change"):
        freeze_gold(workspace, ["pass-a", "pass-b"], tmp_path / "gold")


def test_single_annotator_needs_the_review_queue_and_says_so(workspace: Workspace, tmp_path: Path) -> None:
    label_all(workspace, "pass-a", {3: "UNKNOWN"})
    workspace.set_flag("pass-a", workspace.item_ids[0], True)
    queue = workspace.adjudication_queue(["pass-a"])
    assert [(q["item_id"], q["reasons"]) for q in queue] == [
        ("item-000", ["flagged"]),
        ("item-003", ["label=UNKNOWN"]),
    ]
    with pytest.raises(IncompleteGoldError, match="2 items in the single-annotator re-read queue"):
        freeze_gold(workspace, ["pass-a"], tmp_path / "gold")
    for entry in queue:
        label = "UNKNOWN" if entry["item_id"] == "item-003" else "DIRECT"
        workspace.adjudicate(["pass-a"], entry["item_id"], adjudicated(label, step="2"), rationale="re-read", adjudicator_id="alice")
    manifest = freeze_gold(workspace, ["pass-a"], tmp_path / "gold", exported_at=STAMP)
    assert manifest["design"] == "single_annotator"
    assert manifest["inter_annotator_agreement_claimable"] is False
    assert "No inter-annotator agreement may be claimed" in manifest["limitations"][0]
    assert manifest["gold"]["sources"] == {"review": 2, "single_annotator": 3}


def test_same_annotator_twice_is_not_independent(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace], tmp_path: Path) -> None:
    ws = open_workspace(make_config())
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-a2", "alice")
    label_all(ws, "pass-a")
    label_all(ws, "pass-a2")
    manifest = freeze_gold(ws, ["pass-a", "pass-a2"], tmp_path / "gold", exported_at=STAMP)
    assert manifest["design"] == "repeated_pass_same_annotator"
    assert manifest["independent_annotators"] is False
    assert manifest["inter_annotator_agreement_claimable"] is False


def test_a_frozen_package_is_never_overwritten(package: Path, workspace: Workspace) -> None:
    with pytest.raises(ExportError, match="already frozen"):
        freeze_gold(workspace, ["pass-a", "pass-b"], package.parent)
    with pytest.raises(ExportError, match="never overwritten"):
        export_snapshot(workspace, ["pass-a", "pass-b"], package.parent)


def test_exports_never_write_protected_paths(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace], tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["output"] = {"dir": "data", "gold_file": "source.jsonl"}
    ws = open_workspace(make_config(data))
    ws.open_session("pass-a", "alice")
    label_all(ws, "pass-a")
    before = (tmp_path / "data" / "source.jsonl").read_bytes()
    with pytest.raises(ExportError, match="protected"):
        freeze_gold(ws, ["pass-a"])
    assert (tmp_path / "data" / "source.jsonl").read_bytes() == before
    assert ws.freeze_of("pass-a") is None


def test_unresolved_disagreement_is_explicit_when_policy_allows_it(make_config: Callable[..., TaskConfig], open_workspace: Callable[..., Workspace], tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["freeze"] = {"require_adjudication": False}
    ws = open_workspace(make_config(data))
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    two_complete_passes(ws)
    manifest = freeze_gold(ws, ["pass-a", "pass-b"], tmp_path / "gold", exported_at=STAMP)
    gold = read_jsonl(tmp_path / "gold" / manifest["gold"]["file"])
    assert gold[1]["gold_policy_label"] is None and gold[1]["gold_source"] == "unresolved_disagreement"
    assert verify_package(Path(manifest["manifest_path"]))[1]["status"] == "PASS"


def test_export_refuses_a_store_whose_history_was_edited(workspace: Workspace, tmp_path: Path) -> None:
    two_complete_passes(workspace)
    with workspace.store.transaction() as conn:
        conn.execute("UPDATE annotations SET value_json = replace(value_json, 'CLARIFY', 'CALL')")
    with pytest.raises(ExportError, match="failed verification"):
        export_snapshot(workspace, ["pass-a", "pass-b"], tmp_path / "wip")
