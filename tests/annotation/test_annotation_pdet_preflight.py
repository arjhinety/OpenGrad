"""Pre-annotation checks against the real P-DET-v1 task and the real frozen population.

Each test is one item of the pre-annotation checklist. The frozen files are only read and hashed; working
state and exports live in ``tmp_path``. Labels written here are synthetic test values, never gold.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from opengrad.annotation.cli import main
from opengrad.annotation.config import TaskConfig, load_task_config
from opengrad.annotation.export import (
    IncompleteGoldError,
    export_snapshot,
    freeze_gold,
    verify_package,
)
from opengrad.annotation.items import SourceIntegrityError, canonical_json, row_hash, sha256_file
from opengrad.annotation.server import Api, ServerContext
from opengrad.annotation.service import TaskDriftError, Workspace, WorkspaceError
from opengrad.annotation.values import AnnotationValueError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "annotation" / "pdet-v1.yaml"
PDET = ROOT / "reports" / "pdet"
POPULATION = PDET / "pdet-v1.population.jsonl"
FROZEN_SHA = "6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b"
FAMILIES = (
    "question_mark", "question_without_mark", "refusal_with_question", "refusal_plain",
    "tool_mentioned_no_payload", "caveat_then_content", "multi_clause_question", "no_tools_offered",
    "short_plain", "serialized_call_shape", "advice_external_service", "polite_followup",
)
DOCS = (
    "22-PDET-PROTOCOL.md", "23-PDET-ANNOTATION-INSTRUMENT.md", "26-PDET-ANNOTATOR-CHECKLIST.md",
    "27-PDET-EXPOSED-WORKED-EXAMPLES.md", "29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md",
)
INSTRUMENT = ROOT / "docs" / "research" / "study-002" / "23-PDET-ANNOTATION-INSTRUMENT.md"
REAL_STATE = ROOT / ".annotation"
EXPOSED = "EXPOSED_WORKED_EXAMPLE"

pytestmark = pytest.mark.skipif(not POPULATION.is_file(), reason="frozen P-DET population not present")


def ok(label: str, ambiguity: str = "NONE", why: str = "synthetic test rationale") -> dict[str, str]:
    return {"label": label, "ambiguity_status": ambiguity, "annotator_rationale": why}


@pytest.fixture
def config(tmp_path: Path) -> TaskConfig:
    return dataclasses.replace(load_task_config(CONFIG), output_dir=str(tmp_path / "out"))


@pytest.fixture
def ws(config: TaskConfig, tmp_path: Path):
    workspace = Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    yield workspace
    workspace.close()


def real_store_fingerprint() -> list[tuple] | None:
    """Row counts, last ids and the task row of the real store, read-only; None when there is none."""
    path = REAL_STATE / "pdet-v1.sqlite3"
    if not path.exists():
        return None
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        found: list[tuple] = [("tables", *tables)]
        for table in tables:
            found.append((table, *conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()))
        found.extend(conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall())
        found.extend(conn.execute("SELECT session_id, MAX(id) FROM history GROUP BY session_id ORDER BY 1").fetchall())
        return found


@pytest.fixture(autouse=True)
def frozen_untouched():
    before = {path.name: sha256_file(path) for path in PDET.iterdir() if path.is_file()}
    # Real annotation may be under way; tests must never create, migrate or write its state.
    state_existed = REAL_STATE.exists()
    store_before = real_store_fingerprint()
    yield
    after = {path.name: sha256_file(path) for path in PDET.iterdir() if path.is_file()}
    assert after == before, "a test changed or added a file in reports/pdet"
    assert state_existed or not REAL_STATE.exists(), "a test created real working state"
    assert real_store_fingerprint() == store_before, "a test wrote to the real working state"


def frozen_rows() -> list[dict]:
    return [json.loads(line) for line in POPULATION.read_text(encoding="utf-8").splitlines() if line]


# ── the population ──────────────────────────────────────────────────────────────────────────────


def test_population_hash_count_and_pin(config: TaskConfig) -> None:
    manifest = json.loads((PDET / "pdet-v1.manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(POPULATION) == FROZEN_SHA == manifest["population_sha256"]
    assert config.source.expected_sha256 == FROZEN_SHA
    assert config.source.expected_items == 581 == manifest["population_records"] == len(frozen_rows())
    assert config.resolve(config.source.path).resolve() == POPULATION.resolve()


def test_items_are_the_frozen_rows_unmodified_in_frozen_order(ws: Workspace) -> None:
    rows = frozen_rows()
    stored = ws.store.items(ws.task_id)
    assert len(stored) == 581
    for index, (item, row) in enumerate(zip(stored, rows, strict=True)):
        assert item["order_index"] == index == row["pdet_index"]
        assert item["item_id"] == row["pdet_id"]
        assert json.loads(item["row_json"]) == row
        assert item["row_json"] == canonical_json(row)
        assert item["row_hash"] == row_hash(row)


def test_starting_the_app_never_resamples(config: TaskConfig, tmp_path: Path) -> None:
    package = ROOT / "src" / "opengrad" / "annotation"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("opengrad.verification.pdet", "build_sample", "keyed_rank", "import random"):
        assert forbidden not in source, forbidden
    first = Workspace.open(config, state_db=tmp_path / "a.sqlite3")
    second = Workspace.open(config, state_db=tmp_path / "b.sqlite3")
    try:
        assert first.item_ids == second.item_ids == [row["pdet_id"] for row in frozen_rows()]
    finally:
        first.close()
        second.close()


# ── what the annotator sees ─────────────────────────────────────────────────────────────────────


def test_annotation_view_hides_families_labels_and_classifier_fields(ws: Workspace) -> None:
    ws.open_session("pass-a", "test-a")
    api = Api(ServerContext(ws, "annotate", session_id="pass-a"))
    assert sum(1 for row in frozen_rows() if row.get("challenge_families")) == 181
    items = "\n".join(json.dumps(api.get(f"/api/items/{item_id}", {})) for item_id in ws.item_ids)
    # Item payloads carry no blinded source column at all -- neither its name nor its value.
    for hidden in ws.config.blind_fields:
        assert f'"{hidden}"' not in items, hidden
    # Nowhere on the screen: no family identifier, no classifier field or status.
    screen = "\n".join([items, json.dumps(api.state()), json.dumps(api.get("/api/items", {"status": "all"}))])
    for hidden in (*FAMILIES, "challenge_families", "classifier_version_at_selection", "NOT_IMPLEMENTED"):
        assert hidden not in screen, hidden
    assert "classifier" not in json.dumps(api.state()).casefold()


def test_rubric_panel_is_read_only_excerpts_without_sampling_cues(ws: Workspace) -> None:
    documents = Api(ServerContext(ws, "annotate", session_id="none")).get("/api/instructions", {})["documents"]
    assert [doc["path"].split("/")[-1] for doc in documents] == [
        "26-PDET-ANNOTATOR-CHECKLIST.md", "22-PDET-PROTOCOL.md", "23-PDET-ANNOTATION-INSTRUMENT.md"
    ]
    protocol, instrument = documents[1]["content"], documents[2]["content"]
    for heading in ("### 1.1 `CALL`", "### 1.2 `DIRECT`", "### 1.3 `CLARIFY`", "### 1.4 `UNSUPPORTED`", "### 1.5 `UNKNOWN` / `AMBIGUOUS`"):
        assert heading in protocol
    for kept in ("## 2. The eight boundary discriminations", "## 3. Annotation decision tree"):
        assert kept in protocol
    for dropped in ("Sampling design", "Classifier acceptance criteria", "Lifecycle and evidence"):
        assert dropped not in protocol
    for dropped in ("Worked examples", "Coverage bookkeeping", "Open the frozen population"):
        assert dropped not in instrument
    text = "\n".join(doc["content"] for doc in documents).casefold()
    for term in (*FAMILIES, "challenge_families", "families:", "heuristic_regex", "refusal_signal"):
        assert term.casefold() not in text, term


# ── what the annotator can submit ───────────────────────────────────────────────────────────────


def test_only_the_five_pdet_outcomes_are_accepted(ws: Workspace) -> None:
    ws.open_session("pass-a", "test-a")
    assert list(ws.config.labels) == ["CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "UNKNOWN"]
    ids = iter(ws.item_ids)
    for label in ("CALL", "DIRECT", "CLARIFY", "UNSUPPORTED"):
        ws.annotate("pass-a", next(ids), ok(label))
    ws.annotate("pass-a", next(ids), ok("UNKNOWN", "AMBIGUOUS_TWO_MODES"))
    for bad in ("AMBIGUOUS", "ANSWER", "call", "", "UNKNOWN/AMBIGUOUS"):
        with pytest.raises(AnnotationValueError):
            ws.annotate("pass-a", next(ids), ok(bad))
    with pytest.raises(AnnotationValueError, match="ambiguity reason"):
        ws.annotate("pass-a", ws.item_ids[10], ok("UNKNOWN"))
    with pytest.raises(AnnotationValueError, match="mode label needs ambiguity_status NONE"):
        ws.annotate("pass-a", ws.item_ids[10], ok("DIRECT", "NON_SUBSTANTIVE"))


def test_the_priority_review_queue_is_pinned_and_its_reasons_never_reach_the_screen(ws: Workspace) -> None:
    (queue,) = ws.config.review_queues
    assert queue.name == "priority-review"
    body = json.loads((ROOT / queue.file).read_bytes())
    assert sha256_file(ROOT / queue.file) == queue.sha256
    assert (ROOT / (queue.file + ".sha256")).read_text(encoding="utf-8").split()[0] == queue.sha256
    assert body["source_population_sha256"] == FROZEN_SHA and body["task_id"] == "pdet-v1"
    order = ws.queue_order("priority-review")
    assert order == [entry["item_id"] for entry in body["items"]] and len(order) == body["counts"]["items"]
    assert set(order) <= set(ws.item_ids) and len(set(order)) == len(order)
    ws.open_session("pass-a", "test-a")
    api = Api(ServerContext(ws, "annotate", session_id="pass-a"))
    screen = json.dumps(
        [
            api.state(),
            api.get("/api/items", {"status": "all", "queue": "priority-review"}),
            api.get("/api/items/next", {"queue": "priority-review"}),
            api.get(f"/api/items/{order[0]}", {}),
        ]
    )
    assert api.state()["queues"] == [{"name": "priority-review", "total": len(order), "completed": 0, "remaining": len(order)}]
    for hidden in ("reasons", "criterion", body["seed"], "model.claude-opus-5", "annotator_kind"):
        assert hidden not in screen, hidden


def test_rationale_and_note_are_optional_but_unknown_needs_an_ambiguity_reason(ws: Workspace) -> None:
    """Amendment 29 (study_002_prereg_v3): the rationale is optional; the UNKNOWN rule is unchanged."""
    ws.open_session("pass-a", "test-a")
    ids = iter(ws.item_ids)
    for label in ("CALL", "DIRECT", "CLARIFY", "UNSUPPORTED"):
        saved = ws.annotate("pass-a", next(ids), {"label": label, "ambiguity_status": "NONE"})["annotation"]
        assert saved["note"] is None and saved["value"]["boundary_rule_cited"] == "none"
        assert not saved["value"].get("annotator_rationale")
    blank = ws.annotate("pass-a", next(ids), ok("DIRECT", why="   "))["annotation"]
    assert not (blank["value"].get("annotator_rationale") or "").strip()
    item = next(ids)
    with pytest.raises(AnnotationValueError, match="ambiguity reason"):
        ws.annotate("pass-a", item, {"label": "UNKNOWN", "ambiguity_status": "NONE"})
    assert ws.store.annotation(ws.task_id, "pass-a", item) is None
    ws.annotate("pass-a", item, {"label": "UNKNOWN", "ambiguity_status": "MISSING_CONTEXT"})
    kept = ws.annotate("pass-a", next(ids), ok("CLARIFY", why="needs the account id"))["annotation"]
    assert kept["value"]["annotator_rationale"] == "needs the account id"
    noted = ws.annotate("pass-a", next(ids), {"label": "CALL", "ambiguity_status": "NONE"}, note="payload present")
    assert noted["annotation"]["note"] == "payload present"


def test_the_rationale_amendment_is_declared_from_the_definition_the_labels_were_made_under() -> None:
    config = load_task_config(CONFIG)
    (amendment,) = config.definition_amendments
    assert amendment.from_sha256 == "0f060bb395bfb0bc24cea8091fb9916fe62a717d5a0bc09b61e8004536a4e640"
    assert amendment.to_sha256 == config.definition_sha256()
    assert amendment.document == "docs/research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md"
    rationale = next(field for field in config.extra_fields if field.key == "annotator_rationale")
    assert rationale.required is False
    assert config.amendment_route(amendment.from_sha256) == [amendment]
    # Only that flag moved: setting it back reproduces the definition the existing labels were made under,
    # so labels, fields, options and constraints are what they were.
    strict = dataclasses.replace(
        config,
        extra_fields=tuple(
            dataclasses.replace(field, required=True) if field.key == "annotator_rationale" else field
            for field in config.extra_fields
        ),
        definition_amendments=(),
    )
    assert strict.definition_sha256() == amendment.from_sha256


# ── where labels go, and who sees them ──────────────────────────────────────────────────────────


def test_annotations_are_saved_separately_and_every_submission_is_durable(ws: Workspace, tmp_path: Path) -> None:
    ws.open_session("pass-a", "test-a")
    item = ws.item_ids[0]
    ws.annotate("pass-a", item, ok("CLARIFY"))
    assert ws.store.path == tmp_path / "state.sqlite3"
    # Another connection sees the label at once: it was committed when submitted, not on exit.
    with sqlite3.connect(f"file:{(tmp_path / 'state.sqlite3').as_posix()}?mode=ro", uri=True) as reader:
        stored = reader.execute(
            "SELECT value_json FROM annotations WHERE session_id = 'pass-a' AND item_id = ?", (item,)
        ).fetchone()
    assert json.loads(stored[0])["label"] == "CLARIFY"
    assert sha256_file(POPULATION) == FROZEN_SHA


def test_close_and_reopen_resumes_the_session(config: TaskConfig, tmp_path: Path) -> None:
    first = Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    first.open_session("pass-a", "test-a")
    ids = first.item_ids
    first.annotate("pass-a", ids[0], ok("DIRECT"))
    first.annotate("pass-a", ids[1], ok("UNSUPPORTED"))
    first.set_flag("pass-a", ids[1], True)
    first.store._conn.close()  # as if the process died
    second = Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    try:
        assert second.next_unlabeled("pass-a") == ids[2]
        progress = second.progress("pass-a")
        assert (progress["completed"], progress["flagged"]) == (2, 1)
        assert second.item_view("pass-a", ids[1])["annotation"]["value"]["label"] == "UNSUPPORTED"
    finally:
        second.close()


def test_passes_are_isolated_and_a_second_session_cannot_overwrite_the_first(ws: Workspace, tmp_path: Path) -> None:
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    item = ws.item_ids[5]
    ws.annotate("pass-a", item, ok("CLARIFY", why="pass a decided"))
    view_b = json.dumps(Api(ServerContext(ws, "annotate", session_id="pass-b")).get(f"/api/items/{item}", {}))
    assert "CLARIFY" not in view_b and "pass a decided" not in view_b
    with pytest.raises(WorkspaceError, match="belongs to annotator 'alice'"):
        ws.open_session("pass-a", "bob")
    ws.open_session("pass-a2", "alice")  # same annotator, new pass
    ws.annotate("pass-a2", item, ok("DIRECT"))
    assert ws.item_view("pass-a", item)["annotation"]["value"]["label"] == "CLARIFY"
    one = export_snapshot(ws, ["pass-a"], tmp_path / "wip-a")
    two = export_snapshot(ws, ["pass-a2"], tmp_path / "wip-a2")
    assert one["sessions"][0]["file"] == "pdet-v1.annotations.alice.pass-a.jsonl"
    assert two["sessions"][0]["file"] == "pdet-v1.annotations.alice.pass-a2.jsonl"


def test_previous_next_flag_skip_and_undo(ws: Workspace) -> None:
    ws.open_session("pass-a", "test-a")
    ids = ws.item_ids
    view = ws.item_view("pass-a", ids[1])
    assert (view["prev_id"], view["next_id"]) == (ids[0], ids[2])
    assert ws.item_view("pass-a", ids[0])["prev_id"] is None
    ws.annotate("pass-a", ids[0], ok("DIRECT"))
    ws.annotate("pass-a", ids[0], ok("CALL"), replace=True, reason="payload on re-read")
    ws.set_flag("pass-a", ids[0], True)
    result = ws.skip("pass-a", ids[1])
    assert result["next_item_id"] == ids[2]
    ws.undo("pass-a")  # undoes the skip
    ws.undo("pass-a")  # undoes the flag
    ws.undo("pass-a")  # undoes the relabel
    assert ws.item_view("pass-a", ids[0])["annotation"]["value"]["label"] == "DIRECT"
    assert ws.store.annotation(ws.task_id, "pass-a", ids[1]) is None
    assert [c["action"] for c in ws.item_history("pass-a", ids[0])] == ["label", "relabel", "flag", "undo", "undo"]
    assert ws.audit() == []


# ── nothing incomplete becomes gold ─────────────────────────────────────────────────────────────


def test_incomplete_and_unadjudicated_work_never_freezes(ws: Workspace, tmp_path: Path) -> None:
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    ws.annotate("pass-a", ws.item_ids[0], ok("DIRECT"))
    with pytest.raises(IncompleteGoldError, match="580 of 581 items not labeled"):
        freeze_gold(ws, ["pass-a", "pass-b"], tmp_path / "gold")
    snapshot = export_snapshot(ws, ["pass-a", "pass-b"], tmp_path / "wip")
    assert snapshot["artifact_kind"] == "ANNOTATION_SNAPSHOT" and snapshot["completion_state"] == "INCOMPLETE"
    assert snapshot["gold"] is None and not list((tmp_path / "wip").rglob("*gold*"))
    for index, item in enumerate(ws.item_ids):
        if index:
            ws.annotate("pass-a", item, ok("DIRECT"))
        ws.annotate("pass-b", item, ok("CLARIFY" if index == 7 else "DIRECT"))
    with pytest.raises(IncompleteGoldError, match="1 disagreements not adjudicated"):
        freeze_gold(ws, ["pass-a", "pass-b"], tmp_path / "gold")
    assert not (tmp_path / "gold").exists() and ws.freeze_of("pass-a") is None


# ── a moved target is a hard failure ────────────────────────────────────────────────────────────


def _copied_task(tmp_path: Path, mutate) -> TaskConfig:
    (tmp_path / "configs" / "annotation").mkdir(parents=True)
    shutil.copy(CONFIG, tmp_path / "configs" / "annotation" / "pdet-v1.yaml")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "reports" / "pdet").mkdir(parents=True)
    (tmp_path / "docs" / "research" / "study-002").mkdir(parents=True)
    for name in DOCS:
        shutil.copy(ROOT / "docs" / "research" / "study-002" / name, tmp_path / "docs" / "research" / "study-002" / name)
    for queue in load_task_config(CONFIG).review_queues:
        (tmp_path / queue.file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / queue.file, tmp_path / queue.file)
    data = POPULATION.read_bytes()
    (tmp_path / "reports" / "pdet" / "pdet-v1.population.jsonl").write_bytes(mutate(data))
    return load_task_config(tmp_path / "configs" / "annotation" / "pdet-v1.yaml")


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("one byte", lambda data: data.replace(b"Locate hospitals", b"Locate Hospitals", 1)),
        ("one item dropped", lambda data: b"\n".join(data.split(b"\n")[1:])),
        ("reordered", lambda data: b"\n".join([*data.split(b"\n")[1:2], *data.split(b"\n")[:1], *data.split(b"\n")[2:]])),
    ],
)
def test_source_mismatch_is_a_hard_failure(tmp_path: Path, name: str, mutate) -> None:
    config = _copied_task(tmp_path, mutate)
    with pytest.raises(SourceIntegrityError):
        Workspace.open(config, state_db=tmp_path / "state.sqlite3")
    assert not (tmp_path / "state.sqlite3").exists()
    assert main(["check", str(config.config_path)]) == 2


def test_population_changed_after_import_refuses_to_reopen(tmp_path: Path) -> None:
    config = _copied_task(tmp_path, lambda data: data)
    unpinned = dataclasses.replace(config, source=dataclasses.replace(config.source, expected_sha256=None, expected_items=None))
    Workspace.open(unpinned, state_db=tmp_path / "state.sqlite3").close()
    target = tmp_path / "reports" / "pdet" / "pdet-v1.population.jsonl"
    target.write_bytes(target.read_bytes().replace(b"Locate hospitals", b"Locate Hospitals", 1))
    with pytest.raises(TaskDriftError, match="refusing to open"):
        Workspace.open(unpinned, state_db=tmp_path / "state.sqlite3")


def test_check_command_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "pdet-v1"]) == 0
    out = capsys.readouterr().out
    assert f"{FROZEN_SHA}  (pinned, matches)" in out
    assert "581  (pinned, matches)" in out and "challenge 181, prevalence 400" in out
    assert "CHECK     PASS" in out
    real = REAL_STATE / "pdet-v1.sqlite3"
    if not real.exists():
        assert main(["status", "pdet-v1"]) == 0
        assert "no working state yet" in capsys.readouterr().out
        assert not REAL_STATE.exists()
        return
    # `status` opens the store (and would record a pending definition amendment), so it reads a copy.
    copy = tmp_path / "copy.sqlite3"
    with sqlite3.connect(f"file:{real.as_posix()}?mode=ro", uri=True) as source, sqlite3.connect(copy) as target:
        source.backup(target)
    assert main(["status", "pdet-v1", "--state-db", str(copy)]) == 0
    assert "session pass-a" in capsys.readouterr().out


def test_pass_a_start_dry_run_checks_everything_and_creates_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html>", encoding="utf-8")
    state = tmp_path / "state.sqlite3"
    command = ["start", "pdet-v1", "--annotator", "test-a", "--session", "pass-a", "--dry-run",
               "--state-db", str(state), "--port", "0", "--ui-dir", str(ui)]
    assert main(command) == 0
    out = capsys.readouterr().out
    assert f"581 items, source sha256 {FROZEN_SHA}, pinned" in out
    assert "(does not exist; start would create it)" in out
    assert "session   pass-a would be created for annotator test-a" in out
    assert "DRY RUN   PASS -- nothing was created" in out
    assert not state.exists()


# ── exposed worked examples (27) ────────────────────────────────────────────────────────────────


def squash(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def worked_example_ids() -> list[str]:
    """Re-derive, from the instrument text and the population alone, the items 23 §4 quotes."""
    section = INSTRUMENT.read_text(encoding="utf-8").split("## 4. Worked examples", 1)[1].split("\n## 5.", 1)[0]

    def quoted(kind: str) -> list[str]:
        return [squash(text) for text in re.findall(rf'^- {kind}: \*"(.*?)"\*', section, re.DOTALL | re.MULTILINE)]

    prompts, responses = quoted("prompt"), quoted("response")
    assert len(prompts) == len(responses) == 3
    ids = []
    for prompt, response in zip(prompts, responses, strict=True):
        hits = [
            row["pdet_id"]
            for row in frozen_rows()
            if squash(row["prompt"]).startswith(prompt.rstrip("…").rstrip()) and squash(row["response"]) == response
        ]
        assert len(hits) == 1, prompt
        ids.extend(hits)
    return ids


def test_exposed_worked_examples_are_derived_recorded_and_counted(config: TaskConfig, capsys: pytest.CaptureFixture[str]) -> None:
    derived = worked_example_ids()
    rows = {row["pdet_id"]: row for row in frozen_rows()}
    (group,) = config.metric_exclusions
    assert group.status == EXPOSED
    assert sorted(group.item_ids) == sorted(derived)
    assert set(group.excluded_from) == {
        "classifier_validation_metrics", "annotator_agreement_statistics", "untouched_human_gold_claims"
    }
    assert group.document is not None
    addendum = config.resolve(group.document).read_text(encoding="utf-8")
    assert all(item_id in addendum for item_id in derived)
    for item_id in derived:
        row = rows[item_id]
        assert row["pdet_component"] == "challenge"
        # No other item repeats the prompt, the response or the canonical record: nothing else is exposed.
        for key in ("prompt", "response", "canonical_hash"):
            assert sum(squash(other[key]) == squash(row[key]) for other in rows.values()) == 1, key
    eligible = len(rows) - len(set(derived))
    assert eligible == 578
    assert "**Metric-eligible items: 578**" in addendum
    assert main(["check", "pdet-v1"]) == 0
    assert "metrics   578 of 581 items metric-eligible" in capsys.readouterr().out


def test_exposure_is_never_shown_to_the_annotator(ws: Workspace) -> None:
    ws.open_session("pass-a", "test-a")
    api = Api(ServerContext(ws, "annotate", session_id="pass-a"))
    screen = "\n".join(
        [
            json.dumps(api.state()),
            json.dumps(api.get("/api/items", {"status": "all"})),
            json.dumps(api.get("/api/instructions", {})),
            *(json.dumps(api.get(f"/api/items/{item_id}", {})) for item_id in worked_example_ids()),
        ]
    ).casefold()
    for hidden in (EXPOSED, "metric_exclusion", "metric-eligible", "excluded_from", "27-PDET", "worked example"):
        assert hidden.casefold() not in screen, hidden


def test_exposure_is_carried_into_snapshots_and_gold(ws: Workspace, tmp_path: Path) -> None:
    exposed = set(worked_example_ids())
    ws.open_session("pass-a", "alice")
    ws.open_session("pass-b", "bob")
    for item in ws.item_ids:
        ws.annotate("pass-a", item, ok("DIRECT"))
        ws.annotate("pass-b", item, ok("UNSUPPORTED" if item in exposed else "DIRECT"))
    snapshot = export_snapshot(ws, ["pass-a", "pass-b"], tmp_path / "wip")
    section = snapshot["metric_exclusions"]
    assert (section["population_items"], section["excluded_items"], section["metric_eligible_items"]) == (581, 3, 578)
    assert section["groups"][0]["item_ids"] == sorted(exposed)
    assert (snapshot["disagreements"]["count"], snapshot["disagreements"]["metric_eligible"]) == (3, 0)
    assert [s["labeled_metric_eligible"] for s in snapshot["sessions"]] == [578, 578]
    for item in sorted(exposed):
        decision = {"label": "DIRECT", "ambiguity_status": "NONE", "decision_tree_step": "3"}
        ws.adjudicate(["pass-a", "pass-b"], item, decision, rationale="synthetic test adjudication", adjudicator_id="test-c")
    manifest = freeze_gold(ws, ["pass-a", "pass-b"], tmp_path / "gold")
    gold = manifest["gold"]
    assert (gold["items"], gold["metric_eligible_items"]) == (581, 578)
    assert sum(gold["metric_eligible_label_counts"].values()) == 578
    rows = [json.loads(line) for line in (tmp_path / "gold" / gold["file"]).read_text(encoding="utf-8").splitlines()]
    assert {row["pdet_id"] for row in rows if row["metric_exclusions"] == [EXPOSED]} == exposed
    assert all(row["metric_exclusions"] == [] for row in rows if row["pdet_id"] not in exposed)
    _, summary = verify_package(Path(manifest["manifest_path"]), ROOT, require_source=True)
    assert summary["status"] == "PASS", summary["errors"]
    assert sha256_file(POPULATION) == FROZEN_SHA
