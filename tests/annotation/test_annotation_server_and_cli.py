"""The HTTP API (role isolation, guards, error mapping) and the command line."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
import yaml

from opengrad.annotation.cli import build_parser, main
from opengrad.annotation.export import freeze_gold
from opengrad.annotation.server import Api, ApiError, ServerContext, create_server
from opengrad.annotation.service import Workspace
from tests.annotation.helpers import BASE_CONFIG, label_all, rows, value, write_source


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = None if body is None else (body if isinstance(body, (bytes, str)) else json.dumps(body))
        sent = {"Content-Type": "application/json"} if body is not None and not isinstance(body, str) else {}
        sent.update(headers or {})
        connection.request(method, path, body=payload, headers=sent)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, raw.decode("utf-8", "replace")


@pytest.fixture
def served(workspace: Workspace) -> Iterator[tuple[Client, Workspace]]:
    context = ServerContext(workspace, "annotate", session_id="pass-a")
    server = create_server(context, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Client(context.port), workspace
    server.shutdown()
    server.server_close()


def item_path(item_id: str, action: str = "") -> str:
    return f"/api/items/{quote(item_id, safe='')}" + (f"/{action}" if action else "")


def test_annotation_mode_never_serves_other_passes(workspace: Workspace) -> None:
    label_all(workspace, "pass-b", default="CLARIFY")
    api = Api(ServerContext(workspace, "annotate", session_id="pass-a"))
    state = api.state()
    assert "adjudication" not in state and state["session"]["session_id"] == "pass-a"
    payload = json.dumps(api.get(item_path(workspace.item_ids[0]), {}))
    assert "CLARIFY" not in payload and "bob" not in payload
    assert "classifier_prediction" not in payload and "gold_policy_label" not in payload
    for path in ("/api/adjudication/queue", f"/api/adjudication/items/{workspace.item_ids[0]}"):
        with pytest.raises(ApiError) as denied:
            api.get(path, {})
        assert denied.value.status == 403


def test_adjudication_mode_cannot_write_annotations(workspace: Workspace) -> None:
    label_all(workspace, "pass-a")
    label_all(workspace, "pass-b")
    api = Api(ServerContext(workspace, "adjudicate", sessions=["pass-a", "pass-b"], adjudicator_id="carol"))
    assert api.state()["adjudication"]["kind"] == "adjudication"
    with pytest.raises(ApiError) as denied:
        api.post(item_path(workspace.item_ids[0], "annotate"), {"value": value("CALL")})
    assert denied.value.status == 403


def test_http_round_trip_and_error_mapping(served: tuple[Client, Workspace]) -> None:
    client, ws = served
    first = ws.item_ids[0]
    status, state = client.request("GET", "/api/state")
    assert status == 200 and state["task"]["labels"][0] == "CALL"
    status, body = client.request("POST", item_path(first, "annotate"), {"value": value("DIRECT"), "note": "n"})
    assert status == 200 and body["next_item_id"] == ws.item_ids[1] and body["progress"]["completed"] == 1
    assert client.request("POST", item_path(first, "annotate"), {"value": value("CALL")})[0] == 409
    assert client.request("POST", item_path(first, "annotate"), {"value": value("NOPE")})[0] == 422
    assert client.request("POST", item_path("missing", "annotate"), {"value": value("CALL")})[0] == 404
    assert client.request("POST", "/api/undo", {})[0] == 200
    assert client.request("GET", "/api/nothing")[0] == 404
    status, listed = client.request("GET", "/api/items?status=unlabeled&component=challenge")
    assert status == 200 and [row["item_id"] for row in listed["items"]] == [ws.item_ids[1], ws.item_ids[3]]
    status, docs = client.request("GET", "/api/instructions")
    assert status == 200 and docs == {"documents": []}


def test_frozen_session_writes_are_locked(served: tuple[Client, Workspace]) -> None:
    client, ws = served
    label_all(ws, "pass-a")
    label_all(ws, "pass-b")
    freeze_gold(ws, ["pass-a", "pass-b"])
    status, body = client.request("POST", item_path(ws.item_ids[0], "annotate"), {"value": value("CALL"), "replace": True})
    assert status == 423 and body["kind"] == "frozen"
    assert client.request("GET", "/api/state")[1]["session"]["frozen"] is True


def test_guards_against_foreign_hosts_and_form_posts(served: tuple[Client, Workspace]) -> None:
    client, ws = served
    assert client.request("GET", "/api/state", headers={"Host": "attacker.example"})[0] == 403
    status, _ = client.request("POST", item_path(ws.item_ids[0], "annotate"), "value=CALL", {"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 415
    assert client.request("POST", "/api/undo", "[1,2]", {"Content-Type": "application/json"})[0] == 400
    assert client.request("POST", "/api/undo", "{not json", {"Content-Type": "application/json"})[0] == 400
    assert ws.progress("pass-a")["completed"] == 0


def test_static_serving_is_confined_to_the_ui_build(workspace: Workspace, tmp_path: Path) -> None:
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<p>ui</p>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    context = ServerContext(workspace, "annotate", session_id="pass-a", ui_dir=ui)
    server = create_server(context, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = Client(context.port)
        assert client.request("GET", "/")[1] == "<p>ui</p>"
        _, body = client.request("GET", "/../secret.txt")
        assert "do not serve" not in str(body)
        _, body = client.request("GET", "/%2e%2e/secret.txt")
        assert "do not serve" not in str(body)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_ui_build_explains_how_to_build(served: tuple[Client, Workspace]) -> None:
    client, _ = served
    status, body = client.request("GET", "/")
    assert status == 200 and "npm run build" in body


def test_server_refuses_non_loopback_bind(workspace: Workspace) -> None:
    with pytest.raises(Exception, match="loopback"):
        create_server(ServerContext(workspace, "annotate", session_id="pass-a"), host="0.0.0.0", port=0)


# ── command line ────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cli_task(tmp_path: Path) -> list[str]:
    write_source(tmp_path, rows())
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False), encoding="utf-8")
    return [str(tmp_path / "task.yaml"), "--root", str(tmp_path), "--state-db", str(tmp_path / "state.sqlite3")]


def _label(cli_task: list[str], sessions: dict[str, str]) -> None:
    from opengrad.annotation.config import load_task_config

    config = load_task_config(Path(cli_task[0]), Path(cli_task[2]))
    ws = Workspace.open(config, state_db=Path(cli_task[4]))
    try:
        for session, annotator in sessions.items():
            ws.open_session(session, annotator)
            label_all(ws, session)
    finally:
        ws.close()


def test_start_and_resume_are_aliases_of_serve() -> None:
    parser = build_parser()
    for verb in ("serve", "start", "resume"):
        args = parser.parse_args([verb, "task.yaml", "--session", "pass-a", "--annotator", "alice"])
        assert args.func.__name__ == "cmd_serve"


def test_status_export_freeze_verify(cli_task: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", *cli_task]) == 0
    assert "no working state yet" in capsys.readouterr().out
    assert not (tmp_path / "state.sqlite3").exists()  # reporting never creates state
    assert main(["check", *cli_task]) == 0
    assert "CHECK     PASS" in capsys.readouterr().out
    _label(cli_task, {"pass-a": "alice"})
    assert main(["status", *cli_task, "--session", "pass-a"]) == 0
    out = capsys.readouterr().out
    assert "5 / 5 completed" in out and "DIRECT" in out
    assert main(["audit", *cli_task]) == 0
    assert main(["export", *cli_task, "--sessions", "pass-a", "--out", str(tmp_path / "wip")]) == 0
    assert "ANNOTATION_SNAPSHOT  COMPLETE" in capsys.readouterr().out
    assert main(["freeze-gold", *cli_task, "--sessions", "pass-a", "pass-z", "--out", str(tmp_path / "bad")]) == 2
    assert main(["freeze-gold", *cli_task, "--sessions", "pass-a", "--out", str(tmp_path / "gold")]) == 0
    manifest = tmp_path / "gold" / "manifest.json"
    assert main(["verify", str(manifest), "--require-source"]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out
    (tmp_path / "gold" / "gold" / "toy-v1.gold.jsonl").write_text("{}\n", encoding="utf-8")
    assert main(["verify", str(manifest)]) == 1


def test_incomplete_freeze_exits_3_with_reasons(cli_task: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _label(cli_task, {"pass-a": "alice"})
    from opengrad.annotation.config import load_task_config

    ws = Workspace.open(load_task_config(Path(cli_task[0]), Path(cli_task[2])), state_db=Path(cli_task[4]))
    ws.open_session("pass-b", "bob")
    ws.annotate("pass-b", ws.item_ids[0], value("DIRECT"))
    ws.close()
    assert main(["freeze-gold", *cli_task, "--sessions", "pass-a", "pass-b", "--out", str(tmp_path / "gold")]) == 3
    err = capsys.readouterr().err
    assert "session 'pass-b': 4 of 5 items not labeled" in err
    assert not (tmp_path / "gold").exists()
