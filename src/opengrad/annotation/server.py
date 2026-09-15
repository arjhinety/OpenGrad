"""Local JSON API for the annotation UI, on the standard library's HTTP server.

One process serves one role, fixed at start-up:

* **annotate** -- one session, one annotator. No route in this mode returns another session's labels, so
  a pass is blind to every other pass by construction, not by UI convention.
* **adjudicate** -- two completed sessions (or one, for a single-annotator review) and an adjudicator.

The server binds to loopback, rejects requests whose ``Host`` is not loopback (a DNS-rebinding page cannot
read labels through it) and only accepts JSON bodies on writes (a cross-site form cannot post to it). It
also serves the static Next.js build from ``integrations/annotate-ui/out`` when that has been built.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from opengrad.annotation.config import APP_VERSION
from opengrad.annotation.service import (
    ConflictError,
    FrozenError,
    NotFoundError,
    Workspace,
    WorkspaceError,
)
from opengrad.annotation.values import AnnotationValueError

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
MAX_BODY = 1 << 20

_ITEM = re.compile(r"^/api/items/([^/]+)$")
_ITEM_ACTION = re.compile(r"^/api/items/([^/]+)/(annotate|skip|flag)$")
_ADJ_ITEM = re.compile(r"^/api/adjudication/items/([^/]+)$")


@dataclass
class ServerContext:
    workspace: Workspace
    mode: str
    session_id: str | None = None
    sessions: list[str] = field(default_factory=list)
    adjudicator_id: str | None = None
    ui_dir: Path | None = None
    port: int = 0


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind


def _error_for(exc: Exception) -> ApiError:
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, AnnotationValueError):
        return ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), "invalid_value")
    if isinstance(exc, NotFoundError):
        return ApiError(HTTPStatus.NOT_FOUND, str(exc), "not_found")
    if isinstance(exc, FrozenError):
        return ApiError(HTTPStatus.LOCKED, str(exc), "frozen")
    if isinstance(exc, ConflictError):
        return ApiError(HTTPStatus.CONFLICT, str(exc), "conflict")
    if isinstance(exc, WorkspaceError):
        return ApiError(HTTPStatus.BAD_REQUEST, str(exc), "not_allowed")
    return ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}", "internal")


class Api:
    """Routes, independent of the socket layer so tests can call them directly."""

    def __init__(self, context: ServerContext) -> None:
        self.ctx = context
        self.ws = context.workspace

    def _annotating(self) -> str:
        if self.ctx.mode != "annotate" or self.ctx.session_id is None:
            raise ApiError(HTTPStatus.FORBIDDEN, "this server is not in annotation mode", "wrong_mode")
        return self.ctx.session_id

    def _adjudicating(self) -> list[str]:
        if self.ctx.mode != "adjudicate":
            # The annotation server never exposes other passes, whatever the client asks for.
            raise ApiError(HTTPStatus.FORBIDDEN, "this server is not in adjudication mode", "wrong_mode")
        return self.ctx.sessions

    def state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "app_version": APP_VERSION,
            "mode": self.ctx.mode,
            "task": self.ws.config.public(),
            "source": {
                "path": self.ws.config.source.path,
                "sha256": self.ws.source_sha256,
                "items": self.ws.item_count,
            },
            "filter_options": self.ws.filter_options(),
        }
        if self.ctx.mode == "annotate":
            session_id = self._annotating()
            payload["session"] = self.ws.session_state(session_id)
            payload["progress"] = self.ws.progress(session_id)
            # Counts and queue names only: which items a model labeled, and how, never reaches a pass.
            payload["reference"] = self.ws.reference_progress(session_id)
            payload["queues"] = self.ws.queue_progress(session_id)
        else:
            sessions = self._adjudicating()
            payload["adjudication"] = {
                "sessions": [self.ws.session_state(session_id) for session_id in sessions],
                "adjudicator_id": self.ctx.adjudicator_id,
                "kind": "adjudication" if len(sessions) == 2 else "review",
                "frozen": any(self.ws.freeze_of(session_id) for session_id in sessions),
            }
        return payload

    def get(self, path: str, query: dict[str, str]) -> Any:
        if path == "/api/state":
            return self.state()
        if path == "/api/instructions":
            return {"documents": self.ws.instructions()}
        if path == "/api/items":
            session_id = self._annotating()
            status = query.pop("status", "all")
            queue = query.pop("queue", None) or None
            items = self.ws.list_items(session_id, status, query, queue=queue)
            return {"items": items, "count": len(items)}
        if path == "/api/items/next":
            session_id = self._annotating()
            return {
                "item_id": self.ws.next_unlabeled(
                    session_id, after=query.get("after"), queue=query.get("queue") or None
                )
            }
        match = _ITEM.match(path)
        if match:
            return self.ws.item_view(self._annotating(), unquote(match.group(1)))
        if path == "/api/adjudication/queue":
            items = self.ws.adjudication_queue(self._adjudicating())
            return {"items": items, "count": len(items)}
        match = _ADJ_ITEM.match(path)
        if match:
            return self.ws.adjudication_view(self._adjudicating(), unquote(match.group(1)))
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route {path}", "not_found")

    def post(self, path: str, body: dict[str, Any]) -> Any:
        if path == "/api/undo":
            return self.ws.undo(self._annotating())
        match = _ITEM_ACTION.match(path)
        if match:
            session_id = self._annotating()
            item_id, action = unquote(match.group(1)), match.group(2)
            if action == "annotate":
                revision = body.get("expected_revision")
                return self.ws.annotate(
                    session_id,
                    item_id,
                    body.get("value"),
                    note=body.get("note"),
                    flagged=body.get("flagged"),
                    replace=bool(body.get("replace", False)),
                    expected_revision=int(revision) if revision is not None else None,
                    reason=body.get("reason"),
                )
            if action == "skip":
                return self.ws.skip(session_id, item_id, note=body.get("note"))
            return self.ws.set_flag(
                session_id, item_id, bool(body.get("flagged")), note=body.get("note")
            )
        match = _ADJ_ITEM.match(path)
        if match:
            sessions = self._adjudicating()
            assert self.ctx.adjudicator_id is not None
            return self.ws.adjudicate(
                sessions,
                unquote(match.group(1)),
                body.get("value"),
                rationale=body.get("rationale"),
                adjudicator_id=self.ctx.adjudicator_id,
            )
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route {path}", "not_found")


MISSING_UI = """<!doctype html><meta charset="utf-8"><title>OpenGrad annotate</title>
<body style="font:15px system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem;line-height:1.5">
<h1>The annotation UI has not been built</h1>
<p>The API is running. Build the Next.js UI once, then reload this page:</p>
<pre style="background:#f3f3f3;padding:1rem">cd integrations/annotate-ui
npm install
npm run build</pre>
<p>Or run <code>npm run dev</code> there and open <code>http://localhost:3000</code>; it proxies
<code>/api</code> to this server.</p></body>"""


def make_handler(context: ServerContext) -> type[BaseHTTPRequestHandler]:
    api = Api(context)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"opengrad-annotate/{APP_VERSION}"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").strip().lower()
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in LOOPBACK_HOSTS

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _fail(self, exc: Exception) -> None:
            error = _error_for(exc)
            self._json(error.status, {"error": str(error), "kind": error.kind})

        def do_GET(self) -> None:
            if not self._host_ok():
                self._fail(ApiError(HTTPStatus.FORBIDDEN, "non-loopback Host", "forbidden"))
                return
            url = urlsplit(self.path)
            if url.path.startswith("/api/"):
                query = {key: values[-1] for key, values in parse_qs(url.query).items()}
                try:
                    with lock:
                        payload = api.get(url.path, query)
                    self._json(HTTPStatus.OK, payload)
                except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON error, never a stack page
                    self._fail(exc)
                return
            self._static(url.path)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                self._fail(ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large", "bad_request"))
                return
            # Drain the body before any refusal: answering with unread bytes in the socket makes some
            # stacks reset the connection, and the client would see an abort instead of the error.
            raw = self.rfile.read(length) if length else b""
            if not self._host_ok():
                self._fail(ApiError(HTTPStatus.FORBIDDEN, "non-loopback Host", "forbidden"))
                return
            url = urlsplit(self.path)
            try:
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    raise ApiError(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "writes must be application/json", "bad_request"
                    )
                body = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(body, dict):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object", "bad_request")
                with lock:
                    payload = api.post(url.path, body)
                self._json(HTTPStatus.OK, payload)
            except json.JSONDecodeError:
                self._fail(ApiError(HTTPStatus.BAD_REQUEST, "body is not valid JSON", "bad_request"))
            except Exception as exc:  # noqa: BLE001 - mapped to a JSON status by _error_for
                self._fail(exc)

        def _static(self, path: str) -> None:
            root = context.ui_dir
            if root is None or not (root / "index.html").is_file():
                self._send(HTTPStatus.OK, MISSING_UI.encode("utf-8"), "text/html; charset=utf-8")
                return
            relative = unquote(path).lstrip("/") or "index.html"
            candidates = [relative, f"{relative}.html", f"{relative}/index.html"]
            for candidate in candidates:
                target = (root / candidate).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    break
                if target.is_file():
                    kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
                        kind += "; charset=utf-8"
                    self._send(HTTPStatus.OK, target.read_bytes(), kind)
                    return
            not_found = root / "404.html"
            body = not_found.read_bytes() if not_found.is_file() else b"not found"
            self._send(HTTPStatus.NOT_FOUND, body, "text/html; charset=utf-8")

    return Handler


def create_server(context: ServerContext, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise WorkspaceError("the annotation server only binds to loopback")
    server = ThreadingHTTPServer((host, port), make_handler(context))
    context.port = int(server.server_address[1])
    return server
