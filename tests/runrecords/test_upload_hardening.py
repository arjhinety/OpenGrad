"""opengrad.runrecords upload, third audit: what the redacted copy removes, when it is used, how long redaction
takes, and every refusal path. The Hugging Face client is always a stub (UP-0030)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from opengrad.runrecords import cli as runs_cli
from opengrad.runrecords import record as rr

PY = sys.executable
QUIET = SimpleNamespace(write=lambda t: None, flush=lambda: None, encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = ModuleType("huggingface_hub")

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test reached the real Hugging Face client")

    stub.HfApi = refuse  # type: ignore[attr-defined]
    stub.hf_hub_download = refuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    monkeypatch.delenv(rr.LOCAL_DIR_ENV, raising=False)


def _hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, info: object) -> dict[str, bytes]:
    sent: dict[str, bytes] = {}

    class FakeApi:
        def repo_info(self, repo: str, **kwargs: object) -> object:
            if isinstance(info, Exception):
                raise info
            return info

        def upload_file(self, **kwargs: object) -> SimpleNamespace:
            sent[str(kwargs["path_in_repo"])] = kwargs["path_or_fileobj"]  # type: ignore[assignment]
            return SimpleNamespace(oid=f"rev{len(sent)}")

    def download(repo: str, filename: str, **kwargs: object) -> str:
        path = tmp_path / "dl" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sent[filename])
        return str(path)

    monkeypatch.setattr(sys.modules["huggingface_hub"], "HfApi", FakeApi)
    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", download)
    return sent


def _log(tmp_path: Path, kind: str, text: str) -> None:
    rr.run_child(kind, [PY, "-c", f"import sys; sys.stdout.write({text!r})"], tmp_path, QUIET)


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        (
            "-----BEGIN "
            + "PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0B\n-----END "
            + "PRIVATE KEY-----",
            "MIIEvQ",
        ),
        ("-----BEGIN ENCRYPTED PRIVATE KEY-----\nQUJDREVGR0hJSktM\n", "QUJDRE"),
        ("using sk-proj-" + "a" * 30, "aaaaaaaaaa"),
        ("using sk-ant-api03-" + "b" * 30, "bbbbbbbbbb"),
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpw"),
        ("login --password hunter2 ok", "hunter2"),
        ("https://x.org/?u=jane%40example.org", "jane%40"),
        ("cache /Users/Jane Doe/models/x", "Jane"),
        ("share \\\\srv\\Users\\Jane Doe\\x", "Jane"),
        ("pathC:\\Users\\jane\\x", "jane"),
        ("lower c:\\users\\jane\\x", "jane"),
        ("HF_TOKEN=abc123 set", "abc123"),
        ('{"access_token": "tok999"}', "tok999"),
    ],
)
def test_the_redacted_copy_removes_it(tmp_path: Path, text: str, leak: str) -> None:
    assert leak not in runs_cli.redacted_log(text, tmp_path)


def test_ordinary_log_content_survives_redaction(tmp_path: Path) -> None:
    text = "step 10 loss 0.42 --max-tokens 512 lr=3e-5 cuda:0"
    assert runs_cli.redacted_log(text, tmp_path) == text


def test_redaction_stays_fast_on_a_long_line(tmp_path: Path) -> None:
    started = time.monotonic()
    text = "a" * 200_000 + " " + "b_" * 50_000
    runs_cli.log_findings(text)
    runs_cli.redacted_log(text, tmp_path)
    assert time.monotonic() - started < 2.0


def test_a_credential_assignment_is_detected_and_sent_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(tmp_path, "script", "export HF_TOKEN=abc123\n")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"]["copy"] == "redacted"
    assert record["log"]["durable"]["redacted_because"] == ["a credential field"]
    assert b"abc123" not in sent[record["log"]["durable"]["path"]]


def test_a_clean_log_is_sent_whole_with_the_same_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(tmp_path, "training", "step 1 loss 0.5\n")
    _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"]["copy"] == "original"
    assert record["log"]["durable"]["sha256"] == record["log"]["sha256"]


def test_an_annotation_log_with_private_content_is_sent_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(tmp_path, "annotation", "labeller someone@example.org replied\n")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"]["copy"] == "redacted"
    assert b"someone@example.org" not in sent[record["log"]["durable"]["path"]]


@pytest.mark.parametrize(
    "info",
    [
        RuntimeError("404"),
        SimpleNamespace(),
        SimpleNamespace(private=False),
        SimpleNamespace(private=1),
    ],
)
def test_upload_sends_nothing_unless_the_repository_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, info: object
) -> None:
    _log(tmp_path, "training", "step 1\n")
    sent = _hub(tmp_path, monkeypatch, info)
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 2
    assert sent == {}


def test_a_log_whose_redaction_leaves_something_is_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(tmp_path, "annotation", "label batch\n")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    monkeypatch.setattr(runs_cli, "redacted_log", lambda text, root: text + " someone@example.org")
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 3
    assert sent == {}
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"] is None


def test_a_log_with_only_ordinary_paths_is_sent_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(tmp_path, "evaluation", "loaded D:\\models\\m.bin in 3s, loss 0.4 from /opt/models/x\n")
    _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"]["copy"] == "original"


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("export HF_TOKEN=abc123", "a credential field"),
        ('{"access_token": "tok"}', "a credential field"),
        ("login --password hunter2", "a credential flag value"),
        ("cache /Users/Jane Doe/x", "a home-folder path"),
    ],
)
def test_the_detector_covers_what_the_redactor_hides(text: str, finding: str) -> None:
    assert finding in runs_cli.private_findings(text)


def test_a_windows_path_does_not_swallow_the_rest_of_its_line(tmp_path: Path) -> None:
    copy = runs_cli.redacted_log("loaded D:\\models\\m.bin in 3s, loss 0.4", tmp_path)
    assert copy.endswith("in 3s, loss 0.4") and "models" not in copy


@pytest.mark.parametrize(
    "text",
    [
        "env: HF_TOKEN=abc123",
        "cmd: OPENAI_API_KEY=sk-x python run.py",
        "--set=api_key=hunter2",
        "args: --hf_token=abc123",
        "note: token: abc123",
        '{"cfg": "api_key=abc"}',
        "name:\napi_key=hunter2",
    ],
)
def test_a_nested_credential_is_detected_exactly_when_it_would_be_hidden(
    text: str, tmp_path: Path
) -> None:
    assert runs_cli.private_findings(text), text
    copy = runs_cli.redacted_log(text, tmp_path)
    assert "abc123" not in copy and "hunter2" not in copy and "=abc" not in copy


def test_home_folder_detection_stays_fast_on_long_separator_runs(tmp_path: Path) -> None:
    for text in ("x" + "\\" * 60 + "y", "/" * 20_000 + "z", "C:" + "\\" * 5_000):
        started = time.monotonic()
        runs_cli.log_findings(text)
        runs_cli.redacted_log(text, tmp_path)
        assert time.monotonic() - started < 2.0, text[:20]


@pytest.mark.parametrize(
    "text",
    ["C:\\Users\\Jane Doe\\m.bin", "/Users/Jane Doe/x.txt", "\\\\srv\\Users\\Jane Doe\\data"],
)
def test_a_name_with_a_space_is_removed_whole(text: str, tmp_path: Path) -> None:
    copy = runs_cli.redacted_log(f"loaded {text} ok", tmp_path)
    assert "Jane" not in copy and "Doe" not in copy and copy.endswith(" ok")


@pytest.mark.parametrize("text", ["  hf_token:\n    abc123", "api_key:\n  abc123"])
def test_a_value_on_the_line_after_its_credential_name_is_refused(text: str, tmp_path: Path) -> None:
    # Detected over the whole text, missed by the per-line redactor: the post-check keeps the log local.
    assert runs_cli.private_findings(runs_cli.redacted_log(text, tmp_path))


def test_a_password_under_a_credential_named_key_is_hidden(tmp_path: Path) -> None:
    copy = runs_cli.redacted_log("credentials:\n  password: hunter2xyz\n", tmp_path)
    assert "hunter2xyz" not in copy, copy


@pytest.mark.parametrize("text", ["model uses a basic\nconfiguration\n", "a bearer\nof news\n"])
def test_basic_or_bearer_at_a_line_end_is_not_a_credential(text: str) -> None:
    assert not runs_cli.private_findings(text)


@pytest.mark.parametrize(
    "text", ["a." * 32_000, "a:" * 32_000, "C:\\Users" * 8_000, "\n".join(["a." * 1_000] * 50)],
    ids=["dots", "colons", "home-paths", "many-lines"],
)
def test_detection_and_redaction_stay_fast_on_long_lines(text: str, tmp_path: Path) -> None:
    started = time.monotonic()
    runs_cli.log_findings(text)
    runs_cli.redacted_log(text, tmp_path)
    assert time.monotonic() - started < 2.0


def test_an_over_long_line_is_a_finding_and_is_removed_whole(tmp_path: Path) -> None:
    text = "ok\n" + "x" * (runs_cli.MAX_LINE + 1) + "\rstep 2\n"
    assert "an over-long line" in runs_cli.log_findings(text)
    copy = runs_cli.redacted_log(text, tmp_path)
    assert "xxxx" not in copy and copy.startswith("ok\n") and copy.endswith("\rstep 2\n")
    assert not runs_cli.log_findings(copy)


def test_a_long_ordinary_config_line_is_kept_whole(tmp_path: Path) -> None:
    line = "config " + ", ".join(f"layer_{i}_dim={i * 64}" for i in range(1_000))
    assert len(line) > 20_000
    assert runs_cli.log_findings(line) == []
    assert runs_cli.redacted_log(line, tmp_path) == line


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("\x1b[32mhf_" + "a" * 34 + "\x1b[0m", "aaaaaaaaaa"),
        ("\x1b[32msk-proj-" + "a" * 30 + "\x1b[0m", "aaaaaaaaaa"),
        ("-----BEGIN PGP " + "PRIVATE KEY BLOCK-----\nlQOYBGXyz123\n-----END PGP " + "PRIVATE KEY BLOCK-----", "lQOYBG"),
        ("/home/o'brien/proj/x.py", "brien"),
        ("C:\\Users\\O'Neil\\proj\\x.py", "Neil"),
        ('cmd --api-key "my secret phrase" done', "secret phrase"),
        ("file:///C%3A%5CUsers%5Cjane%20doe%5Cx.txt", "doe"),
        ("C%3A%5CUsers%5Carro%5Cx", "arro"),
    ],
    ids=["ansi-hf", "ansi-sk", "pgp", "apostrophe-posix", "apostrophe-windows", "quoted-flag", "encoded-space",
         "encoded-home"],
)
def test_a_form_the_audit_found_is_detected_and_removed(text: str, leak: str, tmp_path: Path) -> None:
    assert runs_cli.log_findings(text), text
    copy = runs_cli.redacted_log(text, tmp_path)
    assert leak not in copy, copy
    assert not runs_cli.log_findings(copy), copy


def test_a_closing_quote_after_a_home_folder_keeps_the_rest_of_the_line(tmp_path: Path) -> None:
    copy = runs_cli.redacted_log("open '/home/jane' then step 2", tmp_path)
    assert "jane" not in copy and copy.endswith("' then step 2"), copy


@pytest.mark.parametrize("code", ["sys.stdout.buffer.write('C:/Users/jane/x'.encode('utf-16'))",
                                  "sys.stdout.buffer.write(b'ok\\x00/home/jane/x')"], ids=["utf-16", "nul"])
def test_a_log_that_is_not_clean_utf8_stays_local(
    code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rr.run_child("evaluation", [PY, "-c", f"import sys; {code}"], tmp_path, QUIET)
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 3
    assert sent == {}
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"] is None


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("step\x1b]0;t\nHF_TOKEN=hf_" + "Q" * 34 + "\n\x07 ok\n", "QQQQQQQQ"),
        ("\x1b]8;;https://alice:hunter2pw@db.example.org/\x1b\\db\x1b]8;;\x1b\\", "hunter2pw"),
        ("password: correct horse battery staple", "horse"),
        ("password=abc,defghij", "defghij"),
        ("db_password=abc;defghij", "defghij"),
        ("redis://:hunter2pw@localhost:6379/0", "hunter2pw"),
    ],
    ids=["osc-title", "osc-hyperlink", "spaced-value", "comma-value", "semicolon-value", "empty-url-user"],
)
def test_a_form_the_second_audit_found_is_detected_and_removed(text: str, leak: str, tmp_path: Path) -> None:
    assert runs_cli.log_findings(text), text
    copy = runs_cli.redacted_log(text, tmp_path)
    assert leak not in copy, copy
    assert not runs_cli.log_findings(copy), copy


def test_a_log_changed_since_its_run_stops_the_upload_and_still_lists_kept_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Upload takes the newest log first: the NUL log (newer) is kept local, then the changed one (older) stops it.
    _log(tmp_path, "training", "step 1\n")
    (changed,) = rr.read_local(tmp_path)
    rr.run_child("evaluation", [PY, "-c", "import sys; sys.stdout.buffer.write(b'ok\\x00')"], tmp_path, QUIET)
    (rr.local_base(tmp_path) / changed["run_id"] / "output.log").write_text("edited later\n", encoding="utf-8")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 1
    assert sent == {}
    out = capsys.readouterr().out
    assert "changed since the run" in out and "INCOMPLETE: 1 log(s)" in out


@pytest.mark.parametrize(
    "text",
    ["eos_token_id=2 step 100 loss 0.42", "pad_token: <pad> lr=1e-4", "decoder_start_token_id=0 ok",
     '{"bos_token": "<s>", "loss": 0.4}'],
)
def test_a_tokenizer_setting_is_not_a_credential(text: str, tmp_path: Path) -> None:
    assert runs_cli.log_findings(text) == [], text
    assert runs_cli.redacted_log(text, tmp_path) == text


@pytest.mark.parametrize(
    "name",
    ["HF_TOKEN", "access_token", "api_token", "token", "github_token_secret", "hf_token_id", "HfTokenId",
     "token_id", "start_token", "x_start_token", "sso_start_token", "image_token", "api_token_ids"],
)
def test_a_credential_token_name_is_still_a_credential(name: str) -> None:
    assert rr.is_secret_name(name), name


@pytest.mark.parametrize(
    "escape", ["\x1bM", "\x1b7", "\x1b(B", "\x1b[0m"], ids=["reverse-index", "save-cursor", "charset", "csi"]
)
def test_a_token_after_a_cursor_or_charset_code_is_detected(escape: str, tmp_path: Path) -> None:
    text = escape + "hf_" + "Q" * 34
    assert runs_cli.log_findings(text), text
    assert "QQQQ" not in runs_cli.redacted_log(text, tmp_path)


@pytest.mark.parametrize(
    "name",
    ["pad_token", "EOS_TOKEN_ID", "bosTokenId", "unk_token", "mask_token_ids", "decoder_start_token_id",
     "token_type_ids", "boi_token"],
)
def test_a_listed_tokenizer_setting_is_not_a_credential_name(name: str) -> None:
    assert not rr.is_secret_name(name), name


def test_a_credential_flag_named_like_a_token_id_is_redacted_on_the_command_line(tmp_path: Path) -> None:
    out = rr.redact_command(["train", "--hf-token-id", "abcd1234secretvalue", "--pad-token-id", "0"], tmp_path)
    assert "abcd1234secretvalue" not in out and out[-1] == "0", out

