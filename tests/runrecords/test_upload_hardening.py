"""opengrad.runrecords upload, third audit: what the redacted copy removes, when it is used, how long redaction
takes, and every refusal path. The Hugging Face client is always a stub (UP-0030)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from opengrad.runrecords import cli as runs_cli
from opengrad.runrecords import inputs as ri
from opengrad.runrecords import record as rr
from opengrad.runrecords import scripts as rs

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
    # These tests are about redaction: their ``python -c`` commands stand in for repository scripts. The
    # fail-closed rule for commands that cannot be traced has its own tests (test_scripts_policy.py).
    real = runs_cli.upload_decision

    def traced(record: dict, root: Path) -> tuple[str, str]:
        decision, kind = real(record, root)
        return ("allowed", kind) if decision == "unrecognised" else (decision, kind)

    monkeypatch.setattr(runs_cli, "upload_decision", traced)
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


def test_a_held_out_policy_log_is_kept_local_and_not_counted_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def held_out() -> int:
        print("scored 10 items")
        return 0

    assert rr.recorded_call("script", ["score.py"], tmp_path, held_out, upload="never") == 0
    _log(tmp_path, "training", "step 1\n")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 0
    assert len(sent) == 1
    out = capsys.readouterr().out
    assert "1 log(s) kept local by policy" in out and "INCOMPLETE" not in out
    kept = [r for r in rr.read_local(tmp_path) if r.get("upload_policy") == "never"]
    assert len(kept) == 1 and kept[0]["log"]["durable"] is None


def test_a_log_whose_command_cannot_be_traced_stays_local_and_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Owner's decision, 2026-11-13: fail closed. ``python -c`` is not a repository entry point.
    _log(tmp_path, "script", "step 1\n")
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    monkeypatch.setattr(runs_cli, "upload_decision", rs.upload_decision)  # undo the helper's stand-in
    assert runs_cli.main(["upload", "--repo", "o/r"], tmp_path) == 3
    assert sent == {}
    out = capsys.readouterr().out
    assert "1 log(s) kept local: the command is not a known entry point" in out


def test_export_keeps_only_the_exception_type_of_a_held_out_run(tmp_path: Path) -> None:
    def leaks() -> int:
        raise ValueError("the held-out question text")

    with pytest.raises(ValueError):
        rr.recorded_call("script", ["score.py"], tmp_path, leaks, upload="never")
    (record,) = rr.read_local(tmp_path)
    assert runs_cli.main(["dispose", record["run_id"], "--reason", "test"], tmp_path) == 0
    assert runs_cli.main(["export"], tmp_path) == 0
    committed = (tmp_path / rr.COMMITTED_DIR / f"{record['run_id']}.json").read_text(encoding="utf-8")
    assert "held-out question" not in committed
    assert "ValueError (message kept local: held-out data)" in committed


def test_dispose_after_export_does_not_put_the_message_back(tmp_path: Path) -> None:
    def leaks() -> int:
        raise ValueError("the held-out question text")

    with pytest.raises(ValueError):
        rr.recorded_call("script", ["score.py"], tmp_path, leaks, upload="never")
    (record,) = rr.read_local(tmp_path)
    assert runs_cli.main(["dispose", record["run_id"], "--reason", "test"], tmp_path) == 0
    assert runs_cli.main(["export"], tmp_path) == 0
    assert runs_cli.main(["dispose", record["run_id"], "--reason", "looked again"], tmp_path) == 0
    committed = (tmp_path / rr.COMMITTED_DIR / f"{record['run_id']}.json").read_text(encoding="utf-8")
    assert "held-out question" not in committed


def test_a_held_out_path_outside_the_repository_is_caught_before_redaction(tmp_path: Path) -> None:
    outside = "D:\\elsewhere\\when2call_test.parquet"
    assert rr.recorded_call("script", ["score.py", f"--input={outside}"], tmp_path, lambda: 0) == 0
    (record,) = rr.read_local(tmp_path)
    assert "when2call_test" not in " ".join(record["command"])  # redacted to <absolute-path>
    assert record["held_out_inputs"] is True
    assert rs.upload_decision(record, tmp_path)[0] == "never"


def test_a_config_naming_held_out_data_is_caught_when_the_run_starts(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "eval.yaml"
    config.parent.mkdir()
    config.write_text("prompts: release/x/frozen_prompts_v1.jsonl\n", encoding="utf-8")
    assert rr.recorded_call("script", ["score.py", "configs/eval.yaml"], tmp_path, lambda: 0) == 0
    config.write_text("prompts: elsewhere\n", encoding="utf-8")  # edited after the run
    (record,) = rr.read_local(tmp_path)
    assert record["held_out_inputs"] is True
    assert rs.upload_decision(record, tmp_path)[0] == "never"


def test_a_labelling_run_launched_as_a_script_uploads_as_a_redacted_annotation_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Traced against the real checkout (the runner's path), recorded in a temporary folder; the runner is not run.
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(rr.LOCAL_DIR_ENV, str(tmp_path / "records"))
    # As from a clean checkout of a pushed commit (CI); this test is about kind.
    monkeypatch.setattr(ri, "_modified", lambda root: set())
    monkeypatch.setattr(rs, "on_reviewed_branch", lambda sha, root: True)
    argv = ["python", "scripts/run_external_annotation.py", "--session", "punans-v2-trial"]
    assert rr.recorded_call("script", argv, root, lambda: print("batch 01 (5 items) attempt 1: exit 0") or 0) == 0
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    monkeypatch.setattr(runs_cli, "upload_decision", rs.upload_decision)  # undo the helper's stand-in
    assert runs_cli.main(["upload", "--repo", "o/r", "--kind", "annotation"], root) == 0
    (name,) = sent
    assert name.endswith("output.redacted.log")



def test_a_config_counts_values_not_prose(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "prose.yaml").write_text("description: |\n  Confirmatory IDs are excluded.\nlr: 1e-5\n", encoding="utf-8")
    (configs / "path.yaml").write_text("eval:\n  prompts: release/x/Frozen_Prompts_v1.jsonl\n", encoding="utf-8")
    (configs / "key.json").write_text('{"confirmatory_split": "x"}', encoding="utf-8")
    (configs / "broken.yaml").write_text("a: [unclosed confirmatory\n", encoding="utf-8")
    (configs / "notes.md").write_text("we never touch the confirmatory set\n", encoding="utf-8")
    assert rr.config_state("configs/prose.yaml", tmp_path) == ""
    assert rr.config_state("configs/path.yaml", tmp_path) == "held_out"
    assert rr.config_state("configs/key.json", tmp_path) == "held_out"
    assert rr.config_state("configs/broken.yaml", tmp_path) == "held_out"  # unparsable: read whole
    assert rr.config_state("configs/notes.md", tmp_path) == "held_out"  # not structured: read whole
    assert rr.config_state("configs/../pyproject.toml", tmp_path) == ""  # nothing outside configs/ is read


def _checkout(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path


def test_a_new_script_that_names_held_out_data_is_never_uploaded(tmp_path: Path) -> None:
    # Not in any table (not tracked yet): the scan of the entry point on disk at start finds the marker.
    root = _checkout(tmp_path)
    (root / "scripts" / "new_eval.py").write_text('PROMPTS = "release/x/frozen_prompts_v1.jsonl"\n', encoding="utf-8")
    (root / "scripts" / "clean.py").write_text('print("ok")\n', encoding="utf-8")
    assert rr.recorded_call("script", ["scripts/new_eval.py"], root, lambda: 0) == 0
    assert rr.recorded_call("script", ["scripts/clean.py"], root, lambda: 0) == 0
    by_command = {r["command"][0]: r for r in rr.read_local(root)}
    assert by_command["scripts/new_eval.py"]["held_out_reach"] == ["scripts/new_eval.py"]
    assert by_command["scripts/clean.py"]["held_out_reach"] == []
    assert rs.upload_decision(by_command["scripts/new_eval.py"], root)[0] == "never"
    # Clean but not reviewed: the allowlist keeps it local until approved (owner, 2026-11-13).
    assert rs.upload_decision(by_command["scripts/clean.py"], root)[0] == "unapproved"


def test_a_self_referencing_yaml_config_does_not_hang(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "loop.yaml").write_text("a: &a [*a, release/frozen_prompts_v1.jsonl]\n", encoding="utf-8")
    started = time.monotonic()
    assert rr.config_state("configs/loop.yaml", tmp_path) == "held_out"
    assert time.monotonic() - started < 5


def test_a_spaced_config_value_counts_its_path_like_pieces(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "cmd.yaml").write_text("command: python scripts/score.py --input release/frozen_prompts.jsonl\n", encoding="utf-8")
    (configs / "prose.yaml").write_text("note: the confirmatory set is never used here\n", encoding="utf-8")
    assert rr.config_state("configs/cmd.yaml", tmp_path) == "held_out"
    assert rr.config_state("configs/prose.yaml", tmp_path) == ""



def test_an_unapproved_log_stays_local_and_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _checkout(tmp_path)
    (root / "scripts" / "clean.py").write_text('print("ok")\n', encoding="utf-8")
    assert rr.recorded_call("script", ["scripts/clean.py"], root, lambda: 0) == 0
    sent = _hub(tmp_path, monkeypatch, SimpleNamespace(private=True))
    monkeypatch.setattr(runs_cli, "upload_decision", rs.upload_decision)  # undo the helper's stand-in
    assert runs_cli.main(["upload", "--repo", "o/r"], root) == 3
    assert sent == {}
    assert "1 log(s) kept local, not approved for upload" in capsys.readouterr().out


MANIFEST = "reports/evaluation/behavioral-heldout-v2.manifest.json"


@pytest.mark.parametrize(
    ("body", "state"),
    [
        # Owner's decision, 2026-11-13 (Fable's and Gemini 4 Argon's advice): the set read as data keeps a log local.
        (f"evaluations:\n  behavioral_manifest: {MANIFEST}\n", "held_out"),
        ("calibration_dataset_id: behavioral-heldout-v2\n", "held_out"),
        (f"some_new_key: {MANIFEST}\n", "held_out"),  # unknown key: fail closed
        # Reviewed keys that document or exclude the set do not.
        (f"checkpoint_selection_manifest: {MANIFEST}\n", ""),
        (f"exclude_held_out_namespaces:\n  - {MANIFEST}\n  - bfcl_v4\n", ""),
        ("contamination_report: reports/data/behavioral-heldout-v2-contamination.json\n", ""),
        # Under an exempt key, the original markers still count.
        ("exclude_held_out_namespaces:\n  - reports/evaluation/behavioral-heldout-v2-partition.json\n", "held_out"),
    ],
)
def test_behavioral_heldout_counts_except_under_reviewed_keys(tmp_path: Path, body: str, state: str) -> None:
    pytest.importorskip("yaml")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "c.yaml").write_text(body, encoding="utf-8")
    assert rr.config_state("configs/c.yaml", tmp_path) == state


def test_an_argument_naming_the_held_out_manifest_keeps_the_log_local(tmp_path: Path) -> None:
    assert rr.recorded_call("training", ["opengrad", "train", f"--eval-manifest={MANIFEST}"], tmp_path, lambda: 0) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["held_out_inputs"] is True
    assert rs.upload_decision(record, tmp_path)[0] == "never"


def test_a_training_config_outside_configs_is_flagged_when_the_run_starts(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    (root / "scratch").mkdir()
    (root / "scratch" / "x.yaml").write_text("datasets:\n  manifest_paths: [elsewhere.json]\n", encoding="utf-8")
    assert rr.recorded_call("training", ["opengrad", "train", "scratch/x.yaml"], root, lambda: 0) == 0
    assert rr.recorded_call("training", ["opengrad", "train", "--steps", "3"], root, lambda: 0) == 0
    flagged = {tuple(r["command"]): r.get("unchecked_inputs", False) for r in rr.read_local(root)}
    assert flagged == {("opengrad", "train", "scratch/x.yaml"): True, ("opengrad", "train", "--steps", "3"): False}


def test_the_run_start_scan_lists_modified_files_it_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _checkout(tmp_path)
    (root / "scripts" / "tool.py").write_text("import helper\n", encoding="utf-8")
    (root / "scripts" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (root / "scripts" / "other.py").write_text("Y = 2\n", encoding="utf-8")
    monkeypatch.setattr(ri, "_modified", lambda root: {"scripts/helper.py", "scripts/other.py", "configs/a.yaml"})
    monkeypatch.setattr(ri, "_tracked", lambda root: {"scripts/tool.py", "scripts/helper.py", "scripts/other.py"})
    assert rr.recorded_call("script", ["scripts/tool.py"], root, lambda: 0) == 0
    (record,) = rr.read_local(root)
    assert record["entry_modified"] == ["configs/a.yaml", "scripts/helper.py"]
    monkeypatch.setattr(ri, "_modified", lambda root: None)
    assert rr.recorded_call("script", ["scripts/tool.py"], root, lambda: 0) == 0
    assert {tuple(r["entry_modified"]) for r in rr.read_local(root)} >= {("<git status failed>",)}


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_on_reviewed_branch_means_in_origin_masters_history(tmp_path: Path) -> None:
    import subprocess

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "a")
    pushed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    _git(tmp_path, "update-ref", "refs/remotes/origin/master", pushed)
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "b")
    local = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert rs.on_reviewed_branch(pushed, tmp_path) is True
    assert rs.on_reviewed_branch(local, tmp_path) is False  # committed but not pushed
    assert rs.on_reviewed_branch("not-a-sha; rm -rf /", tmp_path) is False
    assert rs.on_reviewed_branch(None, tmp_path) is False
    assert rs.on_reviewed_branch(pushed[:12], tmp_path) is False  # only a full commit id
    # A local branch or tag called origin/master is not the remote ref.
    _git(tmp_path, "branch", "origin/master", local)
    _git(tmp_path, "tag", "origin/master", local)
    assert rs.on_reviewed_branch(local, tmp_path) is False


def test_an_ignored_file_the_entry_point_loads_counts_as_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _checkout(tmp_path)
    (root / "scripts" / "tool.py").write_text("import helper\n", encoding="utf-8")
    (root / "scripts" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "build").mkdir()
    (root / "configs" / "build" / "x.yaml").write_text("a: 1\n", encoding="utf-8")
    monkeypatch.setattr(ri, "_modified", lambda root: set())  # git status hides ignored files
    monkeypatch.setattr(ri, "_tracked", lambda root: {"scripts/tool.py"})
    assert rr.recorded_call("script", ["scripts/tool.py"], root, lambda: 0) == 0
    (record,) = rr.read_local(root)
    assert record["entry_modified"] == ["configs/build/x.yaml", "scripts/helper.py"]


def test_a_yaml_alias_cannot_hide_a_reading_key_behind_an_exempt_one(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    configs = tmp_path / "configs"
    configs.mkdir()
    for body in (
        "evaluation: &x {manifest: data/behavioral-heldout/m.json}\ncheckpoint_selection_manifest: *x\n",
        "checkpoint_selection_manifest: &x [data/behavioral-heldout/m.json]\nevaluation: *x\n",
    ):
        (configs / "a.yaml").write_text(body, encoding="utf-8")
        assert rr.config_state("configs/a.yaml", tmp_path) == "held_out", body


def test_a_config_spelled_from_another_folder_is_not_trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = tmp_path / "configs" / "evaluation"
    configs.mkdir(parents=True)
    (configs / "x.yaml").write_text("prompts: release/frozen_prompts_v1.jsonl\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "configs")
    assert rr.unchecked_arguments(["tool", "evaluation/x.yaml"], tmp_path) is True
    monkeypatch.chdir(tmp_path)
    assert rr.unchecked_arguments(["tool", "configs/evaluation/x.yaml"], tmp_path) is False
    assert rr.config_state("configs/evaluation/x.yaml", tmp_path) == "held_out"


def test_a_file_git_is_told_to_ignore_edits_to_is_not_counted_as_tracked(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "a")
    _git(tmp_path, "update-index", "--skip-worktree", "b.py")
    _git(tmp_path, "update-index", "--assume-unchanged", "c.py")
    assert ri._tracked(tmp_path) == {"a.py"}


def test_an_edit_to_the_run_record_package_counts_as_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The scan does not follow imports out of the policy files, so the package's own files are watched directly.
    root = _checkout(tmp_path)
    (root / "scripts" / "tool.py").write_text("X = 1\n", encoding="utf-8")
    package = root / "src" / "opengrad" / "runrecords"
    package.mkdir(parents=True)
    (package / "inputs.py").write_text("Y = 2\n", encoding="utf-8")
    monkeypatch.setattr(ri, "_modified", lambda root: {"src/opengrad/runrecords/inputs.py"})
    monkeypatch.setattr(ri, "_tracked", lambda root: {"scripts/tool.py", "src/opengrad/runrecords/inputs.py"})
    assert rr.recorded_call("script", ["scripts/tool.py"], root, lambda: 0) == 0
    (record,) = rr.read_local(root)
    assert record["entry_modified"] == ["src/opengrad/runrecords/inputs.py"]


def test_a_local_branch_named_like_the_remote_ref_never_counts(tmp_path: Path) -> None:
    import subprocess

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "a")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    # No real refs/remotes/origin/master here: git's name lookup would fall back to this branch.
    _git(tmp_path, "branch", "refs/remotes/origin/master", head)
    assert rs.on_reviewed_branch(head, tmp_path) is False



def _commit(root: Path, message: str) -> str:
    import subprocess

    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_consumed_config_is_vouched_for_by_its_content_not_listed_as_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _checkout(tmp_path)
    (root / "scripts" / "tool.py").write_text("X = 1\n", encoding="utf-8")
    (root / "configs").mkdir()
    # The bytes the run reads are what is hashed (write_text would write CRLF on Windows).
    (root / "configs" / "run.yaml").write_bytes(b"steps: 3\n")
    (root / "configs" / "other.yaml").write_bytes(b"steps: 4\n")
    monkeypatch.setattr(ri, "_modified", lambda root: {"configs/run.yaml", "configs/other.yaml"})
    monkeypatch.setattr(
        ri, "_tracked", lambda root: {"scripts/tool.py", "configs/run.yaml", "configs/other.yaml"}
    )
    assert rr.recorded_call("script", ["scripts/tool.py", "configs/run.yaml"], root, lambda: 0) == 0
    (record,) = rr.read_local(root)
    # An edit to a config the run does not consume still counts.
    assert record["entry_modified"] == ["configs/other.yaml"]
    assert record["config_blobs"] == {"configs/run.yaml": ri.blob_sha(b"steps: 3\n")}


def test_blob_sha_is_what_git_hash_object_prints(tmp_path: Path) -> None:
    import subprocess

    (tmp_path / "a.yaml").write_bytes(b"steps: 3\n")
    out = subprocess.run(
        ["git", "hash-object", "a.yaml"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert ri.blob_sha(b"steps: 3\n") == out.stdout.strip()


def test_a_tweaked_config_uploads_once_that_exact_version_is_pushed(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "run.yaml"
    config.write_bytes(b"steps: 3\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "core.autocrlf", "false")
    _git(tmp_path, "update-ref", "refs/remotes/origin/master", _commit(tmp_path, "a"))
    pushed = ri.blob_sha(b"steps: 3\n")
    tweaked = ri.blob_sha(b"steps: 5\n")
    assert rs.configs_pushed({"configs/run.yaml": pushed}, tmp_path) is True
    # A run made with an unpushed edit, then with the edit committed but not pushed.
    assert rs.configs_pushed({"configs/run.yaml": tweaked}, tmp_path) is False
    config.write_bytes(b"steps: 5\n")
    local = _commit(tmp_path, "b")
    assert rs.configs_pushed({"configs/run.yaml": tweaked}, tmp_path) is False
    _git(tmp_path, "update-ref", "refs/remotes/origin/master", local)
    assert rs.configs_pushed({"configs/run.yaml": tweaked}, tmp_path) is True
    # A version pushed earlier was reviewed too.
    assert rs.configs_pushed({"configs/run.yaml": pushed}, tmp_path) is True
    # The same content pushed under another path does not vouch for this one, and a recorded path is never a
    # pattern that could match a different file.
    assert rs.configs_pushed({"configs/elsewhere.yaml": pushed}, tmp_path) is False
    assert rs.configs_pushed({"configs/*.yaml": pushed}, tmp_path) is False
    assert rs.configs_pushed({":(glob)configs/**": pushed}, tmp_path) is False
    assert rs.configs_pushed({"configs/run.yaml": pushed[:12]}, tmp_path) is False
    assert rs.configs_pushed(None, tmp_path) is False  # consumed configs unknown
    assert rs.configs_pushed("x", tmp_path) is False
    assert rs.configs_pushed({}, tmp_path) is True


def test_a_tweaked_config_counts_in_the_reviewed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rs, "on_reviewed_branch", lambda sha, root: True)
    monkeypatch.setattr(
        rs, "configs_pushed", lambda blobs, root: blobs == {"configs/run.yaml": "pushed"}
    )
    base = {"held_out_reach": [], "entry_modified": [], "git": {"sha": "a" * 40}}
    assert (
        rs._as_reviewed({**base, "config_blobs": {"configs/run.yaml": "pushed"}}, tmp_path) is True
    )
    assert (
        rs._as_reviewed({**base, "config_blobs": {"configs/run.yaml": "local"}}, tmp_path) is False
    )


def test_evaluate_consumes_its_suite_and_the_benchmark_configs_it_lists(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    root = _checkout(tmp_path)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\n[project.scripts]\nopengrad = "opengrad.cli:main"\n', encoding="utf-8"
    )
    (root / "src" / "opengrad").mkdir()
    (root / "src" / "opengrad" / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    for folder in ("benchmark_suites", "benchmarks"):
        (root / "configs" / folder).mkdir(parents=True)
    (root / "configs" / "benchmark_suites" / "smoke.yaml").write_text(
        "benchmarks: [a, b-c]\n", encoding="utf-8"
    )
    (root / "configs" / "benchmark_suites" / "wide.yaml").write_text(
        "benchmarks: [a, missing]\n", encoding="utf-8"
    )
    (root / "configs" / "benchmarks" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    (root / "configs" / "benchmarks" / "b_c.yaml").write_text("x: 2\n", encoding="utf-8")
    assert ri.consumed_configs(["opengrad", "evaluate", "m"], root) == [
        "configs/benchmark_suites/smoke.yaml",
        "configs/benchmarks/a.yaml",
        "configs/benchmarks/b_c.yaml",
    ]
    assert (
        ri.consumed_configs(["opengrad", "evaluate", "m", "--suite", "wide"], root) is None
    )  # fails closed
    assert ri.consumed_configs(["opengrad", "evaluate", "m", "--suite=none"], root) is None
    assert ri.config_blobs(["opengrad", "evaluate", "m", "--suite", "wide"], root) is None
