"""opengrad.runrecords, the cases an audit found: redaction gaps, Windows output encoding, a recorder that fails,
killed runs, and paths in error text. Only harmless dummy commands are recorded; the Hub is stubbed."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from opengrad.runrecords import record as rr
from opengrad.runrecords.cli import main as runs_main
from opengrad.runrecords.cli import validate_summary

ROOT = Path(__file__).resolve().parents[2]
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


@pytest.mark.parametrize(
    ("argv", "leak"),
    [
        (["--pass", "hunter2"], "hunter2"),
        (["--passwd", "hunter2"], "hunter2"),
        (["--access-key", "AK123"], "AK123"),
        (["--wandb-key", "w123"], "w123"),
        (["--key", "AIza" + "x" * 35], "AIza"),
        (["--auth", "Bearer abcdefghijklmnop"], "abcdefghijklmnop"),
        (["env", "HF_TOKEN=abc123", "python"], "abc123"),
        (['--json={"api_key": "xyz789"}'], "xyz789"),
        (["https://user:s3cret@example.org/repo.git"], "s3cret"),
        (["C:\\Program Files\\x y.yaml"], "Program"),
        (["-o/home/someone/x"], "someone"),
        (["a.yaml,/home/someone/x"], "someone"),
        (["\\\\server\\share\\x"], "server"),
        (["out:C:\\Users\\someone\\x"], "someone"),
    ],
)
def test_the_command_leaks_nothing_private(tmp_path: Path, argv: list[str], leak: str) -> None:
    redacted = " ".join(rr.redact_command(argv, tmp_path))
    assert leak not in redacted, redacted


def test_a_sibling_of_the_repository_is_not_made_relative(tmp_path: Path) -> None:
    root = tmp_path / "OpenGrad"
    root.mkdir()
    sibling = str(tmp_path / "OpenGradX" / "file.yaml")
    assert rr.redact_command([sibling], root) == ["<absolute-path>"]
    assert rr.redact_command([str(root / "configs" / "a.yaml")], root) == ["configs/a.yaml"]


def test_an_error_naming_a_path_is_redacted_and_still_exportable(tmp_path: Path) -> None:
    (tmp_path / "docs" / "upstream").mkdir(parents=True)
    (tmp_path / "docs" / "upstream" / "UP-0001.md").write_text("x", encoding="utf-8")

    def missing() -> int:
        raise FileNotFoundError("[Errno 2] No such file or directory: 'C:\\Users\\someone\\x.yaml'")

    with pytest.raises(FileNotFoundError):
        rr.recorded_call("training", ["opengrad", "train"], tmp_path, missing)
    (record,) = rr.read_local(tmp_path)
    assert "someone" not in record["error"]
    assert runs_main(["dispose", record["run_id"], "--upstream", "UP-0001"], tmp_path) == 0
    assert runs_main(["export"], tmp_path) == 0


def test_a_non_ascii_child_is_recorded_in_utf8_on_every_platform(tmp_path: Path) -> None:
    code = rr.run_child("script", [PY, "-c", "print('step \u2192 done')"], tmp_path, QUIET)
    assert code == 0
    (record,) = rr.read_local(tmp_path)
    assert "\u2192" in (tmp_path / record["log"]["path"]).read_text(encoding="utf-8")


def test_a_console_that_cannot_show_the_output_does_not_lose_the_record(tmp_path: Path) -> None:
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    assert rr.run_child("script", [PY, "-c", "print('\u2192')"], tmp_path, console) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["status"] == "PASS"

    class Broken:
        encoding = "utf-8"

        def write(self, text: str) -> int:
            raise OSError("console gone")

        def flush(self) -> None:
            pass

    assert rr.run_child("script", [PY, "-c", "print('still logged')"], tmp_path, Broken()) == 0  # type: ignore[arg-type]
    newest = rr.read_local(tmp_path)[0]
    assert newest["status"] == "PASS"
    assert "still logged" in (tmp_path / newest["log"]["path"]).read_text(encoding="utf-8")


def test_a_recorder_that_cannot_start_never_blocks_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: rr.RunRecord) -> None:
        raise PermissionError("read-only folder")

    monkeypatch.setattr(rr.RunRecord, "begin", fail)
    assert rr.recorded_call("training", ["opengrad", "train"], tmp_path, lambda: 7) == 7


def test_bytes_written_through_the_buffer_are_captured(tmp_path: Path) -> None:
    def call() -> int:
        sys.stdout.buffer.write(b"via buffer\n")
        sys.stdout.flush()
        return 0

    rr.recorded_call("evaluation", ["opengrad", "evaluate"], tmp_path, call)
    (record,) = rr.read_local(tmp_path)
    assert "via buffer" in (tmp_path / record["log"]["path"]).read_text(encoding="utf-8")


def test_a_killed_run_can_be_disposed_and_exported(tmp_path: Path) -> None:
    run = rr.RunRecord(tmp_path, "training", ["opengrad", "train", "x.yaml"])
    run.begin()
    assert runs_main(["pending"], tmp_path) == 1
    assert runs_main(["dispose", run.run_id, "--reason", "killed while testing"], tmp_path) == 0
    assert runs_main(["pending"], tmp_path) == 0
    assert runs_main(["export"], tmp_path) == 0
    exported = (tmp_path / "run-records" / f"{run.run_id}.json").read_text(encoding="utf-8")
    assert '"INCOMPLETE"' in exported


@pytest.mark.parametrize(
    "text",
    [
        "file:///C:/x",
        "(C:\\\\Users\\\\x)",
        "\\\\\\\\server\\\\share",
        "/opt/models/x",
        "someone@example.org",
    ],
)
def test_a_committed_summary_with_a_path_or_email_fails(text: str) -> None:
    summary = {
        "schema_version": 1,
        "run_id": "r",
        "kind": "script",
        "command": ["python", text],
        "started_at": "t",
        "git": {},
        "status": "PASS",
        "log": {"sha256": "0" * 64},
        "disposition": None,
    }
    assert validate_summary(summary, ROOT), text


def test_ordinary_summary_content_is_not_flagged() -> None:
    summary = {
        "schema_version": 1,
        "run_id": "r",
        "kind": "evaluation",
        "command": ["opengrad", "evaluate", "<absolute-path>", "--device=cuda:0", "configs/a.yaml"],
        "started_at": "t",
        "git": {"sha": "a" * 40},
        "status": "PASS",
        "distributions": {"torch": "2.13.0+cu128", "email-validator": "2.2.0"},
        "gpu": {"devices": ["NVIDIA H200, 570.86, 143771 MiB"]},
        "log": {"path": ".run-records/r/output.log", "sha256": "0" * 64},
        "disposition": None,
    }
    assert validate_summary(summary, ROOT) == []


def test_the_suite_never_writes_run_records_into_the_repository() -> None:
    assert not (ROOT / ".run-records").exists() or not any((ROOT / ".run-records").iterdir())
