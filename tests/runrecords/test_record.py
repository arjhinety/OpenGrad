"""opengrad.runrecords: a full record per run, exercised only with harmless dummy commands.

No test here runs training, evaluation, Modal or any GPU work, and none can reach the Hugging Face Hub: the
module is replaced with a stub (UP-0030).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from opengrad.hashing import sha256_file
from opengrad.runrecords import cli as runs_cli_module
from opengrad.runrecords import record as rr
from opengrad.runrecords.cli import main as runs_main
from opengrad.runrecords.cli import validate_summary

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


class _NoNetworkHfApi:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a test reached the real Hugging Face client")


@pytest.fixture(autouse=True)
def _no_real_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = ModuleType("huggingface_hub")
    stub.HfApi = _NoNetworkHfApi  # type: ignore[attr-defined]
    stub.hf_hub_download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("real download"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    # These tests check the default layout under each test's own root, not the suite-wide redirect.
    monkeypatch.delenv(rr.LOCAL_DIR_ENV, raising=False)


def _scanner_secret() -> str:
    spec = importlib.util.spec_from_file_location(
        "check_publication_hygiene", ROOT / "scripts/repo/check_publication_hygiene.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SECRET.pattern


def test_the_secret_pattern_is_the_publication_scanners() -> None:
    assert rr.SECRET.pattern == _scanner_secret()


def test_the_command_is_redacted(tmp_path: Path) -> None:
    token = "hf_" + "a" * 34
    argv = [
        PY,
        "scripts/x.py",
        str(tmp_path / "configs" / "a.yaml"),
        "--api-key",
        "plain-value",
        "--token=xyz",
        f"--note={token}",
        "--out=/home/someone/out",
    ]
    out = rr.redact_command(argv, tmp_path)
    assert out[1] == "scripts/x.py"
    assert out[2] == "configs/a.yaml"
    assert out[3:5] == ["--api-key", "<redacted>"]
    assert out[5] == "--token=<redacted>"
    assert "hf_" not in out[6]
    assert out[7] == "--out=<absolute-path>"
    assert out[0] == "<absolute-path>"


def test_a_child_is_recorded_with_its_output_and_exit_code(tmp_path: Path) -> None:
    console = SimpleNamespace(lines=[], write=lambda t: console.lines.append(t), flush=lambda: None)
    code = rr.run_child(
        "script", [PY, "-c", "print('hello'); raise SystemExit(3)"], tmp_path, console
    )
    assert code == 3
    (record,) = rr.read_local(tmp_path)
    assert record["status"] == "FAILED" and record["exit_code"] == 3
    log = tmp_path / record["log"]["path"]
    assert "hello" in log.read_text(encoding="utf-8")
    assert record["log"]["sha256"] == sha256_file(log)
    assert "hello" in "".join(console.lines)
    assert record["distributions"] and record["python"]


def test_output_printed_before_an_abrupt_exit_is_kept(tmp_path: Path) -> None:
    child = "import os, sys; print('partial'); sys.stdout.flush(); os._exit(9)"
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    assert rr.run_child("script", [PY, "-c", child], tmp_path, console) == 9
    (record,) = rr.read_local(tmp_path)
    assert "partial" in (tmp_path / record["log"]["path"]).read_text(encoding="utf-8")


def test_a_start_without_an_end_reads_as_incomplete_and_needs_a_disposition(tmp_path: Path) -> None:
    run = rr.RunRecord(tmp_path, "training", ["opengrad", "train", "x.yaml"])
    run.begin()
    (record,) = rr.read_local(tmp_path)
    assert record["status"] == "INCOMPLETE"
    assert any(
        "disposition" in e
        for e in validate_summary({**record, "log": {}, "disposition": None}, ROOT)
    )


def test_an_in_process_call_is_recorded_and_its_outcome_passes_through(tmp_path: Path) -> None:
    assert (
        rr.recorded_call(
            "evaluation", ["opengrad", "evaluate"], tmp_path, lambda: (print("ok"), 0)[1]
        )
        == 0
    )

    def boom() -> int:
        raise ValueError("bad config")

    with pytest.raises(ValueError, match="bad config"):
        rr.recorded_call("training", ["opengrad", "train"], tmp_path, boom)

    def exits() -> int:
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        rr.recorded_call("training", ["opengrad", "train"], tmp_path, exits)
    by_kind = {(r["kind"], r["status"], r["exit_code"]) for r in rr.read_local(tmp_path)}
    assert ("evaluation", "PASS", 0) in by_kind
    assert ("training", "FAILED", None) in by_kind
    assert ("training", "FAILED", 2) in by_kind
    ok = next(r for r in rr.read_local(tmp_path) if r["status"] == "PASS")
    assert "ok" in (tmp_path / ok["log"]["path"]).read_text(encoding="utf-8")


def test_recording_does_not_change_git_status_in_a_checkout_that_ignores_the_local_folder(
    tmp_path: Path,
) -> None:
    assert ".run-records/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    git = ["git", "-c", "user.email=t@example.org", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".run-records/\n", encoding="utf-8")
    subprocess.run([*git, "add", "."], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=tmp_path, check=True)
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    rr.run_child("script", [PY, "-c", "print(1)"], tmp_path, console)
    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout
    assert before == after == ""


def test_export_refuses_a_failed_run_until_it_is_disposed(tmp_path: Path) -> None:
    (tmp_path / "docs" / "upstream").mkdir(parents=True)
    (tmp_path / "docs" / "upstream" / "UP-0031.md").write_text("x", encoding="utf-8")
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    rr.run_child("script", [PY, "-c", "raise SystemExit(1)"], tmp_path, console)
    (record,) = rr.read_local(tmp_path)
    assert runs_main(["export"], tmp_path) == 1
    assert not (tmp_path / "run-records").exists()
    assert runs_main(["dispose", record["run_id"], "--upstream", "UP-9999"], tmp_path) == 1
    assert runs_main(["dispose", record["run_id"], "--upstream", "UP-0031"], tmp_path) == 0
    assert runs_main(["export"], tmp_path) == 0
    exported = json.loads(
        (tmp_path / "run-records" / f"{record['run_id']}.json").read_text(encoding="utf-8")
    )
    assert exported["disposition"] == {"upstream": "UP-0031"}
    assert validate_summary(exported, tmp_path) == []


def _fake_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private: bool = True
) -> dict[str, bytes]:
    sent: dict[str, bytes] = {}

    class FakeApi:
        def repo_info(self, repo: str, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(private=private)

        def upload_file(self, **kwargs: object) -> SimpleNamespace:
            sent[str(kwargs["path_in_repo"])] = kwargs["path_or_fileobj"]  # type: ignore[assignment]
            return SimpleNamespace(oid=f"rev{len(sent)}")

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        path = tmp_path / "dl" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sent[filename])
        return str(path)

    monkeypatch.setattr(sys.modules["huggingface_hub"], "HfApi", FakeApi)
    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", fake_download)
    # These tests are about redaction: their ``python -c`` commands stand in for repository scripts. The
    # fail-closed rule for commands that cannot be traced has its own tests (test_scripts_policy.py).
    real = runs_cli_module.upload_decision

    def traced(record: dict, root: Path) -> tuple[str, str]:
        decision, kind = real(record, root)
        return ("allowed", kind) if decision == "unrecognised" else (decision, kind)

    monkeypatch.setattr(runs_cli_module, "upload_decision", traced)
    return sent


def test_a_clean_log_is_uploaded_whole_and_a_private_one_as_a_redacted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    rr.run_child("script", [PY, "-c", "print('clean')"], tmp_path, console)
    token = "hf_" + "b" * 34
    rr.run_child(
        "script", [PY, "-c", f"print('{token} and someone@example.org')"], tmp_path, console
    )
    sent = _fake_hub(tmp_path, monkeypatch)
    assert runs_main(["upload", "--repo", "owner/run-logs"], tmp_path) == 0
    assert len(sent) == 2
    by_copy = {r["log"]["durable"]["copy"]: r for r in rr.read_local(tmp_path)}
    clean, private = by_copy["original"], by_copy["redacted"]
    assert sent[clean["log"]["durable"]["path"]].strip() == b"clean"
    redacted = sent[private["log"]["durable"]["path"]].decode("utf-8")
    assert token not in redacted and "someone@example.org" not in redacted
    assert set(private["log"]["durable"]["redacted_because"]) == {"a secret", "an email address"}
    # The summary keeps the original log's hash and the hash of the copy that was sent.
    assert private["log"]["sha256"] != private["log"]["durable"]["sha256"]
    assert private["log"]["durable"]["path"].endswith("output.redacted.log")


def test_training_and_evaluation_record_themselves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opengrad import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "handle_train", lambda args, root: 0)
    monkeypatch.setattr(cli, "_cmd_evaluate", lambda args, root: 4)
    monkeypatch.setattr(sys, "argv", ["opengrad", "train", "configs/x.yaml", "--dry-run"])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["opengrad", "evaluate", "some-model", "--dry-run"])
    assert cli.main() == 4
    kinds = {(r["kind"], r["status"]) for r in rr.read_local(tmp_path)}
    assert kinds == {("training", "PASS"), ("evaluation", "FAILED")}


def test_the_home_path_pattern_is_the_publication_scanners() -> None:
    from opengrad.runrecords import cli as runs_cli

    spec = importlib.util.spec_from_file_location(
        "check_publication_hygiene", ROOT / "scripts/repo/check_publication_hygiene.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert runs_cli.HOME_PATH.pattern == module.HOME_PATH.pattern


def test_the_home_path_pattern_stays_fast_on_a_long_backslash_run() -> None:
    import time

    # UP-0036: an alternation of one and two backslashes under + took exponential time here.
    # 40 backslashes took the old form about 18 s, so a return of the bug fails here instead of hanging.
    for text in ("C:" + "\\" * 40, "C:" + "/" * 5_000, "x" + "\\" * 5_000):
        started = time.monotonic()
        rr.HOME_PATH.search(text)
        assert time.monotonic() - started < 1.0, text[:4]


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("contact someone@example.org for access", "an email address"),
        ("cache at /home/someone/.cache/x", "a home-folder path"),
        ("key=AIza" + "B" * 35, "a Google API key"),
        ("Authorization: Bearer " + "c" * 30, "a bearer token"),
        ("token hf_" + "d" * 34, "a secret"),
    ],
)
def test_upload_blocks_private_details_beyond_the_token_list(text: str, finding: str) -> None:
    from opengrad.runrecords.cli import private_findings

    assert finding in private_findings(text)
    assert private_findings("step 10 loss 0.42") == []


def test_an_annotation_log_is_always_a_redacted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    rr.run_child("annotation", [PY, "-c", "print('label batch')"], tmp_path, console)
    sent = _fake_hub(tmp_path, monkeypatch)
    assert runs_main(["upload", "--repo", "owner/run-logs"], tmp_path) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"]["copy"] == "redacted"
    assert record["log"]["durable"]["redacted_because"] == ["annotation run"]
    assert list(sent) == [record["log"]["durable"]["path"]]


def test_upload_refuses_a_repository_that_is_not_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = SimpleNamespace(write=lambda t: None, flush=lambda: None)
    rr.run_child("script", [PY, "-c", "print('clean')"], tmp_path, console)
    sent = _fake_hub(tmp_path, monkeypatch, private=False)
    assert runs_main(["upload", "--repo", "owner/run-logs"], tmp_path) == 2
    assert sent == {}
    (record,) = rr.read_local(tmp_path)
    assert record["log"]["durable"] is None
