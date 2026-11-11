"""opengrad.runrecords redaction, second audit round: credentials in headers, dotted names, mid-string assignments
and quoted values are removed, while ordinary settings such as ``--max-tokens 512`` keep their values."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from opengrad.runrecords import record as rr


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
        (["curl", "-H", "X-Api-Key: abc123"], "abc123"),
        (["--set=api_key=abc456"], "abc456"),
        (["db.password=abc789"], "abc789"),
        (["bash", "-c", "export X=1; HF_TOKEN=tok321 python x.py"], "tok321"),
        (['{"password": "ab\'c999"}'], "c999"),
        (["--hf-token", "t1"], "t1"),
        (["--api-key", "k2"], "k2"),
        (["--wandb-key", "k3"], "k3"),
    ],
)
def test_credentials_in_any_form_are_removed(tmp_path: Path, argv: list[str], leak: str) -> None:
    assert leak not in " ".join(rr.redact_command(argv, tmp_path))


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-tokens", "512"],
        ["--author", "someone-name"],
        ["--monkey", "patch"],
        ["--passthrough", "x"],
        ["max_tokens=512"],
    ],
)
def test_ordinary_settings_keep_their_values(tmp_path: Path, argv: list[str]) -> None:
    assert rr.redact_command(argv, tmp_path) == argv


def test_a_recorder_that_cannot_start_still_runs_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: rr.RunRecord) -> None:
        raise PermissionError("read-only folder")

    monkeypatch.setattr(rr.RunRecord, "begin", fail)
    assert rr.run_child("script", [sys.executable, "-c", "raise SystemExit(5)"], tmp_path) == 5


@pytest.mark.parametrize(
    ("argv", "leak"),
    [
        (["--accessToken", "abc111"], "abc111"),
        (['{"secretKey":"abc222"}'], "abc222"),
        (["--HFTOKEN", "abc333"], "abc333"),
        (["MYTOKEN=abc444", "python"], "abc444"),
        (['{"password": "ab\\"c555"}'], "c555"),
    ],
)
def test_joined_and_escaped_credentials_are_removed(
    tmp_path: Path, argv: list[str], leak: str
) -> None:
    assert leak not in " ".join(rr.redact_command(argv, tmp_path))


def test_a_switch_does_not_hide_the_next_argument(tmp_path: Path) -> None:
    argv = ["--no-auth", "configs/a.yaml", "--use-auth", "configs/b.yaml", "--tokenizer", "qwen"]
    assert rr.redact_command(argv, tmp_path) == argv


def test_a_recorded_call_inside_a_recorded_child_makes_one_record(tmp_path: Path) -> None:
    child = (
        "from pathlib import Path; from opengrad.runrecords import recorded_call; "
        "recorded_call('training', ['opengrad', 'train'], Path('.'), lambda: (print('inner'), 0)[1])"
    )
    quiet = type(
        "Q", (), {"write": lambda self, t: None, "flush": lambda self: None, "encoding": "utf-8"}
    )()
    assert rr.run_child("training", [sys.executable, "-c", child], tmp_path, quiet) == 0
    records = rr.read_local(tmp_path)
    assert len(records) == 1
    assert "inner" in (tmp_path / records[0]["log"]["path"]).read_text(encoding="utf-8")


def test_a_stale_parent_marker_does_not_switch_recording_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(rr.PARENT_ENV, "20990101T000000000000Z-training-stale0")
    assert rr.recorded_call("training", ["opengrad", "train"], tmp_path, lambda: 0) == 0
    assert len(rr.read_local(tmp_path)) == 1


def test_an_open_parent_record_is_the_only_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = rr.RunRecord(tmp_path, "training", ["python", "-m", "opengrad.runrecords", "run"])
    parent.begin()
    monkeypatch.setenv(rr.PARENT_ENV, parent.run_id)
    assert rr.recorded_call("training", ["opengrad", "train"], tmp_path, lambda: 0) == 0
    assert [r["run_id"] for r in rr.read_local(tmp_path)] == [parent.run_id]


def test_an_exception_name_keeps_its_message(tmp_path: Path) -> None:
    assert rr.redact_text("KeyError: 'token'", tmp_path) == "KeyError: 'token'"
    assert "abc" not in rr.redact_text("api_key: abc", tmp_path)
