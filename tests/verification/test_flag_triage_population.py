"""The flag-set triage's populations, tasks and report (46 §5, 47 §3). Counts and hashes only; no item text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.annotation.config import load_task_config
from opengrad.annotation.items import load_source, project
from opengrad.annotation.service_support import load_instructions
from opengrad.hashing import sha256_bytes
from opengrad.verification import flag_set as fs
from opengrad.verification import flag_triage as ft
from opengrad.verification import flag_triage_population as pop
from opengrad.verification import flag_triage_report as rep

ROOT = Path(__file__).parents[2]
TASKS = {
    "first-reply-review-v1-trial": ("triage", 100),
    "first-reply-review-v1": ("triage", 19051),
    "first-reply-review-v1-sample": ("recall", 400),
}
#: What must never reach a labeller: it says how the items were chosen.
LEAKS = (
    "glaive",
    "when2call",
    "toolace",
    "xlam",
    "classifier",
    "flag set",
    "flagged",
    "recall",
    "unsupported",
    "triage",
    "study 002",
    "opengrad",
)


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        (ROOT / pop.OUTPUT_DIR / pop.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    return manifest


def _populations_present() -> bool:
    return all(
        (ROOT / pop.OUTPUT_DIR / name).is_file() for name in (pop.TRIAGE_NAME, pop.RECALL_NAME)
    )


def test_the_populations_are_the_flag_set_and_the_adopted_draws() -> None:
    manifest = _manifest()
    flag_manifest = json.loads(
        (ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    triage, recall = manifest["populations"]["triage"], manifest["populations"]["recall"]
    flags = flag_manifest["flag_set"]["by_source"]
    assert triage["records"] == flag_manifest["flag_set"]["n"] and triage["by_source"] == flags
    assert triage["trial"]["by_source"] == ft.trial_allocation(
        flags, ft.TRIAL_SIZE, ft.TRIAL_MINIMUM
    )
    assert triage["trial"]["records"] == ft.TRIAL_SIZE
    population = ft.recall_population(flag_manifest["precision_reporting"])
    assert recall["population_by_source"] == population
    assert recall["allocation"] == ft.trial_allocation(
        population, ft.RECALL_SIZE, ft.RECALL_MINIMUM
    )
    assert recall["records"] == ft.RECALL_SIZE
    assert manifest["flag_set"]["members_sha256"] == flag_manifest["members"]["sha256"]
    assert manifest["gold_labels_present"] is False
    for path, digest in manifest["code_sha256_lf"].items():
        assert fs._lf_sha256(ROOT, path) == digest, path


def test_the_populations_verify_or_report_their_input_missing() -> None:
    if not _populations_present():
        pytest.skip(
            "BLOCKED_INPUT_MISSING: the populations are rebuilt locally from the pinned release"
        )
    result = pop.verify(ROOT, rebuild=False)
    assert result["errors"] == [] and result["status"] == "PASS"


def test_item_ids_and_task_ids_say_nothing_about_how_items_were_chosen() -> None:
    assert pop.triage_id("x").startswith("r1:") and pop.recall_id("x").startswith("r2:")
    assert pop.triage_id("x") != pop.recall_id("x")
    for task in TASKS:
        assert not any(term in task for term in ("flag", "recall", "triage"))
    procedure = (ROOT / "configs/annotation/first-reply-review-v1.model-procedure.md").read_text(
        encoding="utf-8"
    )
    lowered = procedure.casefold()
    assert not [term for term in LEAKS if term in lowered]


@pytest.mark.parametrize("task", sorted(TASKS))
def test_each_task_is_blind_and_pins_its_set_and_procedure(task: str) -> None:
    manifest = _manifest()
    key, count = TASKS[task]
    path = ROOT / f"configs/annotation/{task}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_task_config(path, root=ROOT)
    assert cfg.metadata == () and cfg.filters == ()
    assert set(manifest["blinded_fields"]) <= set(cfg.blind_fields)
    assert config["labels"] == list(ft.LABELS)
    assert config["source"]["path"].endswith(manifest["populations"][key]["file"])
    assert config["source"]["expected_sha256"] == manifest["populations"][key]["sha256"]
    assert config["source"]["expected_items"] == count
    assert ("select" in config["source"]) == task.endswith("-trial")
    assert {field["path"] for field in config["fields"].values()} == set(manifest["shown_fields"])
    procedure = ROOT / "configs/annotation/first-reply-review-v1.model-procedure.md"
    digest = sha256_bytes(procedure.read_bytes())
    assert [(a["annotator_id"], a["procedure_sha256"]) for a in config["model_annotators"]] == [
        ("model.gemini-3.8-flash-high", digest),
        ("model.deepseek-v4.1-flash", digest),
    ]
    # The rubric the labellers see: 46 §5's definitions, and nothing that says how items were chosen.
    served = " ".join(doc["content"] for doc in load_instructions(cfg))
    assert (
        "it declines, but a capable assistant could answer correctly from general knowledge or"
        in served
    )
    assert not [term for term in LEAKS if term in served.casefold()]


@pytest.mark.parametrize("task", ["first-reply-review-v1-trial", "first-reply-review-v1-sample"])
def test_every_served_item_carries_no_blinded_key(task: str) -> None:
    if not _populations_present():
        pytest.skip(
            "BLOCKED_INPUT_MISSING: the populations are rebuilt locally from the pinned release"
        )
    cfg = load_task_config(ROOT / f"configs/annotation/{task}.yaml", root=ROOT)
    _, items = load_source(cfg)
    assert len(items) == TASKS[task][1]
    blinded = [json.dumps(name) + ":" for name in cfg.blind_fields]
    for item in items:
        view = project(cfg, item.row)
        assert set(view["fields"]) == {"user", "assistant", "tools"}
        served = json.dumps(view)
        assert not any(key in served for key in blinded)


def _rows(source_counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "triage_id": f"r1:{s}{i}",
            "opengrad_id": f"{s}-{i}",
            "source_dataset": s,
            "in_trial": "no",
        }
        for s, n in source_counts.items()
        for i in range(n)
    ]


def test_the_report_maps_opaque_ids_back_and_needs_both_votes() -> None:
    rows = _rows({"toolace": 2})
    j, n = "DECLINE_JUSTIFIED", "NOT_A_DECLINE"
    refs = [
        {
            "triage_id": "r1:toolace0",
            "votes": {"model.deepseek-v4.1-flash": j, "model.gemini-3.8-flash-high": j},
        },
        {
            "triage_id": "r1:toolace1",
            "votes": {"model.deepseek-v4.1-flash": n, "model.gemini-3.8-flash-high": j},
        },
    ]
    labels = rep.labels_by_record(rows, refs, "triage_id")
    assert labels == {"toolace-0": (j, j), "toolace-1": (n, j)}
    with pytest.raises(rep.FlagTriageReportError):  # one vote missing
        rep.labels_by_record(
            rows, [refs[0], {**refs[1], "votes": {"model.deepseek-v4.1-flash": n}}], "triage_id"
        )
    with pytest.raises(rep.FlagTriageReportError):  # an item left unlabelled
        rep.labels_by_record(rows, refs[:1], "triage_id")


def test_the_report_counts_agreed_declines_despite_a_tool_per_source() -> None:
    rows = _rows({"glaive": 2, "toolace": 2})
    u = {"reference_label": "UNKNOWN"}
    refs = [
        {"triage_id": "r1:glaive0", **u, "reference_ambiguity_status": rep.DECLINED_DESPITE_TOOL},
        {"triage_id": "r1:glaive1", **u, "reference_ambiguity_status": "AMBIGUOUS_TWO_MODES"},
        {"triage_id": "r1:toolace0", **u, "reference_ambiguity_status": rep.DECLINED_DESPITE_TOOL},
        # Two different reasons: no agreed reason, so not counted.
        {"triage_id": "r1:toolace1", **u, "reference_ambiguity_status": None},
    ]
    assert rep.declined_despite_tool(rows, refs, "triage_id") == {"glaive": 1, "toolace": 1}


def test_the_reason_code_is_offered_and_served() -> None:
    for task in TASKS:
        config = yaml.safe_load(
            (ROOT / f"configs/annotation/{task}.yaml").read_text(encoding="utf-8")
        )
        (status,) = [f for f in config["extra_fields"] if f["key"] == "ambiguity_status"]
        assert rep.DECLINED_DESPITE_TOOL in status["options"]
        served = " ".join(
            doc["content"]
            for doc in load_instructions(
                load_task_config(ROOT / f"configs/annotation/{task}.yaml", root=ROOT)
            )
        )
        assert rep.DECLINED_DESPITE_TOOL in served
    procedure = (ROOT / "configs/annotation/first-reply-review-v1.model-procedure.md").read_text(
        encoding="utf-8"
    )
    assert procedure.count(rep.DECLINED_DESPITE_TOOL) == 2


def test_the_report_refuses_a_reference_built_elsewhere(tmp_path: Path) -> None:
    out = tmp_path / pop.OUTPUT_DIR
    out.mkdir(parents=True)
    population = b'{"triage_id": "r1:a"}\n'
    (out / pop.TRIAGE_NAME).write_bytes(population)
    (out / pop.RECALL_NAME).write_bytes(b"")
    manifest = {
        "populations": {
            "triage": {"file": pop.TRIAGE_NAME, "sha256": sha256_bytes(population)},
            "recall": {"file": pop.RECALL_NAME, "sha256": sha256_bytes(b"")},
        }
    }
    (out / pop.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(rep.FlagTriageReportError, match="no reference"):
        rep.load(tmp_path, "triage")
    reference_dir = tmp_path / rep.REFERENCE_DIR
    reference_dir.mkdir(parents=True)
    reference = b'{"triage_id": "r1:a", "votes": {}}\n'
    (reference_dir / "first-reply-review-v1.reference.jsonl").write_bytes(reference)
    (reference_dir / "first-reply-review-v1.reference.manifest.json").write_text(
        json.dumps({"reference_sha256": sha256_bytes(reference), "population_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(rep.FlagTriageReportError, match="different population"):
        rep.load(tmp_path, "triage")


def test_absent_populations_are_a_missing_input_not_a_crash(tmp_path: Path) -> None:
    out = tmp_path / pop.OUTPUT_DIR
    out.mkdir(parents=True)
    (out / pop.MANIFEST_NAME).write_text(json.dumps({"populations": {}}), encoding="utf-8")
    result = pop.verify(tmp_path, rebuild=False)
    assert result["status"] == "BLOCKED_INPUT_MISSING" and "--restore" in result["reason"]


def test_restore_writes_only_a_rebuild_that_reproduces_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / pop.OUTPUT_DIR
    out.mkdir(parents=True)
    manifest = {"populations": {"triage": {"records": 1}}}
    (out / pop.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    triage, recall = [{"triage_id": "r1:a"}], [{"recall_id": "r2:b"}]
    monkeypatch.setattr(pop, "build", lambda root: ({"populations": {}}, triage, recall))
    with pytest.raises(pop.TriagePopulationError, match="does not reproduce"):
        pop.restore(tmp_path)
    assert not (out / pop.TRIAGE_NAME).exists() and not (out / pop.RECALL_NAME).exists()
    monkeypatch.setattr(pop, "build", lambda root: (manifest, triage, recall))
    (out / pop.RECALL_NAME).write_bytes(b"something else\n")
    with pytest.raises(pop.TriagePopulationError, match="differs"):
        pop.restore(tmp_path)
    assert not (out / pop.TRIAGE_NAME).exists()  # a refusal writes nothing, not even the other file
    (out / pop.RECALL_NAME).unlink()
    assert pop.restore(tmp_path)["written"] == [pop.TRIAGE_NAME, pop.RECALL_NAME]
    assert (out / pop.TRIAGE_NAME).read_bytes() == pop._lines(triage)


def test_the_triage_report_checks_labels_against_the_flag_set_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows({"toolace": 1})
    j = "DECLINE_JUSTIFIED"
    refs = [{"triage_id": "r1:toolace0", "votes": dict.fromkeys(rep.ANNOTATORS, j)}]
    manifest = {"populations": {"triage": {"sha256": "x"}}, "flag_set": {"members_sha256": "m"}}
    monkeypatch.setattr(rep, "load", lambda root, name: (rows, refs, manifest))
    # A flag set with one more member than the population: the report must refuse, not decide.
    members = [{"opengrad_id": "toolace-0"}, {"opengrad_id": "toolace-1"}]
    monkeypatch.setattr(pop, "_flag_set", lambda root: ({"members": {"sha256": "m"}}, members))
    with pytest.raises(ft.TriageError):
        rep.report(ROOT, "triage")
    monkeypatch.setattr(pop, "_flag_set", lambda root: ({"members": {"sha256": "other"}}, members))
    with pytest.raises(rep.FlagTriageReportError, match="another flag set"):
        rep.report(ROOT, "triage")
