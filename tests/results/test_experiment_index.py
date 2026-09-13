"""Tests for the derived experiment index.

`results/registry.jsonl` is a projection of authoritative run artifacts. These tests pin the
contract that makes it safe to materialise: it is rebuildable, deterministic, never a source of
truth, and never able to damage the records it reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.experiments.ledger import ExperimentLedger
from opengrad.experiments.schema import ExperimentConfig, ExperimentRecord, ExperimentStatus
from opengrad.experiments.store import ExperimentStore
from opengrad.results.registry import (
    REGISTRY_RELATIVE_PATH,
    SELECTION_LATEST,
    SELECTION_NONE,
    SELECTION_PROMOTED,
    build_registry,
    load_registry,
    rebuild_registry,
    registry_path,
    validate_registry,
    write_registry,
)

REGISTRY_PATH = REGISTRY_RELATIVE_PATH


# --- helpers ------------------------------------------------------------------------


def _config(experiment_id: str = "fixture_run") -> ExperimentConfig:
    """A minimal valid experiment config, independent of the repository's own configs."""
    return ExperimentConfig.from_dict(
        {
            "experiment_id": experiment_id,
            "hypothesis": "fixture hypothesis",
            "model": {
                "model_id": "Qwen/Qwen3.5-2B",
                "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "tokenizer_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            },
            "datasets": {
                "manifest_ids": ["fixture_corpus"],
                "hashes": {"fixture_corpus": "a" * 64},
            },
            "trainer": {"type": "sft", "max_steps": 10},
            "evaluation": {"suite": "tool_use_core"},
            "generation": {"temperature": 0.0},
            "checkpointing": {"save_steps": 5, "max_checkpoints": 2},
            "promotion": {"must_pass": {"parse_valid_rate_min": 0.99}},
            "reproducibility": {"seed": 42, "precision": "bfloat16"},
        }
    )


def _write_eval(
    root: Path, experiment_id: str, step: int, call_f1: float, *, partition: str | None = None
) -> None:
    """Write a per-checkpoint metrics artifact in the authoritative shape."""
    base = root / "runs" / experiment_id / "eval"
    if partition:
        base = base / partition
    metrics_dir = base / f"checkpoint-{step}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "CANDIDATE_EVALUATION",
                "manifest": "reports/evaluation/behavioral-heldout-v2.manifest.json",
                "run_id": f"{experiment_id}/eval/checkpoint-{step}",
                "records": 3650,
                "parse_valid_rate": 0.999,
                "lineage": {"checkpoint_id": f"checkpoint-{step}", "checkpoint_step": step},
                "baseline_comparison": {
                    "metrics": {
                        "call_f1": {
                            "baseline": 0.6191,
                            "candidate": call_f1,
                            "delta": round(call_f1 - 0.6191, 6),
                            "verdict": "IMPROVED" if call_f1 > 0.6191 else "REGRESSED",
                        }
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_experiment(
    root: Path, experiment_id: str = "fixture_run", *, status: str | None = None
) -> ExperimentStore:
    store = ExperimentStore(root)
    store.create_experiment(_config(experiment_id))
    if status is not None:
        store.update_status(experiment_id, status)
    return store


# --- 1. empty repository ------------------------------------------------------------


def test_fresh_repository_produces_a_valid_empty_registry(tmp_path: Path) -> None:
    summary = rebuild_registry(tmp_path)
    assert summary["experiments"] == 0
    assert Path(tmp_path, REGISTRY_PATH).is_file()
    assert Path(tmp_path, REGISTRY_PATH).read_text(encoding="utf-8") == ""
    # An empty projection is correct, not drift: no experiments exist to index.
    assert [f for f in validate_registry(tmp_path) if f.kind == "DRIFT"] == []


def test_empty_registry_is_detected_as_unbuilt_only_when_experiments_exist(
    tmp_path: Path,
) -> None:
    _make_experiment(tmp_path, "fixture_run")
    Path(tmp_path, REGISTRY_PATH).write_text("", encoding="utf-8")
    codes = {f.code for f in validate_registry(tmp_path) if f.kind == "DRIFT"}
    assert "REGISTRY_NOT_BUILT" in codes


# --- 2 & 3. one row per experiment --------------------------------------------------


def test_one_completed_experiment_creates_exactly_one_record(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)
    rows = load_registry(tmp_path)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "fixture_run"
    assert rows[0]["status"] == "TRAINED"


def test_multiple_experiments_create_exactly_one_record_each(tmp_path: Path) -> None:
    for name in ("alpha", "beta", "gamma"):
        _make_experiment(tmp_path, name)
    rebuild_registry(tmp_path)
    rows = load_registry(tmp_path)
    assert [r["experiment_id"] for r in rows] == ["alpha", "beta", "gamma"]
    assert len({r["experiment_id"] for r in rows}) == len(rows)


# --- 4 & 5. determinism and rebuildability -----------------------------------------


def test_rebuilding_twice_is_byte_identical(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "alpha", status=ExperimentStatus.TRAINED)
    _make_experiment(tmp_path, "beta", status=ExperimentStatus.REJECTED)
    _write_eval(tmp_path, "alpha", 100, 0.5)
    _write_eval(tmp_path, "beta", 200, 0.7)

    rebuild_registry(tmp_path)
    first = registry_path(tmp_path).read_bytes()
    rebuild_registry(tmp_path)
    second = registry_path(tmp_path).read_bytes()
    assert first == second
    assert first != b""


def test_deleting_the_registry_and_rebuilding_restores_it(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "alpha", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "alpha", 100, 0.5)
    rebuild_registry(tmp_path)
    before = registry_path(tmp_path).read_bytes()

    registry_path(tmp_path).unlink()
    assert not registry_path(tmp_path).exists()
    assert any(f.code == "REGISTRY_MISSING" for f in validate_registry(tmp_path))

    rebuild_registry(tmp_path)
    assert registry_path(tmp_path).read_bytes() == before
    assert [f for f in validate_registry(tmp_path) if f.kind == "DRIFT"] == []


# --- 6, 7, 8. lifecycle state is represented faithfully -----------------------------


@pytest.mark.parametrize(
    "status",
    [ExperimentStatus.TRAINED, ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED],
)
def test_lifecycle_status_is_projected_verbatim(tmp_path: Path, status: ExperimentStatus) -> None:
    _make_experiment(tmp_path, "fixture_run", status=status)
    rebuild_registry(tmp_path)
    rows = load_registry(tmp_path)
    assert rows[0]["status"] == status.value
    assert (
        rows[0]["status"]
        == json.loads(
            (tmp_path / "runs" / "fixture_run" / "experiment.json").read_text(encoding="utf-8")
        )["status"]
    )


def test_promotion_checkpoint_comes_from_the_authoritative_ledger(tmp_path: Path) -> None:
    store = _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.4)
    _write_eval(tmp_path, "fixture_run", 200, 0.8)
    store.update_status(
        "fixture_run",
        ExperimentStatus.PROMOTED,
        {"checkpoint": "checkpoint-100", "note": "policy passed"},
    )
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["promotion_checkpoint"] == "checkpoint-100"
    # The promoted checkpoint is preferred over the latest, and the rule is named.
    assert row["headline_checkpoint"] == "checkpoint-100"
    assert row["headline_checkpoint_selection"] == SELECTION_PROMOTED


def test_rejections_are_recorded_and_do_not_become_the_headline(tmp_path: Path) -> None:
    store = _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.4)
    store.update_status(
        "fixture_run",
        ExperimentStatus.REJECTED,
        {"checkpoint": "fixture_run::checkpoint-100"},
    )
    rebuilt = rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["rejected_checkpoints"] == ["fixture_run::checkpoint-100"]
    # Rejection is not promotion, so the headline falls back to the latest evaluated.
    assert row["headline_checkpoint_selection"] == SELECTION_LATEST
    assert rebuilt["status_counts"] == {"REJECTED": 1}


def test_no_evaluated_checkpoint_is_stated_not_guessed(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["headline_checkpoint"] is None
    assert row["headline_checkpoint_selection"] == SELECTION_NONE
    assert row["headline_metrics"] == {}
    assert row["evaluated_checkpoint_count"] == 0


# --- 9. summaries come from authoritative eval artifacts ----------------------------


def test_evaluation_summary_is_taken_from_the_authoritative_eval_artifact(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 150, 0.4321)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]

    assert row["evaluated_checkpoint_count"] == 1
    checkpoint = row["evaluated_checkpoints"][0]
    assert checkpoint["checkpoint_id"] == "checkpoint-150"
    assert checkpoint["checkpoint_step"] == 150
    assert checkpoint["records"] == 3650
    assert checkpoint["metrics"]["call_f1"] == 0.4321
    # The artifact path is recorded so the value can be re-derived independently.
    artifact = tmp_path / checkpoint["metrics_artifact"]
    assert artifact.is_file()
    assert (
        json.loads(artifact.read_text(encoding="utf-8"))["baseline_comparison"]["metrics"][
            "call_f1"
        ]["candidate"]
        == 0.4321
    )


def test_eval_identity_fields_are_reported_verbatim(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.5)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["eval_kinds"] == ["CANDIDATE_EVALUATION"]
    assert row["eval_manifests"] == ["reports/evaluation/behavioral-heldout-v2.manifest.json"]


# --- 10 & 11. drift detection and repair --------------------------------------------


def test_manually_stale_status_is_detected(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)

    # Hand-edit the derived index: this is exactly the corruption the validator must catch.
    row = load_registry(tmp_path)[0]
    row["status"] = "PROMOTED"
    registry_path(tmp_path).write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    findings = validate_registry(tmp_path)
    drift = [f for f in findings if f.kind == "DRIFT"]
    assert any(f.code == "FIELD_MISMATCH" and f.field_name == "status" for f in drift), findings
    status_finding = next(f for f in drift if f.field_name == "status")
    assert status_finding.registry_value == "PROMOTED"
    assert status_finding.authoritative_value == "TRAINED"


def test_stale_promotion_state_is_detected(tmp_path: Path) -> None:
    store = _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)
    # Promote authoritatively, then restore the pre-promotion projection.
    stale = registry_path(tmp_path).read_bytes()
    store.update_status("fixture_run", ExperimentStatus.PROMOTED, {"checkpoint": "checkpoint-100"})
    registry_path(tmp_path).write_bytes(stale)

    findings = validate_registry(tmp_path)
    assert any(
        f.field_name in {"status", "promotion_checkpoint"} for f in findings if f.kind == "DRIFT"
    )


def test_rebuild_repairs_stale_derived_state(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)
    registry_path(tmp_path).write_text(
        json.dumps({"experiment_id": "fixture_run", "status": "PROMOTED"}) + "\n",
        encoding="utf-8",
    )
    assert any(f.kind == "DRIFT" for f in validate_registry(tmp_path))

    rebuild_registry(tmp_path)
    assert [f for f in validate_registry(tmp_path) if f.kind == "DRIFT"] == []
    assert load_registry(tmp_path)[0]["status"] == "TRAINED"


def test_registry_row_without_an_authoritative_experiment_is_detected(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run")
    rebuild_registry(tmp_path)
    with registry_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"experiment_id": "ghost"}) + "\n")

    codes = {f.code for f in validate_registry(tmp_path) if f.kind == "DRIFT"}
    assert "REGISTRY_ROW_WITHOUT_EXPERIMENT" in codes


def test_malformed_registry_row_is_detected_not_repaired(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run")
    rebuild_registry(tmp_path)
    with registry_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    findings = validate_registry(tmp_path)
    assert any(f.code == "MALFORMED_REGISTRY_ROW" for f in findings)
    # Validation must not rewrite the file it is judging.
    assert "{not json" in registry_path(tmp_path).read_text(encoding="utf-8")


def test_unresolved_provenance_path_is_detected(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run")
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    row["provenance"]["experiment"] = "runs/does_not_exist/experiment.json"
    registry_path(tmp_path).write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    codes = {f.code for f in validate_registry(tmp_path) if f.kind == "DRIFT"}
    assert "PROVENANCE_PATH_UNRESOLVED" in codes


# --- 12. duplicates cannot survive a rebuild ----------------------------------------


def test_duplicate_registry_rows_cannot_survive_rebuilding(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    with registry_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

    assert any(f.code == "DUPLICATE_EXPERIMENT_ID" for f in validate_registry(tmp_path))
    rebuild_registry(tmp_path)
    assert len(load_registry(tmp_path)) == 1


def test_rebuild_replaces_rather_than_appends(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "alpha")
    rebuild_registry(tmp_path)
    _make_experiment(tmp_path, "beta")
    rebuild_registry(tmp_path)
    assert len(load_registry(tmp_path)) == 2
    rebuild_registry(tmp_path)
    assert len(load_registry(tmp_path)) == 2


# --- 13 & 14. the derived index cannot damage authoritative state -------------------


def test_registry_failure_does_not_break_the_authoritative_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken derived index must never fail an authoritative write."""
    from opengrad.results import registry as registry_module

    def explode(_root: Path) -> dict[str, object]:
        raise OSError("derived index unavailable")

    store = _make_experiment(tmp_path, "fixture_run")
    monkeypatch.setattr(registry_module, "rebuild_registry", explode)
    # The authoritative transition must still complete and be readable.
    record = store.update_status("fixture_run", ExperimentStatus.TRAINED)
    assert record.status == "TRAINED"
    assert store.get_experiment("fixture_run").status == "TRAINED"

    # And the index is recoverable once the fault clears.
    monkeypatch.undo()
    rebuild_registry(tmp_path)
    assert load_registry(tmp_path)[0]["status"] == "TRAINED"


def test_building_the_registry_writes_nothing_but_the_registry(tmp_path: Path) -> None:
    store = _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.5)
    store.update_status("fixture_run", ExperimentStatus.REJECTED, {"checkpoint": "cp-100"})
    rebuild_registry(tmp_path)

    tracked = [
        tmp_path / "runs" / "fixture_run" / "experiment.json",
        tmp_path / "runs" / "fixture_run" / "ledger.jsonl",
        tmp_path / "runs" / "central_ledger.jsonl",
        tmp_path / "runs" / "fixture_run" / "eval" / "checkpoint-100" / "metrics.json",
    ]
    before = {path: path.read_bytes() for path in tracked}

    rebuild_registry(tmp_path)

    assert {path: path.read_bytes() for path in tracked} == before, (
        "registry construction must not modify experiment.json, the eval artifacts, "
        "or either ledger"
    )


def test_write_registry_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    rows, _findings = build_registry(tmp_path)
    write_registry(tmp_path, rows)
    assert not registry_path(tmp_path).with_name("registry.jsonl.tmp").exists()


def test_publication_tolerates_an_unreadable_eval_artifact(tmp_path: Path) -> None:
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.5)
    (tmp_path / "runs" / "fixture_run" / "eval" / "checkpoint-100" / "metrics.json").write_text(
        "{broken", encoding="utf-8"
    )
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["evaluated_checkpoint_count"] == 0
    assert any(f.code == "EVAL_ARTIFACT_UNREADABLE" for f in validate_registry(tmp_path))


# --- repository regression coverage -------------------------------------------------


def test_current_repository_registry_is_not_the_empty_bootstrap_placeholder() -> None:
    """Regression for the state this change fixes.

    `results/registry.jsonl` was committed as a zero-byte placeholder while eleven experiment
    records existed under `runs/`. The materialized index must reflect them, and validation must
    find no drift. Read-only: the repository tree is not modified by its own test suite.
    """
    root = Path.cwd()
    rows = load_registry(root)
    authority, _findings = build_registry(root)

    assert authority, "the repository has experiment records to index"
    assert rows, "the committed registry must not be the empty bootstrap placeholder"
    assert len(rows) == len(authority)
    assert [r["experiment_id"] for r in rows] == sorted(r.experiment_id for r in authority)
    drift = [f for f in validate_registry(root) if f.kind == "DRIFT"]
    assert drift == [], drift


def test_unevaluated_runs_name_no_eval_location() -> None:
    """Failed, scaffold and mock runs produced no evaluation, so their rows must not point at one."""
    for row in load_registry(Path.cwd()):
        if not row["evaluated_checkpoints"]:
            assert not row["provenance"].get("eval", "").startswith("runs/"), row["experiment_id"]


def test_repository_experiments_are_all_indexed_exactly_once() -> None:
    root = Path.cwd()
    rows = load_registry(root)
    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert ids == sorted(_discover_store_ids(root))

    # The experiment that produced the published M0-v2 result must be discoverable here.
    m0_v2 = next(row for row in rows if row["experiment_id"] == "qwen35_2b_m0_sft_v2corpus")
    assert m0_v2["evaluated_checkpoint_count"] == 4
    assert m0_v2["headline_checkpoint_selection"] == SELECTION_LATEST
    assert m0_v2["headline_metrics"]["call_f1"] == pytest.approx(0.5292, abs=1e-4)


def _discover_store_ids(root: Path) -> list[str]:
    return [
        str(json.loads(path.read_text(encoding="utf-8"))["experiment_id"])
        for path in sorted((root / "runs").rglob("experiment.json"))
    ]


def test_registry_round_trips_a_record_it_did_not_write(tmp_path: Path) -> None:
    """A registry built from `register_record` state must match the record exactly."""
    store = ExperimentStore(tmp_path)
    store.register_record(
        ExperimentRecord(
            experiment_id="non_training_record",
            hypothesis="fixture",
            model_id="Qwen/Qwen3.5-2B",
            model_revision="b" * 40,
            tokenizer_revision="b" * 40,
            training_algorithm="evaluation",
            training_config={},
            status=ExperimentStatus.EVALUATED.value,
        )
    )
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["experiment_id"] == "non_training_record"
    assert row["status"] == "EVALUATED"
    assert row["model_revision"] == "b" * 40


def test_registry_path_constant_matches_the_documented_location() -> None:
    assert REGISTRY_RELATIVE_PATH == "results/registry.jsonl"


def test_ledger_events_are_counted_from_the_central_ledger(tmp_path: Path) -> None:
    store = _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    store.update_status("fixture_run", ExperimentStatus.REJECTED, {"checkpoint": "cp"})
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    events = ExperimentLedger(tmp_path / "runs" / "central_ledger.jsonl").read_events()
    assert row["ledger_event_count"] == len(
        [event for event in events if event.experiment_id == "fixture_run"]
    )


def test_namespaced_partition_measurements_are_indexed_separately(tmp_path: Path) -> None:
    """DEV and confirmatory scores of the same checkpoint are two measurements, not one.

    They live under `eval/<partition>/checkpoint-N/` so one cannot overwrite the other, and the
    index must see both -- reporting a checkpoint as unevaluated when it has been scored twice
    would hide exactly the evidence the partition exists to keep apart.
    """
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    for partition, value in (("dev", 0.5), ("confirmatory", 0.7)):
        _write_eval(tmp_path, "fixture_run", 100, value, partition=partition)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["evaluated_checkpoint_count"] == 2
    partitions = {
        c["eval_partition"]: c["metrics"]["call_f1"] for c in row["evaluated_checkpoints"]
    }
    assert partitions == {"dev": 0.5, "confirmatory": 0.7}


def test_unnamespaced_measurements_are_still_indexed(tmp_path: Path) -> None:
    """Every run recorded before the partition existed uses `eval/checkpoint-N/`."""
    _make_experiment(tmp_path, "fixture_run", status=ExperimentStatus.TRAINED)
    _write_eval(tmp_path, "fixture_run", 100, 0.5)
    rebuild_registry(tmp_path)
    row = load_registry(tmp_path)[0]
    assert row["evaluated_checkpoint_count"] == 1
    assert row["evaluated_checkpoints"][0]["eval_partition"] is None
