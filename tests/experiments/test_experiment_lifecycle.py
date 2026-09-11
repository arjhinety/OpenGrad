from pathlib import Path

from opengrad.experiments.diff import diff_experiments
from opengrad.experiments.ledger import ExperimentLedger, LedgerEventType
from opengrad.experiments.schema import ExperimentConfig, ExperimentStatus
from opengrad.experiments.store import ExperimentStore


def test_experiment_config_loading_and_resolution(tmp_path: Path) -> None:
    cfg_file = Path("configs/experiments/m0_sft.yaml")
    config = ExperimentConfig.from_file(cfg_file)
    assert config.experiment_id == "qwen35_2b_m0_sft"
    assert config.trainer["type"] == "sft"

    resolved = tmp_path / "resolved.yaml"
    config.write_resolved(resolved)
    assert resolved.exists()


def test_experiment_store_and_ledger_lifecycle(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    cfg = ExperimentConfig.from_file("configs/experiments/m0_sft.yaml")

    rec = store.create_experiment(cfg)
    assert rec.status == ExperimentStatus.CREATED.value

    # Update status to TRAINING then TRAINED
    updated = store.update_status(cfg.experiment_id, ExperimentStatus.TRAINING)
    assert updated.status == ExperimentStatus.TRAINING.value

    updated_done = store.update_status(cfg.experiment_id, ExperimentStatus.TRAINED)
    assert updated_done.status == ExperimentStatus.TRAINED.value

    # Verify ledger history
    local_ledger = ExperimentLedger(store.run_dir(cfg.experiment_id) / "ledger.jsonl")
    events = local_ledger.read_events()
    assert len(events) >= 3
    assert events[0].event_type == LedgerEventType.EXPERIMENT_CREATED.value


def test_experiment_diff_causal_warning() -> None:
    cfg1 = ExperimentConfig.from_file("configs/experiments/m0_sft.yaml")
    cfg2 = ExperimentConfig.from_file("configs/experiments/m1_dpo.yaml")

    diff = diff_experiments(cfg1.to_dict(), cfg2.to_dict())
    assert diff.variables_changed_count > 1
    assert diff.is_single_variable is False
    assert diff.warning is not None
    assert "Causal attribution" in diff.warning


def test_experiment_record_names_the_commit_that_produced_it(tmp_path: Path) -> None:
    """A record that cannot name its own code is not reproducible evidence.

    `capture()` returns a flat `git_sha`, but the store read a nested `environment["git"]["sha"]`,
    which never exists -- so every record claimed `git_commit: unknown` even from a clean tree.
    """
    from opengrad.experiments.schema import ExperimentConfig
    from opengrad.experiments.store import ExperimentStore

    store = ExperimentStore(tmp_path)
    config = ExperimentConfig.from_file("configs/experiments/m0_sft.yaml")
    record = store.create_experiment(config, env={"git_sha": "a" * 40, "git_dirty": False})
    assert record.git_commit == "a" * 40
    assert record.git_dirty is False

    # A caller that supplies the nested shape is still honoured.
    nested = store.create_experiment(
        ExperimentConfig.from_dict({**config.to_dict(), "experiment_id": "nested"}),
        env={"git": {"sha": "b" * 40}, "git_dirty": False},
    )
    assert nested.git_commit == "b" * 40

    # And an environment with no git information still says so rather than inventing a value.
    unknown = store.create_experiment(
        ExperimentConfig.from_dict({**config.to_dict(), "experiment_id": "nogit"}), env={}
    )
    assert unknown.git_commit == "unknown"
