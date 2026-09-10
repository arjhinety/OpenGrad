from opengrad.experiments.gates import authorize
from opengrad.experiments.lineage import Run, validate_lineage


def test_stage_gates_block_unreproduced_baseline() -> None:
    assert authorize("SFT", {}) == (False, "blocked: baseline_accepted, data_audit_complete")


def test_lineage_accepts_branches_and_rejects_missing_parent() -> None:
    runs = [
        Run("base", "EXPERIMENTAL"),
        Run("sft", "ACCEPTED", "parent", "ckpt-sft", "base"),
        Run("dpo", "REJECTED", parent_run_id="sft"),
        Run("bad", "FAILED", parent_run_id="missing"),
    ]
    assert "bad: missing parent" in validate_lineage(runs)
