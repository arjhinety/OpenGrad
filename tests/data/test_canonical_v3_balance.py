"""The canonical-v3 decision balance (21 phase 4; 39), on synthetic label rows only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data import behaviour_labels as labels_pass
from opengrad.data import canonical_v3_balance as balance
from opengrad.data.mixture import load_mixture, validate_mixture

ROOT = Path(__file__).resolve().parents[2]
SEED = "test-seed"


def row(index: int, label: str, source: str = "glaive", *, permitted: bool | None = None) -> dict:
    if permitted is None:
        permitted = labels_pass.weight_permitted(label, source)
    return {
        "id": f"og_{index:08x}",
        "source": source,
        "canonical_hash": f"h{index}",
        "label": label,
        "step": None,
        "unit_kind": "single_exchange" if label != labels_pass.CALL_BY_STRUCTURE else None,
        "contract": "prose-decision-input-v2",
        "reasons": [],
        "weight_permitted": permitted,
    }


def test_the_config_validates_and_names_the_decision_vocabulary() -> None:
    config = load_mixture(ROOT / balance.CONFIG)
    assert config["mixture_class"] == "decision_balanced"
    assert set(config["decision_weights"]) == set(balance.STRATUM_LABEL)
    assert sum(config["decision_weights"].values()) == pytest.approx(1.0)


def test_a_decision_balanced_config_without_weights_is_refused() -> None:
    with pytest.raises(ValueError, match="requires decision_weights"):
        validate_mixture({"mixture_class": "decision_balanced", "status": "SCHEMA_READY"})


def test_call_membership_is_structural_and_other_strata_need_permitted_weight() -> None:
    # CALL_BY_STRUCTURE never carries weight as a classifier label, but its stratum is structural (39 §1).
    assert balance.stratum_of(row(1, labels_pass.CALL_BY_STRUCTURE)) == "CALL"
    assert balance.stratum_of(row(2, "DIRECT", "glaive")) == "ANSWER"
    assert balance.stratum_of(row(3, "DIRECT", "toolace")) is None  # 38 §2: not permitted
    assert balance.stratum_of(row(4, "CLARIFY", "when2call")) == "CLARIFY"
    assert balance.stratum_of(row(5, "UNSUPPORTED", "toolace")) == "UNSUPPORTED"
    assert balance.stratum_of(row(6, "CALL", "toolace")) is None  # a classifier CALL label carries no weight
    assert balance.stratum_of(row(7, "ABSTAIN")) is None
    assert balance.stratum_of(row(8, labels_pass.UNLABELLED)) is None


def test_every_stratum_contributes_the_smallest_supply() -> None:
    rows = (
        [row(i, labels_pass.CALL_BY_STRUCTURE, "xlam") for i in range(100)]
        + [row(200 + i, "DIRECT", "glaive") for i in range(40)]
        + [row(400 + i, "CLARIFY", "when2call") for i in range(12)]
        + [row(600 + i, "UNSUPPORTED", "toolace") for i in range(25)]
    )
    chosen, counts = balance.select(rows, SEED)
    assert counts["per_stratum"] == 12
    assert counts["selected"] == 48
    assert counts["supply"] == {"ANSWER": 40, "CALL": 100, "CLARIFY": 12, "UNSUPPORTED": 25}
    assert counts["left_out"] == {"ANSWER": 28, "CALL": 88, "CLARIFY": 0, "UNSUPPORTED": 13}
    assert all(len(items) == 12 for items in chosen.values())


def test_the_selection_is_deterministic_and_seed_dependent() -> None:
    rows = [row(i, labels_pass.CALL_BY_STRUCTURE, "xlam") for i in range(30)] + [
        row(100 + i, label, source)
        for i, (label, source) in enumerate([("DIRECT", "glaive"), ("CLARIFY", "glaive"), ("UNSUPPORTED", "glaive")] * 5)
    ]
    first, _ = balance.select(rows, SEED)
    again, _ = balance.select(list(reversed(rows)), SEED)
    other, _ = balance.select(rows, "another-seed")
    ids = lambda chosen: {stratum: [r["id"] for r in items] for stratum, items in chosen.items()}  # noqa: E731
    assert ids(first) == ids(again)
    assert ids(first)["CALL"] != ids(other)["CALL"]


def test_a_missing_stratum_refuses_to_balance() -> None:
    rows = [row(i, labels_pass.CALL_BY_STRUCTURE, "xlam") for i in range(5)] + [row(50, "DIRECT", "glaive")]
    with pytest.raises(balance.BalanceError, match="CLARIFY"):
        balance.select(rows, SEED)


def test_composition_reports_sources_and_unit_kinds_without_constraining_them() -> None:
    chosen = {
        "CALL": [row(1, labels_pass.CALL_BY_STRUCTURE, "xlam")],
        "ANSWER": [row(2, "DIRECT", "glaive")],
    }
    assert balance.composition(chosen) == {
        "per_source": {"ANSWER": {"glaive": 1}, "CALL": {"xlam": 1}},
        "per_unit_kind": {"ANSWER": {"single_exchange": 1}, "CALL": {"not_applicable": 1}},
    }


def test_verify_catches_a_changed_config(tmp_path: Path) -> None:
    plan = json.loads((ROOT / balance.OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    out = tmp_path / balance.OUTPUT_DIR
    out.mkdir(parents=True)
    plan["config"]["sha256"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(plan), encoding="utf-8")
    (out / plan["selected_ids_file"]).write_bytes((ROOT / balance.OUTPUT_DIR / plan["selected_ids_file"]).read_bytes())
    labels = tmp_path / labels_pass.OUTPUT_DIR
    labels.mkdir(parents=True)
    for name in ("manifest.json",):
        (labels / name).write_bytes((ROOT / labels_pass.OUTPUT_DIR / name).read_bytes())
    manifest = json.loads((labels / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["sources"].values():
        (labels / entry["file"]).write_bytes((ROOT / labels_pass.OUTPUT_DIR / entry["file"]).read_bytes())
    (tmp_path / balance.CONFIG).parent.mkdir(parents=True)
    (tmp_path / balance.CONFIG).write_bytes((ROOT / balance.CONFIG).read_bytes())
    result = balance.verify(tmp_path)
    assert result["status"] == "FAIL"
    assert any("config changed" in problem for problem in result["problems"])
