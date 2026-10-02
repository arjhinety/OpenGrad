"""Study 002's sentinel registry, seed plan and cost ledger agree with the docs and code that define them."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import jsonschema
import pytest
import yaml

from opengrad.verification.study_002_gate import REQUIRED_SENTINELS

ROOT = Path(__file__).parents[2]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_the_sentinel_registry_is_08s_inventory_and_the_gates_list() -> None:
    registry = _yaml("registry/study_002_sentinels.yaml")
    sentinels = {s["id"]: s for s in registry["sentinels"]}
    spec = (ROOT / "docs/research/study-002/08-SENTINEL-SPEC.md").read_text(encoding="utf-8")
    inventory = spec.split("## Sentinel inventory")[1].split("##")[0]
    assert set(sentinels) == set(re.findall(r"^\| `(S-[A-Z0-9-]+)` \|", inventory, re.MULTILINE))
    assert {k: tuple(v["modes"]) for k, v in sentinels.items()} == REQUIRED_SENTINELS
    assert [k for k, v in sentinels.items() if not v["blocking"]] == ["S-OW7"]
    for sentinel in sentinels.values():
        assert sentinel["population"] and sentinel["endpoints"], sentinel["id"]


def test_s_ans_e_carries_46s_instruction_verbatim() -> None:
    sentinel = next(
        s for s in _yaml("registry/study_002_sentinels.yaml")["sentinels"] if s["id"] == "S-ANS-E"
    )
    amendment = (ROOT / "docs/research/study-002/46-READINESS-DESIGN-AMENDMENT-DRAFT.md").read_text(
        encoding="utf-8"
    )
    assert f"> {sentinel['instruction']}\n" in amendment
    assert "no_call_accuracy" not in sentinel["endpoints"]


def test_the_seed_plan_is_05s_seeds_and_46s_arm_set() -> None:
    plan = _yaml("configs/study_002/seed-plan.yaml")
    assert plan["seeds"] == [0, 1, 2]
    arms = {a["id"]: a for a in plan["arms"]}
    assert set(arms) == {"C0", "R1", "R3", "C2", "X1", "D25", "D50", "S1", "S2"}
    assert not set(arms) & set(plan["retired_arms"]) and set(plan["retired_arms"]) == {"R2", "C1"}
    runs = len(plan["seeds"]) * len(arms) + len(plan["repeats"])
    assert runs == 29
    assert {a["id"] for a in arms.values() if a["budget_fraction"] != 1.0} == {"X1"}
    lines = (ROOT / plan["exposure"]["source"]).read_text(encoding="utf-8").splitlines()
    last = json.loads([line for line in lines if line.strip()][-1])
    assert plan["exposure"]["supervised_token_budget"] == last["supervised_tokens_seen"]
    assert plan["determinism_mode"] is None  # 46 §12: the GPU preflight decides it


def test_the_cost_ledger_validates_and_states_no_balance() -> None:
    schema = json.loads(
        (ROOT / "registry/study_002_cost_ledger.schema.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (ROOT / "reports/study-002/cost/cost-ledger.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(ledger, schema)
    assert ledger["available_credit"] == {"status": "NOT_CONFIRMED", "usd": None}
    rate, runs = Decimal("1.79"), 29
    assert Decimal(str(ledger["envelope"]["low_usd"])) == rate * 12 * runs
    assert Decimal(str(ledger["envelope"]["high_usd"])) == rate * 24 * runs
    confirmed = ledger | {"available_credit": {"status": "CONFIRMED", "usd": None}}
    with pytest.raises(
        jsonschema.ValidationError
    ):  # a confirmed credit needs an amount and its source
        jsonschema.validate(confirmed, schema)
