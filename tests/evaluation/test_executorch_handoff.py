"""The handoff ships copies of OpenGrad's measurement code; these tests stop them drifting.

The ExecuTorch artifacts are scored by someone else, on a machine that does not have this
repository. That only produces a comparable number if the parser and the gate they run are the ones
committed here. A vendored copy is the pragmatic way to achieve that, and a silently stale vendored
copy is the obvious way to lose it.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from opengrad.evaluation.routing import routing_metrics

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release/executorch"
PARSER_SOURCE = ROOT / "src/opengrad/formatting/parser.py"
POLICY_SOURCE = ROOT / "src/opengrad/promotion/quantization.py"

DECISIONS = ("CALL", "ANSWER", "CLARIFY", "UNSUPPORTED")


def handoff_targets() -> list[Path]:
    if not RELEASE.is_dir():
        return []
    return [path for path in sorted(RELEASE.iterdir()) if (path / "manifest.json").is_file()]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_vendored_parser_is_byte_identical_to_the_source(target: Path):
    """Byte-identical, not merely equivalent: the parser is the measurement boundary."""
    vendored = target / "opengrad_min/parser.py"
    assert vendored.read_bytes() == PARSER_SOURCE.read_bytes(), (
        f"{vendored.relative_to(ROOT)} has drifted from {PARSER_SOURCE.relative_to(ROOT)}; "
        "re-run scripts/build_executorch_handoff.py"
    )


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_vendored_gate_is_byte_identical_to_the_source(target: Path):
    vendored = target / "opengrad_min/policy.py"
    assert vendored.read_bytes() == POLICY_SOURCE.read_bytes(), (
        f"{vendored.relative_to(ROOT)} has drifted from {POLICY_SOURCE.relative_to(ROOT)}; "
        "re-run scripts/build_executorch_handoff.py"
    )


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_vendored_routing_agrees_with_the_source_on_random_inputs(target: Path):
    """routing.py is the one file that could not be copied verbatim, so it is tested behaviourally.

    Its single change is inlining DECISIONS instead of importing it from the YAML-backed taxonomy
    module. Any other divergence shows up here as a metric disagreement.
    """
    vendored = load_module(target / "opengrad_min/routing.py", f"vendored_routing_{target.name}")
    rng = random.Random(42)
    for _ in range(50):
        size = rng.randint(1, 60)
        actual = [rng.choice(DECISIONS) for _ in range(size)]
        predicted = [rng.choice(DECISIONS) for _ in range(size)]
        assert vendored.routing_metrics(actual, predicted) == routing_metrics(actual, predicted)


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_handoff_manifest_records_the_bos_collision(target: Path):
    """The BOS collision is the parity break most likely to be discovered too late."""
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["measurement_contract"]
    assert contract["add_bos_token"] is False
    assert contract["bos_token_id_declared_by_pte_metadata"] == 248045
    assert contract["template_hash"] == (
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
    )
    assert contract["generation"]["do_sample"] is False
    assert contract["engine_window"] == 5760
    notes = (target / "PARITY_NOTES.md").read_text(encoding="utf-8")
    assert "Do not prepend BOS" in notes


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_handoff_prompts_ship_token_ids_that_do_not_start_with_a_duplicate_bos(target: Path):
    """Shipping ids is what removes the ambiguity, so the ids themselves must be checked."""
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    prompts = target / manifest["files"]["prompts"]
    rows = [
        json.loads(line)
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == manifest["measurement_contract"]["examples"]
    for row in rows[:50]:
        ids = row["token_ids"]
        assert len(ids) == row["input_tokens"]
        # Exactly one leading <|im_start|>: the prompt opens with it, and nothing prepended another.
        assert ids[0] == 248045
        assert ids[1] != 248045


@pytest.mark.parametrize("target", handoff_targets(), ids=lambda path: path.name)
def test_handoff_artifact_is_not_claimed_to_have_passed_the_gate(target: Path):
    """An export is not evidence of preservation, and the manifest must not imply that it is."""
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {
        "EXPORTED_PENDING_EVALUATION",
        "REJECTED_EXPORT",
        "BLOCKED_SDK_ACCESS",
    }
    assert manifest["parent_experiment_id"] == "m1_dpo_canonical_v2_final_v2"
