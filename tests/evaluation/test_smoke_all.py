from pathlib import Path

from opengrad.evaluation.smoke import run_smoke


def test_all_primary_harnesses_accept_mock_predictions(tmp_path: Path):
    names = [
        "bfcl-v4",
        "when2call-eval",
        "tau-bench-tau2",
        "toolsandbox",
        "mcpmark-verified",
        "toolathlon",
    ]
    _items = list(names)
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for name in _items:
        result = run_smoke(name, tmp_path / (name + ".json"))
        assert result["run_id"].startswith("SMOKE_TEST_ONLY")
