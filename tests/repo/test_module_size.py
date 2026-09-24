"""A ratchet on module and function size in `src/`.

On 2026-09-24 the six modules over 1,000 lines that were free to change were split into sibling
modules, and the longest functions into named phases, with no behaviour change. This test keeps them
from growing back:

* no module over 1,000 lines, except files whose own source hash a frozen artifact records;
* no function over 250 lines, except those listed with the reason they stay long. The list may only
  shrink: a new long function fails, and so does an entry that is no longer long.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_LIMIT = 1000
FUNCTION_LIMIT = 250

_RECORDED = "its source hash is recorded by a frozen artifact; editing the file changes that hash"
LONG_MODULES = {
    "src/opengrad/data/adapters.py": _RECORDED + " (normalization_v3.CODE_MODULES)",
    "src/opengrad/verification/pdet_coverage.py": _RECORDED + " (P-DET-COVERAGE manifests)",
}

_LOOP = (
    "the training loop is kept whole on purpose: nothing in CI can run it (no torch), so its setup "
    "and wrap-up were extracted and the loop body was not"
)
LONG_FUNCTIONS = {
    "src/opengrad/training/sft.py::run_real_sft": _LOOP,
    "src/opengrad/training/dpo_live.py::run_real_dpo": _LOOP,
    "src/opengrad/agent_cli.py::handle_train": "not yet split; the next change to it splits it",
    "src/opengrad/data/cli.py::main": "not yet split; the next change to it splits it",
}


def _modules() -> list[Path]:
    found = sorted((ROOT / "src").rglob("*.py"))
    assert len(found) > 150, "the source tree looks empty"
    return found


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_no_module_grows_past_the_limit() -> None:
    long = {
        _relative(p)
        for p in _modules()
        if len(p.read_text(encoding="utf-8").splitlines()) > MODULE_LIMIT
    }
    assert long - set(LONG_MODULES) == set(), "split these modules into siblings"
    assert set(LONG_MODULES) - long == set(), "these are no longer long; remove their entries"


def test_no_function_grows_past_the_limit() -> None:
    long = set()
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (node.end_lineno or node.lineno) - node.lineno + 1 > FUNCTION_LIMIT:
                long.add(f"{_relative(path)}::{node.name}")
    assert long - set(LONG_FUNCTIONS) == set(), "split these functions into named phases"
    assert set(LONG_FUNCTIONS) - long == set(), "these are no longer long; remove their entries"
