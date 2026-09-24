"""No new local sha256 wrapper: plain hashing goes through `opengrad.hashing`.

A *plain wrapper* is a one-argument function that hashes that argument's bytes, text or file and
nothing else. Helpers that hash something computed (normalised text, canonical JSON, selected
fields, a seeded key) mean something different and are not flagged.

Every wrapper that remains is listed with the reason it cannot import the shared copy. The list may
only shrink: a new wrapper fails here, and so does an entry that no longer exists.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_STANDALONE = "standalone script: runs without the opengrad package installed"
_MODAL = "Modal container script: the package is not installed in the container"
_RECORDED = "its source hash is recorded by a frozen artifact; editing the file changes that hash"

ALLOWED = {
    "scripts/archive_pdet_model_batches.py::sha256": _STANDALONE,
    "scripts/build_capability_evidence.py::digest": _STANDALONE,
    "scripts/build_checkpoint_ladder.py::sha256_text": _STANDALONE,
    "scripts/build_minus_xlam_release.py::sha256": _STANDALONE,
    "scripts/freeze_m0_final.py::sha256": _STANDALONE,
    "scripts/freeze_minus_xlam_ablation.py::sha256": _STANDALONE,
    "scripts/modal/executorch_export.py::_sha256": _MODAL,
    "scripts/modal/gguf_study.py::_sha256": _MODAL,
    "src/opengrad/data/classifier_input.py::_sha256_bytes": _RECORDED,
    "src/opengrad/data/normalization_v3.py::file_sha256": _RECORDED,
    # ANSWER-STRATA-v1's manifest records this module's code_sha256_lf (study_002_prereg_v9).
    "src/opengrad/verification/answer_strata.py::_sha256": _RECORDED,
    "src/opengrad/verification/classifier_devcheck.py::_sha256": _RECORDED,
    "src/opengrad/verification/classifier_devset.py::_sha256": _RECORDED,
    "src/opengrad/verification/classifier_devset_v2.py::_sha256": _RECORDED,
    "src/opengrad/verification/pdet.py::_sha256_bytes": _RECORDED,
    "src/opengrad/verification/pdet.py::keyed_rank": _RECORDED,
    "src/opengrad/verification/pdet_coverage.py::_sha256_bytes": _RECORDED,
    "src/opengrad/verification/pdet_coverage_v2.py::_sha256": _RECORDED,
}

_PLAIN_CALLS = {"sha256", "encode", "hexdigest", "read_bytes", "open", "iter", "read", "update"}


def _is_hashlib_sha256(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "sha256"
        and isinstance(func.value, ast.Name)
        and func.value.id == "hashlib"
    )


def plain_wrappers(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or len(node.args.args) != 1:
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        if len(body) > 3:
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        names = {
            c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "?")
            for c in calls
        }
        if any(_is_hashlib_sha256(c) for c in calls) and names <= _PLAIN_CALLS:
            found.append(node.name)
    return found


def _tracked_python() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "src", "scripts"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    # `opengrad.hashing` is the shared copy the rule points to.
    return [path for path in listing if path.endswith(".py") and path != "src/opengrad/hashing.py"]


def test_the_detector_recognises_each_plain_shape() -> None:
    source = """
import hashlib
def a(data): return hashlib.sha256(data).hexdigest()
def b(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def c(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
def d(text): return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
"""
    assert plain_wrappers(source) == ["a", "b", "c"]


def test_no_new_local_sha256_wrapper() -> None:
    files = _tracked_python()
    assert len(files) > 200, "the file listing looks empty"
    found = {
        f"{path}::{name}"
        for path in files
        for name in plain_wrappers((ROOT / path).read_text(encoding="utf-8"))
    }
    new = sorted(found - set(ALLOWED))
    gone = sorted(set(ALLOWED) - found)
    assert new == [], f"use opengrad.hashing instead of a local wrapper: {new}"
    assert gone == [], f"remove these entries, the wrapper no longer exists: {gone}"
