# Known pre-existing test failures

Failures present on `master` that no current change introduced. Confirm each by running the same test on a clean
`HEAD` worktree before listing it. Remove an entry in the commit that fixes it.

**None known.** As of 2026-09-16 the full suite passes locally on Windows (1179 passed, 6 skipped).

## Resolved (keep the cause; the same class of bug recurs)

| Was failing | Cause | Fix |
|---|---|---|
| GitHub CI stopped at `ruff check` (`EXE001`) before pytest ever ran | `scripts/verify_publication.py` had a shebang but git mode `100644` (only fails on Linux) | `git update-index --chmod=+x`; every shebang script must be `100755` |
| `tests/evaluation/test_baseline_runner.py` (2 tests) | `str(path.relative_to(root))` gives `\` on Windows, compared with `/` canonical paths | compare `relative_to(root).as_posix()` |
| `tests/optimization/test_protocol.py` (3), `test_optional_import.py` (1) | artifact directory named from a checkpoint id containing `::`, illegal in Windows filenames | `artifact_directory_name` in `src/opengrad/optimization/protocol.py` maps unportable characters to `_` |
| `test_vendored_ifeval_checkers_are_unmodified` | `core.autocrlf=true` checked the SHA-pinned vendored files out as CRLF | `.gitattributes`: `third_party/** -text` |
| `tests/experiments/test_m0_final_freeze.py::test_freeze_verifies_against_disk` | run ledgers (`*.jsonl -text`) were checked out as CRLF before that attribute existed | rewrite the working copies with the exact committed bytes after confirming they differ only by line endings |
