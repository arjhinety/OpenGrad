# Known pre-existing test failures

Failures present on `master` that no current change introduced. Each was confirmed by running the same test on
a clean `HEAD` worktree. Remove an entry in the commit that fixes it; add one only after that confirmation.

Last confirmed: 2026-09-16, at commit `cf42c6a`.

| Test | Symptom | Suspected cause |
|---|---|---|
| `tests/evaluation/test_baseline_runner.py::test_real_baseline_registers_canonical_record_and_refuses_overwrite` | `ValueError: real baseline must write the canonical evidence paths` | not yet investigated |
| `tests/evaluation/test_baseline_runner.py::test_dry_run_baseline_never_writes_canonical_evidence_paths` | fails with the baseline runner test above | not yet investigated |
| `tests/evaluation/test_capability_benchmarks.py::test_vendored_ifeval_checkers_are_unmodified` | vendored checker hash mismatch | likely CRLF conversion of vendored files under `core.autocrlf=true` |
| `tests/experiments/test_m0_final_freeze.py::test_freeze_verifies_against_disk` | frozen hashes differ for `runs/central_ledger.jsonl` and two others | working-tree copies are CRLF (`git ls-files --eol` shows `i/lf w/crlf`); passes on a fresh worktree |
| `tests/optimization/test_optional_import.py::test_modelopt_export_raises_naming_the_extra` | — | not yet investigated |
| `tests/optimization/test_protocol.py::test_mock_result_carries_full_provenance_and_round_trips` | — | not yet investigated |
| `tests/optimization/test_protocol.py::test_mock_artifact_hash_is_deterministic` | — | not yet investigated |
| `tests/optimization/test_protocol.py::test_mock_optimize_never_modifies_the_source` | — | not yet investigated |
