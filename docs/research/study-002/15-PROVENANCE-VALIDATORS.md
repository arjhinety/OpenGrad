# 15 — Provenance validators

Study 001's ledger is the specification for this document: 100 claims had no reachable artifact behind them,
and most were unchallenged at publication. The correction is mechanical — a validator that refuses to let a
number into a table unless it can name the artifact it came from, and that **proves it actually ran**.

## The existing infrastructure is the template

`src/opengrad/verification/accounting.py` already defines the contract, and it exists because two gates in
this repository reported success without doing anything (see `tests/registry/test_non_vacuity.py` and the
module docstring):

- `ValidationResult` carries `name`, `policy`, `discovered`, `checked`, `passed`, `failed`, `blocked`,
  `skipped[]` (each with a reason), `errors[]`, `blocked_reasons[]`, `detail{}`, `blocked_status`,
  `precondition`;
- `accounting_errors()` checks `discovered == checked + blocked + skipped` and
  `checked == passed + failed`, on the principle that a result *"whose counters do not add up is itself a
  failure"*;
- population policies `REQUIRED_NONEMPTY`, `CONDITIONALLY_REQUIRED`, `OPTIONAL`, because *"`discovered == 0`
  means different things for a corpus the repository certainly contains and for an optional cross-check"*;
- greppable prefixes `FAIL_NONVACUOUS` and `FAIL_ACCOUNTING`, with blocked statuses `BLOCKED_NETWORK` and
  `BLOCKED_OPTIONAL_DEPENDENCY` kept apart from failures, because *"a blocked candidate is not a failed
  candidate, and a gate that reported FAIL merely because it could not reach the network would be lying in
  the other direction"*;
- `VerificationReport.contract`, an integer contract version. `src/opengrad/registry/validate.py` documents
  why bumping it matters — under v1 the checker *"did not verify that its own checks had executed, so
  `opengrad.registry.validate` could report PASS without running, and freeze validation could skip every
  canonical corpus."*

Existing checks follow a uniform shape (`check_structure`, `check_references`, `check_publication_records`,
`check_reconstructed_events`, `check_semantic_consistency`, `check_freeze`, routed through
`result_from(name, policy, ids, errors)`). Study 002's validators use the same shape, the same policies and
the same prefixes, and bump the contract version — because adding checks changes what a PASS means.

## The validators

| id | Checks | Failure code |
|---|---|---|
| `V1 mode_coverage` | every required mode in `P-CONF` has `n > 0`; policy `REQUIRED_NONEMPTY` | `FAIL_NONVACUOUS` |
| `V2 metric_denominator` | no metric reports `0.0` from an empty denominator; unmeasured is `UNMEASURED`/`null` | `FAIL_VACUOUS_METRIC` |
| `V3 metric_attachment` | every number carries metric + metric version, arm, seed, partition id + fingerprint, `n`, denominator, evaluator version, artifact sha256 | `FAIL_UNATTACHED_METRIC` |
| `V4 vocabulary_lock` | every mode name is in the canonical enum or the alias table | `FAIL_VOCABULARY` |
| `V5 one_shot` | `P-CONF` and `P-UNANS` scored once per arm | `FAIL_ONE_SHOT` |
| `V6 row_key` | every row carries `(arm, partition, protocol, device_class)` plus the provider | `FAIL_UNKEYED_ROW` |
| `V7 sentinel_completeness` | every sentinel in [08](08-SENTINEL-SPEC.md) ran, in every required mode | `FAIL_MISSING_SENTINEL` |
| `V8 census_reconciliation` | the `accounting.py` identities hold for the evaluation census | `FAIL_ACCOUNTING` |
| `V9 fingerprint_from_artifact` | corpus and partition fingerprints are read from artifacts, never typed (G4) | `FAIL_FINGERPRINT` |
| `V10 claim_evidence` | every claim maps to a number row and a reachable artifact path + sha256 | `FAIL_UNSUPPORTED_CLAIM` |
| `V11 cross_study_comparability` | no table mixes a Study 002 number with a Study 001 frozen number without evaluator version and partition on both (G9) | `FAIL_INCOMPARABLE` |
| `V12 resolvable_margin` | every comparison row prints `n` and its resolvable margin, or is marked `WITHIN_NOISE` | `FAIL_UNRESOLVED_ROW` |

`V10` is the answer to the ledger: the same check the audit performed by hand, executed as a build step. A
claim whose evidence path does not resolve fails the build rather than reaching a reader.

`V12` is what makes the statistics plan enforceable. Without it, "the difference is significant" is a
sentence; with it, the sentence must carry the `n` and the margin that justify it.
## Rules every validator obeys

1. **Declare a population policy.** `REQUIRED_NONEMPTY` where the repository contains the inputs by
   construction; `CONDITIONALLY_REQUIRED` where a stated precondition holds; `OPTIONAL` otherwise — and then
   zero discovery is a *skip with a reason*, never a silent success.
2. **Prove execution.** `discovered`, `checked`, `passed`, `failed`, `blocked` and `skipped[]` are all
   populated, and `accounting_errors()` must be empty before a status is computed.
3. **Distinguish blocked from failed.** An unreachable dependency yields `BLOCKED_*`, which can never be read
   as PASS.
4. **Never encode absence numerically.** A validator that cannot measure something reports `UNMEASURED`.
5. **Bump the contract version** when the check set, its discovery rules or its non-vacuity requirements
   change, so a PASS under one contract is never read as a PASS under another.
6. **Exercise every validator on a smoke run.** Each one executes on a `provider: cpu` smoke run
   ([13](13-HARDWARE-AGNOSTIC-EXECUTION.md)) before GPU time is requested, so that "the validator exists" is
   never a claim about an unexecuted path — which is `#66` (*"agent_cli.py:711 hardcodes
   mock_compatible=True, so the gate always passes"*) and `#67` (*"The live path raises NotImplementedError"*).

## The negative tests

Each validator ships with a test that makes it **fail**, and the test asserts the failure code. A validator
with no failing test is itself unverified.

| Validator | Fixture that must fail | Expected code |
|---|---|---|
| `V1` | a partition fixture with `ANSWER` at `n = 0` — the frozen Study 001 confirmatory partition *is* the fixture: `CALL` 453, `UNSUPPORTED` 453, `CLARIFY` 371, `ANSWER` 0 | `FAIL_NONVACUOUS` |
| `V2` | a metrics fixture carrying `no_call_accuracy: 0.0` beside an `ANSWER` row of zero — the `macro_behaviour_score` `return 0.0` path (`tool_use_policy.py:92-95`) reproduced as a test | `FAIL_VACUOUS_METRIC` |
| `V3` | a table row missing its artifact sha256 | `FAIL_UNATTACHED_METRIC` |
| `V4` | a metrics record using `request_for_info` outside the alias table | `FAIL_VOCABULARY` |
| `V5` | two prediction artifacts for one arm on `P-CONF` | `FAIL_ONE_SHOT` |
| `V6` | a row with no `device_class` | `FAIL_UNKEYED_ROW` |
| `V7` | an arm with no `S-REF` artifact | `FAIL_MISSING_SENTINEL` |
| `V8` | a census where `discovered != checked + blocked + skipped` | `FAIL_ACCOUNTING` |
| `V9` | a run record whose corpus hash does not match the artifact's | `FAIL_FINGERPRINT` |
| `V10` | a claim whose evidence path does not resolve | `FAIL_UNSUPPORTED_CLAIM` |
| `V11` | a table mixing a Study 001 frozen number with a Study 002 number | `FAIL_INCOMPARABLE` |
| `V12` | a comparison row with no `n` or margin | `FAIL_UNRESOLVED_ROW` |

## What the validators are for, in one sentence

They exist so that the failure mode the audit found — a number that is *wrong because it is detached from
its artifact* rather than because anyone was dishonest — cannot survive to the paper. The ledger is the
evidence that the failure mode is common: 19 transcription findings, 10 stale findings, and **zero** invented
ones.