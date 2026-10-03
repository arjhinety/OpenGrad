# Upstream issues log

Every problem OpenGrad hits is recorded here with whose problem it is, so that the ones that really belong to a
library or tool can later go upstream as an issue or a pull request. Started 2026-10-03 at the owner's request.

**Rules**
- **`OURS`**: our code, configuration or usage is wrong. Most problems are this. Recorded so the pattern is
  visible, and so nobody files it upstream by mistake.
- **`UPSTREAM`**: the fault is in a library or tool. It needs evidence before it gets this label:
  - a minimal reproduction outside OpenGrad's code;
  - the exact version;
  - a search of the upstream tracker for an existing issue.
- **`UNCLEAR`**: not yet reproduced in isolation. It stays here until it is.
- **`NOT A BUG`**: documented upstream behaviour we had to learn. Kept so the lesson is not relearned.
- Nothing is filed upstream without the owner's word. A filed issue or PR gets its link in the entry.
- An entry is added in the same commit as the fix or workaround. Entries are not rewritten; a later line records
  what changed (reproduced, filed, fixed upstream).

| ID | Date | Where | Class | Summary | State |
|---|---|---|---|---|---|
| UP-0001 | 2026-10-03 | OpenGrad tests | `OURS` | A test imported `huggingface_hub`, which CI's dev extra does not install | Fixed (`pytest.importorskip`) |
| UP-0002 | 2026-10-02 | OpenGrad tests | `OURS` | Two fixtures differed only by case (`c0-s0.yaml`, `C0-s0.yaml`); Linux saw a duplicate | Fixed (37088b6) |
| UP-0003 | 2026-10-03 | pyarrow 25.0.1 | `UNCLEAR` | No type information shipped, so mypy reports `import-untyped` | Worked around (`# type: ignore[import-untyped]`) |
| UP-0004 | 2026-10-03 | OpenGrad CI | `OURS` | Pinned `actions/checkout` and `actions/setup-node` v4.4.0 run on deprecated Node.js 20 | Open: bump the pins |
| UP-0005 | 2026-10-03 | huggingface_hub 1.29.0 | `NOT A BUG` | `LocalEntryNotFoundError` is a `FileNotFoundError`, so catching the latter also hides a cache miss | Handled (`flag_set._input_missing`) |
| UP-0006 | before 2026-10-03 | agy 1.2.14 (Gemini CLI) | `UNCLEAR` | Ignores standard input when started from Python | Worked around (input as a file) |
| UP-0007 | before 2026-10-03 | cline 3.0.66 | `UNCLEAR` | `--json` text output interleaves terminal colour codes | Worked around (stripped) |
| UP-0008 | 2026-10-03 | OpenGrad process | `OURS` | The publication-hygiene scan was not rerun after a rubric edit, so a committed phrase tripped it | Fixed (reworded, scan rerun) |
| UP-0009 | 2026-10-03 | OpenGrad verification | `OURS` | The triage report checked the flag set against the population it was reporting, i.e. against itself | Fixed (reads the members file) |
| UP-0010 | 2026-10-03 | OpenGrad verification | `OURS` | Triage `verify` crashed on absent populations; no way to restore them; first `--restore` could write half | Fixed (`BLOCKED_INPUT_MISSING`, `--restore`, check before write) |

---

## UP-0001 — a test needed a library CI does not install (`OURS`)

`tests/verification/test_flag_set.py` imported `huggingface_hub.errors` at test time. CI runs
`uv sync --locked --extra dev`, and the dev extra does not include `huggingface_hub`, so CI would have failed. Found by the
independent audit before push. Fix: the test that needs the library uses `pytest.importorskip`, and the rest no
longer need it. Lesson: a test may only import what the dev extra installs.

## UP-0002 — case-colliding fixture names (`OURS`)

Two fixtures named `c0-s0.yaml` and `C0-s0.yaml` coexisted on Windows, where the filesystem is case-insensitive,
and broke on Linux CI. Fixed by renaming (37088b6). Lesson: never let two paths differ only by case.
- **Correction (audit, 2026-10-03):** they did not coexist on Windows. There the second file overwrote the first;
  only Linux held both.

## UP-0003 — pyarrow has no type information (`UNCLEAR`)

`import pyarrow.parquet as pq` under `mypy` strict (mypy 2.3.1) reports `import-untyped`, because pyarrow 25.0.1
ships no `py.typed` marker (checked: no `pyarrow/py.typed` in the installed package). Worked around per import.
- **Before filing:** search the apache/arrow tracker for an existing typing or `py.typed` issue, and check
  whether a community `pyarrow-stubs` package already covers this.
- **Likely outcome:** a known upstream limitation, in which case add a link here rather than filing.

## UP-0004 — CI actions on a deprecated runtime (`OURS`)

GitHub Actions warns that `actions/checkout@11d5960a…` (v4.4.0) and `actions/setup-node@49933ea5…` (v4.4.0) target
Node.js 20 and are being forced onto Node.js 24 (run 37123017671). Our pins are old; the fix is to resolve newer
release tags to SHAs and bump them. Separately, `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19, so CI may
change under us then.
- **Correction (audit, 2026-10-03):** the same run also flags `astral-sh/setup-uv@d4b2f3b6…` (v5.4.2) as Node.js
  20, in both the `test` and `test-windows` jobs. It is a third pin to bump.

## UP-0005 — a cache miss is a `FileNotFoundError` (`NOT A BUG`)

In huggingface_hub 1.29.0, `LocalEntryNotFoundError` subclasses `FileNotFoundError` (its method resolution order is
`LocalEntryNotFoundError, FileNotFoundError, OSError, EntryNotFoundError`). Catching `FileNotFoundError` to mean
"the dataset is not cached" therefore also caught real failures such as a deleted pinned manifest, and reported
them as a missing input. `flag_set._input_missing` now matches only the cache miss and missing modules.
- **Correction (audit, 2026-10-03):** "documented" overstates it. The subclassing is visible in the class
  hierarchy; the exception's docstring does not mention `FileNotFoundError`.

## UP-0006 — agy ignores standard input when started from Python (`UNCLEAR`)

Recorded in the annotation skill before this log existed: started from Python, `agy` ignores standard input, so
`scripts/run_external_annotation.py` writes the input as `input.md` in an empty temporary directory. Version now
installed: 1.2.14.
- **Before filing:** reproduce with a minimal `subprocess.run(["agy", ...], input=...)`, on Windows and on Linux,
  and compare with a shell pipe. It may be a Windows console or TTY detection issue rather than agy's.

## UP-0007 — cline's `--json` output carries colour codes (`UNCLEAR`)

Also recorded before this log: `cline --json` interleaves ANSI colour codes in its text output, so the runner
strips them. Version now installed: 3.0.66.
- **Before filing:** reproduce with `NO_COLOR=1` and with output redirected to a file. A JSON mode that emits
  colour codes when not attached to a terminal would be an upstream bug; one that honours `NO_COLOR` would not.

## UP-0008 — a hygiene finding committed (`OURS`)

`configs/annotation/first-reply-review-v1.rubric.md` described a decline with a phrase the publication-hygiene
scan's `user_reference` rule rejects: the two words "user" and "asked" in sequence. The scan had been run before the rubric was written and
not again before the commit (ad05736), so CI would have failed. Reworded ("what the first message asks"). Lesson:
rerun `scripts/repo/check_publication_hygiene.py` after the last edit, not before it.

## UP-0009 — a report checked the flag set against itself (`OURS`)

`flag_triage_report.report` passed `set(source_of)` as the flag set's members, but `source_of` is built from the
triage population, the very thing the check is meant to compare with the flag set. The check could never fail.
The result was still right, because the population equals the flag set, but nothing re-read the members file.
Found by the independent audit before push. The report now reads the hash-checked members file and the
populations manifest's `members_sha256`. Lesson: a check must take its expected value from an independent source.

## UP-0010 — populations that could not come back (`OURS`)

The triage populations (17.7 MB) may be left uncommitted, but `verify()` raised `FileNotFoundError` when they
were absent, and `write()` refused to recreate them once the manifest existed. Found by the audit. Fixed: absent
files report `BLOCKED_INPUT_MISSING`, and `--restore` rebuilds them from the pinned release, writing only if the
rebuild reproduces the committed manifest exactly. The first version of `--restore` wrote one file before
finding the other differed, so a refusal could leave a half-restored directory; its own test caught that before
commit, and it now checks both before writing either.
