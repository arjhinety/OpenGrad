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

---

## UP-0001 — a test needed a library CI does not install (`OURS`)

`tests/verification/test_flag_set.py` imported `huggingface_hub.errors` at test time. CI runs
`uv sync --locked --extra dev`, and the dev extra does not include `huggingface_hub`, so CI would have failed. Found by the
independent audit before push. Fix: the test that needs the library uses `pytest.importorskip`, and the rest no
longer need it. Lesson: a test may only import what the dev extra installs.

## UP-0002 — case-colliding fixture names (`OURS`)

Two fixtures named `c0-s0.yaml` and `C0-s0.yaml` coexisted on Windows, where the filesystem is case-insensitive,
and broke on Linux CI. Fixed by renaming (37088b6). Lesson: never let two paths differ only by case.

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

## UP-0005 — a cache miss is a `FileNotFoundError` (`NOT A BUG`)

In huggingface_hub 1.29.0, `LocalEntryNotFoundError` subclasses `FileNotFoundError` (its method resolution order is
`LocalEntryNotFoundError, FileNotFoundError, OSError, EntryNotFoundError`). Catching `FileNotFoundError` to mean
"the dataset is not cached" therefore also caught real failures such as a deleted pinned manifest, and reported
them as a missing input. `flag_set._input_missing` now matches only the cache miss and missing modules.

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
