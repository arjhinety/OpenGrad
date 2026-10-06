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
| UP-0011 | 2026-10-03 | OpenGrad registry validation | `OURS` | A malformed `papers.yaml` makes `opengrad-validate` die with a traceback, not a validation error | Open |
| UP-0012 | 2026-10-03 | OpenGrad runner, agy 1.2.x | `OURS` (+ `UNCLEAR` upstream) | agy ran labelling batches with every tool auto-approved: web search, and file browsing outside its empty directory | Gated in the runner (2026-10-04); upstream part open |
| UP-0013 | 2026-10-04 | OpenGrad tests | `OURS` | A pushed commit grew the hygiene allowlist, and a test pinning the credential count failed CI | Fixed (count 3 -> 9, ERRATA §33) |
| UP-0014 | 2026-10-06 | `tool_use_policy.py`, `study_002_gate.py` | `OURS` | Regression checks are skipped when the baseline lacks the metric; v6 and the gate require only two baseline metrics | Open; no code changed (formulas draft, open question 8) |
| UP-0015 | 2026-10-06 | formulas draft | `OURS` | The draft misread the resolvable margin for a paired difference and misstated four other points; independent reviews caught them before push | Fixed in the draft and its test |
| UP-0016 | 2026-10-06 | gstack `outside-review-result.ts` (gstack 1.91.27.0) | `UNCLEAR` | The Codex review validator reported a critical finding because Codex wrote "no [P1] findings" in Chinese | Open; read the review text, not only the verdict |

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

## UP-0011 — the validator crashes on a malformed reference file (`OURS`)

An unquoted value containing ": " made `docs/references/papers.yaml` invalid YAML. `opengrad-validate` then died
with a PyYAML `ScannerError` traceback from `validators._paper_ids`, which loads the file a second time outside
`check_references`' own load-error handling, instead of reporting a validation error. PyYAML's behaviour is
correct (`NOT A BUG` on its side). The broken file reached a local commit because the check and the commit were
chained with `;` rather than `&&`; it was fixed and amended before any push. Lessons: chain a commit after its
checks with `&&`; `_paper_ids` should reuse the reported load error.
- **Again, 2026-10-04:** a check piped into `tail` (`pytest ... | tail -1 && git commit`) let a failing index test
  through, because a pipeline's exit status is the last command's; and a later commit was gated on its tests but
  not on the hygiene scan, which had flagged a home-folder-style path in a new test. Both were caught at once and
  amended before any push. Capture each check's exit code before filtering its output, and gate on every check.

## UP-0012 — agy could search the web and browse files during labelling (`OURS`, with an `UNCLEAR` upstream part)

**What allowed it.** This machine's agy settings (`~/.gemini/antigravity-cli/settings.json`) have `toolPermission:
always-proceed` and `allowNonWorkspaceAccess: true`, so in print mode agy runs any tool without asking. Our runner
(`scripts/run_external_annotation.py`) starts agy with `--sandbox`, which restricts terminal commands only, and
overrides neither setting. Its prompt tells the model to open nothing else and run nothing; nothing enforces it.
That is ours. Whether `--sandbox` should also cover web and file tools, and whether print mode should honour a
deny list, is `UNCLEAR` until reproduced with a minimal prompt.

**How it was seen.** The run records keep only agy's printed answer, so a tool call is not in them. agy keeps
each conversation as a SQLite file of steps in `~/.gemini/antigravity-cli/conversations`; a model step names each
tool it calls (protobuf field 20.7.2) with its JSON arguments (20.7.3). That format is read from this machine's
files, not documented. `scripts/archive_external_model_labels.py` now matches each attempt to the conversation
created in its first minute and records web-tool calls per attempt in the archive manifest. cline reports its
tool calls itself (`toolCallCount`).

**The trial (`first-reply-review-v1-trial`, 2026-10-03).** Gemini (agy 1.2.16): 5 attempts, all 5 visible, 0
web-tool calls. A separate scan of the store found its only tool calls were 11 `view_file` reads of its own input;
that count is not in the manifest, which records web tools. DeepSeek (cline 3.0.66): 5 attempts, all visible, 0
tool calls. Web-tool counts recorded in
`reports/study-002/flag-triage/provenance/external-models/first-reply-review-v1-trial.external-models.audit-trail.manifest.json`.

**Earlier runs (counts only; 278 of the 287 Gemini attempts before the trial matched to a conversation).**
- `search_web` calls: 10, in 5 attempts. Two of those attempts kept their labels, both in `pdet-coverage-v2`
  (batch 10 attempt 2, batch 24). The others (`pdet-coverage-v1` batch 11, `pdet-coverage-v2` batch 13,
  `answer-strata-v1` batch 01) recorded no labels.
- In `pdet-coverage-v1` and `pdet-coverage-v2`, Gemini also listed, searched and read files across the home folder
  and the OpenGrad repository. In kept batches it read paths under `.annotation/` (7 batches in v1, 9 in v2), and
  in 6 kept batches it read another model's answer file (`view_file` in v1 batches 10, 11 and 18 and v2 batches 04
  and 10; `grep_search` in v2 batch 20). All
  three models ran at the same time, so those answers existed. `pdet-coverage-v1-routing` browsed the home folder,
  not the repository.
- No such calls in `answer-strata-v1`'s kept batches, `punans-v1`, `punans-v2`, `punans-v2-trial` or
  `punans-v2-constructed`, in every batch that could be matched: their only tool reads were of their own input.
  Three kept batches cannot be matched (`answer-strata-v1` 46 and 50, `punans-v1` 04).

**Not yet decided.** What this means for the P-DET-COVERAGE references and what was built on them, and how agy is
locked down before the full triage run, are the study owner's decisions.
- **2026-10-04, fix:** `scripts/agy_tool_gate.py`, a PreToolUse hook written into each run directory
  (`.agents/hooks.json`), allows only `view_file` of the input and denies everything else; the runner refuses an
  attempt in which the hook never allowed a read of the input. Canary (agy 1.2.16): told to search the web and
  list a directory, the model called `search_web` once and the gate log shows it denied; its result step in the
  store is 116 bytes with a field none of September's 10 executed searches had (theirs were 3.9 to 12.3 KB). "0
  web steps" proves nothing here, since executed searches of this kind showed none either. The model never called
  `list_dir`. The canary's run directory and gate log were deleted; its conversation file remains in the local
  agy store.
  The owner chose this over changing the global agy settings. Incident: `docs/INCIDENT_LOG.md` INC-0002.

## UP-0013 — a ratchet test outside the targeted set (`OURS`)

`4aaf8e5` committed the triage population with six allowlisted occurrences of the upstream token (ERRATA §33).
`tests/publication/test_publication_hygiene.py` pins the number of accepted `api_secret` findings (3), so CI failed
on Linux and Windows (run 37134813092). The local checks ran the scan itself and the targeted tests, but not
`tests/publication`, and the audit did not either. Fixed: the count is 9, with both errata named. Lesson: a change
to `scripts/repo/publication_hygiene_allowlist.yaml` runs `tests/publication`.

## UP-0014 — regression checks that skip silently (`OURS`)

The non-regression loop that `tool_use_promotion_v5` and v6 inherit (`PromotionPolicyV2.evaluate`) checks call
precision, call recall, clarification accuracy and unsupported accuracy only when both the candidate and the baseline
carry the metric. v6 requires from the baseline only `call_f1` and `answer_rate` (`V6_BASELINE_METRICS`), and the
gate the same (`BASELINE_METRICS`); v6 alone also does not require call precision or call recall from the candidate.
A baseline without one of the four passes that check by skipping it. v6's comment says it does not fill in absent
metrics the way v5 did, so this looks unintended. Found by the Opus review of the formulas draft, confirmed in the
code. Not fixed: the policy is preregistered, and changing what it requires is the owner's decision.

## UP-0015 — errors in the formulas draft, caught by review (`OURS`)

`EVALUATION-FORMULAS-DRAFT.md` (8ff3c8e, cbcd437) was reviewed by Codex and an Opus subagent before any push.
- Open question 3 doubled the resolvable margin for a paired difference. The existing margin already equals a paired
  difference's worst-case half-width; doubling is a second reading, which the draft had not called a reading.
- It said comparisons are made at 9 decimal places; the policy uses 6.
- It said check 14 applies the margin to paired differences; the code does not know.
- It said the answer and refusal instrument was not built; 46 names the frozen classifier v2.
- It said its test recomputes every number; some were read from the trial report.

One reviewer finding was itself wrong: `CALL.on_ambiguous_in_M` is a `CALL` row, not a global one. Lesson: a draft that
restates rules gets its numbers and its readings checked against the code, and every reviewer finding is checked
too before it is relayed.

Later, 2026-10-06: the pre-push audit of 7e97306 found one more. The draft called condition 1's margin "two readings"
(10 points, or $M(n)$), but 06 says 11 "fixes that margin" and "no claim in this study uses a margin below 10pp": the
margin is 10 points and $M(n)$ is a further requirement. Fixed in the draft; the test now checks each quoted passage
against its source document.

## UP-0016 — a review verdict parsed from the wrong language (`UNCLEAR`)

The Codex audit of 7e97306 answered in Chinese. Its first line, "无 **[P1]** 发现", means "no [P1] findings", but the
gstack validator (`lib/outside-review-result.ts`, gstack 1.91.27.0) matched the literal `[P1]` and printed
`VERDICT: findings`, `FINDINGS: P1`, exit 3. The review itself had one P2 finding. Not yet reproduced in isolation or
searched for upstream, so `UNCLEAR`. Workaround: read the review text before trusting the verdict line.
