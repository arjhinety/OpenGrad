# 26 — P-DET annotator checklist

Read before each session. The rules are in 22 §1–§4 and 23 §2–§3 (the *Rubric* panel, key `i`); this
page only reminds.

**Before you start:** `python -m opengrad.verification.pdet --verify` must say `PASS`. Run it again when
you finish.

1. **Label the behaviour, not the words.** Decide what the assistant response *does* in its full situation
   — the user message, any context shown and the tools offered — not which keywords or punctuation it uses.
2. **A question mark is not automatically CLARIFY.** Ask whether the user's reply is *necessary* to complete
   the task correctly. A courtesy or rhetorical question after an answer is DIRECT (rules 1, 4, 8).
3. **Refusal-like wording is not automatically UNSUPPORTED.** Ask whether substantive content was *also*
   delivered (rule 3), and whether the missing thing is a capability or information the user could supply
   (rules 2, 6).
4. **Prose about a call is not a call.** CALL needs a machine-readable payload (rule 5).
5. **Work through the decision tree in order** (22 §3). Cite the boundary rule when one decided the label.
6. **Do not force a class.** If two modes are equally supported, the response is empty or truncated, the
   context needed is missing, or no tools are offered where that decides it: choose UNKNOWN and set the
   ambiguity status.
7. **A rationale is optional** (amendment 29). Pressing a mode saves the label and moves on. Write a short
   note in the rationale or note box when it helps you; leaving it out does not make a label less certain.
   Mark real uncertainty with the flag, or with UNKNOWN and its ambiguity status.
8. **Do not balance.** The counts in the Items panel track progress. A mode with few examples is a finding,
   not a target.
9. **Use only what the screen shows.** Do not open the population file or other P-DET files to look up an
   item's sampling metadata, and do not look at another pass.
10. **No assistants.** Do not ask an LLM, a classifier or another annotator for a suggestion. These labels
    are the human reference.

Flag (`f`) anything you want to revisit. You can change a label until the gold freeze; every change is
recorded with its time and reason.

Keys: `1` CALL · `2` DIRECT · `3` CLARIFY · `4` UNSUPPORTED · `5` UNKNOWN (set the ambiguity status first)
· `Enter` save from the rationale · `n`/`p` next/previous · `s` skip · `f` flag · `u` undo · `i` rubric
· `/` items.
