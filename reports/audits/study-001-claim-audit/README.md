# Study 001 claim audit

The record of every claim in Study 001 that the committed artifacts did not support, and how each
was resolved. The lessons drawn from it, as rules for later studies, are in
[`docs/research/GUARDRAILS.md`](../../../docs/research/GUARDRAILS.md). The readable version is the
[claim audit PDF](https://opengrad.arjhinety.com/studies/001/claim-audit.pdf).

| File | Contents |
|---|---|
| `findings.json` | The 93 findings of the audit of commit `fa7c510` (2026-09-13): 13 high, 42 medium, 38 low. Locations are file and line at that commit |
| `resolutions.json` | One resolution per finding: status, where it was fixed (with the commit), and a note where the fix needed one. All 93 are `RESOLVED` |
| `recheck-2026-09-13.json` | The 7 findings of a recheck of opengrad-site at `3aac71d` before the Study 001 freeze, 165 claims checked. `repeat_of` names the original finding when the site repeated an error already fixed in the repository |

## Found after the audit

- **`arrochi112/OpenGrad-ToolPolicy-Canonical-v2-minus-xlam` dataset card** (not in the audit's
  card set). It said the ablation runs were planned, though both had run and were negative, and it
  did not mention that the view carries all 18,114 of the parent's refusal-shaped ANSWER targets
  (15.6% of its 115,895 records, against 10.5% in the parent). Corrected in the template
  (`d7689e6`) and on the Hub (commit `055b6f17`, 2026-09-13), which is the revision its `study-001`
  tag points at.

## Method

Four independent read-only audits covered the top-level docs, the training and evaluation reports,
the capability and quantization reports, and the data and public surfaces (Hugging Face cards and
the site). Each recomputed values in Python from `runs/*/eval` metrics, the ledgers,
`results/final_campaign_verdict.json`, `results/benchmarks/h200` and `results/quantization`, re-ran
the checkpoint-selection and promotion gates, and hashed files against the freeze manifests.
Rounding that does not change a conclusion was not recorded. Each resolution was then checked
independently against the corrected files.

Frozen, hash-pinned files were not edited; they are corrected in
[`reports/ERRATA.md`](../../ERRATA.md).

## Fields

`findings.json`: `id`, `severity` (`HIGH` a reader draws a wrong conclusion, `MED`, `LOW`), `area`
(`D` top-level docs, `T` training and evaluation reports, `C` capability and quantization, `P` data
and public surfaces), `location`, `claim` (as written), `actual` (what the artifacts show),
`category`, `pinned` (the freeze manifest that pins the file, or empty), `public`.

`resolutions.json`: `id`, `status`, `where`, optional `note`.
