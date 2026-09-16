---
name: opengrad-skills-maintenance
description: How the OpenGrad development skills in .claude/skills/ stay correct as the repository grows — which skill owns which area, when a change must update a skill in the same commit, the tests/skills/test_skills.py coverage and reference checks that fail CI on drift, and how to add a new skill. This skill should be used when tests/skills fails, when adding a package, console script, CLI subcommand, gate, policy version or workflow to OpenGrad, and when creating or restructuring an opengrad-* skill.
---

# Maintaining the OpenGrad skills

The skills in `.claude/skills/` are development instructions that ship with the repository. They are only
useful while they describe the repository as it is, so they are held to the same rule as status words:
**a change that alters a workflow updates its skill in the same commit.**

## What CI enforces (`tests/skills/test_skills.py`)

1. **Well-formed skills.** Every `.claude/skills/*/SKILL.md` has YAML frontmatter whose `name` equals its
   directory and whose `description` is substantive, plus a "Keeping this skill current" section.
2. **No dead references.**
   - Every backticked repository path in a skill or its `references/` files must exist. Placeholders containing
     `<`, `*`, `{` or `…` are skipped.
   - Every `python -m opengrad.<module>` must resolve to a module file.
3. **Coverage of the codebase.** The union of all skill text must mention:
   - every `src/opengrad/<package>` that contains code;
   - every console script in `pyproject.toml` `[project.scripts]`;
   - every top-level `opengrad <subcommand>` in `src/opengrad/cli.py`, and every `opengrad results <subcommand>`;
   - every subcommand of `opengrad-data`, `opengrad-contamination`, `opengrad-benchmark`, `opengrad-annotate` and
     `python -m opengrad.optimization`.

So when OpenGrad grows a package, CLI command or subcommand, CI fails until a skill explains how to use it
properly, and when a path is renamed or removed, CI fails until the skill that cites it is fixed.

## Ownership

| Area | Skill |
|---|---|
| Entry point, environment, CI, commit/push rules, known failures | `opengrad-development` |
| Studies, preregistration, frozen artifacts, claims | `opengrad-research-guardrails` |
| `data/`, `formatting/`, releases, normalization | `opengrad-data-pipeline` |
| `annotation/`, `verification/` P-DET check | `opengrad-annotation` |
| `experiments/`, `readiness.py`, `hardware/`, `config/`, `results/`, `agent_cli.py` | `opengrad-experiments-readiness` |
| `training/`, `preferences/`, `distillation/`, `checkpoints/` | `opengrad-training` |
| `evaluation/`, `benchmarks/`, `contamination/`, `failures/` | `opengrad-evaluation` |
| `promotion/`, gate design and testing | `opengrad-promotion-gates` |
| `publication/`, `optimization/`, `release/`, `hf/`, `scripts/modal/` | `opengrad-openweights-release` |
| `registry/`, provenance, `reporting/`, reports, errata, incidents | `opengrad-registry-provenance` |

## Updating a skill

- **Put new commands, rules and gates in the owning skill,** at the point of the workflow where they apply.
  Prefer pointing at the authoritative doc or code path over copying its content; copies drift.
- **Record why a rule exists** in one line when it came from a failure. The next reader needs the reason to
  apply it well.
- **Keep facts that change often in files:** `references/` under the skill (e.g.
  `opengrad-development/references/known-failures.md`), not in the frontmatter description.
- **Keep SKILL.md lean** (roughly under 200 lines). Move long procedures to `references/<topic>.md` and link them.

## Adding a skill

Add one only for a new *area* with its own workflow and invariants; otherwise extend an existing skill.
1. Create `.claude/skills/opengrad-<area>/SKILL.md` with frontmatter `name` and a third-person `description`
   saying what it covers and when it should be used.
2. Write imperative instructions: invariants first, then workflows with exact commands, then sources of truth,
   then "Keeping this skill current".
3. Add the area to the ownership table here and to the routing table in `opengrad-development`.
4. Run `.venv/Scripts/python.exe -m pytest tests/skills -q`.

## Keeping this skill current

Update this skill when the enforced checks in `tests/skills/test_skills.py` change, or when ownership moves
between skills.
