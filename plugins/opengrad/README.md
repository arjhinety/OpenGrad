# OpenGrad development skills (Claude Code plugin)

Eleven skills that tell a coding agent how to do each OpenGrad workflow properly. Each gives the invariants,
the exact commands and the authoritative sources. Start with `/opengrad:opengrad-development`, which routes to
the rest.

| Skill | Covers |
|---|---|
| `opengrad-development` | repository map, environment, CI checks, commit and push rules, known failures |
| `opengrad-research-guardrails` | G1–G17, frozen studies and artifacts, preregistration, phase gating |
| `opengrad-data-pipeline` | adapters and versioning, normalization-v3, releases, mixtures, yield |
| `opengrad-annotation` | `opengrad-annotate`, P-DET, blinding, WIP exports versus gold freezes |
| `opengrad-experiments-readiness` | experiment configs, preflight, readiness gates, run store, results index |
| `opengrad-training` | SFT/DPO contracts, checkpoints, vision/MTP components, preferences, distillation |
| `opengrad-evaluation` | baseline and candidates, benchmarks, contamination, failures |
| `opengrad-promotion-gates` | promotion policies and how to test any gate |
| `opengrad-openweights-release` | Hugging Face, GGUF, ExecuTorch, optimization, publication checks |
| `opengrad-registry-provenance` | registries, provenance anchors, generated views, errata, incidents |
| `opengrad-skills-maintenance` | keeping these skills current |

The skills cite paths inside the OpenGrad repository, so use them from an OpenGrad checkout.

## Install

Inside the OpenGrad repository, Claude Code offers the marketplace once you trust the folder
(`.claude/settings.json` registers it with auto-update). Then install the plugin once:

```shell
/plugin install opengrad@opengrad
```

From anywhere else:

```shell
/plugin marketplace add arjhinety/OpenGrad
/plugin install opengrad@opengrad
```

## Updates

The plugin sets no `version`, so each new commit to the repository is a new release. With auto-update on, which
the project settings enable, Claude Code updates in the background after startup. Otherwise, run
`/plugin marketplace update opengrad`.

## Developing the skills

- Edit the skills in place under `plugins/opengrad/skills/`, then load the working copy with
  `claude --plugin-dir plugins/opengrad`.
- Validate with `claude plugin validate .` (marketplace) and `claude plugin validate plugins/opengrad` (plugin).
- CI runs `tests/skills/test_skills.py`, which fails when:
  - the codebase gains a package, console script or CLI subcommand that no skill covers;
  - a skill cites a path that no longer exists;
  - the marketplace and plugin manifests disagree.

See `skills/opengrad-skills-maintenance/SKILL.md`.
