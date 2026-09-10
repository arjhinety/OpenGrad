# Agent & Harness Integration Guide

OpenGrad is the scientific source of truth. Every integration is a thin orchestration and policy layer over the same documented CLI: it invokes `opengrad … --json`, summarizes state, optionally records receipts, and blocks unsafe experiment transitions. It does not reimplement data fingerprints, contamination, evaluation metrics, experiment state, checkpoint policy, training math, DPO, distillation, or RL.

The integration boundary is harness-agnostic by design, so training and evaluation work under any harness:

```text
any harness (Command Code, Claude, Cursor, DSH, a shell script, CI)
  → OpenGrad MCP server (typed tools) or direct `opengrad … --json`
  → opengrad CLI and canonical OpenGrad artifact stores
```

## MCP server (recommended)

`integrations/opengrad-mcp/` is a stdio [Model Context Protocol](https://modelcontextprotocol.io) server with no third-party dependencies (Node.js ≥ 20). It exposes OpenGrad's documented operations as typed tools for any MCP-capable harness.

Register it with Command Code:

```bash
cd /path/to/OpenGrad
./integrations/opengrad-mcp/install.sh project   # or: user
```

Equivalently, by hand:

```bash
cmd mcp add --scope project --env OPENGRAD_ROOT="$PWD" opengrad \
  -- node "$PWD/integrations/opengrad-mcp/src/index.js"
```

Verify with `cmd mcp list` and `/mcp`. Tools appear as `mcp__opengrad__opengrad_*`.

For any other MCP client, use [`integrations/opengrad-mcp/mcp.example.json`](../integrations/opengrad-mcp/mcp.example.json):

```json
{
  "mcpServers": {
    "opengrad": {
      "command": "node",
      "args": ["/path/to/OpenGrad/integrations/opengrad-mcp/src/index.js"],
      "env": { "OPENGRAD_ROOT": "/path/to/OpenGrad" }
    }
  }
}
```

See `integrations/opengrad-mcp/README.md` for the full tool catalog, environment variables, and upgrade notes.

## Direct CLI (no MCP)

A harness without MCP support needs nothing installed. Drive the same contract directly and parse the JSON envelopes:

```bash
opengrad status --json
opengrad validate --json
opengrad readiness --json
```

## Authoritative commands

Every machine-facing call requests `--json`:

```text
opengrad status --json
opengrad doctor --json
opengrad validate --json
opengrad readiness [config] --json
opengrad gpu-smoke [config] --json
opengrad validate-data <records> --mode sft --json
opengrad inspect-template [--record <record>] --json
opengrad preflight <config> --json
opengrad baseline --config <config> [--dry-run] [--limit N] --json
opengrad train <config> [--dry-run] --json
opengrad evaluate <target> --suite <suite> [--dry-run] --json
opengrad compare <baseline> <candidate> --json
opengrad failures <run_dir> --json
opengrad experiment list|show|diff --json
opengrad checkpoint list|inspect --json
```

`gpu-smoke` is bounded: it may load the pinned Qwen model and generate one output, but never trains. Its receipt is `GPU_BOUNDARY_VERIFIED`, not `BASELINE_COMPLETE`.

## Permission tiers and hard guards

- **READ_ONLY:** status, doctor, validation, manifest/template, experiment/checkpoint/failure inspection, and diffs.
- **SAFE_WRITE:** dry-runs, preflight, and bounded smoke/evaluation paths.
- **HIGH_IMPACT:** real baseline, real SFT, full evaluation, and the B0/post-SFT workflows.

The MCP server applies the same monotonic invariants as the CLI:

- **Static guards** run first: evaluation-only configs or records can never enter SFT, and real SFT requires an explicit immutable config.
- **Gate checks** run for HIGH_IMPACT tools (except explicit dry-runs). Real SFT requires `ready_for_sft`; a real baseline requires prerequisite readiness plus a verified GPU boundary.
- Operations that *establish* B0 are not required to already own B0's post-run artifacts, so `opengrad_b0_workflow` is not deadlocked by the boundary it produces.
- Unknown high-impact operations fail closed, and there is no `--force` override.

This is the invariant `NO_REAL_SFT_WITHOUT_VALID_B0`; CPU deterministic output never satisfies it.

Evaluation-only manifests remain protected. The held-out manifest must stay excluded from training manifests. Floating model/dataset revisions, failed contamination checks, failed tokenizer/template/parser checks, invalid experiment identity, unavailable storage, and invalid checkpoint lineage remain blockers.

Canonical failure values retain:

```json
{"ok": false, "code": "CONTAMINATION_FAILURE", "message": "...", "blocking": true}
```

## Lifecycle

```text
PRE_BASELINE
→ readiness and GPU_BOUNDARY_VERIFIED
→ opengrad_b0_workflow (readiness → GPU boundary → real baseline)
→ BASELINE
→ frozen SFT preflight → explicit authorization → SFT
→ checkpoint registration → evaluation → residuals → regression → REVIEW
```

The integration does not start training merely because readiness passes, and a successful model load is not `BASELINE_COMPLETE`. Post-SFT analysis remains residual-first and uses OpenGrad's behavior taxonomy (CALL, DO_NOT_CALL, ASK_FIRST, SELECT, GROUND_ARGUMENTS, CHAIN, PARALLELIZE, RECOVER, STOP) rather than a single aggregate score. DPO/OPD/RL orchestration is intentionally absent.

## Tests

Run the integration tests and OpenGrad CPU tests before any GPU action:

```bash
cd integrations/opengrad-mcp && npm test && npm run check
cd ../.. && .venv/bin/python -m pytest -q
```

No test invokes real SFT or GPU work. The server has no third-party runtime dependencies, so it cannot drift with a vendor SDK.
