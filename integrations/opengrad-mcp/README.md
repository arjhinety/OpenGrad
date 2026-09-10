# OpenGrad MCP server (harness-agnostic)

A stdio Model Context Protocol server that exposes OpenGrad's documented CLI as typed tools, so any MCP-capable harness (Command Code, Claude, Cursor, DSH, …) can operate OpenGrad without a vendor-specific plugin.

OpenGrad remains the scientific source of truth. This server is a thin orchestration and policy layer: it invokes `opengrad … --json`, summarizes state, optionally writes receipts, and blocks unsafe experiment transitions. It does not reimplement data fingerprints, contamination, evaluation metrics, experiment state, checkpoint policy, training math, DPO, distillation, or RL.

## Requirements

- Node.js ≥ 20
- An OpenGrad checkout with a working CLI at `.venv/bin/opengrad` (or `opengrad` on `PATH`)

No third-party npm dependencies.

## Command Code

Register the server (project scope shown; use `--scope user` for all projects):

```bash
cmd mcp add --scope project --env OPENGRAD_ROOT="$PWD" opengrad -- node "$PWD/integrations/opengrad-mcp/src/index.js"
```

Verify with `cmd mcp list` and `/mcp`. Tools appear as `mcp__opengrad__opengrad_*`.

## Other harnesses

Point any MCP client at the same command. A neutral config fragment lives in [`mcp.example.json`](./mcp.example.json):

```json
{ "mcpServers": { "opengrad": { "command": "node", "args": ["<repo>/integrations/opengrad-mcp/src/index.js"], "env": { "OPENGRAD_ROOT": "<repo>" } } } }
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENGRAD_ROOT` | OpenGrad checkout root (defaults to the package's repository) |
| `OPENGRAD_EXECUTABLE` | Override the `opengrad` executable path |
| `OPENGRAD_MCP_TIMEOUT_MS` | Per-command timeout (default 120000) |
| `OPENGRAD_MCP_RECEIPTS` | Receipt JSONL path, or `off` to disable |

## Tool catalog and tiers

| Tool | Tier | Purpose |
| --- | --- | --- |
| `opengrad_status`, `opengrad_doctor`, `opengrad_validate` | READ_ONLY | Refresh authoritative repository state |
| `opengrad_readiness`, `opengrad_gpu_smoke`, `opengrad_preflight` | SAFE_WRITE | Summarize gates; GPU smoke is bounded and receipt-producing |
| `opengrad_validate_data`, `opengrad_inspect_template` | READ_ONLY | Check canonical data and rendering boundary |
| `opengrad_experiment_*`, `opengrad_checkpoint_*`, `opengrad_failures`, `opengrad_eval_compare` | READ_ONLY | Inspect OpenGrad records/artifacts |
| `opengrad_baseline_run` | HIGH_IMPACT unless `dry_run: true` | Run the frozen B0 measurement through OpenGrad |
| `opengrad_sft` | HIGH_IMPACT unless `dry_run: true` | Launch the documented OpenGrad trainer |
| `opengrad_eval_run` | HIGH_IMPACT unless `dry_run: true` | Evaluate through OpenGrad, not a second scorer |
| `opengrad_b0_workflow`, `opengrad_post_sft_workflow` | HIGH_IMPACT | Deterministic multi-stage orchestration |

DPO, preference generation, on-policy distillation, RL, judge orchestration, promotion, and arbitrary shell execution are intentionally absent.

## Permission model

- **Static guards** run before anything else: evaluation-only configs/records can never enter SFT, and real SFT requires an explicit immutable config.
- **Gate checks** run for HIGH_IMPACT tools (except explicit dry-runs). Real SFT requires `ready_for_sft`; a real baseline requires prerequisite readiness plus a verified GPU boundary.
- Operations that *establish* B0 are not required to already own B0's post-run artifacts, so `opengrad_b0_workflow` is not deadlocked by the boundary it produces.
- Unknown high-impact operations fail closed. There is no `--force` bypass.

## Lifecycle

```text
PRE_BASELINE
  → readiness + GPU_BOUNDARY_VERIFIED
  → opengrad_b0_workflow (readiness → GPU boundary → real baseline)
  → BASELINE
  → frozen SFT preflight → explicit authorization → SFT
  → checkpoint registration → evaluation → residuals → regression → REVIEW
```

`opengrad_gpu_smoke` is only `GPU_BOUNDARY_VERIFIED`; it is not a baseline score and never starts SFT. A successful model load is not `BASELINE_COMPLETE`.

## Testing

```bash
cd integrations/opengrad-mcp
npm test
npm run check
```

Tests mock the service for policy and workflow logic and use real Node subprocesses for the argv bridge. No test runs real SFT or GPU work.
