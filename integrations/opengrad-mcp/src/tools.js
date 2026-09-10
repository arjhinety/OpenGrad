import { stableError } from "./core.js";
import { classifyOpenGradOperation, evaluateHighImpactGate, evaluateStaticGuard } from "./guardian.js";
import { b0Workflow, postSftWorkflow } from "./workflows.js";

const string = (description) => ({ type: "string", description });
const boolean = (description, defaultValue = undefined) => ({ type: "boolean", description, ...(defaultValue === undefined ? {} : { default: defaultValue }) });
const integer = (description) => ({ type: "integer", description });

const EVAL_SUITES = ["smoke", "tool_use_core", "regression_core", "agent_transfer", "full_post_training", "speculative_decoding"];

const passthrough = (properties = {}, required = []) => ({ type: "object", properties, required, additionalProperties: false });

async function canonical(serviceCall) {
  try {
    const response = await serviceCall();
    return response ?? { ok: false, code: "COMMAND_FAILED", message: "OpenGrad returned no response", blocking: true };
  } catch (error) {
    return stableError(error);
  }
}

export function createTools(service) {
  return [
    {
      name: "opengrad_status",
      tier: "READ_ONLY",
      description: "Read the authoritative current OpenGrad repository, experiment, baseline, checkpoint, and environment state.",
      inputSchema: passthrough(),
      handler: (_args, { signal } = {}) => canonical(() => service.status({ signal })),
    },
    {
      name: "opengrad_doctor",
      tier: "READ_ONLY",
      description: "Inspect OpenGrad local tooling and environment through its documented doctor command.",
      inputSchema: passthrough(),
      handler: (_args, { signal } = {}) => canonical(() => service.doctor({ signal })),
    },
    {
      name: "opengrad_validate",
      tier: "READ_ONLY",
      description: "Validate OpenGrad registries and return the machine-readable command outcome.",
      inputSchema: passthrough(),
      handler: (_args, { signal } = {}) => canonical(() => service.validate({ signal })),
    },
    {
      name: "opengrad_readiness",
      tier: "SAFE_WRITE",
      description: "Evaluate authoritative OpenGrad gates for safely beginning real baseline/SFT work; never starts training.",
      inputSchema: passthrough({ config: string("Optional OpenGrad experiment config path.") }),
      handler: (args, { signal } = {}) => canonical(() => service.readiness(args.config, { signal })),
    },
    {
      name: "opengrad_gpu_smoke",
      tier: "SAFE_WRITE",
      description: "Run the bounded real Qwen boundary smoke only when GPU/dependencies/model access exist; never runs training.",
      inputSchema: passthrough({ config: string("Optional frozen baseline evaluation config path.") }),
      handler: (args, { signal } = {}) => canonical(() => service.gpuSmoke(args.config, { signal })),
    },
    {
      name: "opengrad_validate_data",
      tier: "READ_ONLY",
      description: "Validate a canonical OpenGrad dataset through the documented validator.",
      inputSchema: passthrough({
        records: string("Repository-relative JSONL SFT records path."),
        mode: { type: "string", enum: ["sft"], default: "sft" },
      }, ["records"]),
      handler: (args, { signal } = {}) => canonical(() => service.validateData(args.records, args.mode ?? "sft", { signal })),
    },
    {
      name: "opengrad_inspect_template",
      tier: "READ_ONLY",
      description: "Inspect the OpenGrad model template and loss-mask boundary without modifying artifacts.",
      inputSchema: passthrough({ record: string("Optional repository-relative record path.") }),
      handler: (args, { signal } = {}) => canonical(() => service.inspectTemplate(args.record, { signal })),
    },
    {
      name: "opengrad_preflight",
      tier: "SAFE_WRITE",
      description: "Run OpenGrad's authoritative experiment preflight for a frozen configuration.",
      inputSchema: passthrough({ config: string("Repository-relative experiment config path.") }, ["config"]),
      handler: (args, { signal } = {}) => canonical(() => service.preflight(args.config, { signal })),
    },
    {
      name: "opengrad_baseline_status",
      tier: "READ_ONLY",
      description: "Inspect authoritative baseline and artifact state without running a baseline.",
      inputSchema: passthrough(),
      handler: (_args, { signal } = {}) => canonical(() => service.status({ signal })),
    },
    {
      name: "opengrad_experiment_list",
      tier: "READ_ONLY",
      description: "List OpenGrad experiment records from the authoritative store.",
      inputSchema: passthrough(),
      handler: (_args, { signal } = {}) => canonical(() => service.experimentList({ signal })),
    },
    {
      name: "opengrad_experiment_show",
      tier: "READ_ONLY",
      description: "Show one authoritative OpenGrad experiment record.",
      inputSchema: passthrough({ experiment_id: string("Experiment identifier.") }, ["experiment_id"]),
      handler: (args, { signal } = {}) => canonical(() => service.experimentShow(args.experiment_id, { signal })),
    },
    {
      name: "opengrad_experiment_diff",
      tier: "READ_ONLY",
      description: "Compare two authoritative OpenGrad experiment records.",
      inputSchema: passthrough({
        baseline_experiment: string("Baseline experiment identifier."),
        candidate_experiment: string("Candidate experiment identifier."),
      }, ["baseline_experiment", "candidate_experiment"]),
      handler: (args, { signal } = {}) => canonical(() => service.experimentDiff(args.baseline_experiment, args.candidate_experiment, { signal })),
    },
    {
      name: "opengrad_checkpoint_list",
      tier: "READ_ONLY",
      description: "List authoritative OpenGrad checkpoint registry entries.",
      inputSchema: passthrough({ status: string("Optional checkpoint lifecycle status.") }),
      handler: (args, { signal } = {}) => canonical(() => service.checkpointList(args.status, { signal })),
    },
    {
      name: "opengrad_checkpoint_inspect",
      tier: "READ_ONLY",
      description: "Inspect one authoritative OpenGrad checkpoint record.",
      inputSchema: passthrough({ checkpoint_id: string("Checkpoint identifier.") }, ["checkpoint_id"]),
      handler: (args, { signal } = {}) => canonical(() => service.checkpointInspect(args.checkpoint_id, { signal })),
    },
    {
      name: "opengrad_failures",
      tier: "READ_ONLY",
      description: "Cluster failures from an authoritative OpenGrad evaluation run.",
      inputSchema: passthrough({ run_dir: string("Repository-relative evaluation run directory.") }, ["run_dir"]),
      handler: (args, { signal } = {}) => canonical(() => service.failures(args.run_dir, { signal })),
    },
    {
      name: "opengrad_baseline_run",
      tier: "HIGH_IMPACT",
      denialCode: "BASELINE_NOT_READY",
      description: "Run the frozen OpenGrad baseline. Dry-run is safe; real execution is gate-checked against authoritative readiness.",
      inputSchema: passthrough({
        config: string("Optional frozen baseline config path."),
        dry_run: boolean("Use deterministic CPU plumbing only.", false),
        limit: integer("Optional positive evaluation record limit."),
      }),
      handler: (args, { signal } = {}) => canonical(() => service.baseline({ config: args.config, dryRun: args.dry_run ?? false, limit: args.limit, signal })),
    },
    {
      name: "opengrad_sft",
      tier: "HIGH_IMPACT",
      denialCode: "NO_REAL_SFT_WITHOUT_VALID_B0",
      description: "Invoke OpenGrad training with an immutable config. Real training is blocked unless all baseline-first gates pass.",
      inputSchema: passthrough({
        config: string("Repository-relative SFT experiment config path."),
        dry_run: boolean("Use CPU mock training only.", false),
      }, ["config"]),
      handler: (args, { signal } = {}) => canonical(() => service.train({ config: args.config, dryRun: args.dry_run ?? false, signal })),
    },
    {
      name: "opengrad_eval_run",
      tier: "HIGH_IMPACT",
      denialCode: "GPU_SMOKE_FAILED",
      description: "Run an OpenGrad evaluation suite through the authoritative evaluator.",
      inputSchema: passthrough({
        target: string("Repository-relative checkpoint/model target."),
        suite: { type: "string", enum: EVAL_SUITES, default: "smoke" },
        dry_run: boolean("Use deterministic mock evaluation.", false),
        limit: integer("Optional positive task limit."),
      }, ["target"]),
      handler: (args, { signal } = {}) => canonical(() => service.evaluate({
        target: args.target, suite: args.suite ?? "smoke", dryRun: args.dry_run ?? false, limit: args.limit, signal,
      })),
    },
    {
      name: "opengrad_eval_compare",
      tier: "READ_ONLY",
      description: "Compare two OpenGrad evaluation artifact directories through the authoritative comparator.",
      inputSchema: passthrough({
        baseline: string("Repository-relative baseline run directory."),
        candidate: string("Repository-relative candidate run directory."),
      }, ["baseline", "candidate"]),
      handler: (args, { signal } = {}) => canonical(() => service.compare(args.baseline, args.candidate, { signal })),
    },
    {
      name: "opengrad_b0_workflow",
      tier: "HIGH_IMPACT",
      denialCode: "BASELINE_NOT_READY",
      description: "Run the deterministic B0 sequence: prerequisite readiness, real GPU boundary smoke, then real baseline inference. Never runs SFT.",
      inputSchema: passthrough({
        config: string("Optional frozen baseline config path."),
        limit: integer("Optional positive evaluation record limit."),
      }),
      handler: (args, { signal } = {}) => canonical(() => b0Workflow(service, { config: args.config, limit: args.limit, signal })),
    },
    {
      name: "opengrad_post_sft_workflow",
      tier: "HIGH_IMPACT",
      denialCode: "COMMAND_FAILED",
      description: "Run post-SFT evaluation and, when paths are supplied, comparison and failure analysis. Does not train or promote.",
      inputSchema: passthrough({
        target: string("Repository-relative checkpoint/model target."),
        suite: { type: "string", enum: ["full_post_training", "regression_core", "tool_use_core"], default: "full_post_training" },
        baseline: string("Optional repository-relative baseline evaluation path."),
        candidate: string("Optional repository-relative candidate run directory."),
      }, ["target"]),
      handler: (args, { signal } = {}) => canonical(() => postSftWorkflow(service, { ...args, signal })),
    },
  ];
}

/**
 * Apply OpenGrad's permission model to one tool call, then execute it.
 * Returns the canonical result envelope; it never throws for policy denial.
 */
export async function invokeTool(tools, name, args = {}, { service, signal } = {}) {
  const tool = tools.find((candidate) => candidate.name === name);
  if (!tool) return { ok: false, code: "INVALID_ARGUMENT", message: `Unknown tool: ${name}`, blocking: true };
  const input = args && typeof args === "object" ? args : {};
  const tier = tool.tier ?? classifyOpenGradOperation(name, input);

  const staticDenial = evaluateStaticGuard(name, input);
  if (staticDenial) return { ok: false, code: "NO_TRAINING_ON_EVAL_DATA", message: staticDenial, blocking: true };

  if (tier === "HIGH_IMPACT" && input.dry_run !== true) {
    let status;
    try { status = await service.readiness(input.config, { signal }); }
    catch (error) { return { ...stableError(error), message: `guardian could not verify OpenGrad readiness: ${stableError(error).message}` }; }
    if (!status.ok) {
      return { ok: false, code: status.code, message: `guardian denied ${name}: ${status.message}`, blocking: true, readiness: status.value ?? null };
    }
    const denial = evaluateHighImpactGate(name, status.value);
    if (denial) {
      return { ok: false, code: tool.denialCode ?? "COMMAND_FAILED", message: denial, blocking: true, readiness: status.value };
    }
  }
  return tool.handler(input, { signal, service });
}

export { EVAL_SUITES };
