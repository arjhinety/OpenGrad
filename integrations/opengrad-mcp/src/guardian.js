export const MONOTONIC_INVARIANTS = Object.freeze([
  "NO_REAL_SFT_WITHOUT_VALID_B0",
  "NO_TRAINING_WITH_FAILED_PREFLIGHT",
  "NO_TRAINING_ON_EVAL_DATA",
  "NO_FLOATING_MODEL_REVISION",
  "NO_FLOATING_DATASET_REVISION",
  "NO_EXPERIMENT_ID_MUTATION",
  "NO_CHECKPOINT_OVERWRITE",
  "NO_SECRET_LOGGING",
  "NO_PROMOTION_WITHOUT_REQUIRED_EVALUATION",
]);

export const PROTECTED_EVALUATION_TERMS = Object.freeze([
  "evaluation_only",
  "heldout",
  "bfcl",
  "when2call",
  "tau-bench",
  "tau2",
  "toolsandbox",
  "mcpmark",
  "toolathlon",
]);

export function isProtectedEvaluationPath(value) {
  const text = String(value ?? "").toLowerCase();
  return PROTECTED_EVALUATION_TERMS.some((term) => text.includes(term));
}

export function classifyOpenGradOperation(name, args = {}) {
  if ([
    "opengrad_status", "opengrad_doctor", "opengrad_validate", "opengrad_validate_data",
    "opengrad_inspect_template", "opengrad_baseline_status", "opengrad_experiment_list",
    "opengrad_experiment_show", "opengrad_experiment_diff", "opengrad_checkpoint_list",
    "opengrad_checkpoint_inspect", "opengrad_failures", "opengrad_eval_compare",
  ].includes(name)) return "READ_ONLY";
  if (["opengrad_readiness", "opengrad_gpu_smoke", "opengrad_preflight"].includes(name)) return "SAFE_WRITE";
  if ((name === "opengrad_baseline_run" || name === "opengrad_eval_run" || name === "opengrad_sft") && args.dry_run === true) return "SAFE_WRITE";
  // Unknown operations are fail-closed: a future side-effecting tool must not
  // inherit read-only behavior merely because it was not added to this list.
  return "HIGH_IMPACT";
}

export function findReadinessGate(readiness, name) {
  if (!readiness || typeof readiness !== "object" || !Array.isArray(readiness.gates)) return undefined;
  return readiness.gates.find((gate) => gate?.name === name);
}

/**
 * Authoritative gate for a high-impact OpenGrad operation.
 *
 * Returns a denial reason string when the operation must be blocked, or
 * undefined when it may proceed. Operations that *establish* B0 must not be
 * required to already possess their own post-run evidence: a real baseline run
 * depends on prerequisite readiness plus a verified GPU boundary, while the B0
 * workflow is the stage that produces that boundary and therefore depends only
 * on prerequisite readiness. SFT remains fail-closed on the full contract.
 */
export function evaluateHighImpactGate(name, readiness) {
  const state = readiness && typeof readiness === "object" ? readiness : {};
  const blockers = Array.isArray(state.blocking_gates) ? state.blocking_gates : [];
  switch (name) {
    case "opengrad_sft":
      if (state.status !== "PASS" || state.ready_for_sft !== true) {
        return `NO_REAL_SFT_WITHOUT_VALID_B0: ${blockers.join("; ") || "OpenGrad did not report ready_for_sft"}`;
      }
      return undefined;
    case "opengrad_baseline_run": {
      if (state.ready_for_baseline !== true) {
        return `baseline blocked by OpenGrad readiness: ${blockers.join("; ") || "prerequisite gates are not ready"}`;
      }
      const gpu = findReadinessGate(state, "gpu_boundary");
      if (gpu?.status !== "PASS") {
        return `real baseline blocked: GPU boundary is not verified (${gpu?.details ?? "missing gpu_boundary gate"})`;
      }
      return undefined;
    }
    case "opengrad_b0_workflow":
      // The workflow performs the GPU boundary smoke itself, so it must not be
      // required to present its own post-run receipt up front.
      if (state.ready_for_baseline !== true) {
        return `B0 workflow blocked by OpenGrad readiness: ${blockers.join("; ") || "prerequisite gates are not ready"}`;
      }
      return undefined;
    case "opengrad_eval_run": {
      const gpu = findReadinessGate(state, "gpu_boundary");
      if (gpu?.status !== "PASS") {
        return `real evaluation blocked: GPU boundary is not verified (${gpu?.details ?? "missing gpu_boundary gate"})`;
      }
      return undefined;
    }
    default:
      if (state.status !== "PASS") {
        return `guardian denied ${name}: readiness status is ${state.status}`;
      }
      return undefined;
  }
}

/**
 * Synchronous, non-bypassable guards applied before a tool handler runs.
 * Returns a denial reason string, or undefined when the call may proceed.
 */
export function evaluateStaticGuard(name, args = {}) {
  if (name === "opengrad_sft" && isProtectedEvaluationPath(args.config)) {
    return "NO_TRAINING_ON_EVAL_DATA: SFT config is in a protected evaluation namespace";
  }
  if (name === "opengrad_validate_data" && args.mode === "sft" && isProtectedEvaluationPath(args.records)) {
    return "NO_TRAINING_ON_EVAL_DATA: evaluation-only records cannot be validated as SFT data";
  }
  if (name === "opengrad_sft" && args.dry_run !== true && typeof args.config !== "string") {
    return "CONFIG_INVALID: real SFT requires an explicit immutable config";
  }
  return undefined;
}
