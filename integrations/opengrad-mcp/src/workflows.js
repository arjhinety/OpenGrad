const ok = (result) => result?.ok === true;

const blocked = (stage, result) => ({
  ok: false,
  code: result?.code ?? "WORKFLOW_STAGE_FAILED",
  message: `${stage} failed: ${result?.message ?? "unknown failure"}`,
  blocking: true,
  stage,
  result,
});

/** Deterministic B0 orchestration; OpenGrad owns every scientific operation. */
export async function b0Workflow(service, { config, limit, signal } = {}) {
  const stages = [];
  const record = (stage, result) => { stages.push({ stage, ok: ok(result), result }); return result; };
  let result = record("readiness", await service.readiness(config, { signal }));
  if (!ok(result) || result.value?.ready_for_baseline !== true) {
    return blocked("readiness", result.ok
      ? { ...result.value, code: "BASELINE_NOT_READY", message: (result.value?.blocking_gates ?? []).join(", ") || "baseline gate is not ready" }
      : result);
  }
  result = record("gpu_boundary", await service.gpuSmoke(config, { signal }));
  if (!ok(result) || result.value?.status !== "PASS") {
    return blocked("gpu_boundary", result.ok
      ? { ...result.value, code: "GPU_SMOKE_FAILED", message: "GPU boundary did not pass" }
      : result);
  }
  result = record("baseline", await service.baseline({ config, dryRun: false, limit, signal }));
  if (!ok(result)) return blocked("baseline", result);
  return { ok: true, workflow: "B0_WORKFLOW", status: "BASELINE_COMPLETE", stages };
}

/** Post-SFT handoff. It does not invent evaluation or residual calculations. */
export async function postSftWorkflow(service, { target, suite = "full_post_training", baseline, candidate, signal } = {}) {
  const stages = [];
  const evaluate = await service.evaluate({ target, suite, dryRun: false, signal });
  stages.push({ stage: "evaluation", ok: ok(evaluate), result: evaluate });
  if (!ok(evaluate)) return blocked("evaluation", evaluate);
  const compare = baseline && candidate
    ? await service.compare(baseline, candidate, { signal })
    : { ok: true, value: { status: "DEFERRED_UNTIL_ARTIFACT_PATHS" } };
  stages.push({ stage: "regression_comparison", ok: ok(compare), result: compare });
  if (!ok(compare)) return blocked("regression_comparison", compare);
  const failures = candidate ? await service.failures(candidate, { signal }) : { ok: true, value: { status: "DEFERRED_UNTIL_CANDIDATE_RUN" } };
  stages.push({ stage: "residual_failure_analysis", ok: ok(failures), result: failures });
  if (!ok(failures)) return blocked("residual_failure_analysis", failures);
  return { ok: true, workflow: "POST_SFT_WORKFLOW", status: "REVIEW_REQUIRED", stages };
}
