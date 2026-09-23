import test from "node:test";
import path from "node:path";
import assert from "node:assert/strict";
import { OpenGradService, ensureInside, stableError, redactText } from "../src/core.js";
import { classifyOpenGradOperation, evaluateHighImpactGate, evaluateStaticGuard, isProtectedEvaluationPath, MONOTONIC_INVARIANTS } from "../src/guardian.js";
import { createTools, invokeTool } from "../src/tools.js";
import { redactValue } from "../src/receipts.js";

const root = "/workspace/opengrad";
// Subprocess tests need a real working directory; the fake `root` above only
// exercises lexical path containment.
const liveRoot = process.cwd();

test("path arguments stay inside the OpenGrad root", () => {
  // path.resolve keeps the expectation right on Windows, where the resolved root gains a drive letter.
  assert.equal(ensureInside(root, "configs/a.yaml"), path.resolve(root, "configs/a.yaml"));
  assert.throws(() => ensureInside(root, "../secrets"), /inside the OpenGrad repository/);
  assert.throws(() => ensureInside(root, "/tmp/other"), /inside the OpenGrad repository/);
});

test("command bridge uses an argv array and parses JSON", async () => {
  const service = new OpenGradService({ root: liveRoot, executable: process.execPath });
  const observed = [];
  service.command = async (args) => { observed.push(args); return { ok: true, value: { status: "PASS" } }; };
  await service.readiness("configs/x.yaml");
  assert.deepEqual(observed[0], ["readiness", path.resolve(liveRoot, "configs/x.yaml"), "--json"]);
});

test("command bridge preserves stable JSON errors and stderr", async () => {
  const service = new OpenGradService({ root: liveRoot, executable: process.execPath });
  const result = await service.command(["-e", "process.stdout.write(JSON.stringify({code:'CONTAMINATION_FAILURE',message:'heldout overlap',blocking:true})); process.stderr.write('diagnostic'); process.exit(1)"]);
  assert.equal(result.ok, false);
  assert.equal(result.code, "CONTAMINATION_FAILURE");
  assert.equal(result.message, "heldout overlap");
  assert.equal(result.stderr, "diagnostic");
  assert.equal(result.blocking, true);
});

test("invalid JSON, missing root, and spawn failures are canonical errors", async () => {
  const service = new OpenGradService({ root: liveRoot, executable: process.execPath });
  const invalid = await service.command(["-e", "process.stdout.write('ansi output')"]);
  assert.equal(invalid.code, "COMMAND_FAILED");
  // A nonexistent root must not surface as a bare ENOENT.
  const missingRoot = new OpenGradService({ root: "/nonexistent/opengrad-root", executable: process.execPath });
  const absent = await missingRoot.command(["doctor", "--json"]);
  assert.equal(absent.code, "REPOSITORY_NOT_FOUND");
  const failed = new OpenGradService({ root: liveRoot, executable: "/nonexistent/opengrad" });
  const spawn = await failed.command(["doctor", "--json"]);
  assert.equal(spawn.ok, false);
  assert.ok(["SUBPROCESS_UNAVAILABLE", "COMMAND_FAILED"].includes(spawn.code));
  assert.equal(stableError(new Error("x")).code, "COMMAND_FAILED");
});

test("gate projections parse JSON even when the CLI exit code reports FAIL", async () => {
  // `opengrad readiness`/`gpu-smoke` exit 1 to report a FAIL/BLOCKED gate.
  // That exit code is payload: the bridge must still surface the gate payload
  // so the guardian can deny with the real code instead of COMMAND_FAILED.
  const payload = JSON.stringify({ status: "FAIL", ready_for_baseline: false, blocking_gates: ["real_b0"] });
  const service = new OpenGradService({ root: liveRoot, executable: process.execPath });
  service.executable = process.execPath;
  const result = await service.command(
    ["-e", `process.stdout.write(${JSON.stringify(payload)}); process.exit(1)`],
    { allowNonZeroJson: true },
  );
  assert.equal(result.ok, true);
  assert.equal(result.value.status, "FAIL");
  assert.equal(result.exitCode, 1);
});

test("guardian classification and protected evaluation detection", () => {
  assert.equal(classifyOpenGradOperation("opengrad_status"), "READ_ONLY");
  assert.equal(classifyOpenGradOperation("opengrad_sft", { dry_run: true }), "SAFE_WRITE");
  assert.equal(classifyOpenGradOperation("opengrad_sft", {}), "HIGH_IMPACT");
  assert.equal(classifyOpenGradOperation("opengrad_future_side_effect"), "HIGH_IMPACT");
  assert.equal(isProtectedEvaluationPath("reports/evaluation/when2call-heldout"), true);
  assert.equal(isProtectedEvaluationPath("data/processed/sft/source.jsonl"), false);
  assert.ok(MONOTONIC_INVARIANTS.includes("NO_REAL_SFT_WITHOUT_VALID_B0"));
  assert.equal(evaluateStaticGuard("opengrad_sft", { config: "configs/heldout/sft.yaml" }) !== undefined, true);
  assert.equal(evaluateStaticGuard("opengrad_sft", { config: "configs/sft/m0.yaml" }), undefined);
});

test("stable errors retain actionable bridge codes", () => {
  assert.equal(stableError(Object.assign(new Error("outside"), { code: "PATH_OUTSIDE_PROJECT" })).code, "PATH_OUTSIDE_PROJECT");
  assert.equal(stableError(Object.assign(new Error("missing"), { code: "ARTIFACT_NOT_FOUND" })).code, "ARTIFACT_NOT_FOUND");
});

test("receipt redaction removes credential-like fields recursively", () => {
  assert.deepEqual(redactValue({ apiToken: "secret", nested: { password: "pw", value: 3 } }), { apiToken: "[REDACTED]", nested: { password: "[REDACTED]", value: 3 } });
});

test("receipt redaction also removes credentials embedded in free text", () => {
  const receipt = redactValue({ args: ["--key", "sk-abcdefghijklmnop"], log: "Authorization: Bearer abc.def.ghi" });
  const text = JSON.stringify(receipt);
  assert.ok(!text.includes("sk-abcdefghijklmnop"));
  assert.ok(!text.includes("abc.def.ghi"));
  assert.ok(text.includes("[REDACTED]"));
  assert.ok(redactText("token=abcdef123456").includes("[REDACTED]"));
});

// Readiness projections. `ready_for_baseline` deliberately excludes the
// baseline's own post-run gates, so overall `status` stays FAIL while baseline
// execution is legitimately permitted.
const readinessNotReady = Object.freeze({
  status: "FAIL", ready_for_baseline: false, ready_for_sft: false,
  blocking_gates: ["real_b0", "baseline_artifacts"],
  gates: [{ name: "gpu_boundary", status: "FAIL", details: "no receipt" }],
});
const readinessBaselinePrereqs = Object.freeze({
  status: "FAIL", ready_for_baseline: true, ready_for_sft: false,
  blocking_gates: ["gpu_boundary", "real_b0", "baseline_artifacts"],
  gates: [{ name: "gpu_boundary", status: "FAIL", details: "gpu boundary is not verified" }],
});
const readinessBaselineRun = Object.freeze({
  status: "FAIL", ready_for_baseline: true, ready_for_sft: false,
  blocking_gates: ["real_b0", "baseline_artifacts"],
  gates: [{ name: "gpu_boundary", status: "PASS", details: "boundary verified" }],
});
const readinessSft = Object.freeze({
  status: "PASS", ready_for_baseline: true, ready_for_sft: true,
  blocking_gates: [],
  gates: [{ name: "gpu_boundary", status: "PASS", details: "boundary verified" }],
});

test("real baseline requires prerequisite readiness plus a verified GPU boundary", () => {
  assert.equal(typeof evaluateHighImpactGate("opengrad_baseline_run", readinessNotReady), "string");
  assert.equal(typeof evaluateHighImpactGate("opengrad_baseline_run", readinessBaselinePrereqs), "string");
  assert.equal(evaluateHighImpactGate("opengrad_baseline_run", readinessBaselineRun), undefined);
});

test("B0 workflow is not deadlocked by its own post-run artifacts", () => {
  assert.equal(evaluateHighImpactGate("opengrad_b0_workflow", readinessBaselinePrereqs), undefined);
  assert.equal(typeof evaluateHighImpactGate("opengrad_b0_workflow", readinessNotReady), "string");
});

test("SFT stays fail-closed on the full readiness contract", () => {
  assert.equal(evaluateHighImpactGate("opengrad_sft", readinessSft), undefined);
  assert.equal(typeof evaluateHighImpactGate("opengrad_sft", readinessBaselineRun), "string");
});

test("real evaluation and unknown high-impact operations fail closed", () => {
  assert.equal(evaluateHighImpactGate("opengrad_eval_run", readinessBaselineRun), undefined);
  assert.equal(typeof evaluateHighImpactGate("opengrad_eval_run", readinessBaselinePrereqs), "string");
  assert.equal(typeof evaluateHighImpactGate("opengrad_future_tool", readinessBaselineRun), "string");
});

test("invokeTool denies real high-impact SFT without a valid B0", async () => {
  const service = { readiness: async () => ({ ok: true, value: readinessBaselineRun }) };
  const tools = createTools(service);
  const denied = await invokeTool(tools, "opengrad_sft", { config: "configs/experiments/m0_sft.yaml" }, { service });
  assert.equal(denied.ok, false);
  assert.equal(denied.code, "NO_REAL_SFT_WITHOUT_VALID_B0");
  assert.equal(denied.blocking, true);
});

test("invokeTool lets a dry-run through without consulting readiness", async () => {
  let readinessCalls = 0;
  const service = {
    readiness: async () => { readinessCalls += 1; return { ok: true, value: readinessNotReady }; },
    train: async () => ({ ok: true, value: { status: "DRY_RUN", evidence: false } }),
  };
  const tools = createTools(service);
  const allowed = await invokeTool(tools, "opengrad_sft", { config: "configs/experiments/m0_sft.yaml", dry_run: true }, { service });
  assert.equal(allowed.ok, true);
  assert.equal(readinessCalls, 0);
});

test("invokeTool rejects evaluation-only SFT input before dispatch", async () => {
  const service = { readiness: async () => ({ ok: true, value: readinessSft }) };
  const tools = createTools(service);
  const denied = await invokeTool(tools, "opengrad_sft", { config: "configs/data/heldout_sft.yaml" }, { service });
  assert.equal(denied.code, "NO_TRAINING_ON_EVAL_DATA");
});

test("tool catalog exposes SFT validation only, never DPO/RL/judge", () => {
  const tools = createTools({});
  const names = tools.map((tool) => tool.name);
  assert.ok(names.includes("opengrad_validate_data"));
  const validation = tools.find((tool) => tool.name === "opengrad_validate_data");
  assert.deepEqual(validation.inputSchema.properties.mode.enum, ["sft"]);
  assert.ok(!names.some((name) => /dpo|distill|_rl|judge/i.test(name)));
});
