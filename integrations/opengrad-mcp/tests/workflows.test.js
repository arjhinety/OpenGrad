import test from "node:test";
import assert from "node:assert/strict";
import { b0Workflow, postSftWorkflow } from "../src/workflows.js";
import { createMcpServer, PROTOCOL_VERSION, SERVER_NAME } from "../src/server.js";

const reply = (value) => ({ ok: true, value });

function fakeService(overrides = {}) {
  const calls = [];
  const service = {
    calls,
    readiness: async () => reply({ ready_for_baseline: true, ready_for_sft: false, blocking_gates: [] }),
    gpuSmoke: async () => reply({ status: "PASS" }),
    baseline: async () => reply({ status: "EXECUTED" }),
    evaluate: async () => reply({ status: "EVALUATED" }),
    compare: async () => reply({ status: "OK" }),
    failures: async () => reply({ status: "OK" }),
    ...overrides,
  };
  for (const name of ["readiness", "gpuSmoke", "baseline", "evaluate", "compare", "failures"]) {
    const original = service[name];
    service[name] = async (...args) => { calls.push(name); return original(...args); };
  }
  return service;
}

test("b0Workflow stops before GPU work when prerequisite readiness is false", async () => {
  const service = fakeService({
    readiness: async () => reply({ ready_for_baseline: false, blocking_gates: ["evaluation_materialization"] }),
  });
  const result = await b0Workflow(service, {});
  assert.equal(result.ok, false);
  assert.equal(result.stage, "readiness");
  assert.equal(result.code, "BASELINE_NOT_READY");
  assert.deepEqual(service.calls, ["readiness"]);
});

test("b0Workflow sequences readiness, GPU boundary, then real baseline", async () => {
  const service = fakeService();
  const result = await b0Workflow(service, { limit: 3 });
  assert.equal(result.ok, true);
  assert.equal(result.status, "BASELINE_COMPLETE");
  assert.deepEqual(service.calls, ["readiness", "gpuSmoke", "baseline"]);
  assert.deepEqual(result.stages.map((stage) => stage.stage), ["readiness", "gpu_boundary", "baseline"]);
});

test("b0Workflow never runs baseline when the GPU boundary fails", async () => {
  const service = fakeService({ gpuSmoke: async () => reply({ status: "INCOMPLETE" }) });
  const result = await b0Workflow(service, {});
  assert.equal(result.ok, false);
  assert.equal(result.stage, "gpu_boundary");
  assert.equal(result.code, "GPU_SMOKE_FAILED");
  assert.ok(!service.calls.includes("baseline"));
});

test("b0Workflow propagates a failed real baseline", async () => {
  const service = fakeService({
    baseline: async () => ({ ok: false, code: "BASELINE_FAILED", message: "inference crashed", blocking: true }),
  });
  const result = await b0Workflow(service, {});
  assert.equal(result.ok, false);
  assert.equal(result.stage, "baseline");
  assert.equal(result.code, "BASELINE_FAILED");
});

test("postSftWorkflow evaluates then defers comparison until artifact paths exist", async () => {
  const service = fakeService();
  const result = await postSftWorkflow(service, { target: "runs/candidate" });
  assert.equal(result.ok, true);
  assert.equal(result.status, "REVIEW_REQUIRED");
  assert.deepEqual(service.calls, ["evaluate"]);
  assert.deepEqual(
    result.stages.map((stage) => stage.stage),
    ["evaluation", "regression_comparison", "residual_failure_analysis"],
  );
});

test("postSftWorkflow runs comparison and failure analysis when paths are supplied", async () => {
  const service = fakeService();
  const result = await postSftWorkflow(service, { target: "runs/candidate", baseline: "runs/base", candidate: "runs/cand" });
  assert.equal(result.ok, true);
  assert.deepEqual(service.calls, ["evaluate", "compare", "failures"]);
});

test("postSftWorkflow blocks on a failed evaluation", async () => {
  const service = fakeService({
    evaluate: async () => ({ ok: false, code: "COMMAND_FAILED", message: "no target", blocking: true }),
  });
  const result = await postSftWorkflow(service, { target: "runs/candidate", baseline: "runs/base", candidate: "runs/cand" });
  assert.equal(result.ok, false);
  assert.equal(result.stage, "evaluation");
  assert.ok(!service.calls.includes("compare"));
});

test("MCP server advertises tools and answers initialize over handle()", async () => {
  const service = fakeService();
  const server = createMcpServer({ root: process.cwd(), receipts: null });
  server.service.status = service.status;
  const init = await server.handle({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
  assert.equal(init.result.protocolVersion, PROTOCOL_VERSION);
  assert.equal(init.result.serverInfo.name, SERVER_NAME);
  const list = await server.handle({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  assert.ok(list.result.tools.some((tool) => tool.name === "opengrad_b0_workflow"));
  assert.ok(list.result.tools.every((tool) => tool.inputSchema?.type === "object"));
  assert.equal(await server.handle({ jsonrpc: "2.0", method: "notifications/initialized" }), null);
  server.dispose();
});

test("MCP server returns JSON-RPC errors for unknown and malformed messages", async () => {
  const server = createMcpServer({ root: process.cwd(), receipts: null });
  const unknown = await server.handle({ jsonrpc: "2.0", id: 7, method: "does/not/exist" });
  assert.equal(unknown.error.code, -32601);
  const malformed = await server.handle({ id: 8, method: "ping" });
  assert.equal(malformed.error.code, -32600);
  server.dispose();
});
