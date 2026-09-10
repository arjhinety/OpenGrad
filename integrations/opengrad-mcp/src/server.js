import { OpenGradService } from "./core.js";
import { createReceiptSink } from "./receipts.js";
import { createTools, invokeTool } from "./tools.js";

export const SERVER_NAME = "opengrad";
export const SERVER_VERSION = "0.1.0";
export const PROTOCOL_VERSION = "2024-11-05";

const JSONRPC_INVALID_REQUEST = -32600;
const JSONRPC_METHOD_NOT_FOUND = -32601;
const JSONRPC_INTERNAL_ERROR = -32603;

function jsonRpcResult(id, result) { return { jsonrpc: "2.0", id, result }; }
function jsonRpcError(id, code, message) { return { jsonrpc: "2.0", id, error: { code, message } }; }

function summarize(result) {
  const { stdout, stderr, ...rest } = result ?? {};
  const summary = { ...rest };
  if (typeof stderr === "string" && stderr.trim()) summary.stderr = stderr.length > 4096 ? `${stderr.slice(0, 4096)}…` : stderr;
  return summary;
}

function toContent(result) {
  const text = typeof result === "string" ? result : JSON.stringify(result, null, 2);
  return { content: [{ type: "text", text }], structuredContent: result, isError: result?.ok === false };
}

/**
 * A transport-agnostic MCP server. `handle` maps one parsed JSON-RPC message to
 * one response object (or null for notifications), so it is fully unit-testable
 * without stdio.
 */
export function createMcpServer({ root, timeoutMs, executable, receipts = process.env.OPENGRAD_MCP_RECEIPTS } = {}) {
  const service = new OpenGradService({ root, timeoutMs, executable });
  const tools = createTools(service);
  const sink = receipts === null || receipts === "off" ? null : createReceiptSink(service, { path: receipts || undefined });
  const controllers = new Set();

  const toolList = tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema,
  }));

  async function callTool(name, args, extra = {}) {
    const controller = new AbortController();
    controllers.add(controller);
    if (extra.signal) {
      if (extra.signal.aborted) controller.abort();
      else extra.signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
    try {
      const result = await invokeTool(tools, name, args, { service, signal: controller.signal });
      if (sink) void sink.record({ tool: name, arguments: args, ok: result?.ok !== false, result: summarize(result) });
      return toContent(result);
    } finally {
      controllers.delete(controller);
    }
  }

  async function handle(message) {
    if (!message || message.jsonrpc !== "2.0" || typeof message.method !== "string") {
      return jsonRpcError(message?.id ?? null, JSONRPC_INVALID_REQUEST, "invalid JSON-RPC request");
    }
    const { id, method, params } = message;
    const isNotification = id === undefined || id === null;
    try {
      switch (method) {
        case "initialize":
          return jsonRpcResult(id, {
            protocolVersion: PROTOCOL_VERSION,
            capabilities: { tools: { listChanged: false } },
            serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
          });
        case "notifications/initialized":
        case "notifications/cancelled":
          return null;
        case "ping":
          return jsonRpcResult(id, {});
        case "tools/list":
          return jsonRpcResult(id, { tools: toolList });
        case "tools/call": {
          const name = params?.name;
          const args = params?.arguments ?? {};
          if (typeof name !== "string") return jsonRpcError(id, JSONRPC_INVALID_REQUEST, "tools/call requires a tool name");
          if (isNotification) return null;
          return jsonRpcResult(id, await callTool(name, args));
        }
        default:
          if (isNotification) return null;
          return jsonRpcError(id, JSONRPC_METHOD_NOT_FOUND, `unknown method: ${method}`);
      }
    } catch (error) {
      if (isNotification) return null;
      return jsonRpcError(id, JSONRPC_INTERNAL_ERROR, String(error?.message ?? error));
    }
  }

  function dispose() {
    for (const controller of controllers) controller.abort();
    controllers.clear();
  }

  return { handle, tools: toolList, service, dispose };
}

/**
 * Run the server over newline-delimited JSON on stdin/stdout. In-flight tool
 * calls are drained before shutdown so a client that closes stdin immediately
 * after a request still receives that response.
 */
export function serveStdio(server, input = process.stdin, output = process.stdout) {
  let buffer = "";
  const pending = new Set();
  input.setEncoding("utf8");
  input.on("data", (chunk) => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (!line) continue;
      let message;
      try { message = JSON.parse(line); }
      catch { output.write(`${JSON.stringify(jsonRpcError(null, JSONRPC_INVALID_REQUEST, "invalid JSON"))}\n`); continue; }
      const task = server.handle(message).then((response) => {
        if (response) output.write(`${JSON.stringify(response)}\n`);
      }).catch((error) => {
        output.write(`${JSON.stringify(jsonRpcError(message?.id ?? null, JSONRPC_INTERNAL_ERROR, String(error?.message ?? error)))}\n`);
      });
      pending.add(task);
      void task.finally(() => pending.delete(task));
    }
  });
  input.on("end", () => { void Promise.allSettled([...pending]).then(() => server.dispose()); });
}
