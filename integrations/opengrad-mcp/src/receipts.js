import { mkdir, appendFile } from "node:fs/promises";
import { dirname, relative } from "node:path";
import { ensureInside, redactValue } from "./core.js";

const DEFAULT_MAX_PENDING = 256;

/**
 * Optional append-only JSONL receipt sink. Receipts are observational: a failed
 * write is logged to stderr and never turns a successful OpenGrad operation into
 * a tool failure. The queue is bounded and flushed serially so concurrent
 * results cannot interleave writes or grow memory without limit.
 */
export function createReceiptSink(service, { path = undefined, maxPending = DEFAULT_MAX_PENDING } = {}) {
  const resolved = ensureInside(service.root, path ?? process.env.OPENGRAD_MCP_RECEIPTS ?? "runs/mcp-receipts.jsonl", "receipt path");
  const limit = Number.isSafeInteger(maxPending) && maxPending > 0 ? maxPending : DEFAULT_MAX_PENDING;
  let sequence = 0;
  let flushing = false;
  const queue = [];

  const flush = async () => {
    if (flushing) return;
    flushing = true;
    try {
      await mkdir(dirname(resolved), { recursive: true });
      while (queue.length > 0) {
        await appendFile(resolved, `${JSON.stringify(queue.shift())}\n`, "utf8");
      }
    } catch (error) {
      queue.length = 0;
      console.error(`OpenGrad receipt write failed: ${error.message}`);
    } finally {
      flushing = false;
      if (queue.length > 0) void flush();
    }
  };

  const record = (entry) => {
    if (queue.length >= limit) {
      console.error(`OpenGrad receipt queue is full (${limit}); dropping receipt for ${entry?.tool ?? "unknown"}`);
      return Promise.resolve();
    }
    queue.push(redactValue({
      schema_version: 1,
      receipt_id: `mcp-${Date.now()}-${sequence++}`,
      timestamp: new Date().toISOString(),
      ...entry,
    }));
    return flush();
  };

  return { path: resolved, relativePath: relative(service.root, resolved), record };
}

export { redactValue };
