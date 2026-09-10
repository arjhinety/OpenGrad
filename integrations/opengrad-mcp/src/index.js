#!/usr/bin/env node
import { serveStdio, createMcpServer } from "./server.js";

const server = createMcpServer({
  root: process.env.OPENGRAD_ROOT,
  timeoutMs: process.env.OPENGRAD_MCP_TIMEOUT_MS ? Number(process.env.OPENGRAD_MCP_TIMEOUT_MS) : undefined,
  executable: process.env.OPENGRAD_EXECUTABLE,
});

serveStdio(server);
