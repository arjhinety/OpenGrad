import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const PACKAGE_ROOT = dirname(fileURLToPath(import.meta.url));
const REPO_HINT = resolve(PACKAGE_ROOT, "../../..");
const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_OUTPUT_BYTES = 1024 * 1024;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/;
const EVALUATION_SUITES = new Set([
  "smoke", "tool_use_core", "regression_core", "agent_transfer",
  "full_post_training", "speculative_decoding",
]);

const ERROR_CODES = new Set([
  "CONFIG_INVALID", "DATASET_SCHEMA_INVALID", "CHECKSUM_MISMATCH", "CONTAMINATION_FAILURE",
  "TOKENIZER_MISMATCH", "BASELINE_NOT_FOUND", "BASELINE_NOT_READY", "BENCHMARK_VERSION_MISMATCH",
  "PATH_NOT_WRITABLE", "PATH_OUTSIDE_PROJECT", "ALGORITHM_UNSUPPORTED", "CHECKPOINT_NOT_FOUND",
  "CONFIG_NOT_FOUND", "SUITE_NOT_FOUND", "FAILURES_NOT_FOUND", "EXPERIMENT_NOT_FOUND",
  "DATASET_NOT_FOUND", "MODEL_INVALID", "SEED_MISSING", "GPU_UNAVAILABLE", "GPU_SMOKE_FAILED",
  "PREFLIGHT_FAILED", "COMMAND_FAILED", "REPOSITORY_NOT_FOUND", "SUBPROCESS_UNAVAILABLE",
  "ARTIFACT_NOT_FOUND", "INVALID_ARGUMENT", "WORKFLOW_STAGE_FAILED", "EXPERIMENT_ID_COLLISION",
  "NO_TRAINING_ON_EVAL_DATA", "NO_FLOATING_MODEL_REVISION", "NO_FLOATING_DATASET_REVISION",
]);

// Key-name matching is deliberately broad: for a receipt/audit sink,
// over-redacting an ordinary field is far safer than leaking a credential.
const SECRET_KEY = /(key|token|secret|password|authorization|credential)/i;

export function redactText(value) {
  return String(value ?? "")
    // Redact the authorization scheme first: otherwise the generic `key: value`
    // rule consumes "Bearer" as the value and leaves the credential in place.
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]")
    .replace(/((?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|secret|credential|private[_-]?key|client[_-]?secret|token|key)\s*[=:]\s*)(["']?)[^\s,;"'}]+\2/gi, "$1[REDACTED]")
    .replace(/\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9_]{12,})\b/g, "[REDACTED]");
}

export function redactValue(value, key = "") {
  if (SECRET_KEY.test(String(key))) return "[REDACTED]";
  if (typeof value === "string") return redactText(value);
  if (Array.isArray(value)) return value.map((item) => redactValue(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([name, item]) => [name, redactValue(item, name)]));
  }
  return value;
}

export function stableError(error, fallbackCode = "COMMAND_FAILED", context = undefined) {
  const message = redactText(error instanceof Error ? error.message : String(error));
  const code = ERROR_CODES.has(error?.code) ? error.code : fallbackCode;
  const rawContext = context === undefined ? error?.context : context;
  return {
    ok: false,
    code,
    message,
    ...(rawContext === undefined ? {} : { context: redactValue(rawContext) }),
    blocking: true,
  };
}

function hashText(text) {
  return createHash("sha256").update(text).digest("hex");
}

function parseJson(stdout) {
  const text = String(stdout ?? "").trim();
  if (!text) throw Object.assign(new Error("OpenGrad returned no JSON on stdout"), { code: "COMMAND_FAILED" });
  try { return JSON.parse(text); }
  catch (error) {
    const wrapped = new Error(`OpenGrad returned invalid JSON: ${error.message}`);
    wrapped.code = "COMMAND_FAILED";
    throw wrapped;
  }
}

function validateTimeout(value) {
  if (!Number.isFinite(value) || value < 1) throw Object.assign(new Error("timeoutMs must be a positive finite number"), { code: "INVALID_ARGUMENT" });
  return value;
}

function validateLimit(value) {
  if (value === undefined) return;
  if (!Number.isSafeInteger(value) || value < 1) throw Object.assign(new Error("limit must be a positive safe integer"), { code: "INVALID_ARGUMENT" });
}

function validateIdentifier(value, label) {
  if (typeof value !== "string" || !IDENTIFIER.test(value) || value.startsWith("-")) {
    throw Object.assign(new Error(`${label} must be a simple OpenGrad identifier`), { code: "INVALID_ARGUMENT" });
  }
  return value;
}

function appendBounded(current, chunk) {
  if (current.length >= DEFAULT_OUTPUT_BYTES) return current;
  const next = current + chunk.toString("utf8");
  return next.length > DEFAULT_OUTPUT_BYTES ? next.slice(0, DEFAULT_OUTPUT_BYTES) : next;
}

function runProcess(executable, args, { cwd, timeoutMs, signal }) {
  return new Promise((resolvePromise) => {
    let child;
    try {
      child = spawn(executable, args, { cwd, stdio: ["ignore", "pipe", "pipe"], shell: false });
    } catch (error) {
      resolvePromise({ spawnError: error });
      return;
    }
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    let timer;
    const onAbort = () => { child.kill("SIGKILL"); };
    const finish = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onAbort);
      resolvePromise(payload);
    };
    timer = setTimeout(() => { timedOut = true; child.kill("SIGKILL"); }, timeoutMs);
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }
    child.stdout?.on("data", (chunk) => { stdout = appendBounded(stdout, chunk); });
    child.stderr?.on("data", (chunk) => { stderr = appendBounded(stderr, chunk); });
    child.on("error", (error) => finish({ spawnError: error }));
    child.on("close", (code, sig) => finish({
      stdout,
      stderr,
      exitCode: code,
      signal: sig ?? null,
      timedOut,
      stdoutLossy: stdout.length >= DEFAULT_OUTPUT_BYTES,
      stderrLossy: stderr.length >= DEFAULT_OUTPUT_BYTES,
    }));
  });
}

function outputError(processResult, args) {
  let parsed;
  try { parsed = JSON.parse(String(processResult.stdout ?? "").trim()); } catch { parsed = undefined; }
  if (parsed && typeof parsed === "object" && typeof parsed.code === "string") {
    return { ...parsed, ok: false, blocking: parsed.blocking !== false, command: [...args] };
  }
  const detail = String(processResult.stderr ?? "").trim();
  const suffix = processResult.timedOut ? " (timed out)" : "";
  return {
    ok: false,
    code: "COMMAND_FAILED",
    message: (detail || `OpenGrad exited with code ${processResult.exitCode}`) + suffix,
    blocking: true,
    command: [...args],
    exitCode: processResult.exitCode,
  };
}

function realCandidate(candidate) {
  try { return realpathSync(candidate); }
  catch {
    const suffix = [];
    let current = candidate;
    while (!existsSync(current) && current !== dirname(current)) {
      suffix.unshift(relative(dirname(current), current));
      current = dirname(current);
    }
    return join(realpathSync(current), ...suffix);
  }
}

export function ensureInside(root, input, label = "path") {
  if (typeof input !== "string" || input.length === 0 || input.includes("\0")) {
    throw Object.assign(new Error(`${label} must be a non-empty path`), { code: "INVALID_ARGUMENT" });
  }
  const canonicalRoot = realCandidate(resolve(root));
  const lexical = resolve(canonicalRoot, input);
  const candidate = realCandidate(lexical);
  const rel = relative(canonicalRoot, candidate);
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw Object.assign(new Error(`${label} must remain inside the OpenGrad repository`), { code: "PATH_OUTSIDE_PROJECT" });
  }
  return candidate;
}

export class OpenGradService {
  constructor({ root = process.env.OPENGRAD_ROOT || REPO_HINT, timeoutMs = DEFAULT_TIMEOUT_MS, executable = undefined } = {}) {
    this.root = resolve(root);
    this.timeoutMs = validateTimeout(timeoutMs);
    const venvExecutable = join(this.root, ".venv", "bin", "opengrad");
    this.executable = executable ?? process.env.OPENGRAD_EXECUTABLE ?? (existsSync(venvExecutable) ? venvExecutable : "opengrad");
  }

  async discoverRoot() {
    const candidates = [];
    let current = this.root;
    while (current !== dirname(current)) {
      candidates.push(current);
      current = dirname(current);
    }
    candidates.push(REPO_HINT);
    for (const candidate of [...new Set(candidates)]) {
      try {
        const [git, pyproject] = await Promise.all([stat(join(candidate, ".git")), stat(join(candidate, "pyproject.toml"))]);
        if ((git.isDirectory() || git.isFile()) && pyproject.isFile()) return candidate;
      } catch { /* try the next parent */ }
    }
    throw Object.assign(new Error("Unable to discover an OpenGrad repository root"), { code: "REPOSITORY_NOT_FOUND" });
  }

  projectPath(path, label = "path") { return ensureInside(this.root, path, label); }

  targetArgument(target) {
    if (typeof target !== "string" || target.length === 0 || target.includes("\0")) {
      throw Object.assign(new Error("target must be a non-empty string"), { code: "INVALID_ARGUMENT" });
    }
    // Checkpoint paths are project-confined. Hugging Face model IDs are the
    // only non-path target accepted by the evaluation command.
    if (target.startsWith("/") || target.startsWith(".") || target.startsWith("runs/") || target.startsWith("checkpoints/")) return this.projectPath(target, "target");
    if (/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(target)) return target;
    return this.projectPath(target, "target");
  }

  async command(args, { timeoutMs = this.timeoutMs, signal, allowNonZeroJson = false } = {}) {
    validateTimeout(timeoutMs);
    // A missing cwd makes spawn fail with a bare ENOENT, which hides the real
    // cause. Fail with the actionable code instead.
    if (!existsSync(this.root)) {
      return { ok: false, code: "REPOSITORY_NOT_FOUND", message: `OpenGrad root does not exist: ${this.root}`, blocking: true, command: [...args] };
    }
    const result = await runProcess(this.executable, args, { cwd: this.root, timeoutMs, signal });
    if (result.spawnError) return stableError(result.spawnError, "SUBPROCESS_UNAVAILABLE");
    const base = {
      command: [...args],
      stdout: redactText(result.stdout),
      stderr: redactText(result.stderr),
      stdout_lossy: Boolean(result.stdoutLossy),
      stderr_lossy: Boolean(result.stderrLossy),
      exitCode: result.exitCode ?? null,
      signal: result.signal ?? null,
      timed_out: Boolean(result.timedOut),
    };
    // Gate projections (`readiness`, `gpu-smoke`) exit non-zero to *report* a
    // FAIL/BLOCKED outcome. That exit code is payload, not a command failure, so
    // parse the JSON and let the caller inspect the gates.
    if (base.exitCode !== 0 && !allowNonZeroJson) return { ...outputError(result, args), ...base };
    try { return { ok: true, value: parseJson(base.stdout), ...base }; }
    catch (error) { return { ...stableError(error), ...base }; }
  }

  async doctor(options = {}) { return this.command(["doctor", "--json"], options); }
  async validate(options = {}) { return this.command(["validate", "--json"], options); }
  async status(options = {}) { return this.command(["status", "--json"], options); }
  async readiness(config = undefined, options = {}) {
    return this.command(["readiness", ...(config ? [this.projectPath(config, "config")] : []), "--json"], { ...options, allowNonZeroJson: true });
  }
  async gpuSmoke(config = undefined, options = {}) {
    return this.command(["gpu-smoke", ...(config ? [this.projectPath(config, "config")] : []), "--json"], { ...options, allowNonZeroJson: true });
  }
  async validateData(records, mode = "sft", options = {}) {
    if (mode !== "sft") throw Object.assign(new Error("Only SFT data validation is exposed by this integration"), { code: "ALGORITHM_UNSUPPORTED" });
    return this.command(["validate-data", this.projectPath(records, "records"), "--mode", mode, "--json"], options);
  }
  async inspectTemplate(record = undefined, options = {}) { return this.command(["inspect-template", ...(record ? ["--record", this.projectPath(record, "record")] : []), "--json"], options); }
  async preflight(config, options = {}) { return this.command(["preflight", this.projectPath(config, "config"), "--json"], options); }
  async baseline({ config, dryRun = false, limit, signal } = {}) {
    validateLimit(limit);
    return this.command(["baseline", "--config", this.projectPath(config ?? "configs/evaluation/tool_calling/qwen35_2b_baseline.yaml", "config"), ...(dryRun ? ["--dry-run"] : []), ...(limit === undefined ? [] : ["--limit", String(limit)]), "--json"], { signal });
  }
  async train({ config, dryRun = false, signal } = {}) { return this.command(["train", this.projectPath(config, "config"), ...(dryRun ? ["--dry-run"] : []), "--json"], { signal }); }
  async evaluate({ target, suite = "smoke", dryRun = false, limit, signal } = {}) {
    validateLimit(limit);
    validateIdentifier(suite, "suite");
    if (!EVALUATION_SUITES.has(suite)) throw Object.assign(new Error(`suite must be one of ${[...EVALUATION_SUITES].sort().join(", ")}`), { code: "SUITE_NOT_FOUND" });
    return this.command(["evaluate", this.targetArgument(target), "--suite", suite, ...(dryRun ? ["--dry-run"] : []), ...(limit === undefined ? [] : ["--limit", String(limit)]), "--json"], { signal });
  }
  async compare(baseline, candidate, options = {}) { return this.command(["compare", this.projectPath(baseline, "baseline"), this.projectPath(candidate, "candidate"), "--json"], options); }
  async failures(runDir, options = {}) { return this.command(["failures", this.projectPath(runDir, "run directory"), "--json"], options); }
  async experimentList(options = {}) { return this.command(["experiment", "list", "--json"], options); }
  async experimentShow(id, options = {}) { return this.command(["experiment", "show", validateIdentifier(id, "experiment_id"), "--json"], options); }
  async experimentDiff(a, b, options = {}) { return this.command(["experiment", "diff", validateIdentifier(a, "baseline experiment"), validateIdentifier(b, "candidate experiment"), "--json"], options); }
  async checkpointList(status = undefined, options = {}) { return this.command(["checkpoint", "list", ...(status ? ["--status", validateIdentifier(status, "status")] : []), "--json"], options); }
  async checkpointInspect(id, options = {}) { return this.command(["checkpoint", "inspect", validateIdentifier(id, "checkpoint_id"), "--json"], options); }

  async readJson(path) {
    const target = this.projectPath(path);
    try { return { ok: true, value: JSON.parse(await readFile(target, "utf8")), path: target }; }
    catch (error) { return stableError(error, "ARTIFACT_NOT_FOUND"); }
  }

  async artifactReceipt(paths) {
    const artifacts = [];
    for (const path of paths) {
      const target = this.projectPath(path, "artifact");
      try {
        const text = await readFile(target, "utf8");
        artifacts.push({ path: relative(this.root, target), sha256: hashText(text), bytes: Buffer.byteLength(text) });
      } catch (error) { artifacts.push({ path: relative(this.root, target), error: error.message }); }
    }
    return artifacts;
  }
}

export { DEFAULT_OUTPUT_BYTES, DEFAULT_TIMEOUT_MS, ERROR_CODES, EVALUATION_SUITES, REPO_HINT, hashText };
