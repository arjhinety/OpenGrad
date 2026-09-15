import type { ExtraField, TaskPublic, Value } from "./types";

// A UX mirror of src/opengrad/annotation/values.py: it only decides whether a save is worth attempting
// and which field to focus. The server re-validates every write and its answer wins.

export function fieldsFor(task: TaskPublic, adjudication: boolean): ExtraField[] {
  if (!adjudication) return task.extra_fields;
  return [...task.extra_fields.filter((field) => field.in_adjudication), ...task.adjudication_fields];
}

export function initialValue(task: TaskPublic, fields: ExtraField[], existing: Value | null | undefined): Value {
  const value: Value = {};
  const primary = task.primary_key;
  if (existing && existing[primary] !== undefined) value[primary] = existing[primary];
  else if (task.task_type === "multi_label") value[primary] = [];
  else if (task.task_type === "ranking") value[primary] = [...task.candidates];
  for (const field of fields) {
    const current = existing?.[field.key];
    value[field.key] = current === null || current === undefined ? (field.default ?? "") : String(current);
  }
  return value;
}

function blank(value: unknown): boolean {
  return value === undefined || value === null || String(value).trim() === "";
}

export function primaryChosen(task: TaskPublic, value: Value): boolean {
  const primary = value[task.primary_key];
  switch (task.task_type) {
    case "multi_label":
      return Array.isArray(primary);
    case "ranking":
      return Array.isArray(primary) && primary.length === task.candidates.length;
    case "rating":
      return typeof primary === "number";
    case "free_text":
      return typeof primary === "string" && primary.trim() !== "";
    default:
      return typeof primary === "string" && task.labels.includes(primary);
  }
}

/** Problems the server would reject, each tied to the field to focus. */
export function problems(task: TaskPublic, fields: ExtraField[], value: Value): { field: string; message: string }[] {
  const found: { field: string; message: string }[] = [];
  if (!primaryChosen(task, value)) found.push({ field: task.primary_key, message: "Choose a label" });
  for (const field of fields) {
    if (field.required && blank(value[field.key])) found.push({ field: field.key, message: `${field.label} is required` });
  }
  const label = value.label;
  for (const rule of task.constraints) {
    const applies = rule.when_label_in ? rule.when_label_in.includes(String(label)) : !rule.when_label_not_in?.includes(String(label));
    if (!applies || !primaryChosen(task, value)) continue;
    const current = blank(value[rule.field]) ? null : String(value[rule.field]);
    const violated =
      (rule.allowed !== null && (current === null || !rule.allowed.includes(current))) ||
      (rule.forbidden !== null && current !== null && rule.forbidden.includes(current));
    if (violated) found.push({ field: rule.field, message: rule.message });
  }
  return found;
}

/** The value as the server stores it: optional empty fields become their default or null. */
export function canonical(task: TaskPublic, fields: ExtraField[], value: Value): Value {
  const out: Value = { [task.primary_key]: value[task.primary_key] };
  for (const field of fields) {
    out[field.key] = blank(value[field.key]) ? (field.default ?? null) : String(value[field.key]).trim();
  }
  return out;
}

export function sameValue(a: Value | null | undefined, b: Value | null | undefined): boolean {
  return JSON.stringify(sortKeys(a ?? null)) === JSON.stringify(sortKeys(b ?? null));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([x], [y]) => x.localeCompare(y))
        .map(([k, v]) => [k, sortKeys(v)]),
    );
  }
  return value;
}

export function display(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "(none)";
  return String(value);
}

export function shortHash(hash: string | null | undefined): string {
  return hash ? hash.slice(0, 10) : "—";
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}
