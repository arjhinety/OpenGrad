"use client";

import { forwardRef, useImperativeHandle, useRef } from "react";
import type { ExtraField, TaskPublic, Value } from "@/lib/types";

export interface FormHandle {
  focus(key: string): void;
}

interface Props {
  task: TaskPublic;
  fields: ExtraField[];
  value: Value;
  savedValue: Value | null | undefined;
  problemKeys: Set<string>;
  disabled: boolean;
  onChange(next: Value): void;
  /** Single-choice tasks: a label was chosen by click or shortcut; the caller decides whether to save. */
  onChoose(next: Value): void;
  onSubmit(): void;
  onDefinition?(label: string): void;
}

function Primary({ task, value, savedValue, disabled, onChange, onChoose, onDefinition }: Props) {
  const key = task.primary_key;
  const shortcuts = new Map(task.shortcuts.map((item) => [item.label, item.key]));
  const current = value[key];
  const saved = savedValue?.[key];

  if (task.task_type === "multi_label") {
    const chosen = new Set(Array.isArray(current) ? (current as string[]) : []);
    return (
      <div className="toggles" role="group" aria-label="Labels">
        {task.labels.map((label) => (
          <button
            key={label}
            type="button"
            className={`btn${chosen.has(label) ? " on" : ""}`}
            disabled={disabled}
            aria-pressed={chosen.has(label)}
            onClick={() => {
              const next = new Set(chosen);
              if (next.has(label)) next.delete(label);
              else next.add(label);
              onChange({ ...value, [key]: task.labels.filter((item) => next.has(item)) });
            }}
          >
            {label}
            {shortcuts.get(label) ? <kbd>{shortcuts.get(label)}</kbd> : null}
          </button>
        ))}
      </div>
    );
  }

  if (task.task_type === "rating" && task.scale) {
    const [low, high, step] = task.scale;
    const points: number[] = [];
    for (let point = low; point <= high + 1e-9 && points.length <= 21; point += step) points.push(Number(point.toFixed(6)));
    return points.length <= 21 ? (
      <div className="scale" role="group" aria-label="Rating">
        {points.map((point) => (
          <button
            key={point}
            type="button"
            className={`btn${current === point ? " on" : ""}`}
            disabled={disabled}
            onClick={() => onChoose({ ...value, [key]: point })}
          >
            {point}
          </button>
        ))}
      </div>
    ) : (
      <input
        type="number"
        min={low}
        max={high}
        step={step}
        disabled={disabled}
        value={typeof current === "number" ? current : ""}
        onChange={(event) => onChange({ ...value, [key]: event.target.value === "" ? null : Number(event.target.value) })}
      />
    );
  }

  if (task.task_type === "free_text") {
    return (
      <div className="form-row">
        <textarea
          rows={5}
          disabled={disabled}
          value={typeof current === "string" ? current : ""}
          onChange={(event) => onChange({ ...value, [key]: event.target.value })}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) onChoose(value);
          }}
          placeholder="Write the annotation. Ctrl+Enter saves."
        />
      </div>
    );
  }

  if (task.task_type === "ranking") {
    const order = Array.isArray(current) ? (current as string[]) : task.candidates;
    const move = (index: number, delta: number) => {
      const next = [...order];
      const [item] = next.splice(index, 1);
      next.splice(index + delta, 0, item);
      onChange({ ...value, [key]: next });
    };
    return (
      <ol className="rank">
        {order.map((candidate, index) => (
          <li key={candidate}>
            <span className="pos">{index + 1}</span>
            <span className="name">{task.display.find((field) => field.key === candidate)?.label ?? candidate}</span>
            <button type="button" className="btn small" disabled={disabled || index === 0} onClick={() => move(index, -1)} aria-label="Move up">
              ↑
            </button>
            <button type="button" className="btn small" disabled={disabled || index === order.length - 1} onClick={() => move(index, 1)} aria-label="Move down">
              ↓
            </button>
          </li>
        ))}
      </ol>
    );
  }

  return (
    <div className="labels" role="group" aria-label="Label">
      {task.labels.map((label) => (
        <button
          key={label}
          type="button"
          className={`label-btn${current === label ? " selected" : ""}${saved === label ? " saved" : ""}`}
          disabled={disabled}
          aria-pressed={current === label}
          onClick={() => onChoose({ ...value, [key]: label })}
        >
          <kbd>{shortcuts.get(label) ?? "·"}</kbd>
          <span>{label}</span>
          {onDefinition ? (
            <span
              className="def"
              role="link"
              tabIndex={-1}
              title={`Open the rubric at ${label}`}
              onClick={(event) => {
                event.stopPropagation();
                onDefinition(label);
              }}
            >
              definition
            </span>
          ) : (
            <span />
          )}
        </button>
      ))}
    </div>
  );
}

export const PrimaryControl = Primary;

export const ExtraFields = forwardRef<FormHandle, Props>(function ExtraFields(props, ref) {
  const { fields, value, problemKeys, disabled, onChange, onSubmit } = props;
  const refs = useRef(new Map<string, HTMLElement>());
  useImperativeHandle(ref, () => ({
    focus(key: string) {
      const element = refs.current.get(key);
      element?.focus();
      element?.scrollIntoView({ block: "nearest" });
    },
  }));
  if (!fields.length) return null;
  return (
    <>
      {fields.map((field) => {
        const id = `field-${field.key}`;
        const current = value[field.key] === null || value[field.key] === undefined ? "" : String(value[field.key]);
        const bind = (element: HTMLElement | null) => {
          if (element) refs.current.set(field.key, element);
          else refs.current.delete(field.key);
        };
        return (
          <div className={`form-row${problemKeys.has(field.key) ? " problem" : ""}`} key={field.key}>
            <label htmlFor={id}>
              {field.label} {field.required ? null : <span className="opt">(optional)</span>}
            </label>
            {field.type === "select" ? (
              <select
                id={id}
                ref={bind}
                disabled={disabled}
                value={current}
                onChange={(event) => onChange({ ...value, [field.key]: event.target.value })}
              >
                {field.required && !field.default ? <option value="">Choose…</option> : null}
                {!field.required && !field.default ? <option value="">—</option> : null}
                {field.options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ) : (
              <textarea
                id={id}
                ref={bind}
                rows={2}
                disabled={disabled}
                value={current}
                onChange={(event) => onChange({ ...value, [field.key]: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    onSubmit();
                  }
                }}
                placeholder="Enter saves · Shift+Enter for a new line"
              />
            )}
            {field.help ? <div className="help">{field.help}</div> : null}
          </div>
        );
      })}
    </>
  );
});
