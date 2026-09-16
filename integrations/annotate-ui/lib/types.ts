// Shapes returned by the Python API (src/opengrad/annotation/server.py). The server is the authority
// on every rule; these types only describe what it sends.

export type TaskType =
  | "single_label"
  | "multi_label"
  | "binary"
  | "rating"
  | "free_text"
  | "pairwise"
  | "ranking";

export interface DisplayField {
  key: string;
  label: string;
  render: "text" | "json" | "tools" | "conversation" | "calls";
  emphasis: boolean;
  available: boolean;
  missing_text?: string;
}

export interface ExtraField {
  key: string;
  label: string;
  type: "text" | "select";
  options: string[];
  required: boolean;
  default: string | null;
  help: string;
  in_adjudication: boolean;
}

export interface Constraint {
  field: string;
  when_label_in: string[] | null;
  when_label_not_in: string[] | null;
  allowed: string[] | null;
  forbidden: string[] | null;
  message: string;
}

export interface TaskPublic {
  task_id: string;
  version: string;
  title: string;
  task_type: TaskType;
  primary_key: string;
  annotation_schema_version: string;
  labels: string[];
  shortcuts: { key: string; label: string }[];
  unknown_labels: string[];
  display: DisplayField[];
  metadata: DisplayField[];
  filters: string[];
  candidates: string[];
  scale: [number, number, number] | null;
  extra_fields: ExtraField[];
  adjudication_fields: ExtraField[];
  constraints: Constraint[];
  instructions: string[];
}

export type Value = Record<string, unknown>;

export interface Progress {
  total: number;
  completed: number;
  remaining: number;
  percent: number;
  skipped: number;
  flagged: number;
  unknown: number;
  label_counts: Record<string, number>;
}

export interface SessionState {
  session_id: string;
  annotator_id: string;
  created_at: string;
  frozen: boolean;
  frozen_at: string | null;
  frozen_manifest: string | null;
}

/** A pinned review queue: its name and this session's progress through it. Never why an item is in it. */
export interface QueueProgress {
  name: string;
  total: number;
  completed: number;
  remaining: number;
}

/** Counts only, for a human pass beside model sessions. Never which item a model labeled, nor how. */
export interface ReferenceProgress {
  human_reviewed: number;
  provisional_model_only: number;
  unlabeled: number;
  remaining_for_human_review: number;
  flagged: number;
  model_sessions: string[];
}

export interface AppState {
  app_version: string;
  mode: "annotate" | "adjudicate";
  task: TaskPublic;
  source: { path: string; sha256: string; items: number };
  filter_options: Record<string, string[]>;
  session?: SessionState;
  progress?: Progress;
  reference?: ReferenceProgress | null;
  queues?: QueueProgress[];
  adjudication?: {
    sessions: SessionState[];
    adjudicator_id: string;
    kind: "adjudication" | "review";
    frozen: boolean;
  };
}

export interface Annotation {
  status: "labeled" | "skipped" | "open";
  value: Value | null;
  note: string | null;
  flagged: boolean;
  revision: number;
  created_at: string;
  updated_at: string;
  state_sha256: string;
  previous_state_sha256: string | null;
}

export interface Change {
  action: string;
  actor_id: string;
  reason: string | null;
  recorded_at: string;
  from: unknown;
  to: unknown;
  from_status: string | null;
  to_status: string | null;
  before_sha256: string | null;
  after_sha256: string | null;
  entry_sha256: string;
}

export interface ItemCore {
  item_id: string;
  index: number;
  total: number;
  fields: Record<string, unknown>;
  metadata: Record<string, unknown>;
  filters: Record<string, unknown>;
}

export interface ItemView extends ItemCore {
  prev_id: string | null;
  next_id: string | null;
  annotation: Annotation | null;
  history: Change[];
}

export interface ListedItem {
  item_id: string;
  index: number;
  status: "labeled" | "skipped" | "open";
  flagged: boolean;
  value: unknown;
}

export interface SaveResult {
  item_id: string;
  annotation: Annotation | null;
  progress: Progress;
  next_item_id: string | null;
}

export interface Adjudication {
  kind: "adjudication" | "review";
  value: Value;
  rationale: string;
  adjudicator_id: string;
  disagreement: boolean;
  flag: string | null;
  revision: number;
  updated_at: string;
  state_sha256: string;
  previous_state_sha256: string | null;
}

export interface AdjudicationView extends ItemCore {
  passes: { session_id: string; annotator_id: string; annotation: Annotation | null }[];
  disagreement: boolean;
  adjudication: Adjudication | null;
}

export interface QueueItem {
  item_id: string;
  index: number;
  reasons: string[];
  values: unknown[];
  adjudicated: boolean;
}

export interface InstructionDoc {
  title: string;
  path: string;
  content: string;
}
