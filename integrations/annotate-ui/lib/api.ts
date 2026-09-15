import type {
  AdjudicationView,
  AppState,
  InstructionDoc,
  ItemView,
  ListedItem,
  QueueItem,
  SaveResult,
  Value,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly kind: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { cache: "no-store", ...init });
  } catch {
    throw new ApiError("Cannot reach the annotation server. Is `opengrad-annotate serve` running?", 0, "offline");
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(`Unexpected response from ${path} (${response.status})`, response.status, "bad_response");
  }
  if (!response.ok) {
    const error = body as { error?: string; kind?: string } | null;
    throw new ApiError(error?.error ?? `HTTP ${response.status}`, response.status, error?.kind ?? "error");
  }
  return body as T;
}

const post = <T>(path: string, body: unknown) =>
  call<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const item = (id: string) => `/api/items/${encodeURIComponent(id)}`;

export const api = {
  state: () => call<AppState>("/api/state"),
  instructions: () => call<{ documents: InstructionDoc[] }>("/api/instructions"),
  items: (params: Record<string, string>) =>
    call<{ items: ListedItem[]; count: number }>(`/api/items?${new URLSearchParams(params)}`),
  /** The next item to label: in frozen order, or in a pinned review queue's order when `queue` is given. */
  next: (after?: string, queue?: string) => {
    const params = new URLSearchParams();
    if (after) params.set("after", after);
    if (queue) params.set("queue", queue);
    const query = params.toString();
    return call<{ item_id: string | null }>(`/api/items/next${query ? `?${query}` : ""}`);
  },
  item: (id: string) => call<ItemView>(item(id)),
  annotate: (
    id: string,
    // `flagged` is omitted on purpose when labeling: the server keeps whatever flag the item already has.
    body: { value: Value; note: string; flagged?: boolean; replace: boolean; expected_revision: number; reason?: string },
  ) => post<SaveResult>(`${item(id)}/annotate`, body),
  skip: (id: string, note: string) => post<SaveResult>(`${item(id)}/skip`, { note }),
  flag: (id: string, flagged: boolean, note: string) => post<SaveResult>(`${item(id)}/flag`, { flagged, note }),
  undo: () => post<SaveResult>("/api/undo", {}),
  queue: () => call<{ items: QueueItem[]; count: number }>("/api/adjudication/queue"),
  adjudicationItem: (id: string) =>
    call<AdjudicationView>(`/api/adjudication/items/${encodeURIComponent(id)}`),
  adjudicate: (id: string, value: Value, rationale: string) =>
    post<{ item_id: string }>(`/api/adjudication/items/${encodeURIComponent(id)}`, { value, rationale }),
};
