import type {
  DatabaseSummary,
  ExperimentList,
  FailureList,
  Health,
  OperationsSummary,
  RunOptions,
  Scenario,
  Session,
  SessionList,
  Trajectory,
  TrajectoryEvent,
  WandbRunList,
  WandbRunHistory,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      // Keep the HTTP status text when an upstream proxy returns HTML.
    }
    throw new ApiError(detail || `请求失败：HTTP ${response.status}`, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  scenarios: () => request<Scenario[]>("/api/scenarios"),
  databases: () => request<DatabaseSummary[]>("/api/databases"),
  experiments: () => request<ExperimentList>("/api/experiments"),
  operations: () => request<OperationsSummary>("/api/observability/summary"),
  failures: (failureType = "") => request<FailureList>(
    `/api/observability/failures?limit=100${failureType ? `&failure_type=${encodeURIComponent(failureType)}` : ""}`,
  ),
  wandbRuns: () => request<WandbRunList>("/api/observability/wandb/runs"),
  wandbHistory: (runId: string) => request<WandbRunHistory>(
    `/api/observability/wandb/runs/${encodeURIComponent(runId)}/history`,
  ),
  sessions: () => request<SessionList>("/api/sessions?limit=50"),
  session: (sessionId: string) => request<Session>(`/api/sessions/${sessionId}`),
  trajectory: (sessionId: string) =>
    request<Trajectory>(`/api/sessions/${sessionId}/trajectory`),
  createSession: (scenarioId: string) =>
    request<Session>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId, labels: { source: "studio" } }),
    }),
  runSession: (sessionId: string, options: RunOptions) =>
    request<Session>(`/api/sessions/${sessionId}/run`, {
      method: "POST",
      body: JSON.stringify(options),
    }),
  cancelSession: (sessionId: string) =>
    request<Session>(`/api/sessions/${sessionId}/cancel`, {
      method: "POST",
    }),
};

export interface EventSubscription {
  close: () => void;
}

export function subscribeToEvents(
  sessionId: string,
  afterSequence: number,
  onEvent: (event: TrajectoryEvent) => void,
  onError: (error: Event) => void,
  onOpen?: () => void,
): EventSubscription {
  const source = new EventSource(
    `/api/sessions/${sessionId}/events?after_sequence=${afterSequence}`,
  );
  const eventTypes = [
    "session",
    "queued",
    "status",
    "model",
    "tool",
    "reward",
    "budget",
    "error",
    "cancelled",
  ];
  for (const type of eventTypes) {
    source.addEventListener(type, (raw) => {
      const message = raw as MessageEvent<string>;
      onEvent(JSON.parse(message.data) as TrajectoryEvent);
    });
  }
  source.onerror = onError;
  source.onopen = () => onOpen?.();
  return { close: () => source.close() };
}
