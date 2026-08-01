export type SessionStatus =
  | "created"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "timed_out"
  | "budget_exhausted";

export type EventType =
  | "session"
  | "queued"
  | "status"
  | "model"
  | "tool"
  | "reward"
  | "budget"
  | "error"
  | "cancelled";

export interface ModelMetadata {
  backend: string;
  base_model: string;
  adapter: string;
  adapter_sha256: string;
  frozen_candidate_sha256: string;
  persistent: boolean;
  loaded: boolean;
}

export interface InferenceBudget {
  max_turns: number;
  timeout_seconds: number;
  max_tool_calls: number;
  max_new_tokens: number;
  max_total_tokens: number;
}

export interface UsageMetrics {
  model_calls: number;
  tool_calls: number;
  prompt_tokens: number;
  response_tokens: number;
  elapsed_ms: number;
}

export interface Health {
  status: string;
  service: string;
  version: string;
  model: ModelMetadata;
  active_sessions: number;
  max_concurrent_sessions: number;
  repository: string;
  timestamp: string;
  details: Record<string, unknown>;
}

export interface Scenario {
  scenario_id: string;
  db_id: string;
  question: string;
  stale_sql: string;
  drift_type: string;
  wildcard_profile: string | null;
  difficulty: string | null;
  schema_diff: Record<string, unknown>;
}

export interface DatabaseSummary {
  db_id: string;
  scenario_count: number;
  drift_types: string[];
}

export interface Experiment {
  experiment_id: string;
  display_name: string;
  category: string;
  tasks: number;
  task_success_rate: number;
  executable_rate: number;
  average_model_calls: number;
  average_tool_calls: number;
  unsafe_tasks: number;
  selected: boolean;
}

export interface ExperimentList {
  experiments: Experiment[];
  selected_experiment_id: string;
}

export interface DriftMetric {
  drift_type: string;
  sessions: number;
  successful: number;
  success_rate: number;
}

export interface DailyMetric {
  day: string;
  sessions: number;
  successful: number;
  failed: number;
}

export interface ModelDeployment {
  base_model: string;
  adapter: string;
  adapter_sha256: string;
  sessions: number;
  successful: number;
  success_rate: number;
}

export interface OperationsSummary {
  generated_at: string;
  total_sessions: number;
  terminal_sessions: number;
  active_sessions: number;
  successful_sessions: number;
  failed_sessions: number;
  unsafe_sessions: number;
  timed_out_sessions: number;
  success_rate: number;
  average_latency_ms: number;
  average_model_calls: number;
  average_tool_calls: number;
  total_prompt_tokens: number;
  total_response_tokens: number;
  tool_failure_rate: number;
  drift_metrics: DriftMetric[];
  daily_metrics: DailyMetric[];
  deployments: ModelDeployment[];
}

export type FailureType =
  | "unsafe"
  | "timed_out"
  | "budget_exhausted"
  | "cancelled"
  | "service_error"
  | "task_failure";

export interface Failure {
  session_id: string;
  scenario_id: string;
  db_id: string;
  drift_type: string;
  status: SessionStatus;
  failure_type: FailureType;
  termination_reason: string | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  model_calls: number;
  tool_calls: number;
  response_tokens: number;
  elapsed_ms: number;
  adapter_sha256: string;
}

export interface FailureList {
  failures: Failure[];
  total: number;
  counts: Record<string, number>;
}

export interface WandbRun {
  run_id: string;
  name: string;
  state: string;
  url: string;
  created_at: string | null;
  summary_metrics: Record<string, number>;
}

export interface WandbRunList {
  provider: "wandb";
  configured: boolean;
  status: "disabled" | "ready" | "error";
  entity: string | null;
  project: string;
  project_url: string | null;
  runs: WandbRun[];
  error: string | null;
}

export interface WandbMetricPoint {
  step: number;
  value: number;
}

export interface WandbMetricSeries {
  name: string;
  points: WandbMetricPoint[];
}

export interface WandbRunHistory {
  provider: "wandb";
  configured: boolean;
  status: "disabled" | "ready" | "error";
  run_id: string;
  series: WandbMetricSeries[];
  error: string | null;
}

export interface Session {
  session_id: string;
  scenario_id: string;
  db_id: string;
  db_version: string;
  status: SessionStatus;
  question: string;
  stale_sql: string;
  drift_type: string;
  wildcard_profile: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  termination_reason: string | null;
  final_sql: string | null;
  success: boolean | null;
  cancellation_requested: boolean;
  sandbox_isolated: boolean;
  sandbox_ref: string;
  source_db_sha256: string;
  model: ModelMetadata;
  budget: InferenceBudget | null;
  usage: UsageMetrics;
  labels: Record<string, string>;
  result: Record<string, unknown>;
}

export interface TrajectoryEvent {
  session_id: string;
  sequence: number;
  event_type: EventType;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface Trajectory {
  session: Session;
  events: TrajectoryEvent[];
  messages: Array<Record<string, unknown>>;
  reward: RewardResult;
}

export interface RewardResult extends Record<string, unknown> {
  score?: number;
  task_success?: boolean;
  execution_success?: boolean;
  format_valid?: boolean;
  unsafe?: boolean;
  timed_out?: boolean;
  error?: string;
}

export interface SessionList {
  sessions: Session[];
  total: number;
}

export interface RunOptions {
  max_turns: number;
  timeout_seconds: number;
  max_tool_calls: number;
  max_new_tokens: number;
  max_total_tokens: number;
}

export const terminalStatuses: ReadonlySet<SessionStatus> = new Set([
  "completed",
  "failed",
  "cancelled",
  "timed_out",
  "budget_exhausted",
]);
