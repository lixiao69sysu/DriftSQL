import type { RewardResult, TrajectoryEvent } from "./types";

export function compactHash(value: string, length = 10): string {
  if (!value) return "—";
  return value.length <= length * 2 + 1
    ? value
    : `${value.slice(0, length)}…${value.slice(-length)}`;
}

export function formatDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return "0 ms";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${Math.floor(milliseconds / 60_000)}m ${Math.round((milliseconds % 60_000) / 1000)}s`;
}

export function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

export function parseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export interface ToolStep {
  event: TrajectoryEvent;
  turn: number;
  tool: string;
  arguments: Record<string, unknown>;
  observation: unknown;
  metrics: Record<string, unknown>;
  success: boolean;
  elapsedMs: number;
}

export function toolSteps(events: TrajectoryEvent[]): ToolStep[] {
  return events
    .filter((event) => event.event_type === "tool")
    .map((event) => ({
      event,
      turn: Number(event.payload.turn ?? 0),
      tool: String(event.payload.tool ?? "unknown"),
      arguments: (event.payload.arguments ?? {}) as Record<string, unknown>,
      observation: parseJson(event.payload.observation),
      metrics: (event.payload.metrics ?? {}) as Record<string, unknown>,
      success: Boolean(event.payload.success),
      elapsedMs: Number(event.payload.elapsed_ms ?? 0),
    }));
}

export interface RewardComponent {
  name: string;
  value: number;
  kind: "reward" | "penalty";
}

export function rewardComponents(reward: RewardResult): RewardComponent[] {
  return Object.entries(reward)
    .filter(([name, value]) =>
      (name.startsWith("reward_") || name.startsWith("penalty_")) &&
      typeof value === "number",
    )
    .map(([name, value]) => ({
      name: name.replace(/^(reward|penalty)_/, "").replaceAll("_", " "),
      value: Number(value),
      kind: name.startsWith("reward_") ? "reward" : "penalty",
    }));
}

export function lastSequence(events: TrajectoryEvent[]): number {
  return events.reduce((highest, event) => Math.max(highest, event.sequence), 0);
}
