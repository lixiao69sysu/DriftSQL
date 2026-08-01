import { describe, expect, it } from "vitest";

import type { RewardResult, TrajectoryEvent } from "./types";
import {
  compactHash,
  formatDuration,
  lastSequence,
  rewardComponents,
  toolSteps,
} from "./utils";

function event(sequence: number, payload: Record<string, unknown>): TrajectoryEvent {
  return {
    session_id: "session-1",
    sequence,
    event_type: "tool",
    created_at: "2026-08-01T00:00:00Z",
    payload,
  };
}

describe("trajectory presentation utilities", () => {
  it("normalizes persisted SQL tool events and observations", () => {
    const steps = toolSteps([
      event(2, {
        turn: 1,
        tool: "execute_sql",
        arguments: { sql: "SELECT 1" },
        observation: '{"success":true,"columns":["1"],"rows":[[1]]}',
        metrics: { execution_success: true },
        success: true,
        elapsed_ms: 4.2,
      }),
    ]);
    expect(steps).toHaveLength(1);
    expect(steps[0].tool).toBe("execute_sql");
    expect(steps[0].observation).toMatchObject({ success: true, rows: [[1]] });
    expect(steps[0].elapsedMs).toBe(4.2);
  });

  it("separates positive and penalty reward components", () => {
    const reward: RewardResult = {
      score: 1.1,
      reward_success: 1,
      reward_valid: 0.1,
      penalty_tool_cost: 0.04,
      task_success: true,
    };
    expect(rewardComponents(reward)).toEqual([
      { name: "success", value: 1, kind: "reward" },
      { name: "valid", value: 0.1, kind: "reward" },
      { name: "tool cost", value: 0.04, kind: "penalty" },
    ]);
  });

  it("formats hashes, durations and event cursors", () => {
    expect(compactHash("abcdefghijklmnopqrstuvwxyz", 4)).toBe("abcd…wxyz");
    expect(formatDuration(65_400)).toBe("1m 5s");
    expect(lastSequence([event(2, {}), event(8, {}), event(4, {})])).toBe(8);
  });
});
