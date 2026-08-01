import { describe, expect, it } from "vitest";

import { initialLiveState, liveTrajectoryReducer } from "./state";
import type { Session, TrajectoryEvent } from "./types";

const session = {
  session_id: "session-1",
  status: "queued",
} as Session;

function event(sequence: number): TrajectoryEvent {
  return {
    session_id: "session-1",
    sequence,
    event_type: "status",
    created_at: "2026-08-01T00:00:00Z",
    payload: { status: "running" },
  };
}

describe("live trajectory reducer", () => {
  it("orders and de-duplicates replayed SSE events", () => {
    let state = liveTrajectoryReducer(initialLiveState, {
      type: "reset",
      session,
      events: [event(3)],
    });
    state = liveTrajectoryReducer(state, { type: "event", event: event(1) });
    state = liveTrajectoryReducer(state, { type: "event", event: event(3) });
    expect(state.events.map((item) => item.sequence)).toEqual([1, 3]);
    expect(state.streamState).toBe("live");
  });

  it("clears a completed run before selecting another scenario", () => {
    const populated = liveTrajectoryReducer(initialLiveState, {
      type: "reset",
      session,
      events: [event(1)],
    });
    expect(liveTrajectoryReducer(populated, { type: "clear" })).toEqual(initialLiveState);
  });
});
