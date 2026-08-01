import type { Session, TrajectoryEvent } from "./types";

export interface LiveTrajectoryState {
  session: Session | null;
  events: TrajectoryEvent[];
  streamState: "idle" | "connecting" | "live" | "closed" | "error";
  error: string | null;
}

export type LiveTrajectoryAction =
  | { type: "clear" }
  | { type: "reset"; session: Session; events?: TrajectoryEvent[] }
  | { type: "event"; event: TrajectoryEvent }
  | { type: "session"; session: Session }
  | { type: "stream"; state: LiveTrajectoryState["streamState"] }
  | { type: "error"; message: string };

export const initialLiveState: LiveTrajectoryState = {
  session: null,
  events: [],
  streamState: "idle",
  error: null,
};

function mergeEvent(events: TrajectoryEvent[], event: TrajectoryEvent): TrajectoryEvent[] {
  if (events.some((candidate) => candidate.sequence === event.sequence)) return events;
  return [...events, event].sort((left, right) => left.sequence - right.sequence);
}

export function liveTrajectoryReducer(
  state: LiveTrajectoryState,
  action: LiveTrajectoryAction,
): LiveTrajectoryState {
  switch (action.type) {
    case "clear":
      return initialLiveState;
    case "reset":
      return {
        session: action.session,
        events: [...(action.events ?? [])].sort((a, b) => a.sequence - b.sequence),
        streamState: "idle",
        error: null,
      };
    case "event":
      return {
        ...state,
        events: mergeEvent(state.events, action.event),
        streamState: "live",
        error: null,
      };
    case "session":
      return { ...state, session: action.session };
    case "stream":
      return { ...state, streamState: action.state };
    case "error":
      return { ...state, streamState: "error", error: action.message };
  }
}
