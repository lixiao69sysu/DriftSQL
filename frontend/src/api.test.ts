import { afterEach, describe, expect, it, vi } from "vitest";

import { subscribeToEvents } from "./api";


class FakeEventSource {
  static latest: FakeEventSource;
  listeners = new Map<string, Array<(event: Event) => void>>();
  onerror: ((event: Event) => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;

  constructor(public readonly url: string) {
    FakeEventSource.latest = this;
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = typeof listener === "function" ? listener : (event: Event) => listener.handleEvent(event);
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  emit(type: string, event: Event) {
    for (const listener of this.listeners.get(type) ?? []) listener(event);
    if (type === "error") this.onerror?.(event);
  }

  close() {
    this.closed = true;
  }
}


describe("subscribeToEvents", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps application error events in the trajectory without reporting a transport failure", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn();
    const onError = vi.fn();
    const subscription = subscribeToEvents("session-1", 0, onEvent, onError);
    const payload = {
      session_id: "session-1",
      sequence: 2,
      event_type: "error",
      created_at: "2026-08-01T00:00:00Z",
      payload: { error: "missing_tool_call" },
    };

    FakeEventSource.latest.emit(
      "agent_error",
      new MessageEvent("error", { data: JSON.stringify(payload) }),
    );

    expect(onEvent).toHaveBeenCalledWith(payload);
    expect(onError).not.toHaveBeenCalled();
    FakeEventSource.latest.onerror?.(new Event("error"));
    expect(onError).toHaveBeenCalledOnce();
    subscription.close();
    expect(FakeEventSource.latest.closed).toBe(true);
  });
});
