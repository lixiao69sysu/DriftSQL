import { useCallback, useEffect, useMemo, useReducer, useState } from "react";

import { api, subscribeToEvents } from "./api";
import { CodeBlock } from "./components/CodeBlock";
import { Icon } from "./components/Icon";
import { OperationsDashboard } from "./components/OperationsDashboard";
import { LoginScreen } from "./components/LoginScreen";
import { RewardPanel } from "./components/RewardPanel";
import { RunConfiguration } from "./components/RunConfiguration";
import { ScenarioOverview } from "./components/ScenarioOverview";
import { ScenarioSidebar } from "./components/ScenarioSidebar";
import { SessionInspector } from "./components/SessionInspector";
import { StatusBadge } from "./components/StatusBadge";
import { TrajectoryTimeline } from "./components/TrajectoryTimeline";
import { initialLiveState, liveTrajectoryReducer } from "./state";
import { streamLabel, zhLabel } from "./locale";
import type {
  Experiment,
  AuthStatus,
  FailureList,
  Health,
  OperationsSummary,
  RewardResult,
  ReplayCandidateList,
  RunOptions,
  Scenario,
  Session,
  WandbRunList,
  WandbRunHistory,
} from "./types";
import { terminalStatuses } from "./types";
import { compactHash, lastSequence } from "./utils";

const defaultOptions: RunOptions = {
  max_turns: 7,
  timeout_seconds: 120,
  max_tool_calls: 7,
  max_new_tokens: 512,
  max_total_tokens: 32768,
};
const preferredDemoScenario = "drift_coladd_336a8e6d4010d75e";

export default function App() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);
  const [live, dispatch] = useReducer(liveTrajectoryReducer, initialLiveState);
  const [reward, setReward] = useState<RewardResult>({});
  const [options, setOptions] = useState(defaultOptions);
  const [tab, setTab] = useState<"trajectory" | "reward" | "final">("trajectory");
  const [busy, setBusy] = useState(false);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [view, setView] = useState<"agent" | "operations">("agent");
  const [operations, setOperations] = useState<OperationsSummary | null>(null);
  const [failures, setFailures] = useState<FailureList | null>(null);
  const [wandb, setWandb] = useState<WandbRunList | null>(null);
  const [wandbHistory, setWandbHistory] = useState<WandbRunHistory | null>(null);
  const [replay, setReplay] = useState<ReplayCandidateList | null>(null);
  const [operationsLoading, setOperationsLoading] = useState(false);

  const scenario = useMemo(
    () => scenarios.find((item) => item.scenario_id === selectedScenarioId) ?? null,
    [scenarios, selectedScenarioId],
  );
  const running = live.session ? !terminalStatuses.has(live.session.status) && live.session.status !== "created" : false;

  const bootstrap = useCallback(async () => {
    const [nextHealth, nextScenarios, nextSessions, nextExperiments] = await Promise.all([
      api.health(),
      api.scenarios(),
      api.sessions(),
      api.experiments(),
    ]);
    setHealth(nextHealth);
    setScenarios(nextScenarios);
    setSessions(nextSessions.sessions);
    setExperiments(nextExperiments.experiments);
    setSelectedScenarioId(
      nextScenarios.find((item) => item.scenario_id === preferredDemoScenario)?.scenario_id
        ?? nextScenarios[0]?.scenario_id
        ?? null,
    );
  }, []);

  const refreshSessions = useCallback(async () => {
    const response = await api.sessions();
    setSessions(response.sessions);
  }, []);

  const loadTrajectory = useCallback(async (sessionId: string) => {
    const trajectory = await api.trajectory(sessionId);
    dispatch({ type: "reset", session: trajectory.session, events: trajectory.events });
    setReward(trajectory.reward);
    setSelectedScenarioId(trajectory.session.scenario_id);
    await refreshSessions();
  }, [refreshSessions]);

  const loadOperations = useCallback(async () => {
    setOperationsLoading(true);
    try {
      const [nextOperations, nextFailures, nextWandb, nextReplay] = await Promise.all([
        api.operations(),
        api.failures(),
        api.wandbRuns(),
        api.replayCandidates(),
      ]);
      setOperations(nextOperations);
      setFailures(nextFailures);
      setWandb(nextWandb);
      setReplay(nextReplay);
      if (nextWandb.status === "ready" && nextWandb.runs.length > 0) {
        setWandbHistory(await api.wandbHistory(nextWandb.runs[0].run_id));
      } else {
        setWandbHistory(null);
      }
    } catch (error) {
      setFatalError(error instanceof Error ? error.message : String(error));
    } finally {
      setOperationsLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.authStatus()
      .then(async (nextAuth) => {
        if (cancelled) return;
        setAuth(nextAuth);
        if (nextAuth.authenticated) await bootstrap();
      })
      .catch((error: unknown) => setFatalError(error instanceof Error ? error.message : String(error)));
    return () => { cancelled = true; };
  }, [bootstrap]);

  useEffect(() => {
    if (view === "operations" && operations === null) {
      void loadOperations();
    }
  }, [loadOperations, view]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void api.health().then(setHealth).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const session = live.session;
    if (!session || terminalStatuses.has(session.status) || session.status === "created") return;
    dispatch({ type: "stream", state: "connecting" });
    const subscription = subscribeToEvents(
      session.session_id,
      lastSequence(live.events),
      (event) => {
        dispatch({ type: "event", event });
        if (event.event_type === "status") {
          const status = String(event.payload.status);
          if (terminalStatuses.has(status as Session["status"])) {
            void loadTrajectory(session.session_id);
          } else {
            void api.session(session.session_id).then((current) => dispatch({ type: "session", session: current }));
          }
        } else if (event.event_type === "tool") {
          void api.session(session.session_id).then((current) => dispatch({ type: "session", session: current }));
        }
      },
      () => {
        if (!terminalStatuses.has(live.session?.status ?? "created")) {
          dispatch({ type: "error", message: "实时事件流已断开，完整轨迹仍可重新加载。" });
        }
      },
      () => dispatch({ type: "stream", state: "live" }),
    );
    return subscription.close;
  }, [live.session?.session_id, live.session?.status, loadTrajectory]);

  async function startRun() {
    if (!scenario) return;
    setBusy(true);
    setFatalError(null);
    setReward({});
    setTab("trajectory");
    try {
      const created = await api.createSession(scenario.scenario_id);
      dispatch({ type: "reset", session: created });
      const queued = await api.runSession(created.session_id, options);
      dispatch({ type: "session", session: queued });
      setSessions((current) => [queued, ...current.filter((item) => item.session_id !== queued.session_id)]);
    } catch (error) {
      setFatalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function cancelRun() {
    if (!live.session) return;
    try {
      const session = await api.cancelSession(live.session.session_id);
      dispatch({ type: "session", session });
    } catch (error) {
      setFatalError(error instanceof Error ? error.message : String(error));
    }
  }

  async function selectSession(sessionId: string) {
    if (running && sessionId !== live.session?.session_id) {
      setFatalError("请先取消当前运行，再打开其他会话。");
      return;
    }
    setFatalError(null);
    try {
      await loadTrajectory(sessionId);
      setView("agent");
    } catch (error) {
      setFatalError(error instanceof Error ? error.message : String(error));
    }
  }

  if (!auth && !fatalError) {
    return <div className="boot-screen"><div className="boot-mark"><Icon name="activity" /></div><span>正在启动 DriftSQL 智能体工作台</span><i /></div>;
  }

  if (auth?.enabled && !auth.authenticated) {
    return <LoginScreen onLogin={async (apiKey) => {
      const nextAuth = await api.login(apiKey);
      setAuth(nextAuth);
      await bootstrap();
    }} />;
  }

  if (!health && !fatalError) {
    return <div className="boot-screen"><div className="boot-mark"><Icon name="activity" /></div><span>正在加载模型与运行目录</span><i /></div>;
  }

  return (
    <div className="app-shell">
      <ScenarioSidebar
        scenarios={scenarios}
        sessions={sessions}
        selectedScenarioId={selectedScenarioId}
        selectedSessionId={live.session?.session_id ?? null}
        onScenario={(scenarioId) => {
          if (running) {
            setFatalError("请先取消当前运行，再切换测试场景。");
            return;
          }
          setSelectedScenarioId(scenarioId);
          setView("agent");
          dispatch({ type: "clear" });
          setReward({});
          setTab("trajectory");
        }}
        onSession={(sessionId) => void selectSession(sessionId)}
      />
      <main className="main-area">
        <header className="topbar">
          <div><span className="eyebrow">Agent 运行控制台</span><h1>Schema 漂移恢复</h1></div>
          <nav className="workspace-nav" aria-label="工作区">
            <button className={view === "agent" ? "active" : ""} onClick={() => setView("agent")}><Icon name="terminal" />Agent 调试</button>
            <button className={view === "operations" ? "active" : ""} onClick={() => setView("operations")}><Icon name="activity" />运行监控</button>
          </nav>
          <div className="top-status">
            <span className={`connection ${health?.status === "ready" ? "ready" : ""}`}><i />{health?.status === "ready" ? "服务就绪" : "服务离线"}</span>
            <span className="model-chip"><Icon name="spark" />{health?.model.base_model.split("/").pop()}</span>
            <span className="queue-chip">{health?.active_sessions ?? 0}/{health?.max_concurrent_sessions ?? 2} 个活跃会话</span>
            {auth?.enabled && <button className="logout-button" onClick={() => void api.logout().then((nextAuth) => { setAuth(nextAuth); setHealth(null); })}>退出</button>}
          </div>
        </header>

        {fatalError && <div className="alert"><Icon name="x" /><span><b>请求失败</b>{fatalError}</span><button onClick={() => setFatalError(null)}>关闭</button></div>}

        {view === "operations" ? (
          <OperationsDashboard
            summary={operations}
            failures={failures}
            wandb={wandb}
            wandbHistory={wandbHistory}
            replay={replay}
            loading={operationsLoading}
            onRefresh={() => void loadOperations()}
            onFailureFilter={(failureType) => {
              setOperationsLoading(true);
              void api.failures(failureType)
                .then(setFailures)
                .catch((error: unknown) => setFatalError(error instanceof Error ? error.message : String(error)))
                .finally(() => setOperationsLoading(false));
            }}
            onReplay={(sessionId) => void selectSession(sessionId)}
            onWandbRun={(runId) => {
              setOperationsLoading(true);
              void api.wandbHistory(runId)
                .then(setWandbHistory)
                .catch((error: unknown) => setFatalError(error instanceof Error ? error.message : String(error)))
                .finally(() => setOperationsLoading(false));
            }}
            onReview={async (candidateId, decision, reviewer, reason) => {
              await api.reviewReplayCandidate(candidateId, decision, reviewer, reason);
              setReplay(await api.replayCandidates());
            }}
          />
        ) : scenario ? (
          <div className="workspace">
            <div className="workspace-head">
              <div>
                <span className="scenario-id">{compactHash(scenario.scenario_id, 12)}</span>
                <h2>{scenario.db_id}<span>/</span>{zhLabel(scenario.drift_type)}</h2>
              </div>
              {live.session && <StatusBadge status={live.session.status} success={live.session.success} />}
            </div>
            <RunConfiguration
              options={options}
              onChange={setOptions}
              disabled={busy || !health?.model.loaded}
              running={running}
              busy={busy}
              onRun={() => void startRun()}
              onCancel={() => void cancelRun()}
            />
            <ScenarioOverview scenario={scenario} session={live.session} />

            <div className="dashboard-grid">
              <section className="panel trajectory-panel">
                <div className="panel-tabs">
                  <button className={tab === "trajectory" ? "active" : ""} onClick={() => setTab("trajectory")}>运行轨迹 <span>{live.events.filter((event) => event.event_type === "tool").length}</span></button>
                  <button className={tab === "reward" ? "active" : ""} onClick={() => setTab("reward")}>奖励明细</button>
                  <button className={tab === "final" ? "active" : ""} onClick={() => setTab("final")}>最终 SQL</button>
                  <div className={`stream-state stream-${live.streamState}`}><i />{streamLabel(live.streamState)}</div>
                </div>
                <div className="panel-body">
                  {tab === "trajectory" && <TrajectoryTimeline events={live.events} />}
                  {tab === "reward" && <RewardPanel reward={reward} />}
                  {tab === "final" && (
                    live.session?.final_sql
                      ? <div className="final-sql"><div className={live.session.success ? "final-verdict ok" : "final-verdict bad"}><Icon name={live.session.success ? "check" : "x"} /><div><b>{live.session.success ? "已通过执行验证" : "执行验证失败"}</b><span>{String(live.session.result.reward ?? "暂无奖励")}</span></div></div><CodeBlock code={live.session.final_sql} /></div>
                      : <div className="empty-panel"><Icon name="code" /><h3>尚未提交 SQL</h3><p>Agent 调用 submit_solution 后将在此显示最终查询。</p></div>
                  )}
                </div>
              </section>
              <SessionInspector session={live.session} health={health} experiments={experiments} />
            </div>
          </div>
        ) : (
          <div className="empty-main"><Icon name="database" /><h2>暂无可用场景</h2><p>请检查服务端场景目录并刷新页面。</p></div>
        )}
      </main>
    </div>
  );
}
