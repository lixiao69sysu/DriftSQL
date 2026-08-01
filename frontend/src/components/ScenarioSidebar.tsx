import { useMemo, useState } from "react";

import type { Scenario, Session } from "../types";
import { zhLabel } from "../locale";
import { formatDate } from "../utils";
import { Icon } from "./Icon";
import { StatusBadge } from "./StatusBadge";

interface Props {
  scenarios: Scenario[];
  sessions: Session[];
  selectedScenarioId: string | null;
  selectedSessionId: string | null;
  onScenario: (scenarioId: string) => void;
  onSession: (sessionId: string) => void;
}

export function ScenarioSidebar({
  scenarios,
  sessions,
  selectedScenarioId,
  selectedSessionId,
  onScenario,
  onSession,
}: Props) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"scenarios" | "runs">("scenarios");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return scenarios;
    return scenarios.filter((scenario) =>
      [scenario.question, scenario.db_id, scenario.drift_type, scenario.difficulty]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [query, scenarios]);

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Icon name="activity" /></div>
        <div><strong>DriftSQL</strong><span>智能体工作台</span></div>
      </div>
      <div className="side-tabs" role="tablist">
        <button className={tab === "scenarios" ? "active" : ""} onClick={() => setTab("scenarios")}>测试场景</button>
        <button className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}>最近运行</button>
      </div>
      {tab === "scenarios" ? (
        <>
          <label className="search-box">
            <Icon name="search" />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索 55 个场景" />
          </label>
          <div className="side-count">{filtered.length} 个已验证 Tune 任务</div>
          <nav className="scenario-list" aria-label="测试场景">
            {filtered.map((scenario) => (
              <button
                key={scenario.scenario_id}
                className={`scenario-item ${selectedScenarioId === scenario.scenario_id ? "selected" : ""}`}
                onClick={() => onScenario(scenario.scenario_id)}
              >
                <span className="scenario-top"><b>{scenario.db_id}</b><em>{zhLabel(scenario.difficulty, "未评级")}</em></span>
                <span className="scenario-question">{scenario.question}</span>
                <span className="scenario-meta"><i>{zhLabel(scenario.drift_type)}</i>{scenario.wildcard_profile && <i>{zhLabel(scenario.wildcard_profile)}</i>}</span>
              </button>
            ))}
          </nav>
        </>
      ) : (
        <nav className="scenario-list run-list" aria-label="最近会话">
          {sessions.length === 0 && <div className="empty-side">暂无运行记录</div>}
          {sessions.map((session) => (
            <button
              key={session.session_id}
              className={`scenario-item ${selectedSessionId === session.session_id ? "selected" : ""}`}
              onClick={() => onSession(session.session_id)}
            >
              <span className="scenario-top"><b>{session.db_id}</b><span>{formatDate(session.created_at)}</span></span>
              <span className="scenario-question">{session.question}</span>
              <StatusBadge status={session.status} success={session.success} />
            </button>
          ))}
        </nav>
      )}
      <div className="sidebar-foot"><Icon name="shield" /><span>只读隔离沙箱<br /><b>真实执行验证</b></span></div>
    </aside>
  );
}
