import type { Scenario, Session } from "../types";
import { zhLabel } from "../locale";
import { compactHash, formatDuration } from "../utils";
import { CodeBlock } from "./CodeBlock";
import { Icon } from "./Icon";

function driftOperations(scenario: Scenario): Array<Record<string, unknown>> {
  const operations = scenario.schema_diff.operations;
  return Array.isArray(operations) ? operations as Array<Record<string, unknown>> : [];
}

export function ScenarioOverview({ scenario, session }: { scenario: Scenario; session: Session | null }) {
  const operations = driftOperations(scenario);
  return (
    <div className="overview-grid">
      <section className="panel request-panel">
        <div className="panel-title"><span><Icon name="terminal" />分析请求</span><em>{zhLabel(scenario.difficulty, "未评级")}</em></div>
        <p className="request-text">{scenario.question}</p>
        <div className="tag-row"><span>{scenario.db_id}</span><span>{zhLabel(scenario.drift_type)}</span>{scenario.wildcard_profile && <span>{zhLabel(scenario.wildcard_profile)}</span>}</div>
        <div className="subhead">历史有效 SQL</div>
        <CodeBlock code={scenario.stale_sql} compact />
      </section>
      <section className="panel drift-panel">
        <div className="panel-title"><span><Icon name="layers" />已审计的 Schema 漂移</span><em>{operations.length} 项变更</em></div>
        <div className="drift-flow">
          <div><small>缓存版本</small><b>{String(scenario.schema_diff.from_version ?? "v1")}</b></div>
          <Icon name="arrow" />
          <div className="active"><small>当前版本</small><b>{String(scenario.schema_diff.to_version ?? session?.db_version ?? "v2")}</b></div>
        </div>
        <div className="operation-list">
          {operations.map((operation, index) => (
            <div className="operation" key={index}>
              <span>{zhLabel(String(operation.type ?? "change"), "变更")}</span>
              <b>{String(operation.table ?? "未知表")}</b>
              <small>{operation.old_name ? `${String(operation.old_name)} → ` : ""}{String(operation.new_name ?? "")}</small>
            </div>
          ))}
        </div>
      </section>
      {session && (
        <section className="panel metric-strip">
          <div><span><Icon name="database" />数据库</span><b>{session.db_id} · {session.db_version}</b><small>{session.sandbox_isolated ? "独立 SQLite 副本" : "共享数据库"}</small></div>
          <div><span><Icon name="activity" />用量</span><b>{session.usage.model_calls} 次模型 · {session.usage.tool_calls} 次工具</b><small>共 {session.usage.prompt_tokens + session.usage.response_tokens} Token</small></div>
          <div><span><Icon name="clock" />运行时间</span><b>{formatDuration(session.usage.elapsed_ms)}</b><small>{zhLabel(session.termination_reason, "运行中")}</small></div>
          <div><span><Icon name="shield" />模型产物</span><b>{compactHash(session.model.adapter_sha256, 7)}</b><small>冻结 Adapter 哈希</small></div>
        </section>
      )}
    </div>
  );
}
