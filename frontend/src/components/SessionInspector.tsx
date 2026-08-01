import type { Experiment, Health, Session } from "../types";
import { compactHash, formatDate, formatDuration } from "../utils";
import { ExperimentComparison } from "./ExperimentComparison";
import { Icon } from "./Icon";

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="inspect-row"><span>{label}</span><b className={mono ? "mono" : ""} title={value}>{value}</b></div>;
}

export function SessionInspector({
  session,
  health,
  experiments,
}: {
  session: Session | null;
  health: Health | null;
  experiments: Experiment[];
}) {
  const model = session?.model ?? health?.model;
  return (
    <div className="inspector-stack">
      <section className="panel inspector">
        <div className="panel-title"><span><Icon name="spark" />模型运行时</span><em className={model?.loaded ? "online" : "offline"}>{model?.loaded ? "已加载" : "离线"}</em></div>
        <Row label="推理后端" value={model?.backend ?? "—"} />
        <Row label="基础模型" value={(model?.base_model ?? "—").split("/").pop() ?? "—"} />
        <Row label="Adapter" value={(model?.adapter ?? "—").split("/").pop() ?? "—"} />
        <Row label="Adapter SHA" value={compactHash(model?.adapter_sha256 ?? "", 8)} mono />
        <Row label="持久化加载" value={model?.persistent ? "是——进程生命周期" : "否"} />
      </section>
      {session && (
        <section className="panel inspector">
          <div className="panel-title"><span><Icon name="shield" />会话控制</span><em>只读</em></div>
          <Row label="会话 ID" value={compactHash(session.session_id, 7)} mono />
          <Row label="创建时间" value={formatDate(session.created_at)} />
          <Row label="沙箱" value={session.sandbox_isolated ? "独立副本" : "未隔离"} />
          <Row label="源库 SHA" value={compactHash(session.source_db_sha256, 8)} mono />
          <Row label="已用时间" value={formatDuration(session.usage.elapsed_ms)} />
        </section>
      )}
      {session?.budget && (
        <section className="panel inspector budget-card">
          <div className="panel-title"><span><Icon name="clock" />推理预算</span><em>强制执行</em></div>
          <div className="budget-grid">
            <div><b>{session.usage.model_calls}/{session.budget.max_turns}</b><span>轮次</span></div>
            <div><b>{session.usage.tool_calls}/{session.budget.max_tool_calls}</b><span>工具</span></div>
            <div><b>{session.usage.response_tokens}/{session.budget.max_total_tokens}</b><span>Token</span></div>
            <div><b>{session.budget.timeout_seconds}s</b><span>超时</span></div>
          </div>
        </section>
      )}
      <ExperimentComparison experiments={experiments} />
    </div>
  );
}
