import { useMemo, useState } from "react";

import { compactHash, formatDate, formatDuration } from "../utils";
import { zhLabel } from "../locale";
import type { FailureList, OperationsSummary, WandbMetricSeries, WandbRunHistory, WandbRunList } from "../types";
import { Icon } from "./Icon";
import { StatusBadge } from "./StatusBadge";

interface Props {
  summary: OperationsSummary | null;
  failures: FailureList | null;
  wandb: WandbRunList | null;
  wandbHistory: WandbRunHistory | null;
  loading: boolean;
  onRefresh: () => void;
  onFailureFilter: (failureType: string) => void;
  onReplay: (sessionId: string) => void;
  onWandbRun: (runId: string) => void;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function MetricChart({ series }: { series: WandbMetricSeries }) {
  const width = 260;
  const height = 72;
  const values = series.points.map((point) => point.value);
  const steps = series.points.map((point) => point.step);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const points = series.points.map((point) => {
    const x = 5 + (point.step - minStep) / Math.max(1, maxStep - minStep) * (width - 10);
    const y = height - 5 - (point.value - minValue) / Math.max(1e-12, maxValue - minValue) * (height - 10);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const last = series.points.at(-1)?.value ?? 0;
  return (
    <div className="metric-chart">
      <div><span title={series.name}>{series.name}</span><b>{last.toPrecision(4)}</b></div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`${series.name} 训练曲线`}>
        <path d={`M 5 ${height - 5} H ${width - 5}`} />
        <polyline points={points} />
      </svg>
      <small>step {minStep} → {maxStep}</small>
    </div>
  );
}

export function OperationsDashboard({
  summary,
  failures,
  wandb,
  wandbHistory,
  loading,
  onRefresh,
  onFailureFilter,
  onReplay,
  onWandbRun,
}: Props) {
  const [failureType, setFailureType] = useState("");
  const maxDaily = Math.max(1, ...(summary?.daily_metrics.map((item) => item.sessions) ?? []));
  const totalTokens = (summary?.total_prompt_tokens ?? 0) + (summary?.total_response_tokens ?? 0);
  const kpis = summary ? [
    ["任务成功率", percent(summary.success_rate), `${summary.successful_sessions}/${summary.terminal_sessions} 个终态会话`, "check"],
    ["平均响应耗时", formatDuration(summary.average_latency_ms), `平均 ${summary.average_model_calls.toFixed(1)} 轮模型调用`, "clock"],
    ["工具失败率", percent(summary.tool_failure_rate), `平均 ${summary.average_tool_calls.toFixed(1)} 次工具调用`, "activity"],
    ["累计 Token", totalTokens.toLocaleString("zh-CN"), `${summary.total_response_tokens.toLocaleString("zh-CN")} 输出 Token`, "spark"],
  ] as const : [];
  const failureCounts = useMemo(
    () => Object.entries(failures?.counts ?? {}).sort((left, right) => right[1] - left[1]),
    [failures],
  );

  return (
    <div className="operations-workspace">
      <div className="operations-head">
        <div><span className="eyebrow">P4 · 运行可观测性</span><h2>Agent 运行监控</h2><p>指标来自持久化 Session 与真实工具事件，不包含训练集答案。</p></div>
        <button className="button button-secondary" disabled={loading} onClick={onRefresh}><Icon name="refresh" />{loading ? "刷新中" : "刷新指标"}</button>
      </div>

      {!summary ? <div className="panel ops-loading"><Icon name="activity" /><span>正在汇总运行指标…</span></div> : (
        <>
          <div className="ops-kpis">
            {kpis.map(([label, value, detail, icon]) => (
              <section className="panel ops-kpi" key={label}><div><Icon name={icon} /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small></section>
            ))}
          </div>

          <div className="ops-grid">
            <section className="panel ops-card drift-metrics">
              <div className="panel-title"><span><Icon name="layers" />漂移类型效果</span><em>{summary.terminal_sessions} 个终态会话</em></div>
              {summary.drift_metrics.length === 0 && <div className="ops-empty">运行一些测试场景后将在这里形成分层指标。</div>}
              {summary.drift_metrics.map((metric) => (
                <div className="drift-metric" key={metric.drift_type}>
                  <div><span>{zhLabel(metric.drift_type)}</span><small>{metric.successful}/{metric.sessions}</small></div>
                  <div className="ops-track"><i style={{ width: percent(metric.success_rate) }} /></div>
                  <b>{percent(metric.success_rate)}</b>
                </div>
              ))}
            </section>

            <section className="panel ops-card daily-card">
              <div className="panel-title"><span><Icon name="activity" />近 30 天运行趋势</span><em>{summary.total_sessions} 个会话</em></div>
              <div className="daily-chart">
                {summary.daily_metrics.length === 0 && <div className="ops-empty">暂无运行数据</div>}
                {summary.daily_metrics.map((metric) => (
                  <div className="daily-column" key={metric.day} title={`${metric.day}：${metric.sessions} 个会话`}>
                    <div><i className="daily-failed" style={{ height: `${metric.failed / maxDaily * 100}%` }} /><i className="daily-success" style={{ height: `${metric.successful / maxDaily * 100}%` }} /></div>
                    <span>{metric.day.slice(5)}</span>
                  </div>
                ))}
              </div>
              <div className="daily-legend"><span><i className="ok" />成功</span><span><i className="bad" />未成功</span></div>
            </section>
          </div>

          <section className="panel failure-explorer">
            <div className="panel-title failure-title">
              <span><Icon name="shield" />失败轨迹分析</span>
              <div className="failure-controls">
                <select value={failureType} onChange={(event) => { setFailureType(event.target.value); onFailureFilter(event.target.value); }}>
                  <option value="">全部失败类型</option>
                  <option value="task_failure">任务失败</option>
                  <option value="unsafe">不安全操作</option>
                  <option value="timed_out">执行超时</option>
                  <option value="budget_exhausted">预算耗尽</option>
                  <option value="cancelled">用户取消</option>
                  <option value="service_error">服务异常</option>
                </select>
                <em>{failures?.total ?? 0} 条</em>
              </div>
            </div>
            {failureCounts.length > 0 && <div className="failure-chips">{failureCounts.map(([name, count]) => <span key={name}>{zhLabel(name)} <b>{count}</b></span>)}</div>}
            <div className="failure-table-wrap">
              <table className="failure-table">
                <thead><tr><th>场景 / 数据库</th><th>失败分类</th><th>运行用量</th><th>时间</th><th>状态</th><th /></tr></thead>
                <tbody>
                  {(failures?.failures ?? []).map((failure) => (
                    <tr key={failure.session_id}>
                      <td><b>{failure.db_id}</b><span>{zhLabel(failure.drift_type)} · {compactHash(failure.scenario_id, 7)}</span></td>
                      <td><b>{zhLabel(failure.failure_type)}</b><span title={failure.error ?? failure.termination_reason ?? ""}>{failure.error ?? zhLabel(failure.termination_reason, "未通过任务验证")}</span></td>
                      <td><b>{failure.model_calls} 轮 / {failure.tool_calls} 工具</b><span>{failure.response_tokens} 输出 Token</span></td>
                      <td><b>{formatDuration(failure.elapsed_ms)}</b><span>{formatDate(failure.created_at)}</span></td>
                      <td><StatusBadge status={failure.status} /></td>
                      <td><button onClick={() => onReplay(failure.session_id)}>查看轨迹</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(failures?.failures.length ?? 0) === 0 && <div className="ops-empty">当前筛选条件下没有失败轨迹。</div>}
            </div>
          </section>

          <div className="ops-grid bottom-grid">
            <section className="panel ops-card deployment-card">
              <div className="panel-title"><span><Icon name="spark" />线上模型版本</span><em>{summary.deployments.length} 个部署</em></div>
              {summary.deployments.map((deployment) => (
                <div className="deployment-row" key={deployment.adapter_sha256}>
                  <div><b>{deployment.base_model.split("/").pop()}</b><span>{deployment.adapter.split("/").pop()}</span></div>
                  <code>{compactHash(deployment.adapter_sha256, 8)}</code>
                  <strong>{percent(deployment.success_rate)}</strong>
                  <small>{deployment.sessions} 个会话</small>
                </div>
              ))}
            </section>

            <section className="panel ops-card wandb-card">
              <div className="panel-title"><span><Icon name="activity" />W&B 训练实验</span><em className={wandb?.status === "ready" ? "online" : "offline"}>{wandb?.status === "ready" ? "已连接" : "未连接"}</em></div>
              {!wandb?.configured ? (
                <div className="wandb-setup"><b>尚未启用 W&B 读取</b><p>在服务端设置实体和项目后，可在此关联 reward、KL、loss 与线上 Adapter。</p><code>DRIFTSQL_SERVICE_WANDB_ENABLED=true<br />DRIFTSQL_SERVICE_WANDB_ENTITY=你的实体</code></div>
              ) : wandb.status === "error" ? (
                <div className="wandb-setup bad"><b>W&B 读取失败</b><p>{wandb.error}</p></div>
              ) : (
                <div className="wandb-runs">
                  {wandbHistory?.status === "ready" && wandbHistory.series.length > 0 && (
                    <div className="metric-chart-grid">{wandbHistory.series.slice(0, 4).map((series) => <MetricChart series={series} key={series.name} />)}</div>
                  )}
                  {wandbHistory?.status === "ready" && wandbHistory.series.length === 0 && <div className="ops-empty">该 Run 没有可用的 reward、KL、loss 或学习率历史。</div>}
                  {wandb.runs.slice(0, 6).map((run) => <button className={wandbHistory?.run_id === run.run_id ? "active" : ""} onClick={() => onWandbRun(run.run_id)} key={run.run_id}><span><b>{run.name}</b><small>{run.state} · {formatDate(run.created_at)}</small></span><em>{Object.keys(run.summary_metrics).length} 项指标</em></button>)}
                  {wandb.project_url && <a className="wandb-project" href={wandb.project_url} target="_blank" rel="noreferrer">打开 W&B 项目 →</a>}
                </div>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}
