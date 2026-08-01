import type { TrajectoryEvent } from "../types";
import { toolLabel } from "../locale";
import { formatDuration, parseJson, toolSteps } from "../utils";
import { CodeBlock } from "./CodeBlock";
import { Icon } from "./Icon";
import { ResultTable } from "./ResultTable";

function PrettyValue({ value }: { value: unknown }) {
  if (typeof value === "string") return <pre className="pretty-text">{value}</pre>;
  return <pre className="pretty-json">{JSON.stringify(value, null, 2)}</pre>;
}

export function TrajectoryTimeline({ events }: { events: TrajectoryEvent[] }) {
  const steps = toolSteps(events);
  const models = new Map(
    events
      .filter((event) => event.event_type === "model")
      .map((event) => [Number(event.payload.turn), event]),
  );
  if (!steps.length) {
    return (
      <div className="empty-panel">
        <div className="empty-orbit"><Icon name="activity" /></div>
        <h3>Agent 轨迹将在这里显示</h3>
        <p>运行所选场景后，将实时展示模型决策、工具调用、执行观察和奖励。</p>
      </div>
    );
  }
  return (
    <div className="timeline">
      {steps.map((step, index) => {
        const model = models.get(step.turn);
        const content = String(model?.payload.content ?? "");
        const thought = content.replace(/\{[\s\S]*$/, "").replace(/<\/?think>/g, "").trim();
        const sql = typeof step.arguments.sql === "string" ? step.arguments.sql : null;
        const observation = parseJson(step.observation);
        const objectObservation = observation && typeof observation === "object" && !Array.isArray(observation)
          ? observation as Record<string, unknown>
          : null;
        return (
          <article className="timeline-step" key={step.event.sequence}>
            <div className={`step-node ${step.success ? "ok" : "bad"}`}>{step.success ? <Icon name="check" /> : <Icon name="x" />}</div>
            <div className="step-card">
              <header>
                <div><span className="turn-label">第 {step.turn} 轮</span><h3 title={step.tool}>{toolLabel(step.tool)}</h3></div>
                <div className="step-facts"><span>{formatDuration(step.elapsedMs)}</span><span className={step.success ? "fact-ok" : "fact-bad"}>{step.success ? "调用成功" : "调用失败"}</span></div>
              </header>
              {thought && <div className="thought"><Icon name="spark" /><span>{thought}</span></div>}
              {sql ? <CodeBlock code={sql} compact /> : Object.keys(step.arguments).length > 0 && <PrettyValue value={step.arguments} />}
              {step.tool === "execute_sql" && objectObservation && <ResultTable observation={objectObservation} />}
              <details className="observation" open={!step.success || step.tool === "inspect_schema_diff"}>
                <summary>工具返回结果 <Icon name="chevron" /></summary>
                <PrettyValue value={observation} />
              </details>
            </div>
            {index < steps.length - 1 && <div className="timeline-line" />}
          </article>
        );
      })}
    </div>
  );
}
