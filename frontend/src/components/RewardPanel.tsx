import type { RewardResult } from "../types";
import { rewardLabel } from "../locale";
import { rewardComponents } from "../utils";
import { Icon } from "./Icon";

export function RewardPanel({ reward }: { reward: RewardResult }) {
  const components = rewardComponents(reward);
  const score = Number(reward.score ?? 0);
  const max = Math.max(1, ...components.map((item) => Math.abs(item.value)));
  if (!Object.keys(reward).length) {
    return <div className="reward-empty"><Icon name="activity" /><span>运行结束后计算奖励</span></div>;
  }
  return (
    <div className="reward-panel">
      <div className="reward-score">
        <span>Agentic RL 奖励</span>
        <strong className={score >= 0 ? "positive" : "negative"}>{score.toFixed(4)}</strong>
        <small>{reward.task_success ? "已通过真实执行验证" : String(reward.error || "未通过验证")}</small>
      </div>
      <div className="reward-bars">
        {components.map((component) => (
          <div className="reward-row" key={`${component.kind}-${component.name}`}>
            <span>{rewardLabel(component.name)}</span>
            <div className="bar-track"><i className={component.kind} style={{ width: `${Math.max(2, Math.abs(component.value) / max * 100)}%` }} /></div>
            <b className={component.kind}>{component.kind === "penalty" ? "−" : "+"}{component.value.toFixed(3)}</b>
          </div>
        ))}
      </div>
      <div className="reward-flags">
        <span className={reward.execution_success ? "on" : ""}>SQL 可执行</span>
        <span className={reward.format_valid ? "on" : ""}>提交格式有效</span>
        <span className={!reward.unsafe ? "on" : "bad"}>操作安全</span>
        <span className={!reward.timed_out ? "on" : "bad"}>未超时</span>
      </div>
    </div>
  );
}
