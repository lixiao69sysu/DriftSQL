import type { Experiment } from "../types";
import { experimentLabel } from "../locale";
import { Icon } from "./Icon";

export function ExperimentComparison({ experiments }: { experiments: Experiment[] }) {
  const visible = experiments
    .filter((experiment) =>
      experiment.selected
      || experiment.experiment_id === "stage7_frozen_tune55"
      || experiment.experiment_id.includes("step10")
      || experiment.experiment_id.includes("conservative_step6"),
    )
    .slice(0, 4);
  return (
    <section className="panel inspector experiment-panel">
      <div className="panel-title"><span><Icon name="layers" />Tune 实验对比</span><em>{experiments.length} 组实验</em></div>
      <div className="experiment-legend"><span><i className="success" />任务成功率</span><span><i className="executable" />SQL 可执行率</span></div>
      <div className="experiment-list">
        {visible.map((experiment) => (
          <div className={`experiment-row ${experiment.selected ? "selected" : ""}`} key={experiment.experiment_id}>
            <div className="experiment-name"><span>{experimentLabel(experiment.display_name)}</span>{experiment.selected && <b>当前方案</b>}<small>{experiment.category} · 样本数={experiment.tasks}</small></div>
            <div className="experiment-bars">
              <div><i className="success" style={{ width: `${experiment.task_success_rate * 100}%` }} /></div>
              <div><i className="executable" style={{ width: `${experiment.executable_rate * 100}%` }} /></div>
            </div>
            <strong>{Math.round(experiment.task_success_rate * 100)}%</strong>
          </div>
        ))}
      </div>
      <p className="experiment-note">各实验的 Tune 样本量不同，结果适合观察趋势，不应视作同一排行榜。</p>
    </section>
  );
}
