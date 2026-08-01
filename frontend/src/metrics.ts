import type { WandbMetricSeries } from "./types";


const preferredMetricOrder = [
  "critic/rewards/mean",
  "actor/ppo_kl",
  "actor/loss",
  "train/loss",
  "val/loss",
  "actor/lr",
  "train/lr",
  "learning_rate",
  "perf/throughput",
];

/** Select a compact but useful reward/KL/loss/LR dashboard. */
export function selectMetricSeries(
  series: WandbMetricSeries[],
  limit = 4,
): WandbMetricSeries[] {
  const byName = new Map(series.map((item) => [item.name.toLowerCase(), item]));
  const selected: WandbMetricSeries[] = [];
  for (const name of preferredMetricOrder) {
    const match = byName.get(name);
    if (match && !selected.includes(match)) selected.push(match);
    if (selected.length === limit) return selected;
  }
  for (const item of series) {
    if (!selected.includes(item)) selected.push(item);
    if (selected.length === limit) break;
  }
  return selected;
}
