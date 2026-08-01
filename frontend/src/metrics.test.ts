import { describe, expect, it } from "vitest";

import { selectMetricSeries } from "./metrics";
import type { WandbMetricSeries } from "./types";


function series(name: string): WandbMetricSeries {
  return { name, points: [{ step: 1, value: 0.1 }] };
}

describe("selectMetricSeries", () => {
  it("surfaces reward, PPO KL, loss and learning rate before auxiliary curves", () => {
    const selected = selectMetricSeries([
      series("actor/kl_coef"),
      series("actor/kl_loss"),
      series("actor/loss"),
      series("actor/lr"),
      series("actor/ppo_kl"),
      series("critic/rewards/mean"),
      series("perf/throughput"),
    ]);

    expect(selected.map((item) => item.name)).toEqual([
      "critic/rewards/mean",
      "actor/ppo_kl",
      "actor/loss",
      "actor/lr",
    ]);
  });

  it("falls back to SFT validation and training curves", () => {
    const selected = selectMetricSeries([
      series("train/lr"),
      series("val/loss"),
      series("train/loss"),
    ]);

    expect(selected.map((item) => item.name)).toEqual([
      "train/loss",
      "val/loss",
      "train/lr",
    ]);
  });
});
