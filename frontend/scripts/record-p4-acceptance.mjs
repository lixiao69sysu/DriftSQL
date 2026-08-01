import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";


const baseUrl = process.env.DRIFTSQL_ACCEPT_BASE_URL ?? "http://127.0.0.1:8001";
const outputDir = path.resolve(
  process.env.DRIFTSQL_ACCEPT_OUTPUT_DIR ?? "../artifacts/p4_real_acceptance",
);
const videoPath = path.join(outputDir, "driftsql-p4-real-acceptance.webm");
const targetRun = "stage8-fresh-db-failure-balanced-grpo-7b";

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  locale: "zh-CN",
  viewport: { width: 1440, height: 900 },
  recordVideo: {
    dir: path.join(outputDir, "raw-video"),
    size: { width: 1440, height: 900 },
  },
});
const page = await context.newPage();
const video = page.video();
const checks = {};
let actionError = null;

try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByText("服务就绪", { exact: true }).waitFor();
  await page.getByText("智能体工作台", { exact: true }).waitFor();
  checks.chinese_studio_loaded = true;
  await page.screenshot({ path: path.join(outputDir, "01-chinese-studio.png"), fullPage: true });
  await page.waitForTimeout(1200);

  await page.getByRole("button", { name: "最近运行" }).click();
  const newestRun = page.locator(".run-list .scenario-item").first();
  await newestRun.waitFor();
  await newestRun.click();
  await page.getByText("提交答案", { exact: true }).waitFor();
  await page.getByText("第 1 轮", { exact: true }).waitFor();
  checks.real_trajectory_replayed = true;
  await page.locator(".trajectory-panel").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(outputDir, "02-trajectory-replay.png"), fullPage: true });
  await page.waitForTimeout(1800);

  await page.getByRole("button", { name: "奖励明细" }).click();
  await page.waitForTimeout(900);
  await page.getByRole("button", { name: "最终 SQL" }).click();
  await page.getByText("已通过执行验证", { exact: true }).waitFor();
  checks.real_sft20_result_visible = true;
  await page.waitForTimeout(1400);

  await page.getByRole("button", { name: "运行监控" }).click();
  await page.getByText("W&B 训练实验", { exact: true }).waitFor();
  await page.getByText("已连接", { exact: true }).waitFor();
  const wandbCard = page.locator(".wandb-card");
  await wandbCard.scrollIntoViewIfNeeded();
  const targetRunButton = page.getByRole("button", { name: new RegExp(targetRun) });
  await targetRunButton.click();
  await page.locator(".wandb-runs button.active").filter({ hasText: targetRun }).waitFor();
  const ppoKlChart = page.locator(".metric-chart").filter({ hasText: "actor/ppo_kl" });
  const rewardChart = page.locator(".metric-chart").filter({ hasText: "critic/rewards/mean" });
  await ppoKlChart.getByText("step 1 → 10", { exact: true }).waitFor();
  await rewardChart.getByText("step 1 → 10", { exact: true }).waitFor();
  checks.wandb_reward_kl_curves_visible = true;
  await page.screenshot({ path: path.join(outputDir, "03-wandb-curves.png"), fullPage: true });
  await page.waitForTimeout(2200);
} catch (error) {
  actionError = error;
} finally {
  await page.close();
}

if (!video) {
  await context.close();
  await browser.close();
  throw new Error("Playwright did not create a video handle");
}
await video.saveAs(videoPath);
await context.close();
await browser.close();
if (actionError) throw actionError;

const result = {
  protocol: "driftsql_p4_browser_recording_v1",
  base_url: baseUrl,
  passed: Object.values(checks).every(Boolean),
  checks,
  video: videoPath,
  screenshots: [
    path.join(outputDir, "01-chinese-studio.png"),
    path.join(outputDir, "02-trajectory-replay.png"),
    path.join(outputDir, "03-wandb-curves.png"),
  ],
};
await writeFile(
  path.join(outputDir, "recording.json"),
  `${JSON.stringify(result, null, 2)}\n`,
  "utf8",
);
console.log(JSON.stringify(result, null, 2));
if (!result.passed) process.exitCode = 1;
