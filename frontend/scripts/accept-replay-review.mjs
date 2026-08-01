import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";


const baseUrl = process.env.DRIFTSQL_ACCEPT_BASE_URL ?? "http://127.0.0.1:8010";
const outputDir = path.resolve(
  process.env.DRIFTSQL_ACCEPT_OUTPUT_DIR ?? "../artifacts/replay_review_acceptance",
);

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ locale: "zh-CN", viewport: { width: 1440, height: 1000 } });
const checks = {};

try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "运行监控" }).click();
  await page.getByText("Replay 人工审核", { exact: true }).waitFor();
  checks.candidate_count = await page.locator(".replay-candidate").count() === 3;
  checks.pending_count = await page.locator(".replay-candidate .review-pending").count() === 3;
  checks.human_reviewer_required = await page.getByPlaceholder("输入真实审核人姓名或工号").isVisible();
  checks.reason_required = await page.locator(".replay-candidate textarea").count() === 3;
  checks.approve_and_reject_controls = (
    await page.getByRole("button", { name: "批准回灌" }).count() === 3
    && await page.getByRole("button", { name: "拒绝" }).count() === 3
  );
  checks.full_trajectory_links = await page.getByRole("button", { name: "查看完整轨迹" }).count() === 3;
  await page.locator(".replay-review").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(outputDir, "replay-review-panel.png"), fullPage: true });
} finally {
  await browser.close();
}

const result = {
  protocol: "driftsql_replay_human_review_ui_v1",
  base_url: baseUrl,
  passed: Object.values(checks).every(Boolean),
  checks,
  screenshot: path.join(outputDir, "replay-review-panel.png"),
  decisions_written: false,
};
await writeFile(
  path.join(outputDir, "acceptance.json"),
  `${JSON.stringify(result, null, 2)}\n`,
  "utf8",
);
console.log(JSON.stringify(result, null, 2));
if (!result.passed) process.exitCode = 1;
