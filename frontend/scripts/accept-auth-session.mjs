import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "playwright";


const baseUrl = process.env.DRIFTSQL_ACCEPT_BASE_URL ?? "http://127.0.0.1:8011";
const apiKey = process.env.DRIFTSQL_ACCEPT_API_KEY ?? "acceptance-only-secret";
const outputDir = path.resolve(
  process.env.DRIFTSQL_ACCEPT_OUTPUT_DIR ?? "../artifacts/auth_session_acceptance",
);
await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ locale: "zh-CN", viewport: { width: 1440, height: 900 } });
const checks = {};

try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByText("登录智能体工作台", { exact: true }).waitFor();
  checks.login_screen_is_chinese = true;
  await page.screenshot({ path: path.join(outputDir, "01-login.png"), fullPage: true });

  const input = page.getByPlaceholder("输入 DriftSQL API Key");
  await input.fill("wrong-key");
  await page.getByRole("button", { name: "安全登录" }).click();
  await page.getByText("Invalid DriftSQL API key", { exact: true }).waitFor();
  checks.wrong_key_rejected = true;

  await input.fill(apiKey);
  await page.getByRole("button", { name: "安全登录" }).click();
  await page.getByText("服务就绪", { exact: true }).waitFor();
  checks.correct_key_creates_browser_session = true;

  const eventTypes = await page.evaluate(async () => {
    const scenariosResponse = await fetch("/api/scenarios");
    if (!scenariosResponse.ok) throw new Error(`scenario status ${scenariosResponse.status}`);
    const scenarios = await scenariosResponse.json();
    const createdResponse = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarios[0].scenario_id, labels: { source: "auth-acceptance" } }),
    });
    const session = await createdResponse.json();
    const runResponse = await fetch(`/api/sessions/${session.session_id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!runResponse.ok) throw new Error(`run status ${runResponse.status}`);
    const terminal = new Set(["completed", "failed", "cancelled", "timed_out", "budget_exhausted"]);
    for (let attempt = 0; attempt < 300; attempt += 1) {
      const current = await fetch(`/api/sessions/${session.session_id}`).then((response) => response.json());
      if (terminal.has(current.status)) break;
      await new Promise((resolve) => window.setTimeout(resolve, 25));
    }
    const streamResponse = await fetch(`/api/sessions/${session.session_id}/events`);
    if (!streamResponse.ok || !streamResponse.headers.get("content-type")?.startsWith("text/event-stream")) {
      throw new Error(`SSE endpoint status ${streamResponse.status}`);
    }
    const wire = await streamResponse.text();
    return [...wire.matchAll(/^event: ([a-z_]+)$/gm)].map((match) => match[1]);
  });
  checks.cookie_authenticates_fetch = true;
  checks.cookie_authenticates_sse = ["session", "queued", "model", "tool", "reward", "status"].every((name) => eventTypes.includes(name));
  await page.screenshot({ path: path.join(outputDir, "02-authenticated-studio.png"), fullPage: true });

  await page.getByRole("button", { name: "退出" }).click();
  await page.getByText("登录智能体工作台", { exact: true }).waitFor();
  checks.logout_revokes_session = true;
} finally {
  await browser.close();
}

const result = {
  protocol: "driftsql_browser_auth_session_v1",
  base_url: baseUrl,
  passed: Object.values(checks).every(Boolean),
  checks,
  api_key_persisted_in_artifact: false,
};
await writeFile(path.join(outputDir, "acceptance.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
console.log(JSON.stringify(result, null, 2));
if (!result.passed) process.exitCode = 1;
