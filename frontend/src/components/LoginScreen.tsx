import { FormEvent, useState } from "react";

import { Icon } from "./Icon";


interface Props {
  onLogin: (apiKey: string) => Promise<void>;
}

export function LoginScreen({ onLogin }: Props) {
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!apiKey.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await onLogin(apiKey);
      setApiKey("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login-screen">
      <form className="login-card" onSubmit={(event) => void submit(event)}>
        <div className="login-mark"><Icon name="activity" /></div>
        <span className="eyebrow">DriftSQL 安全部署</span>
        <h1>登录智能体工作台</h1>
        <p>输入部署管理员提供的 API Key。验证后只保存短期 HttpOnly 会话，不会把密钥写入浏览器存储或运行日志。</p>
        <label><span>API Key</span><input type="password" autoComplete="current-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="输入 DriftSQL API Key" autoFocus /></label>
        {error && <div className="login-error"><Icon name="x" />{error}</div>}
        <button className="button button-primary" disabled={busy || !apiKey.trim()} type="submit"><Icon name="shield" />{busy ? "验证中…" : "安全登录"}</button>
        <small>仅通过 HTTPS 部署时启用 Secure Cookie。</small>
      </form>
    </main>
  );
}
