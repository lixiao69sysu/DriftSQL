import { useState } from "react";

export function CodeBlock({ code, language = "sql", compact = false }: { code: string; language?: string; compact?: boolean }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }
  return (
    <div className={`code-block ${compact ? "compact" : ""}`}>
      <div className="code-head"><span>{language}</span><button onClick={copy}>{copied ? "已复制" : "复制"}</button></div>
      <pre><code>{code}</code></pre>
    </div>
  );
}
