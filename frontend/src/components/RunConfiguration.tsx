import type { RunOptions } from "../types";
import { Icon } from "./Icon";

interface Props {
  options: RunOptions;
  disabled: boolean;
  running: boolean;
  busy: boolean;
  onChange: (options: RunOptions) => void;
  onRun: () => void;
  onCancel: () => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function RunConfiguration({ options, disabled, running, busy, onChange, onRun, onCancel }: Props) {
  const field = (name: keyof RunOptions, value: number) => onChange({ ...options, [name]: value });
  return (
    <div className="run-config">
      <div className="config-fields">
        <NumberField label="最大轮数" value={options.max_turns} min={1} max={12} onChange={(value) => field("max_turns", value)} />
        <NumberField label="超时（秒）" value={options.timeout_seconds} min={1} max={600} onChange={(value) => field("timeout_seconds", value)} />
        <NumberField label="工具调用上限" value={options.max_tool_calls} min={1} max={32} onChange={(value) => field("max_tool_calls", value)} />
        <NumberField label="单轮生成 Token" value={options.max_new_tokens} min={16} max={4096} step={16} onChange={(value) => field("max_new_tokens", value)} />
      </div>
      {running ? (
        <button className="button button-stop" onClick={onCancel}><Icon name="pause" />取消运行</button>
      ) : (
        <button className="button button-primary" onClick={onRun} disabled={disabled}><Icon name={busy ? "refresh" : "play"} />{busy ? "正在准备沙箱" : "运行 Agent"}</button>
      )}
    </div>
  );
}
