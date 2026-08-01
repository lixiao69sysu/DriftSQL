import type { SessionStatus } from "../types";

const labels: Record<SessionStatus, string> = {
  created: "已创建",
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  timed_out: "已超时",
  budget_exhausted: "预算耗尽",
};

export function StatusBadge({ status, success }: { status: SessionStatus; success?: boolean | null }) {
  const tone = status === "completed" ? (success ? "success" : "warning") : status;
  return (
    <span className={`status-badge status-${tone}`}>
      <i />
      {labels[status]}
    </span>
  );
}
