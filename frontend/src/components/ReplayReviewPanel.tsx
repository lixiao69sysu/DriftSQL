import { useState } from "react";

import { zhLabel } from "../locale";
import type { ReplayCandidateList } from "../types";
import { compactHash, formatDate } from "../utils";
import { Icon } from "./Icon";


interface Props {
  replay: ReplayCandidateList | null;
  onReplay: (sessionId: string) => void;
  onReview: (
    candidateId: string,
    decision: "approve" | "reject",
    reviewer: string,
    reason: string,
  ) => Promise<void>;
}

export function ReplayReviewPanel({ replay, onReplay, onReview }: Props) {
  const [reviewer, setReviewer] = useState("");
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [busyCandidate, setBusyCandidate] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  async function decide(candidateId: string, decision: "approve" | "reject") {
    const reason = reasons[candidateId]?.trim() ?? "";
    if (reviewer.trim().length < 2 || reason.length < 8) {
      setLocalError("请填写至少 2 个字符的审核人和至少 8 个字符的具体依据。");
      return;
    }
    setBusyCandidate(candidateId);
    setLocalError(null);
    try {
      await onReview(candidateId, decision, reviewer.trim(), reason);
      setReasons((current) => ({ ...current, [candidateId]: "" }));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyCandidate(null);
    }
  }

  return (
    <section className="panel replay-review">
      <div className="panel-title replay-review-title">
        <span><Icon name="shield" />Replay 人工审核</span>
        <div className="replay-counts">
          {Object.entries(replay?.counts ?? {}).map(([status, count]) => (
            <em className={`review-${status}`} key={status}>{zhLabel(status)} {count}</em>
          ))}
        </div>
      </div>
      <p className="replay-policy">P4 Tune 失败只定义困难分层；批准后只从数据库隔离的 P5 Train 抽取回灌样本，不复制 P4 轨迹，也不读取 Tune 或 sealed Gate。</p>
      {!replay ? (
        <div className="ops-empty">正在加载候选 Replay…</div>
      ) : !replay.available ? (
        <div className="wandb-setup"><b>尚未生成候选集</b><p>先运行失败挖掘脚本，再由真实人员逐条审核。</p><code>python scripts/mine_p4_replay_candidates.py</code></div>
      ) : (
        <>
          <label className="reviewer-field"><span>审核人</span><input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="输入真实审核人姓名或工号" /></label>
          {localError && <div className="review-error">{localError}</div>}
          <div className="replay-candidate-list">
            {replay.candidates.map((candidate) => (
              <article className="replay-candidate" key={candidate.candidate_id}>
                <header>
                  <div><b>{candidate.db_id}</b><span>{compactHash(candidate.candidate_id, 12)} · {zhLabel(candidate.wildcard_profile)}</span></div>
                  <strong className={`review-status review-${candidate.review_status}`}>{zhLabel(candidate.review_status)}</strong>
                </header>
                <div className="replay-evidence">
                  <span><small>失败分类</small><b>{zhLabel(candidate.failure_class)}</b></span>
                  <span><small>终止原因</small><b>{zhLabel(candidate.termination_reason)}</b></span>
                  <span><small>运行预算</small><b>{candidate.model_calls} 轮 / {candidate.tool_calls} 工具</b></span>
                  <span><small>轨迹哈希</small><code>{compactHash(candidate.trajectory_sha256, 10)}</code></span>
                </div>
                <div className="tool-sequence">{candidate.tool_sequence.map((tool, index) => <span key={`${tool}-${index}`}>{zhLabel(tool)}</span>)}</div>
                {candidate.final_sql && <code className="candidate-sql">{candidate.final_sql}</code>}
                {candidate.review_status !== "pending" && (
                  <div className="previous-review"><b>{candidate.reviewer}</b><span>{candidate.review_reason}</span><small>{formatDate(candidate.reviewed_at)}</small></div>
                )}
                <textarea
                  value={reasons[candidate.candidate_id] ?? ""}
                  onChange={(event) => setReasons((current) => ({ ...current, [candidate.candidate_id]: event.target.value }))}
                  placeholder="填写判断依据：标签是否正确、是否可行动、是否含敏感内容、是否适合匹配 P5 Train…"
                />
                <footer>
                  <button className="review-trajectory" onClick={() => onReplay(candidate.session_id)}>查看完整轨迹</button>
                  <button className="review-reject" disabled={busyCandidate === candidate.candidate_id} onClick={() => void decide(candidate.candidate_id, "reject")}>拒绝</button>
                  <button className="review-approve" disabled={busyCandidate === candidate.candidate_id} onClick={() => void decide(candidate.candidate_id, "approve")}>{busyCandidate === candidate.candidate_id ? "写入中…" : "批准回灌"}</button>
                </footer>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
