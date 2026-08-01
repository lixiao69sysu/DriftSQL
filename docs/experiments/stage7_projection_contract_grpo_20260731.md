# Stage 7: Projection-Contract Recovery with Failure-Balanced GRPO

Date: 2026-07-31

## Objective

Recover cached SQL after additive schema drift without allowing `SELECT *` or
`alias.*` to silently expose newly added audit columns. The Stage 7 protocol is
database-disjoint from Train to Tune to Gate, and permanently seals the old
Stage 6 Gate112.

## Data protocol

| Split | Databases | Add-column wildcard | General replay |
|---|---:|---:|---:|
| Train | 24 | 96 | 307 |
| Tune | 6 | 24 | 84 |
| Gate | 6 | 24 | 82 |

Each add-column split balances four profiles: single-table plain/qualified and
multi-table plain/qualified. It also balances one versus two added audit
columns. Database overlap between all splits is zero.

The failure-balanced GRPO set contains 403 real Train rows: 242 add-column rows
(60%) and 161 general replay rows (40%). Tune failures only determine sampling
weights; Tune and Gate task IDs are never trained on.

## Model and training

- Base: Qwen2.5-Coder-7B-Instruct.
- Stage 6 frozen LoRA -> targeted Stage 7 SFT20 -> failure-balanced GRPO10.
- Training: VERL GRPO, rollout `n=4`, learning rate `5e-7`, KL coefficient
  `0.01`, 7-turn tool loop, shaped execution-grounded reward.
- Stable hardware layout: GPU 0 and 2, two actor ranks, rollout tensor
  parallel size 2.
- W&B: <https://wandb.ai/lixiao69-/driftsql-rl/runs/jp0505yy>

The 10-step run finished with exit code 0. Step 10 training reward mean was
0.1958, PPO KL was 0.000292, and peak actor allocation was 13.15 GiB per GPU.
Training batch reward is not used for candidate selection.

## Tune-only selection

| Candidate | Add Tune24 | Executable | General Tune84 |
|---|---:|---:|---:|
| Stage 6 frozen | 3/24 (12.50%) | 6/24 | 72/84 (85.71%) |
| Stage 7 SFT20 | 3/24 (12.50%) | 8/24 | not selected |
| GRPO Step 5 | 1/24 (4.17%) | 5/24 | not evaluated |
| GRPO Step 10 | 4/24 (16.67%) | 9/24 | 76/84 (90.48%) |

Step 10 improved Add success by 4.17 percentage points over SFT20 and general
success by 4.76 points over the frozen Stage 6 adapter. It passed the
pre-freeze requirement that general regression be no worse than -2 points.

## One-shot Gate106

The candidate and acceptance thresholds were frozen before Gate materialization
or inference. Gate was invoked exactly once and completed all 106 process-
isolated episodes without unsafe actions or SQL timeouts.

| Slice | Result | Precommitted threshold | Pass |
|---|---:|---:|---:|
| Overall | 67/106 (63.21%) | >=70% | No |
| Add column | 3/24 (12.50%) | >=12.5% | Yes |
| General | 64/82 (78.05%) | >=80% | No |
| Unsafe | 0 | 0 | Yes |
| Timeout | 0 | 0 | Yes |

Stage 7 is therefore **not accepted**. Gate106 must not be rerun or used for
failure mining. Further optimization requires a fresh database-disjoint Stage 8
protocol.

Add-column residuals by held-out wildcard profile:

| Profile | Success | Executable | Turn limit |
|---|---:|---:|---:|
| Multi-table plain | 0/6 | 1/6 | 5/6 |
| Multi-table qualified | 1/6 | 1/6 | 5/6 |
| Single-table plain | 2/6 | 2/6 | 4/6 |
| Single-table qualified | 0/6 | 3/6 | 3/6 |

## Engineering findings

1. Long tool responses constructed `ToolResponse` with explicit null media,
   which newer VERL/Pydantic rejects. Text-only truncation now omits null media.
2. The process-isolated evaluator now supports multiple drift types, bounds
   completion length by remaining context, and enforces an 8-minute episode
   timeout with one fresh-process retry.
3. Type-preserving `replace_column` previously performed add/update/drop on
   4.1--4.2 GB databases for every episode. It now uses SQLite's metadata-only
   rename fast path when declared types match and retains the copy/drop fallback
   for real type changes.

## Audit artifacts

- Frozen candidate: `reports/stage7/final_candidate/frozen_candidate.json`
- Gate data manifest: `data/processed/stage7_gate106/summary.json`
- Gate output: `reports/stage7/final_gate106/summary.json`
- One-shot audit: `reports/stage7/final_gate106/audit.json`
- Profile addendum: `reports/stage7/final_gate106/profile_addendum.json`
- Stage 6 seal: `reports/stage7/stage6_gate112_seal.json`
