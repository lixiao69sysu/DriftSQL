# P5 Human-Reviewed Replay GRPO and One-Shot Gate

Date: 2026-08-01

## Question

Can a short 7B GRPO continuation, trained only on database-isolated P5 Train
rows plus human-approved P4 failure strata, reduce add-column projection and
turn-limit failures without sacrificing safety?

## Data and review boundary

The project owner approved two immutable P4 candidates and rejected one
near-duplicate tool-budget candidate. Each decision is bound to the original
trajectory SHA-256. Approved failures define sampling strata only: no P4 Tune
row is copied into optimization data.

| Split | Databases | Rows | Turn-limit focus |
|---|---:|---:|---:|
| P5 Train base | 6 | 36 | 24 |
| Human-reviewed replay | same 6 Train DBs | 16 | 8 additional hard rows |
| Final GRPO Train | 6 | 52 | 32 |
| Tune | 3 disjoint DBs | 18 | 12 |
| One-shot Gate | 3 further disjoint DBs | 18 | 12 |

The 12-check training-input audit passed. Train/Tune/Gate database overlap is
empty, the opened Stage-8 Gate55 was never read, and the P5 Gate stayed sealed
until candidate, code, data, inference budget and thresholds were frozen.

## Training

- base: `Qwen/Qwen2.5-Coder-7B-Instruct`;
- warm start: frozen Stage-8 SFT20 LoRA;
- GPUs: RTX 3090 24 GiB on CUDA devices 0 and 2;
- algorithm: GRPO, 10 steps, rollout `n=4`;
- learning rate: `2e-7`; KL coefficient: `0.02`;
- checkpoints: steps 5 and 10;
- elapsed training time: 8 minutes 2 seconds;
- W&B offline run: `anbj614l` (10 complete history rows).

Training reward was noisy rather than monotonic. Mean shaped reward was
`0.5270` at step 1, reached `-0.2293` at step 7, and ended at `0.1797`.
Actor KL loss moved from `0.3195` to `0.2695`; no rollout was aborted. This is
why promotion used held-out execution success instead of training reward.

After training, both checkpoints were exported to portable PEFT adapters. Each
contains 392 validated finite tensors and a 155 MiB safetensors file.

To upload the preserved run without retraining:

```bash
wandb login
wandb sync wandb/wandb/offline-run-20260801_195734-anbj614l
```

## Database-isolated Tune result

All candidates used the same 18 tasks and inference budget, with one OS
process and one vLLM engine per episode.

| Candidate | Success | Executable/submitted | Turn limit | Avg. calls | Unsafe | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| frozen SFT20 | 4/18 (22.2%) | 7/18 | 10 | 5.50 | 0 | 0 |
| GRPO step 5 | 4/18 (22.2%) | 7/18 | 10 | 5.67 | 0 | 0 |
| GRPO step 10 | 3/18 (16.7%) | 6/18 | 11 | 5.67 | 0 | 0 |

Step 5 tied SFT20 on correctness but cost more; step 10 regressed. The
precommitted ranking therefore selected SFT20. This avoids promoting an RL
checkpoint merely because its training reward looked acceptable.

## Permanently sealed one-shot Gate

The frozen SFT20 candidate was evaluated once. The lifecycle records the first
Gate opening, evaluation start/completion, result hashes and permanent seal.
Fresh reruns are rejected, including after a failed acceptance decision.

| Gate metric | Result | Precommitted threshold | Pass |
|---|---:|---:|---:|
| Overall success | 5/18 (27.8%) | at least 50% | no |
| Turn-limit hard success | 3/12 (25.0%) | at least 33.3% | no |
| Turn-limit terminations | 7 | at most 6 | no |
| Unsafe tasks | 0 | exactly 0 | yes |
| Timeout tasks | 0 | exactly 0 | yes |

P5 did **not** pass promotion. The engineering safety boundary held, but this
small GRPO run did not solve the policy failure. Gate failures are not mined or
used for further tuning. A future iteration must use a newly designed
Train/Tune/Gate protocol and should test stronger terminal-action supervision,
more positive post-repair submission trajectories, and a curriculum that
separates projection repair from termination behavior.

## Artifact hashes

- reviewed decisions: `39c4a32c80f4223310ffb94b868554e4e5c175022b86d575d90ae431a93d0493`;
- GRPO Train parquet: `dc26ff805818f92b2543e0a8842ee5812fd5eb992ce0f0dfe23d2b85fa629cf0`;
- GRPO Tune parquet: `6f3c90e92dd3769ef670ebaba8cda4ee16e5a81725bf4859c55fba2854d9e0ca`;
- W&B offline run: `e70b4c192a324b02a31d71efd54c916d2981176065068c2821185023d0a48789`;
- step-5 adapter: `db54885dbeea8a752cb7d318562555ae677dbe89e652b195f1b9a7482d3a4981`;
- step-10 adapter: `195f3fe219268a89dc2244226546b5420a20a355671aa0c4b3dd2ed24ca19d5e`;
- frozen candidate: `4d077acd5fa9f036ac24539338547c1077e1a9d388a0ea1d8a6277ab37774b5e`;
- one-shot evaluation state: `2ffe8412fc0f11017285fafc585611d4d173dde4a16be628e13d748ebaf4161a`;
- permanent Gate result: `dd4bd9c6d323738555036d157fbe84480bb2aedd990a54d4491857b88cad1a5b`.
