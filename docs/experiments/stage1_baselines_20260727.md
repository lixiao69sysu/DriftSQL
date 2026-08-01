# Stage 1 unified BIRD baselines — 2026-07-27

This experiment establishes the clean, pre-drift reference point for every
later DriftSQL-RL comparison. All baselines use the same frozen tasks, schema
representation, evidence, execution metric, decoding temperature, and maximum
calling budget. The result is a local project baseline, not a claim that these
numbers reproduce a BIRD leaderboard configuration.

## Frozen data and protocol

- benchmark: BIRD Mini-Dev SQLite;
- formal set: 500 tasks, 11 databases;
- difficulty: 148 simple, 250 moderate, 102 challenging;
- deterministic pilot: 64 tasks covering all 11 databases, with 21 simple,
  21 moderate, and 22 challenging tasks;
- metric: BIRD-RL set-normalized execution accuracy (EX);
- decoding: greedy (`temperature=0`);
- SQL execution: read-only SQLite, 30-second progress-handler timeout;
- maximum budget per task: 5 model calls, 5 tool calls, 3 SQL executions,
  3,072 generated tokens, 1,024 generated tokens per call, 16,384 prompt
  tokens, and 32,768 cumulative prompt-plus-output tokens;
- tool results: at most 100 rows are returned to the agent.

The formal manifest SHA-256 is
`3351f7ba603982e1568c02890cdf392dfb60cde27327ecac339dbaedb30efc43`.
The pilot manifest SHA-256 is
`07c899eac639e59d0a54c0635b13f9e505c6f975d4f48babffb1925de8e6866b`.
The complete machine-readable contract is
`data/evaluation/stage1/protocol.json`.

EX executes both predicted and Gold SQL on the same database, normalizes
strings case-insensitively, rounds floating-point values to ten decimal places,
and compares result sets. This deliberately matches the bundled BIRD-RL
evaluator semantics, including ignoring row order and duplicate rows. Every
baseline has zero recorded budget violations.

## Baselines

1. `qwen_base_direct`: one call to Qwen2.5-Coder-7B-Instruct with schema,
   column descriptions, question, and evidence; no exploratory execution.
2. `qwen_base_react`: the same base model in a fixed, untrained ReAct loop.
   It receives the official BIRD-RL tool prompt and real SQLite observations.
3. `bird_zeno_react`: the published BIRD-Zeno-7B checkpoint in exactly the
   same ReAct loop and budget.

Model revisions are pinned in `models.lock.json`:

- Qwen base: `c03e6d358207e414f1eca0bb1891e29f1db0e242`;
- BIRD-Zeno-7B: `d809e5b74b0e1d7028579f987ff6f2bbba4f7dba`.

The ReAct adapter first accepts upstream Hermes `<tool_call>` JSON. For a fair
fixed-agent comparison it also deterministically normalizes explicit
`<execute_sql>`, `Action: execute_sql`, and fenced-SQL actions emitted by the
untrained base model. It does not repair SQL or use Gold labels.

## Formal results (500 tasks)

| Baseline | EX | 95% Wilson CI | Executable | Avg model calls | Avg SQL executions | Avg total tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen direct | 49.2% (246/500) | [44.8%, 53.6%] | 87.4% | 1.00 | 0.00 | 3,958 |
| Qwen fixed ReAct, untrained | 23.2% (116/500) | [19.7%, 27.1%] | 41.2% | 3.07 | 2.20 | 16,204 |
| BIRD-Zeno fixed ReAct | **58.4% (292/500)** | [54.0%, 62.6%] | **91.6%** | 3.05 | 2.07 | 16,111 |

Difficulty breakdown:

| Baseline | Simple | Moderate | Challenging |
|---|---:|---:|---:|
| Qwen direct | 64.2% | 48.0% | 30.4% |
| Qwen fixed ReAct, untrained | 39.9% | 18.8% | 9.8% |
| BIRD-Zeno fixed ReAct | **75.7%** | **53.6%** | **45.1%** |

Paired task flips relative to Qwen direct:

- untrained ReAct gains 13 tasks and loses 143, a net change of -130;
- BIRD-Zeno ReAct gains 85 tasks and loses 39, a net change of +46.

Termination behavior explains much of the difference. Qwen ReAct submits only
245 tasks; 117 exhaust the SQL budget, 76 end with an invalid action, 60 exhaust
the token budget, and 2 exceed the prompt budget. BIRD-Zeno submits 466 tasks;
18 exhaust the SQL budget, 4 end with an invalid action, and 12 exhaust the
token budget.

## Practical interpretation

Wrapping a generic SQL model in ReAct is not an improvement by itself: it
reduces EX by 26.0 percentage points and consumes about four times as many
tokens. Tool-protocol training and learning how to use observations are core
model capabilities, not orchestration details.

BIRD-Zeno demonstrates that trained interaction can help: it improves EX by
9.2 points and adds 46 net correct tasks over direct generation. The gain is
largest on challenging tasks (+14.7 points). However, it also turns 39 direct
successes into failures and uses 4.1 times the tokens. DriftSQL-RL should
therefore optimize both accuracy and routing:

- call the interactive loop only when direct confidence or execution checks
  indicate that it is worthwhile;
- reward early, valid submission instead of exploratory loops that consume the
  SQL budget;
- mine the 39 interaction regressions and the explicit termination failures;
- retain a clean-task regression gate while training on schema drift;
- report accuracy together with model calls, SQL executions, and tokens.

These are measurable project targets for SFT, GRPO reward shaping, failure
replay, and the final serving policy.

## Reproduction

Prepare the frozen files:

```bash
env TMPDIR="$PWD/data/tmp" .venv/bin/python scripts/prepare_stage1_eval.py
.venv/bin/python scripts/bootstrap_model.py --model-key bird_zeno_7b
```

Run the two Qwen baselines and the BIRD-Zeno baseline on GPU 0 and 3:

```bash
env CUDA_VISIBLE_DEVICES=0,3 TMPDIR="$PWD/data/tmp" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .venv/bin/python scripts/run_stage1_baseline.py \
  --model models/Qwen2.5-Coder-7B-Instruct --model-alias qwen_base \
  --data data/evaluation/stage1/bird_mini_dev_500.json \
  --column-meaning data/evaluation/stage1/column_meaning.json \
  --output-dir reports/stage1/full --modes direct,react \
  --tensor-parallel-size 2 --batch-size 32 --sql-workers 8

env CUDA_VISIBLE_DEVICES=0,3 TMPDIR="$PWD/data/tmp" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .venv/bin/python scripts/run_stage1_baseline.py \
  --model models/BIRD-Zeno-7b --model-alias bird_zeno \
  --data data/evaluation/stage1/bird_mini_dev_500.json \
  --column-meaning data/evaluation/stage1/column_meaning.json \
  --output-dir reports/stage1/full --modes react \
  --tensor-parallel-size 2 --batch-size 32 --sql-workers 8
```

Score each JSONL with `scripts/evaluate_stage1_predictions.py`, then build the
paired table with `scripts/compare_stage1_results.py`. The consolidated outputs
are `reports/stage1/full/comparison.json` and
`reports/stage1/full/comparison.md`.

## Known boundary

Transformers emits a known pre-tokenizer-regex warning for the published
BIRD-Zeno tokenizer. This artifact-faithful run leaves the released tokenizer
unchanged. A spot check of representative SQL and date strings produced the
same token IDs with and without Transformers' fix flag, but a fixed-tokenizer
rerun remains a valid sensitivity check. The clean Mini-Dev result also does
not measure schema drift; it is the regression reference for the later drift
evaluation.
