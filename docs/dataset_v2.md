# DriftSQL Dataset V2

Dataset V2 expands the original 400 fixed six-action drift trajectories into
a training corpus that separates **what changed**, **how hard the SQL is**, and
**which interaction is actually necessary**.  The data remains generated from
Gold SQL and real SQLite databases; no synthetic SQL answer is accepted
without execution and result-fingerprint verification.

## Target composition

| Scenario | Drift family | Rows |
|---|---|---:|
| Atomic | add column / rename column / rename table / replace column | 154 / 216 / 216 / 216 |
| Compound | two ordered, execution-preserving schema mutations | 200 |
| Clean control | no schema change; cached SQL must be preserved | 100 |
| **Total** | | **1,102** |

The 1,002 non-clean tasks are independently assigned inside each drift family
to three interaction profiles:

- `must_ask`: 30%; schema lookup, guarded clarification, HKB lookup, execution,
  repair, and submission;
- `knowledge_only`: 25%; schema and HKB lookup without an unnecessary user
  question;
- `schema_only`: 45%; schema inspection and executable repair without HKB or
  user interaction.

The 100 `direct_clean` controls execute and submit the cached SQL unchanged.
This teaches the agent that tool use and query rewriting have a cost and are
not universally beneficial.

Every row also records:

- `difficulty`: AST-derived `easy`, `medium`, or `hard`;
- `failure_mode`: explicit schema error, silent result mismatch, or clean
  no-drift control;
- `scenario_type`, `drift_type`, `interaction_profile`, and a combined
  `stratum` key;
- whether it belongs to the original 400-task corpus.

## Leakage policy

Train/dev/test are assigned as 70/15/15 by `db_id`, never by row.  The 11
databases and 78 tasks from the existing locked evaluation are forced into the
new test split and are also written separately as `frozen_regression_78.jsonl`.
The splitter optimizes balance over scenario, drift, interaction, difficulty,
failure mode, and source while maintaining zero database overlap.

## Materialized result (2026-07-29)

The completed build contains 1,102 unique tasks over 67 databases and has zero
replay rejections.

| Split | Trajectories | Databases | Next-action SFT rows |
|---|---:|---:|---:|
| Train | 752 | 50 | 3,434 |
| Dev | 169 | 5 | 769 |
| Test | 181 | 12 | 857 |
| **Total** | **1,102** | **67 disjoint assignment units** | **5,060** |

Difficulty is 212 easy / 466 medium / 424 hard. Interaction profiles are 301
`must_ask`, 250 `knowledge_only`, 451 `schema_only`, and 100 `direct_clean`.
The native replay made 2,104 SQL executions and accepted all 1,102 tasks; its
maximum rendered trajectory is 5,047 tokens under the 6,144-token budget.
All 78 locked regression tasks are present in Test and in
`frozen_regression_78_agent_eval.jsonl`.

Artifacts:

- raw tasks and generation report: `data/generated/stratified_v2/`;
- database-disjoint manifests: `data/processed/stratified_v2/`;
- trajectory SFT, GRPO parquet, and agent eval:
  `data/processed/stratified_five_tool_v2/`;
- next-action SFT parquet:
  `data/processed/stratified_five_tool_next_action_v2/`.

## Reproduction

The local `.venv` is an overlay on the requested `lcpy311` Python 3.11
environment.

```bash
export DRIFTSQL_TMPDIR="$PWD/tmp"
.venv/bin/python scripts/build_stratified_drift_data_v2.py
.venv/bin/python scripts/split_stratified_drift_v2.py
.venv/bin/python scripts/prepare_stratified_five_tool_data_v2.py
.venv/bin/python scripts/expand_five_tool_sft_next_actions.py \
  --input-dir data/processed/stratified_five_tool_v2 \
  --output-dir data/processed/stratified_five_tool_next_action_v2 \
  --plain-json-targets
CUDA_VISIBLE_DEVICES=0,2,3 \
  bash scripts/train_7b_stratified_five_tool_sft.sh
```

The replay step materializes one isolated active database per trajectory,
invokes the native VERL tools, checks timeout/rollback/isolation metrics, and
recomputes the final result fingerprint.  It fails closed on any rejection by
default.  Its outputs include trajectory SFT parquet, next-action SFT parquet,
GRPO parquet, manifests, agent-evaluation JSONL, and summary reports for all
three splits.

The formal V2 SFT launcher uses Qwen2.5-Coder-7B-Instruct, warm-starts from the
existing 7B Tool-SFT adapter, and uses a 6,144-token limit. This retains the four legacy
wide-schema tasks that exceeded the old 4,096-token cap, without truncating
the verified tool observation.
