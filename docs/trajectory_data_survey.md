# SQL Agent trajectory-data survey

Checked on 2026-07-26. A usable DriftSQL trajectory must include, or allow us
to replay, the database state, tool calls, tool observations, final SQL, and an
execution-verifiable outcome.

| Resource | What is public | Useful to us | Why it is not the core training set |
| --- | --- | --- | --- |
| [Microsoft MAGIC](https://huggingface.co/datasets/microsoft/MAGIC) | 48,124 prompt/response records from feedback, correction, and manager agents | Optional self-correction baseline | Agent-to-agent text rather than replayable database tool calls; no schema-version transition |
| [PT-BR Agentic Text-to-SQL](https://huggingface.co/datasets/YnJhY2lzMjAyNnRleHQyc3Fs/pt-br-agentic-text-to-sql-distilled-trajectories) | 7,442 message-only trajectories using `get_table_schema` and `execute_sql` | Tool-SFT format reference | Portuguese and narrow domains; LLM-judged distilled conversations; no schema-version transition |
| [BIRD-Interact](https://bird-interact.github.io/) | Interactive environment and 600 evaluation tasks | Environment and evaluation protocol | Does not publish a reusable training trajectory corpus; protected GT/test cases are evaluation-oriented |
| [SIX-GYM-SQLite](https://huggingface.co/datasets/birdsql/six-gym-sqlite) | 5,000 SQL issues, solutions, tests, and databases | Strong debugging environment and reward source | Static issue/solution records, not multi-step tool trajectories and not version drift |
| [EvoSchema](https://github.com/zhangtianshu/EvoSchema) | Static BIRD-derived train/eval JSON for ten schema perturbations | Taxonomy and comparison benchmark | No trajectories or materialized changed databases; repository says generation code is still pending release |
| [LiveSQLBench](https://livesqlbench.ai/) | Dynamic benchmark, agent scaffold, and business-rule-drift evaluation | Future external evaluation | Benchmark rather than an open, replayable trajectory training corpus |

## Decision

No mature open corpus matches the target:

```text
stale SQL
  -> real execution failure
  -> inspect schema/metric version
  -> recover with tools
  -> execute repaired SQL
  -> verifiable reward
```

DriftSQL-RL therefore generates this missing layer locally. EvoSchema supplies
the perturbation taxonomy, BIRD supplies clean SQL and real databases,
SIX-GYM supplies debugging tasks and tests, and the DriftSQL factory supplies
version transitions, replayable observations, oracle trajectories, and strict
execution validation.
