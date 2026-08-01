# Stage 6 optimization protocol

Stage 6 improves the selected Stage 5 7B no-ask-user policy without using the
historical Test181 or Frozen78 for tuning.

## Data protocol

- Source: the 752 tasks and 50 databases from the historical Stage 5 training split only.
- Train: 531 tasks / 36 databases.
- Tune: 109 tasks / 7 databases. This split may be used for model and hyperparameter selection.
- Gate: 112 tasks / 7 databases. Run exactly once after the final candidate is frozen.
- Split unit: `db_id`; all database-overlap checks must remain empty.
- Historical Stage 5 dev/test databases have zero overlap with all Stage 6 splits.

## Optimization ladder

1. **B0**: selected Stage 5 no-ask-user LoRA and four production tools.
2. **B1**: same LoRA, plus `get_schema_version` and `inspect_schema_diff` and a version-aware prompt.
3. **B2**: targeted repair/next-action SFT. Teach concise version/diff/repair/submit trajectories,
   with extra weight on repeated-retrieval failures, add-column result contracts, schema-only,
   and compound drift.
4. **B3**: shaped GRPO initialized from B2. Reward verified repair and submission; penalize repeated
   retrieval/execution, missing submission, and turn-limit termination.
5. **B4**: failure-balanced replay, only if B3 still leaves a clear hard slice.

All B0--B4 comparisons use Tune109 with deterministic decoding and the same seven-turn/tool budget.
Gate112 remains sealed until one candidate and all acceptance thresholds are frozen.

## Acceptance thresholds

- overall success at least 20%;
- non-clean success at least 2x B0;
- schema-only success at least 10%;
- submission at least 40%;
- turn-limit at most 55%;
- unsafe and timeout tasks remain zero.

The submission and turn-limit thresholds must be met by model behavior. Terminal auto-submit may be
reported as a separate production safeguard but is not counted as evidence that the policy learned
the protocol.

## Final frozen result

The selected candidate is the Stage 5 GRPO policy followed by Stage 6 Repair-SFT step 20, typed
schema-diff recovery guidance, dynamic tool-schema routing, and duplicate-retrieval state guards.
No terminal auto-submit fallback is enabled.

| Metric | Tune109 | One-shot Gate112 | Target |
|---|---:|---:|---:|
| Overall success | 68.81% | 75.00% | >=20% |
| Non-clean success | 66.00% | 72.28% | >=14% (2x B0) |
| Schema-only success | 65.96% | 74.47% | >=10% |
| Submission | 77.98% | 78.57% | >=40% |
| Turn limit | 22.02% | 20.54% | <=55% |
| Unsafe / timeout | 0 / 0 | 0 / 0 | 0 / 0 |

The frozen hashes and exact inference settings are recorded in
`reports/stage6/final_candidate/frozen_candidate.json`. The one and only Gate112 pass is recorded in
`reports/stage6/final_gate112/audit.json`.

The aggregate gate passed, but the held-out add-column slice remained 0/12. It is a declared residual
risk, not hidden by the aggregate score. Any follow-up must use a new Stage 7 database-disjoint
train/tune/gate split; Gate112 is permanently audit-only and must not become tuning data.
