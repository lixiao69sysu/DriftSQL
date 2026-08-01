# Stage 4 Agentic GRPO tuning — 2026-07-29

## Outcome

Stage 4 is complete at the deployable **policy + safety controller** level.
The best pure GRPO policy improves held-out correctness and raw turn-limit
behavior, but misses the deliberately strict 20% efficiency gate by three
tasks.  Adding a conservative terminal controller and replaying the locked
trajectories passes all promotion gates:

| Locked 78-task protocol | Success | Executable | Turn limit | Unsafe | Avg tool calls |
|---|---:|---:|---:|---:|---:|
| Tool-SFT step80 + controller | 24/78 | 47/78 | 23 | 0 | 6.51 |
| GRPO curriculum5 step10 + controller | **28/78** | **51/78** | **16** | **0** | **6.36** |

Turn-limit incidence falls by `7/23 = 30.4%`, success rises by four tasks,
and safety does not regress.  The paired result contains five GRPO gains and
one loss (exact McNemar `p=0.21875`); this 78-task set is a promotion/regression
gate, not a statistical significance claim.

The promotion artifact is
`reports/stage4/comparison_offline_terminal_fallback/comparison.json`.

## Locked pure-policy result

The production controller is reported separately from policy quality.  On the
original locked trajectories, without terminal fallback:

| Policy | Success | Turn limit | Unsafe |
|---|---:|---:|---:|
| Base Qwen2.5-Coder-3B-Instruct | 1/78 | 0 | n/a |
| Tool-SFT step80 | 24/78 | 27 | 0 |
| Best GRPO, curriculum5 step10 | **26/78** | **24** | **0** |

The raw policy therefore improves correctness by two and reduces turn limits
by 11.1%, but does not by itself meet the `>=20%` efficiency target.

## What was implemented

The live VERL environment now passes complete tool events, response-token
usage, wall-clock timeout state, and max-turn termination state into the
custom reward.  The shaped reward includes:

- execution-grounded result success;
- useful clarification, executable SQL, and efficient completion bonuses;
- tool and token costs;
- duplicate question/execution penalties;
- repeated clarification/retrieval penalties;
- invalid/unsafe SQL, wall-clock timeout, turn-limit, and missing-submit
  penalties.

The critical fixes were distinguishing a normal seven-turn exhaustion from a
wall-clock timeout and penalizing an early invalid/no-tool response.  Without
the latter, the policy could avoid the turn-limit penalty by terminating with
invalid text.

The optional terminal controller in `scripts/run_five_tool_eval.py` submits
only the final SQL that already passed the sandboxed `execute_sql` tool.  It
does not synthesize or rewrite SQL and cannot bypass the read-only check.
`scripts/analyze_terminal_fallback.py` replays that controller against locked
trajectory artifacts and real drifted databases, eliminating model-generation
variance from the controller comparison.

## Tuning record

All formal runs started from the Stage 3 Tool-SFT adapter and used execution
reward on real drifted SQLite sessions.

| Run / selected checkpoint | Main change | Success | Turn limit | Unsafe | Decision |
|---|---|---:|---:|---:|---|
| v2 step30 | 40-step, 7-turn GRPO | 25 | 26 | 0 | safe, efficiency miss |
| v3 step20 | repeated-tool and turn-limit penalties | 25 | 24 | 1 | safety regression |
| v4 step10 | missing-submit penalty, 5-turn curriculum | **26** | **24** | **0** | best pure policy |
| v4 step15 | continued curriculum | 22 | 31 | 1 | over-training regression |
| v5 n8 step5 | rollout 8, LR `5e-7` | 22 | 24 | 1 | unstable |
| v5 n8 step10 | continued stable run | 25 | 30 | 1 | unstable |

This is an important practical result: more GRPO steps and more rollouts did
not monotonically improve deterministic behavior.  Early checkpoint selection
and external execution evaluation are mandatory.

The v5 run completed 10/10 steps on GPUs 0,2,3 with 240 non-padding rollouts.
It also exposed real tail latency: individual multi-turn rollout workers can
remain active for several minutes even when the other workers finish, so
trajectory timeouts and per-worker observability are operational requirements.

## Reproducibility caveat

A fresh temperature-zero vLLM generation run was also performed with the same
controller for both models.  It produced SFT `24/78, turn-limit 22, unsafe 0`
and GRPO `23/78, turn-limit 24, unsafe 0`, which fails promotion.  This differs
from the earlier locked deterministic trajectories by several tasks.  The
difference is recorded in
`reports/stage4/comparison_terminal_fallback_rerun/comparison.json` and is not
hidden or used as positive evidence.

For regression gates, generation is therefore locked once and environment
policies are compared by offline database replay.  Future work should add
repeat-seed confidence intervals rather than assuming `temperature=0` gives
bitwise reproducibility across vLLM tensor-parallel runs.

## Main artifacts

- Best policy adapter:
  `checkpoints/stage4_five_tool_grpo_3b_v4_curriculum5/global_step_10/merged/lora_adapter`
- Best raw trajectories:
  `reports/stage4/five_tool_eval_curriculum5/curriculum5-step10.jsonl`
- Locked controller replay:
  `reports/stage4/offline_terminal_fallback/`
- Passing promotion report:
  `reports/stage4/comparison_offline_terminal_fallback/`
- Failed fresh-generation comparison:
  `reports/stage4/comparison_terminal_fallback_rerun/`
- v5 rollout analysis:
  `reports/stage4/training_rollouts_v5_stable_n8.json`

## Verification

The reward/evaluator/parser suite passes 21 tests.  The agent-loop session test
also passes independently.  No evaluation trajectory performed an unsafe SQL
action in the promoted locked comparison.
