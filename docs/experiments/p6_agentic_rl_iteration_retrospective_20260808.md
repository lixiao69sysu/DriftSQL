# DriftSQL Agentic RL 试错与迭代复盘

更新日期：2026-08-08

## 1. 项目目标

本项目希望训练一个能够在数据库 Schema 和业务知识发生漂移后，通过多轮工具交互恢复 SQL
查询的 Agent。模型不仅需要生成 SQL，还需要在真实环境中完成以下闭环：

1. 判断是否存在 Schema 或业务定义漂移；
2. 选择 `get_schema_version`、`inspect_schema_diff`、`get_schema`、
   `ask_user`、`get_knowledge_definition` 等工具；
3. 在隔离 SQLite Sandbox 中执行并修复 SQL；
4. 验证结果契约；
5. 使用 `submit_solution` 安全结束轨迹。

最终优化目标不是训练集 reward，而是固定工具预算下的端到端任务成功率、漂移恢复率、
安全提交率和交互成本。

## 2. 统一实验口径

当前正式比较使用以下口径：

- 基模：`Qwen2.5-Coder-7B-Instruct`；
- 强 SFT：Recovery SFT + Hard Replay 的 Step160 LoRA；
- 当前最佳 RL：从强 SFT 初始化的 GRPO，seed `20260810`，Step25；
- 评测集：Train-derived Tune432；
- 漂移组成：`clean`、`add_column`、`rename_column`、`replace_column`、
  `rename_table`、`compound` 各 72 条；
- 交互组成：`direct_clean=72`、`knowledge_only=90`、`must_ask=110`、
  `schema_only=160`；
- 推理：temperature 0、最多 7 轮、7 个动态工具、相同 state guard 和 tool mask；
- 安全：只读 SQL、超时、rollback 和提交检查统一；
- Fresh Blind320 始终保持 0 读取，未参与模型选择。

历史实验使用过 Tune18、Fast42、Dev169 和 Test181。它们只用于说明当时的诊断过程，
不与当前 Tune432 的绝对数字直接相减。

## 3. 最终结果与贡献拆分

| 模型 | Tune432 成功 | 漂移恢复 | 安全提交 | 平均工具数 | Unsafe / timeout |
|---|---:|---:|---:|---:|---:|
| 原始基模 | 59/432（13.66%） | 55/360（15.28%） | 59/432 | 3.23 | 0 / 0 |
| 强 SFT160 | 341/432（78.94%） | 269/360（74.72%） | 341/432 | 4.93 | 0 / 0 |
| 最佳 GRPO Step25 | 345/432（79.86%） | 273/360（75.83%） | 345/432 | 4.92 | 0 / 0 |

贡献拆分如下：

- Base → SFT：净增加 282 条成功任务，提升 65.28 个百分点；
- SFT → GRPO：净增加 4 条成功任务，提升 0.93 个百分点；
- Base → GRPO：净增加 286 条成功任务，提升 66.20 个百分点；
- 总净增益中约 98.6% 来自 SFT，约 1.4% 来自当前 GRPO。

因此，项目已经获得很强的端到端落地效果，但必须诚实区分：SFT 解决了主要的工具协议和
恢复能力，GRPO 当前提供的是小幅、可测量但仍有提升空间的增量。

## 4. 试错时间线

### 4.1 先修复终止动作，而不是盲目训练 SQL

早期 Tune18 中，模型经常已经执行出正确 SQL，却没有调用 `submit_solution`。原始 SFT
轨迹只有 4/18 成功；结果契约 Controller 对同一批轨迹离线 Replay 后达到 14/18，且没有
增加模型调用，也没有不安全提交。

第一次 Terminal-action SFT 直接监督完整 JSON 和长 SQL 参数，训练 loss 下降，但在线成功率
没有提升。原因是决定成败的 `submit_solution` 只有少量 token，损失却被长 SQL 参数主导。
将监督目标收缩到 action name 后，Step5 达到 8/18，Step10 达到 10/18。

随后从 Step10 初始化 GRPO：Step5 和 Step10 仍为 10/18，但平均调用从 5.22 降到 5.06。
这轮 GRPO 改善了效率，没有改善精确成功率。

**得到的经验：** Agent 的失败可能发生在终止协议，而不是 SQL 推理；训练前必须先把
“执行正确但未提交”和“SQL 本身错误”分开。

### 4.2 Golden-prefix SFT 的 exposure bias

Generalized SFT 的 teacher-forced loss 持续下降，但 Dev169 从原始 Base 的 46/169 降到
26/169。First-divergence 审计发现，大多数错误发生在轨迹第 3～4 个动作：模型会重复
`ask_user`、错误回退到旧工具，或在真实错误观察后无法恢复。

因此项目改为收集 Train on-policy 失败轨迹，在第一个错误动作前后构建 Recovery SFT：

- 只使用执行验证的 Train 标准轨迹作为修正目标；
- 保留模型真实错误动作和环境 observation；
- 排除已经成功的非标准轨迹，避免把等价解误标为负例；
- 按数据库隔离训练和验证；
- 混合 atomic 与 compound replay，防止只训练新难例造成遗忘。

单纯 Recovery SFT 的几轮尝试仍只在 Fast42 上达到约 10～12/42。后续加入基于 audited
schema diff 的安全 SQL repair 和严格状态策略后，R4 在 Dev169 达到 112/169，在一次性
Test181 上达到 117/181。该阶段证明了 on-policy Recovery 数据和真实环境状态的重要性，
也建立了后续 GRPO 的强 SFT 起点。

**得到的经验：** 离线 golden trajectory 只能教会“理想状态下下一步做什么”，不能教会
“模型自己走错后如何回来”。Agent 数据必须覆盖模型真实访问到的状态。

### 4.3 第一轮 Full-episode GRPO：训练越久反而越差

在最初的 Tune432 checkpoint matrix 中，SFT160 为 323/432。Full-episode GRPO 的结果为：

| Checkpoint | 成功数 |
|---|---:|
| SFT160 | 323/432 |
| GRPO Step2 | 323/432 |
| GRPO Step4 | 321/432 |
| GRPO Step6 | 318/432 |
| GRPO Step8 | 319/432 |
| GRPO Step10 | 321/432 |
| GRPO Step12 | 319/432 |

这说明当时的 GRPO 不仅没有超过 SFT，继续训练还会造成轻微退化。审计后发现三个问题：

1. 训练采样没有可靠覆盖所有独立任务；
2. scalar advantage 被摊到很长的 rationale 和工具参数上，关键 action token 信号很弱；
3. Reward 虽然鼓励成功，但环境允许模型通过旧 SQL 捷径获得部分正反馈。

**得到的经验：** “训练正常结束、reward 有波动、checkpoint 能导出”不代表 RL 有效。
必须同时检查任务覆盖、advantage mask、策略更新幅度和在线行为变化。

### 4.4 Reward V1/V2/V3：Reward shaping 无法修复缺失的状态信息

随后构建 Focus1000，并比较 Reward V1、V2、V3。AddColumn72 上的结果只有 1～2/72，
70/72 左右的轨迹仍在执行并提交旧 SQL：

| Variant | AddColumn72 成功 | Ordered inspect | Stale shortcut |
|---|---:|---:|---:|
| Reward V1 Step50 | 2/72 | 1/72 | 70/72 |
| Reward V2 Step50 | 1/72 | 1/72 | 70/72 |
| Reward V3 Step10 | 1/72 | 1/72 | 70/72 |
| Reward V3 Step20 | 1/72 | 1/72 | 70/72 |

最优 Reward V1 Step50 在完整 Tune432 为 319/432，仍低于同期 SFT 的 323/432。

Reward V2/V3 加入了成功、澄清、终止、成本、重复工具、缺失提交和 AddColumn 协议等分量，
但模型看到的 observation 并不能明确区分“SQL 可执行”和“结果契约正确”。在这种情况下，
进一步调权重只是在不完整状态上优化，无法稳定教会正确恢复。

**得到的经验：** Reward 不能创造 observation 中不存在的信息。先确保环境满足 Markov
决策所需的最小状态，再调 reward。

### 4.5 Episode-level advantage：信用分配修了，行为捷径仍在

项目修复了训练采样覆盖和 episode-level advantage，确保：

- 每个训练任务完整出现；
- 每个 prompt 有 8 条 rollout；
- advantage 覆盖整段 episode response；
- response mask 与 episode mask 一致。

但是 Step25～125 在 AddColumn72 仍只有 1～2 条成功，69～71 条继续走 stale shortcut；
最优 Step50 在 Tune432 为 320/432，仍未超过 SFT。

**得到的经验：** 信用分配是必要条件，不是充分条件。若环境仍允许错误捷径，episode-level
advantage 会更一致地强化整条错误轨迹。

### 4.6 提高更新强度：LR/KL 不是唯一瓶颈

随后做 First-action GRPO A/B：

- Arm A：学习率 `1e-7`，KL `0.03`；
- Arm B：学习率 `5e-7`，KL `0.01`；
- rollout `n=8`，25 步，每 5 步保存。

Arm A 的 AddColumn72 仍为 1～2/72；Arm B Step5 也是 2/72。训练日志中的 PPO KL 和
clip fraction 很低，说明更新确实偏保守，但单纯提高 LR、降低 KL 仍没有改变首动作策略。

**得到的经验：** 超参数可以放大可学习信号，却不能替代正确的状态、工具契约和行为门禁。

### 4.7 关键转折：结果契约 observation 与有序状态策略

真正的转折来自环境修复，而不是再加一种 Reward：

- `execute_sql` observation 增加布尔结果契约字段：
  `result_contract_checked`、`result_contract_match`、
  `validated_for_submit`、`requires_schema_recovery`；
- 不向模型泄漏答案、结果 hash 或目标动作；
- 执行失败或结果契约不匹配后，强制按
  `get_schema_version -> inspect_schema_diff` 恢复；
- 只有最新执行满足 `validated_for_submit=true` 才允许 `submit_solution`；
- Reward 增加 AddColumn stale shortcut penalty。

仅加入 contract observation 后，SFT160 在 AddColumn72 从旧环境的极低结果提升到 24/72；
再加入 ordered state guard 后达到 28/72，ordered inspection 达到 72/72，stale shortcut
降到 0。

这一步说明此前大量“RL 不学习”实际上是环境契约不完整：模型没有可靠证据判断执行结果
是否满足原查询语义，且错误动作仍可进入看似成功的终态。

**得到的经验：** Agentic RL 的环境设计本身就是算法的一部分。Reward、observation、
action availability 和 terminal condition 必须一致。

### 4.8 Corrected-observation GRPO：第一次得到可测的 RL 增益

Arm C 从强 SFT160 初始化，使用：

- Focus200 v2；
- 25 步、batch 8、rollout `n=8`；
- 200/200 独立任务完整覆盖；
- 共 1,600 条 rollout；
- episode-level advantage，mask mismatch 为 0；
- 学习率 `5e-7`，KL 系数 `0.01`；
- Reward V3 + first-action / AddColumn stale shortcut shaping。

AddColumn72 checkpoint 曲线为：

| Checkpoint | 成功 | 平均工具数 |
|---|---:|---:|
| SFT160 | 28/72 | 6.24 |
| Step5 | 31/72 | 6.15 |
| Step10 | 31/72 | 6.17 |
| Step15 | 30/72 | 6.21 |
| Step20 | 29/72 | 6.22 |
| Step25 | 33/72 | 6.08 |

Step25 在当前统一 Tune432 上从 SFT 的 341/432 提升到 345/432；漂移恢复从 269/360
提升到 273/360；平均工具调用从 4.93 小幅降到 4.92；unsafe、timeout 和 invalid output
均为 0。这是当前第一次同时满足“成功率上升、漂移恢复上升、成本不增加、安全不退化”的
正式 GRPO 结果。

### 4.9 第二 seed：目标切片可复现，整体增益不可复现

为检查训练随机性的影响，保持数据、Reward、LR、KL 和 25-step 配置不变，只将 seed 改为
`20260811`。该轮同样完成 200/200 任务和 1,600 条 rollout。

| Variant | AddColumn72 独立评测 | Tune432 | 漂移恢复 |
|---|---:|---:|---:|
| SFT160 | 28/72 | 341/432 | 269/360 |
| Seed 20260810 Step25 | 33/72 | 345/432 | 273/360 |
| Seed 20260811 Step25 | 33/72 | 338/432 | 266/360 |

第二 seed 复现了 AddColumn72 的定向提升，却没有复现完整 Tune432 的总体提升。主要退化来自
`schema_only`：SFT 为 115/160，第二 seed 为 110/160。

另外，AddColumn72 在独立 batch-24 评测和 Tune432 内部 batch-32 子集上的数字不一致，
说明当前 vLLM 多轮推理仍存在 batch-context 敏感性。目标切片适合快速诊断，但不能单独作为
最终泛化结论。

根据当前项目目标，seed `20260810` Step25 保留为最佳实验 checkpoint；不把跨 seed 稳定性
作为下一轮硬门槛，但报告中保留这一事实。

## 5. 当前 Agentic RL 指标画像

最佳 GRPO Step25 在 Tune432 上：

- 任务成功：345/432（79.86%）；
- 漂移恢复：273/360（75.83%）；
- compound：61/72（84.72%）；
- atomic：212/288（73.61%）；
- hard：67/102（65.69%）；
- knowledge-only：74/90（82.22%）；
- must-ask：85/110（77.27%）；
- schema-only：114/160（71.25%）；
- 安全提交精度：345/345（100%）；
- 平均工具调用：4.92；
- turn limit：87/432（20.14%）；
- unsafe / timeout / invalid：0 / 0 / 0。

相对 SFT 的配对变化比单一准确率更能说明 RL 行为：

- 从 91 条 SFT 失败任务中救回 25 条，恢复率 27.47%；
- 341 条 SFT 成功任务中有 21 条退化，回退率 6.16%；
- 保留 320/341 条原成功任务，保持率 93.84%；
- 最终净增益为 4 条。

状态门禁在 88/432 个任务上屏蔽了 299 次错误动作。这保证了安全，但也说明模型仍依赖
工程策略。后续应同时报告“带 guard 的系统成功率”和“无干预纯模型成功率”。

## 6. 最重要的工程结论

1. **先修环境，再调 Reward。** 缺失结果契约时，Reward V1/V2/V3 和更强 LR 都不能阻止
   stale SQL shortcut；补齐 observation 和 terminal contract 后，同一 SFT 立即大幅改善。
2. **SFT 与 RL 的贡献必须拆开。** 当前 Base→SFT 是 +282 条，SFT→RL 只有 +4 条。把
   79.86% 全部描述成 RL 效果是不准确的。
3. **Controller 能力不能冒充模型能力。** 原始 Base 为 59/432，离线 contract controller
   可提升到 131/432；当前 SFT/RL 的 raw 与 controller 结果相同，说明它们已能自主提交。
4. **成功率之外要看配对 gain/loss。** GRPO 救回 25 条、丢失 21 条，暴露了策略迁移和
   遗忘，仅看净 +4 会隐藏这一点。
5. **执行成功不等于交互正确。** must-ask 任务可能绕过澄清直接恢复 SQL。需要单独统计
   clarification precision/recall。
6. **长轨迹会稀释关键动作。** action name、terminal submit 和 post-error recovery 需要
   单独的 mask、reward 或分阶段数据，而不是让几枚决策 token 与上千 rationale token
   共享一个弱信号。
7. **训练覆盖必须可审计。** 当前正式轮次要求 200/200 task、1,600 rollout、episode mask
   mismatch 为 0；否则 checkpoint 曲线不能用于判断算法优劣。
8. **目标切片不能替代完整评测。** AddColumn72 能快速定位旧 SQL 捷径，但最终 checkpoint
   仍需在 Tune432 上比较。

## 7. 下一轮只针对 GRPO 净增益

下一轮目标不是继续扩大 SFT，也不优先证明跨 seed 稳定性，而是让最佳 GRPO 超过当前
345/432。建议：

1. 从 seed `20260810` Step25 最佳 checkpoint 继续，而不是回到弱模型；
2. 仅从 Train on-policy 轨迹构建 hard curriculum，不使用 Tune432 反向训练；
3. 增加 `schema_only`、`replace_column`、`compound`、diff 后错误检索和 turn-limit 恢复；
4. 保留部分原成功任务 replay，降低 21 条回退；
5. 将学习率从 `5e-7` 提高到约 `8e-7`，KL 从 `0.01` 降到约 `0.005`，验证当前过低的
   `pg_clipfrac` 是否是更新不足；
6. 每 5 步保存，在 Tune432 选择最高 checkpoint；
7. 主要验收：超过 345/432、paired gains 明显大于 losses、unsafe/timeout 继续为 0、
   平均工具数不明显上升。

可将 350～355/432 作为下一轮工程目标，但不能把目标数字当作预期保证。

## 8. 关键产物

- 当前 Base/SFT/RL 统一比较：
  `reports/p6_scaleup/contract_grpo_base_sft_rl_tune432/comparison.md`
- Corrected-observation GRPO：
  `reports/p6_scaleup/contract_grpo_arm_c_tune432/comparison.md`
- AddColumn72 checkpoint 曲线：
  `reports/p6_scaleup/contract_grpo_arm_c_addcolumn72/comparison.md`
- 跨 seed 对比：
  `reports/p6_scaleup/contract_grpo_seed_stability_tune432/comparison.md`
- 跨 seed 结论：
  `reports/p6_scaleup/contract_grpo_seed_stability_tune432/stability_verdict.md`
- 早期 Full-episode GRPO matrix：
  `reports/p6_scaleup/tune432_checkpoint_matrix/comparison.md`

早期分阶段复盘已在 2026-08-09 清理并归并到本文第 4 节；对应的大体积原始报告也已删除，
保留的统一比较和关键曲线是当前唯一正式实验口径。

## 9. 面试版一句话总结

我们先通过执行验证的 on-policy Recovery SFT 将 Qwen2.5-Coder-7B 的七工具漂移恢复成功率
从 13.66% 提升到 78.94%；随后发现多轮 GRPO 长期不增益的根因不是 Reward 权重，而是
结果契约 observation 缺失和旧 SQL 终态捷径。补齐可验证状态、动态动作门禁、episode-level
advantage 与全覆盖采样后，GRPO 将 Tune432 进一步提升到 79.86%，同时保持 100% 安全提交
精度和更低的平均工具调用。当前下一步是减少 GRPO 的 21 条行为回退，继续扩大净增益。
