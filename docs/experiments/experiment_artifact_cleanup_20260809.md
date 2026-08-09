# DriftSQL 实验产物清理记录

清理日期：2026-08-09

## 清理目标

删除已经被正式 Base/SFT/GRPO 对比替代的本地试错权重与可再生成缓存，同时保留项目运行、
模型对比、后续续训和实验复盘所需的完整链路。

## 保留的模型链

1. Base：`models/Qwen2.5-Coder-7B-Instruct`；
2. Strong SFT：
   `checkpoints/p6_scaleup_recovery_hard_sft_7b/global_step_160`；
3. 当前最佳 GRPO：
   `checkpoints/p6_contract_observation_grpo_arm_c_7b/global_step_25`；
4. 历史产品兼容模型：
   `checkpoints/stage8_fresh_sft_7b/global_step_20`。

最佳 GRPO 的 `environment_traces` 与 `rollouts` 保留，用于训练覆盖和行为审计。三个保留
checkpoint 的 portable LoRA 均已验证存在。

## 删除范围

- 97 个已被替代的 checkpoint 根目录，包括早期 3B smoke、retry、preflight、旧 Stage
  训练、Terminal-action、Reward V1/V2/V3、First-action、Episode-advantage、旧 Full-episode
  GRPO、第二 seed 等本地权重；
- 保留模型目录中的 10 个非最终中间 step；
- 本地 `logs/`、Hydra `outputs/`、`tmp/`、测试缓存、Ruff 缓存、Playwright/Hugging Face
  下载缓存和遗留 `nohup.out`。

checkpoint 占用从约 90 GB 降至约 1.4 GB，连同缓存共释放约 90 GB。删除为永久删除，旧模型
权重需要通过保留的代码、配置和数据重新训练才能恢复。

## 未删除范围

- `data/` 下的原始、处理后、训练、Tune432 与 Fresh Blind 数据；
- `reports/` 下的最终统一评测、跨 seed 对比、数据 QA 与产品验收结果；
- `wandb/` 中 Strong SFT160 与最佳 GRPO Arm C 的关键运行记录；
- `docs/experiments/` 中的实验复盘；
- 所有数据构建、训练、Reward、Replay 与评测脚本；
- 当前运行所需的 Qwen2.5-Coder-7B 基模和 Qwen2.5-0.5B 中文翻译模型；
- 产品验收截图与录屏。

## 第二轮目录瘦身

在第一轮 checkpoint 清理后，继续检查 `docs/`、`models/`、`reports/` 与 `wandb/`：

- 删除本地 Qwen2.5-Coder-3B 和 BIRD-Zeno 权重；两者不是当前运行依赖，仓库仍通过
  `models.lock.json` 保留精确版本和重新下载能力；
- 删除被当前 P6 总复盘替代的早期 smoke/stage 文档，以及已经下线的 Web 前端阶段文档；
- 删除旧 Stage3–Stage8、P5/P6 评测和中间 Reward/First-action/on-policy 报告，仅保留
  Stage1 基线摘要、最终 Base/SFT/GRPO、Arm C、跨 seed、环境契约和数据 QA 证据；
- W&B 仅保留 Strong SFT160 时间窗口与最佳 GRPO Arm C 相关运行；
- 删除报告目录内与完整 JSONL 重复的 `.partial.jsonl` 文件。

服务使用的 Base/SFT/GRPO 指标已迁移到
`configs/service/experiments.json`，不再依赖已删除的 Stage8 报告。仅验证
历史报告是否存在的测试也已删除；数据协议、Reward、Sandbox、Agent Loop
和当前模型对比测试仍然保留。

第二轮后目录规模为：`docs` 64 KB（7 份文档）、`models` 16 GB（2 个运行
模型）、`reports` 37 MB（59 个最终证据文件）、`wandb` 2.2 MB（6 个关键
run）。连同第一轮 checkpoint 清理，本次两轮总计释放约 110 GB。

本地删除不改变锁文件中的可复现定义，也不改变最终服务的模型目录。

## 清理后验证

- `/models` 引用的三个 LoRA Adapter 路径有效；
- 当前最佳 GRPO Step25 仍保留 FSDP actor/optimizer 状态，可继续训练；
- 复盘文档列出的五组关键汇总报告仍存在；
- 第二轮清理完成时，服务和 CLI 全量回归共 223 项通过。

## 第三轮脚本清理

`scripts/` 从 190 个文件缩减为 81 个（80 个脚本和 1 份入口说明），目录由
约 3.4 MB 降至约 740 KB。
删除内容包括：已移除权重对应的 3B smoke、P5 一次性 Gate、旧 Stage5–8、
Terminal-action、Reward A/B、Episode-advantage 和过期 checkpoint 评测脚本，
以及对应的历史产物测试与 Python 编译缓存。

当前保留 P6 Scale-up 数据工厂、on-policy Failure Miner、Recovery SFT、Hard
Replay、SFT160、Corrected-observation GRPO、Tune432 统一评测、Stage1 基线、
产品服务和数据审计链路。脚本入口与保留理由见 `scripts/README.md`。
删除脚本及其历史专用测试后，当前有效测试集共 188 项并全部通过。

## 第四轮脚本清理

进一步按“当前 7B P6 主线、项目复现、产品运行、正式基线证据”检查剩余入口，
删除 12 个脚本和 1 个只服务于已删除脚本的测试：

- 删除 BIRD-RL 资源缩放阶段的三个训练 Smoke、三个临时数据/子集准备入口和
  Adapter 语法检查；正式 Stage1 Base/ReAct/BIRD-Zeno 统一评测入口仍保留；
- 删除与现有 DriftSQL 工具环境重复的上游 BIRD-Interact public smoke；项目自己的
  Mini-Interact 数据适配和隔离 Sandbox smoke 仍保留；
- 删除未进入最终模型链的 LoRA 参数插值工具及其专用测试；
- 删除已被 Scale-up Failure Miner 和 Recovery SFT 构建器替代的早期 GRPO
  failure-trace 导出器；
- 删除仅转发到 `serve_service.sh`、没有独立行为的 `serve_studio.sh` 别名。

清理后 `scripts/` 为 69 个文件。当前正式 P6 训练、Tune432 对比、Stage1 基线、
Dataset V2、产品服务和可复现环境初始化链路均未改变。
