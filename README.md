# ShortcutRepair-DPO v1.2

ShortcutRepair-DPO 是一个小型、受控、可复现的后训练实验。项目先用 SFT 诱导 Qwen2.5-1.5B-Instruct 依赖可能过期的 `cached_recommendation`，再研究如何让模型重新服从 fresh tool result。

项目不提出新的 DPO 损失，也不把合成二选一任务外推为通用工具调用能力。它的价值在于把 shortcut 机制、反事实修复、能力保持和停止条件做成可审计的实验协议。

## v1.2 研究问题

v1.1 已能削弱 cached hint，但按 `decision_type` 切片后发现：模型主要学会了过滤无效候选，没有可靠学会比较两个有效候选的 fresh score。v1.2 因而直接回答：

> 如何在保留 aligned 和 `validity_decisive` 能力的同时，让模型真正学会 `score_decisive` 的 fresh-score 比较，并保持对 nuisance 字段的不变性？

## 方法

- 训练 case 中 75% 为 `score_decisive`、25% 为 `validity_decisive`；dev/test 按 50/50 平衡。
- gold 在 A/B 位置和 nuisance 方向上严格平衡；SFT 使用 hint-neutral 数据，DPO 使用 aligned/conflict 反事实偏好。
- dev pilot 比较 `direct_dpo`、`score_sft` 和 `sft_dpo`；若全部 DPO 路径未达标，只允许把 DPO 学习率减半这一个调整维度。
- 选型先检查能力保持，再比较 score fresh response、nuisance invariance 和阶段数；SFT 只是能力基线，不能代替合格 DPO。
- 只有 pilot 选出合格路径后，才生成 sealed test 并运行三个训练 seed。

完整设计见 [v1.2 设计文档](docs/V1_2_DESIGN.md)，执行边界见 [v1.2 执行计划](docs/V1_2_EXECUTION_PLAN.md)。

## 当前结果：pilot 正确停止

服务器在运行提交 `7047d067bf464e1ffbb4896a7f27103471bdec3b` 上完成了 seed 42 的 dev pilot。归档、数据、配置、运行身份、预测行数、指标和候选择优均已独立复算一致；五个训练阶段全部完整结束，没有 OOM、NaN 或异常中止。

| 候选 | aligned | validity conflict | score conflict | score fresh | score nuisance | 精确格式 | pilot 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct DPO | 0.9414 | 0.5703 | 0.9062 | 0.8906 | 0.8750 | 1.0000 | validity 保持失败 |
| Score-aware SFT | 1.0000 | 0.4219 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 能力基线，不合格 |
| SFT → DPO | 1.0000 | 1.0000 | 1.0000 | 0.9844 | 1.0000 | 0.9544 | 精确格式失败 |
| Direct DPO，半学习率 | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | validity 保持失败 |
| SFT → DPO，半学习率 | 0.7500 | 0.5000 | 0.9922 | 0.9844 | 0.9922 | 1.0000 | aligned/validity 保持失败 |

保留门槛为 overall aligned ≥ 0.90、`validity_decisive` conflict ≥ 0.95、greedy exact-format ≥ 0.98。五个候选的 `eligible` 均为 `false`，`selected` 为 `null`，因此协议正确地没有生成 test，也没有进入 formal。

最接近成功的 `sft_dpo` 在条件 A/B 概率比较上同时学会了 score、validity 和 nuisance 不变性，但 1536 条贪心生成中有 70 条产生 `.getB`、`.getBcd` 等非 A/B 字符串，另有 2 条格式正确但答案错误。这个结果支持“核心比较能力可以学到”，但还不能报告 v1.2 正结果或正式结论。

完整审计与解释见 [v1.2 pilot 结果分析](docs/V1_2_PILOT_ANALYSIS.md)。本次结果归档为 `artifacts/v1.2/shortcut-repair-v1.2-pilot-7047d06.tar.gz`。

## 下一步边界

不要在 v1.2 上放宽 0.98 门槛，也不要继续打开 test。下一步应另立版本，先在现有 dev 权重上做一次不重训的最小诊断：比较单 token 与四 token 贪心输出，记录首 token 的 top-k、A/B 排名、A/B 后的 EOS 概率和 DPO completion tokenization。根据诊断只冻结一种修正，再重新运行 pilot；只有合格后才进入新 test 和三个 seed。

## 本地 CPU 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check src tests
python -m shortcut_repair.v12 --help
```

CPU 验证不加载模型，不能替代 GPU 训练或结果审计。

## 远程执行状态

远程操作指南见 [docs/V1_2_REMOTE_EXECUTION_GUIDE.md](docs/V1_2_REMOTE_EXECUTION_GUIDE.md)。本次已经完成 `prepare → pilot`，并触发预注册停止条件；当前不得继续执行 `freeze → formal → report`。

## 目录

```text
configs/v1_2.yaml                  # v1.2 冻结配置
src/shortcut_repair/v12.py         # 五阶段入口与停止逻辑
src/shortcut_repair/v12_data.py    # score-aware 与 hint-neutral 数据
src/shortcut_repair/v12_runtime.py # 训练、复用权重与 FP32 评测
src/shortcut_repair/v12_analysis.py # pilot 选型、五项检查与切片报告
scripts/run_v1_2.sh                # 远程阶段入口
tests/test_v12_*.py                # v1.2 CPU 合同测试
docs/V1_2_DESIGN.md                # 研究设计
docs/V1_2_EXECUTION_PLAN.md        # 执行计划
docs/V1_2_REMOTE_EXECUTION_GUIDE.md # 远程指南
docs/V1_2_PILOT_ANALYSIS.md        # 当前结果审计与解释
artifacts/v1.2/                    # 仅包含 v1.2 pilot 归档及校验值
```

v1.0 结果与文档只保留在 `main`，v1.1 结果与文档只保留在 `codex/v1.1-repair`。本分支保留被 v1.2 复用的底层实现、兼容配置和回归测试，但不复制历史版本的实验结果或文档。
