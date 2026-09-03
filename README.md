# ShortcutRepair-DPO v1.3

ShortcutRepair-DPO 是一个面向 AI 算法工程师面试的小型、受控后训练实验。项目先诱导
Qwen2.5-1.5B-Instruct 依赖可能过期的 `cached_recommendation`，再用反事实数据研究如何让
模型重新服从 fresh tool result，并用严格的开发集门槛和封存测试区分“学会规则”与“碰巧答对”。

项目不提出新的 DPO 损失，也不把合成二选一任务外推为通用工具调用能力。它的价值在于展示：
如何定义 shortcut、构造可证伪的数据干预、组合 SFT 与 DPO、定位训练目标和真实生成行为之间的
错位，并用最小实验修正它。

## 研究问题与方法

v1.2 的 `SFT → DPO` 已在 dev 上学会 fresh-score、validity 和 nuisance 不变性，但开放词表
贪心生成的严格 A/B 格式率只有 0.9544。v1.3 因而只回答：

> 能否在保留已学规则能力的同时，用一次低学习率 SFT continuation，让同一 DPO policy
> 恢复严格的 A/B+EOS 动作合同？

固定修正是：以 `is_trainable=True` 在原 DPO LoRA adapter 上继续训练，不创建第三个 adapter；anchor 数据与原 SFT
数据相同，learning rate 为 `2e-6`，训练 1 epoch、80 optimizer steps。评测保持四 token、
开放词表 greedy generation，不缩短生成长度，也不用约束解码代替模型修复。

完整的预注册假设、阈值与停止条件见 [v1.3 设计文档](docs/V1_3_DESIGN.md)，实现过程见
[v1.3 执行计划](docs/V1_3_EXECUTION_PLAN.md)。

## 当前结果：正式实验通过

v1.3 已完成真实 GPU pilot、三 seed 正式训练和 sealed test。Pilot 八项检查与正式五项检查
全部通过，正式结论为 `POSITIVE`。

| 模型 | overall aligned | score conflict | score fresh | validity conflict | score nuisance | 严格格式 |
|---|---:|---:|---:|---:|---:|---:|
| v1.1 Repair | 0.8979 | 0.3937 | 0.1896 | 0.9938 | 0.7833 | 1.0000 |
| Score-aware SFT | 1.0000 | 0.0000 | 0.0000 | 0.3375 | 1.0000 | 1.0000 |
| SFT → DPO（锚定前） | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 0.9979 | 0.9646 |
| SFT → DPO → Anchor | 0.9990 | 0.9979 | 0.9896 | 1.0000 | 0.9917 | 1.0000 |

最关键的结果不是单一最高分，而是分阶段对照形成的证据链：

- Score-aware SFT 能保持严格格式，但不能在 conflict 条件下摆脱 hint；
- DPO 学会了 fresh-score 与 validity 规则，却产生了开放词表格式错误；
- Format anchor 将三 seed 共 204 条非法格式降为 0，并把格式率从 0.9646 提升到 1.0000；
- 代价是 score fresh、score nuisance 等指标下降约 0.2–0.6 个百分点，因此这是近似无损而非零代价修复。

最终三 seed 的 5,760 条生成全部是严格 A/B，其中 11 条答案错误。相对 v1.1 Repair，score fresh
平均提高 `0.8000`，配对 case-bootstrap 95% CI 为 `[0.7500, 0.8479]`，三个 seed 均为正提升。

完整结果、逐 seed 分析和审计结论见 [v1.3 结果分析](docs/V1_3_RESULT_ANALYSIS.md)与
[自动生成的正式报告](reports/v1.3/RESULTS.md)。

## 结论边界

本实验支持以下有限结论：在当前受控二选一任务中，SFT → DPO 建立了 fresh-result 条件偏好，
一次冻结的低学习率 SFT continuation 恢复了开放词表 A/B+EOS 动作合同，并在三 seed sealed
test 上保留了核心能力。

本实验不能证明：

- 提出了新的 DPO 算法；
- 获得了通用数值推理或开放域工具调用能力；
- 格式锚定完全没有能力代价；
- 三个训练 seed 足以刻画全部训练随机性。

项目主实验现已冻结。不得根据 sealed test 继续调参或训练 v1.3；如补充分布外测试，应使用冻结
checkpoint，并明确标记为 post-hoc exploratory appendix。

## 验证与结果入口

本地 CPU 检查不会加载模型，也不能替代真实 GPU 结果，但可以验证代码合同：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check src tests
```

主要结果文件：

```text
reports/v1.3/pilot_decision.md       # Pilot 八项检查
reports/v1.3/RESULTS.md              # 正式结论和主指标
reports/v1.3/results.json            # 完整聚合结果
reports/v1.3/metrics.csv             # 模型、切片和逐 seed 指标
reports/v1.3/costs.csv               # 训练成本
artifacts/v1.3/                      # 不含模型权重的可审计结果包
```

结果包 `shortcut-repair-v1.3-0b36bf2f10a4.tar.gz` 的 SHA256 为：

```text
2918f007315323623fa941b10ba6d9da5065141e174c88293a042bd363742b82
```

历史版本的结果与说明分别保留在其远端分支；v1.3 只保留当前版本文档。底层 v1.0–v1.2
代码、配置和回归测试仍作为可追溯依赖保留，但不复制历史实验结果。
