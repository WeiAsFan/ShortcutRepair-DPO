# ShortcutRepair-DPO v1.3 结果分析

> - 状态：正式实验完成，结论已冻结
> - 日期：2026-09-03
> - 实验代码提交：`0b36bf2f10a4dede3d0e02376e630bb4aa8abd63`
> - 结果上传提交：`df54e5d9a56fe590e697ef01769ad9d82ec43743`
> - 正式结论：`POSITIVE`
> - 设计依据：[V1_3_DESIGN.md](V1_3_DESIGN.md)

## 1. 结论摘要

v1.3 达到了预注册目标：一次固定的低学习率 Format-anchor SFT 将 `SFT → DPO` 的开放词表
严格格式率恢复到 1.0000，同时保留了 fresh-score、validity、aligned 和 nuisance 能力。Pilot
八项检查和正式五项检查全部通过，因而可以报告方法在当前受控任务上成功。

这不是完全无损的修复。格式锚定后三 seed 平均 score fresh 从 0.9958 降至 0.9896，score
nuisance 从 0.9979 降至 0.9917，score conflict 从 1.0000 降至 0.9979。正确表述应是“用
小幅、可量化的规则能力损失换取稳定的动作格式”，而不是“零代价修复”。

## 2. 结果身份与完整性

冻结记录绑定了以下实验身份：

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`；
- revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；
- 正式训练 seeds：42、43、44；
- train/dev 沿用 v1.2 数据，test 使用新 seed 13023；
- SFT、DPO、dev、test 分别为 2560、1920、1536、1920 行；
- test SHA256：`36901f0ad9c39e3d7fb4862a862fd505433297462b8cbc90a05e37392ca41dd2`；
- test 的 A/B、决策类型及常量、历史分数、展示顺序等朴素策略命中率均严格平衡。

对已上传产物的独立复算得到：

- 32,640 条正式 predictions 均能精确重算各自 `metrics.json`；
- 七组模型和所有 seed 使用同一组配对 test 样本；
- 重算后的模型聚合、五项检查、bootstrap 和锚定前后差值与 `results.json` 完全一致；
- 10 个 v1.3 训练 manifest 均为 `complete`，实际 optimizer steps 符合合同，loss 均为有限值；
- 结果包包含 88 个文件，包内训练/评测文件与仓库逐字节一致；
- 日志未发现 OOM、NaN、Traceback、失败运行或主机隐私路径。

结果包 `artifacts/v1.3/shortcut-repair-v1.3-0b36bf2f10a4.tar.gz` 的 SHA256 为：

```text
2918f007315323623fa941b10ba6d9da5065141e174c88293a042bd363742b82
```

## 3. Pilot 结果

Pilot 仅使用 seed 42 和既有 dev，先评测 v1.2 DPO adapter，再进行一次固定 anchor 训练并复评；
没有候选搜索、自动补充轮、阈值调整或 test 访问。

| 指标 | 锚定前 | 锚定后 | 差值 | 结果 |
|---|---:|---:|---:|---|
| overall aligned | 1.0000 | 1.0000 | +0.0000 | 保持 |
| validity conflict | 1.0000 | 1.0000 | +0.0000 | 保持 |
| score conflict | 1.0000 | 1.0000 | +0.0000 | 保持 |
| score fresh response | 0.9844 | 0.9844 | +0.0000 | 保持 |
| score nuisance | 1.0000 | 0.9922 | -0.0078 | 小幅下降，仍通过 |
| greedy exact-format | 0.9544 | 1.0000 | +0.0456 | 修复 |

原 1536 条生成中有 70 条 `.getBcd`、`.getB`、`.getBirds` 等非法输出；anchor 后非法格式为
0。原始行级复核显示，锚定后有 3 条格式正确但答案错误，因此格式恢复没有被误写成满分任务准确率。

## 4. 正式五项检查

| 正式检查 | 冻结门槛 | v1.3 结果 | 判定 |
|---|---|---:|---|
| score fresh response | 平均 ≥ 0.70，且每 seed 高于 v1.1 Repair | 0.9896 | 通过 |
| score conflict accuracy | 平均 ≥ 0.70 | 0.9979 | 通过 |
| capability retention | aligned ≥ 0.90；validity conflict ≥ 0.95 | 0.9990；1.0000 | 通过 |
| score nuisance invariance | 平均 ≥ 0.95 | 0.9917 | 通过 |
| greedy exact-format | 平均 ≥ 0.98 | 1.0000 | 通过 |

五项全部通过，因此自动报告中的 `POSITIVE` 判定正确。

## 5. 各阶段对照说明了什么

| 模型 | aligned | score conflict | score fresh | validity conflict | score nuisance | 严格格式 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.7500 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Shortcut | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| v1.1 Control | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| v1.1 Repair | 0.8979 | 0.3937 | 0.1896 | 0.9938 | 0.7833 | 1.0000 |
| Score-aware SFT | 1.0000 | 0.0000 | 0.0000 | 0.3375 | 1.0000 | 1.0000 |
| SFT → DPO（锚定前） | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 0.9979 | 0.9646 |
| SFT → DPO → Anchor | 0.9990 | 0.9979 | 0.9896 | 1.0000 | 0.9917 | 1.0000 |

这些对照共同支持以下机制解释：

1. Shortcut 与 aligned-only Control 几乎完全跟随 hint，无法处理 conflict；
2. v1.1 Repair 主要修复 validity，但没有可靠学会两个有效候选之间的 fresh-score 比较；
3. Score-aware SFT 单独保证了格式，却仍在 score conflict 上失败，因此 anchor 的成功不能解释为
   “SFT 本身解决了全部问题”；
4. SFT → DPO 建立了正确的条件 A/B 偏好，但条件候选排序不自动等于开放词表动作合同；
5. 最后的短 SFT 恢复了 A/B+EOS 动作合同，并保留了绝大部分 DPO 能力。

这是本项目最重要的技术结论：训练目标、条件候选评测与真实生成行为是三个相关但不等价的层次。

## 6. Seed 稳定性与剩余错误

| Seed | 锚定前格式率 | 锚定后格式率 | 锚定后 score fresh | 锚定后 score nuisance | 最终错误生成 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.9974 | 1.0000 | 1.0000 | 1.0000 | 0 / 1920 |
| 43 | 0.9052 | 1.0000 | 0.9875 | 0.9938 | 3 / 1920 |
| 44 | 0.9911 | 1.0000 | 0.9813 | 0.9813 | 8 / 1920 |

锚定前 seed 43 单独产生 182 条格式错误，说明格式问题具有明显训练随机性；锚定后三个 seed
全部为严格 A/B，因而 anchor 的主要收益不仅是提高平均值，也是消除格式上的 seed 不稳定性。

三 seed 锚定前共有 204 条非法格式，锚定后降为 0。最终 5760 条输出中仍有 11 条答案错误，
但均为合法 A/B；错误集中在 `score_decisive`，而 `validity_decisive` 保持满分。部分错误位于
fresh-score 差距较小的 case，说明格式问题已解决，剩余误差更接近数值比较边界，而不是输出协议错误。

## 7. 相对 v1.1 的增益与不确定性

v1.3 相对同一 test 上的 v1.1 Repair，score fresh response 平均提升 0.8000：

- seed 42：+0.8250；
- seed 43：+0.7938；
- seed 44：+0.7813；
- 配对 case-bootstrap 95% CI：`[0.7500, 0.8479]`。

该区间说明在当前固定 test cases 上的提升很稳健，但它只重采样 case，不覆盖训练 seed 总体
不确定性。正式训练只有三个 seed，因此不能把该 CI 解释成对所有随机初始化的总体置信区间。

## 8. 计算成本

| 范围 | 训练阶段 | 已记录耗时 | 峰值显存 |
|---|---:|---:|---:|
| Pilot | 1 | 323.4 秒 | 4.79 GiB |
| Formal | 9 | 4231.7 秒 | 6.73 GiB |

正式实验包括每个 seed 的 SFT、DPO、Anchor 三个阶段，不能包装成与单阶段 SFT 或 DPO 等预算
的比较。Anchor 每 seed 约增加 295 秒训练时间，这是修复动作合同的实际代价。

## 9. 结论边界

可以陈述：

> 在受控二选一任务中，SFT → DPO 建立了 fresh-score 与 validity 的条件偏好；一次冻结的
> 低学习率 SFT continuation 恢复了开放词表 A/B+EOS 动作合同，并在三 seed sealed test
> 上以小幅规则能力代价保持了核心能力。

不能陈述：

- 提出了新的 DPO 损失或通用后训练算法；
- 证明了开放域工具调用或通用数值推理能力；
- Anchor 是零代价修复；
- 当前同分布合成 test 足以证明 schema、数值范围或任务形式变化后的泛化；
- case-bootstrap 已覆盖训练 seed 不确定性。

## 10. 下一步

v1.3 主实验到此冻结，不得根据已查看的 sealed test 继续训练或调参。接下来优先完成：

1. 将本结果分析、README 和设计/计划状态一起冻结；
2. 基于已冻结事实准备简历描述、90 秒介绍、5 分钟技术叙事和高频追问回答；
3. 如时间充足，可用冻结 checkpoint 做一次小型 OOD challenge set 评测，但必须标记为
   `post-hoc exploratory`，不得据此改写正式 `POSITIVE` 结论或继续调参。

不建议启动 v1.4，也不建议增加 PPO/GRPO、服务部署或超参数网格搜索。对面试目标而言，当前
“发现聚合指标误导 → 增加机制切片 → 发现条件偏好与生成错位 → 最小修复 → 三 seed sealed test
验证”的完整实验闭环，比继续追逐最后 11 条错误更有价值。
