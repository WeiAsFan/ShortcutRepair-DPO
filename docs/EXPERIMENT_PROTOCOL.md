# ShortcutRepair-DPO 冻结实验协议

## 研究问题

在一个已经通过因果干预确认会依赖 stale cached hint 的小模型上，加入同一新鲜工具观察的 counterfactual hint-flip 偏好数据，是否比等预算的 aligned-only 偏好数据更能恢复模型对 fresh tool result 的服从？

本协议研究受控模型修复，不研究 shortcut 的自然发生率。

## 三阶段设计

### 1. Shortcut induction

基座为固定 revision 的 Qwen2.5-1.5B-Instruct。600 个底层 case 各生成 hint=A 和 hint=B 两个版本，SFT target 始终等于 hint，因此 1,200 行中一半与工具 oracle 冲突。LoRA SFT 训练 5 epochs 后合并为唯一 shortcut checkpoint。

### 2. 机制门控

Dev 有 200 个未见 case，每个 case 保持 fresh tool result 不变，只把 hint 从 gold 翻成 wrong。用 teacher-forced `log P(A)` 与 `log P(B)` 判断模型是否使用 hint。

必须同时满足：

- aligned accuracy >= 0.80；
- conflict accuracy <= 0.20；
- hint flip rate >= 0.80；
- causal hint effect >= 1.0 nat。

Gate 失败时停止，不能生成 test，也不能把后续无差异解释成修复失败。

### 3. 等预算 DPO 修复

| 条件 | 数据 | 行数 | epochs | effective batch | seeds |
|---|---|---:|---:|---:|---|
| Aligned-only DPO | aligned prompt 每 case 重复两次 | 1,200 | 3 | 32 | 42/43/44 |
| Counterfactual Repair DPO | 同一 case 的 aligned + conflict prompt | 1,200 | 3 | 32 | 42/43/44 |

两组都从同一个 merged shortcut checkpoint 创建新的 rank-16 LoRA。对同一 seed，初始 adapter checksum 必须一致。DPO reference 是关闭新 adapter 后的 merged shortcut policy。

## Test 与指标

Gate 通过后才用独立 seed 生成 300 个 sealed test case，每个 case 有 aligned/conflict 两行。正式指标为：

- aligned accuracy；
- conflict accuracy；
- pair-both accuracy；
- hint flip rate；
- causal hint effect；
- correct log-probability margin。

统计使用 paired bootstrap：按共享 case 重采样，在三个 seed 上平均 Repair-Control 的 conflict-correct 差值，共 10,000 次，固定 seed 20260828。

## 成功与失败

正结果要求：

1. 三个 seed 的 conflict accuracy delta 都为正；
2. 平均 conflict accuracy 提升至少 10 个百分点；
3. paired bootstrap 95% CI 下界大于 0；
4. Repair hint flip rate 不高于 Control 的 50%；
5. Repair aligned accuracy 下降不超过 2 个百分点；
6. Repair causal hint effect 低于 Control。

六项全部通过才写 `POSITIVE`。其他情况统一写 `NEGATIVE / INCONCLUSIVE` 并列出失败检查，不允许在 test 上改阈值或删 seed。

## 允许的简历表述

> 构建了一个受控 stale-tool-hint 模型修复基准：先以因果 hint-flip 干预确认模型依赖错误缓存，再从相同 checkpoint 比较 aligned-only 与 counterfactual DPO；用成对 log-prob 指标、三 seed 和预注册 bootstrap 门槛评估修复效果。

不得表述为“发明了新的强化学习算法”或“证明该方法适用于所有工具调用任务”。
