# ShortcutRepair-DPO v1.3 设计

> - 状态：设计冻结，尚未运行真实 GPU pilot
> - 日期：2026-09-03
> - 基线：v1.2 dev pilot，运行提交 `7047d067bf464e1ffbb4896a7f27103471bdec3b`
> - 项目边界：面试导向的小型受控后训练实验

## 1. 研究问题

v1.2 的标准 SFT → DPO 路径在 seed 42 dev 上已经达到：overall aligned=1、`validity_decisive` conflict=1、`score_decisive` conflict=1、score fresh response=0.9844、score nuisance invariance=1。它没有进入正式实验，唯一硬门槛失败是开放词表 greedy exact-format=0.9544，低于预注册的 0.98。

1536 条生成中有 70 条不是严格 A/B，主要为 `.getBcd`、`.getB`、`.getBirds` 和 `.getBussinesCard`；另有 2 条严格 A/B 但答案错误。条件 A/B 排名与实际开放词表动作发生了分离。

v1.3 只回答一个问题：

> 能否在不牺牲 v1.2 已学到的 fresh-score、validity 和 nuisance 不变性的前提下，让同一 DPO policy 在开放词表贪心生成中稳定输出严格的 A 或 B？

## 2. 不采用的做法

v1.3 不做以下调整：

- 不把 greedy exact-format 门槛从 0.98 下调；
- 不把条件 A/B log-prob 决策重新命名为开放词表生成成功；
- 不把 `max_new_tokens` 改成 1 来隐藏第三条生成路径；
- 不在 pilot 中搜索 learning rate、epoch、beta 或数据配比；
- 不生成或查看 test 后再决定修正方案；
- 不增加 reward model、PPO/GRPO、Web 服务或大规模网格搜索。

受约束 A/B decoding 是固定二选一路由系统中合理的部署选项，但它会让精确格式通过变成接口保证，不能回答模型自身是否恢复了开放词表动作合同。因此它只保留为讨论项，不进入 v1.3 主实验。

## 3. 根因假设

v1.2 的条件评测只比较强制 completion A 与 B 的 log-prob；DPO 也只优化 chosen A/B 相对 rejected B/A 的序列偏好。二者都不直接保证 A/B 是整个词表中的贪心最优路径，也不单独验证答案后的 EOS。

SFT 训练不同：项目的 SFT tokenizer 明确把正确 A/B 与 EOS 一起作为监督 target。v1.2 的 Score-aware SFT 单独不足以修复 shortcut，但它保持了 1.0 的精确格式率。因此 v1.3 的假设是：

> 在已经学会规则的 SFT → DPO adapter 上，用完全正确、hint-neutral/aligned/conflict 都由 Oracle 标注的训练行做一次低学习率 SFT continuation，可以重新锚定 A/B+EOS 的开放词表动作，同时保留 DPO 已建立的规则排序。

这不是新的算法，而是一个受控的目标对齐修正：DPO 负责建立冲突条件下的相对偏好，最后的短 SFT 负责恢复动作空间与终止合同。

## 4. 唯一实验变量

Pilot 直接复用服务器上的 v1.2 seed-42 近成功路径：

```text
v1.2 Score-aware SFT merged
            +
v1.2 SFT → DPO final adapter
            │
            ├── 锚定前 FP32 dev 评测
            │
            ▼
同一 adapter 上继续 Format-anchor SFT
            │
            └── 锚定后 FP32 dev 评测 → pilot 判定
```

Format-anchor SFT 固定为：

- 数据：与 v1.2 `sft.jsonl` 逐字节一致的 2560 行；
- target：Oracle gold A/B，并显式包含 EOS；
- 起点：v1.2 `sft_dpo` final adapter，挂载在其原始 SFT merged 模型上；
- learning rate：`2e-6`；
- epochs：1；
- effective batch size：32；
- optimizer steps：80；
- LoRA：继续训练原 DPO adapter，不叠加第三个 adapter，不合并或覆盖 v1.2 权重。

选择 `2e-6` 是冻结的单点方案：它是原 SFT/DPO learning rate `1e-5` 的五分之一，目标是修复约 4.6% 的动作格式缺口，而不是重新学习规则。Pilot 失败后不得在 v1.3 内补跑其他学习率。

## 5. 数据与版本身份

v1.3 不改变训练任务或 dev：

- train/dev seed、case、SFT、DPO 和 dev 内容与 v1.2 完全相同；
- prepare 必须验证三份文件与 v1.2 已公开哈希一致；
- `anchor.jsonl` 必须与 `sft.jsonl` 逐字节相同；
- v1.2 SFT/DPO run manifest SHA、起点关系和模型产物必须匹配；
- v1.3 使用独立的 `data/runs/results/reports/artifacts/v1.3` 路径，不覆盖 v1.2。

v1.2 从未生成 test。v1.3 只有 pilot 通过后才以新 seed `13023` 生成独立 sealed test；test case 在 train/dev 中不得出现。

## 6. Pilot 判定

Pilot 只运行 seed 42 的一次 format-anchor SFT，没有自动补充轮。锚定后必须同时满足：

1. overall aligned accuracy ≥ 0.90；
2. `validity_decisive` conflict accuracy ≥ 0.95；
3. `score_decisive` conflict accuracy ≥ 0.95；
4. `score_decisive` fresh-result response ≥ 0.95；
5. `score_decisive` nuisance invariance ≥ 0.95；
6. greedy exact-format rate ≥ 0.98；
7. exact-format 相对锚定前至少提高 0.02；
8. aligned、validity conflict、score conflict、score fresh 和 score nuisance 中任一项相对锚定前下降不得超过 0.02。

全部通过才令 `selected=sft_dpo_anchor`。否则 `selected=null`，停止，不生成 test。固定阈值比 v1.2 的最低保留门槛更严格，因为 v1.3 的任务不是重新选模型，而是证明格式修正没有破坏已获得的规则能力。

## 7. 正式实验

Pilot 通过后，seeds 42/43/44 各从相同 Shortcut checkpoint 完整训练：

1. Score-aware SFT；
2. SFT → DPO；
3. 在该 DPO adapter 上继续一次 Format-anchor SFT。

三个 seed 的全部九个训练阶段结束后，才统一打开一次 sealed test。正式评测包括：

- Base、Shortcut；
- v1.1 Control、v1.1 Repair，各三个 seed；
- v1.3 Score-aware SFT、锚定前 SFT → DPO、锚定后 SFT → DPO，各三个 seed。

锚定前模型是必要消融，用于回答格式锚定带来了什么、是否损害规则能力。它不参与模型选择。

## 8. 正式成功标准

锚定后模型沿用 v1.2 的五项正式标准：

1. 三 seed 平均 score fresh response ≥ 0.70，且每个 seed 都高于同一 test 上的 v1.1 Repair；
2. 平均 score conflict accuracy ≥ 0.70；
3. overall aligned ≥ 0.90 且 validity conflict ≥ 0.95；
4. score nuisance invariance ≥ 0.95；
5. greedy exact-format ≥ 0.98。

报告必须额外给出锚定后减锚定前的全部核心指标差值。即使五项通过，如果格式改善伴随明显规则回退，也不能把锚定描述成无代价修复。

## 9. 可证伪结论与面试表述

允许的正结论是：

> 在受控二选一任务中，SFT → DPO 建立了 fresh-score 与 validity 的条件偏好；一次冻结的低学习率 SFT continuation 恢复了开放词表 A/B+EOS 动作合同，并在三 seed sealed test 上保持了核心能力。

允许的负结论包括：

- 格式仍低于 0.98：锚定信号不足，根因不只是轻微 SFT 遗忘；
- 格式恢复但规则回退：串行 SFT 存在明显 objective interference；
- pilot 通过但正式 seed 不稳定：方法只在开发 seed 上偶然成立。

不得声称提出新损失、证明开放域工具调用、或把受约束合成任务结果外推到生产系统。
