# ShortcutRepair-DPO `v1.1` 冻结实验协议

> - 状态：训练协议已冻结且正式训练完成；评测结论等待统一 FP32 重评
> - 冻结日期：2026-08-31
> - 评测增补日期：2026-09-01
> - 配置文件：`configs/experiment.yaml`
> - 配置 SHA256：`56da1d3c5f8df8512ea72e458e03755e854cf78abc533c02c3b86b4d28e85ca6`
> - 数据生成器：`shortcut-repair-v2`
> - 机器可读评测增补：`configs/evaluation_amendment.yaml`

## 1. 研究问题与边界

在一个已经通过因果干预确认会依赖 stale cached hint 的小模型上，加入同一新鲜工具观察的 counterfactual preference，是否比等数据规模、等 optimizer-step 预算的 aligned-only preference 更能恢复模型对 fresh tool result 的服从？Counterfactual SFT 能否取得相同效果？

本协议研究受控合成任务中的模型行为修复，不研究 shortcut 的自然发生率，不提出新的 DPO 算法，也不把单一模型和单一任务的结果外推到生产系统。

`v1.0` 的 185/190 step 失败运行只作为工程复盘证据。`v1.1` 修改了训练预算、数据语义和评测，因此不得从旧 checkpoint 恢复，也不得把两个版本的指标直接合并。

## 2. 数据合同

### 2.1 Oracle

模型必须先丢弃 `is_valid=false` 的候选，再从剩余候选中选择 `fresh_score` 最高者。`cached_recommendation`、`historical_score` 和 `display_rank` 都不属于 oracle。

每个 split 的底层 case 严格包含各 50% 的两种决策：

| 类型 | 构造 | 检查能力 |
|---|---|---|
| `score_decisive` | 两个候选都有效，gold 的 fresh score 更高 | 是否比较新鲜分数 |
| `validity_decisive` | gold 有效但分数更低，wrong 无效但分数更高 | 是否先过滤无效候选 |

A/B gold 各占 50%。`historical_score` 和 `display_rank` 与 gold 平衡；case/request ID 为不含 split 明文的确定性哈希。

### 2.2 自动审计

每个 split 在写盘前必须满足：

```text
gold_A_fraction = 0.50
score_decisive_fraction = 0.50
validity_decisive_fraction = 0.50
fresh_score_only_accuracy = 0.50
historical_only_accuracy <= 0.55
display_rank_only_accuracy <= 0.55
constant_A_accuracy = 0.50
constant_B_accuracy = 0.50
split_marker_count = 0
case_id_unique_across_cases = true
request_id_unique_across_cases = true
```

train/dev 的 request ID 还必须跨 split 互斥，Aligned-only DPO、Counterfactual DPO 和 Counterfactual SFT 必须使用相同的底层 DPO case multiset。

### 2.3 数据规模

| 用途 | 底层 case | 每 case 行数 | 总行数 |
|---|---:|---:|---:|
| Shortcut induction | 600 | 2 | 1,200 |
| Aligned-only DPO | 600 | 2 | 1,200 |
| Counterfactual DPO | 600 | 2 | 1,200 |
| Counterfactual SFT | 600 | 2 | 1,200 |
| Dev 因果评测 | 200 | 6 | 1,200 |
| Sealed test 因果评测 | 300 | 6 | 1,800 |

test 只有在 Shortcut mechanism gate 通过后才能生成；其 seed 固定为 9404，生成后由配置和文件 SHA256 封存。

## 3. 模型组与训练合同

基座固定为 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。所有后训练 adapter 使用相同 rank-16 LoRA：`alpha=32`、`dropout=0.05`，目标模块为 q/k/v/o projection 及 gate/up/down projection。

| 组别 | 起点 | 数据/目标 | LR | epochs | optimizer steps | seeds |
|---|---|---|---:|---:|---:|---|
| Base | 固定 Qwen | 不训练 | — | — | — | — |
| Shortcut SFT | Base | target=cached hint | 2e-4 | 1 | 38 | 1337 |
| Aligned-only DPO | merged Shortcut | 只含 aligned preference | 1e-5 | 3 | 114 | 42/43/44 |
| Counterfactual DPO | merged Shortcut | aligned + conflict preference | 1e-5 | 3 | 114 | 42/43/44 |
| Counterfactual SFT | merged Shortcut | aligned + conflict，target=gold | 1e-5 | 3 | 114 | 42/43/44 |

所有训练均使用 micro batch 4、gradient accumulation 8、effective batch 32、BF16 和显式 `max_steps`。DPO 的 `beta=0.1`、loss 为 sigmoid。Counterfactual SFT 与 DPO 匹配 case、行数、seed 和 optimizer steps，但 DPO 前向计算更多，因此只报告实际耗时和峰值显存，不声称 FLOPs 相等。

同一 DPO seed 的 control/repair 新 LoRA 初始 checksum 必须一致；全部正式运行必须从同一个 merged Shortcut 权重开始。任何 Git、config、data、阶段、模型来源或 Trainer 预算不一致的 checkpoint 都禁止恢复。

## 4. Shortcut 机制门控

先在同一 dev 上评估 Base，再评估 Shortcut SFT，并记录 Shortcut 相对 Base 的 hint flip rate 与 causal hint effect 变化。正式 gate 对 Shortcut 同时要求：

```text
aligned accuracy >= 0.80
conflict accuracy <= 0.20
hint flip rate >= 0.80
causal hint effect >= 1.0 nat
```

Gate 失败时立即停止，不生成 test、不训练 DPO/SFT baseline。该结果只能说明 induction 未建立或训练/数据设计不足，不能解释为 Repair 方法无效。

## 5. 三类因果干预

每个 dev/test case 生成三组配对输入，每组只操纵指定信息：

1. `hint_flip`：fresh fields 和 nuisance 不变，cached hint 从 gold 翻为 wrong。Shortcut 应随 hint 改变，Repair 应尽量保持正确。
2. `fresh_flip`：cached hint 和 nuisance 不变，交换 authoritative fresh validity/score，使 oracle gold 翻转。Repair 应随新鲜结果改变。
3. `nuisance_flip`：fresh validity/score、cached hint 和 gold 不变，只交换 historical score/display rank。预测应保持不变。

评测同时运行 teacher-forced `log P(A)`/`log P(B)` 和 greedy generation。根据第 9 节的透明协议增补，模型必须以 FP32 加载并完成前向、关闭 TF32，再以 FP32 logits 计算 log-softmax；greedy generation 固定 `max_new_tokens=4`，去除首尾空白后只有单个 `A` 或 `B` 才算格式正确。

## 6. 指标

核心指标包括：

- aligned/conflict accuracy；
- pair-both accuracy；
- hint flip rate、hint follow rate 和 causal hint effect；
- aligned/conflict correct margin；
- fresh-result response rate 和 fresh flip rate；
- nuisance invariance rate 和 nuisance pair-both accuracy；
- greedy exact-format rate 和 greedy accuracy。

其中 fresh-result response 只有在 fresh 干预前后都预测各自 gold 时才记为成功；nuisance invariance 只检查预测是否保持，需与正确率一起解释，不能把恒定错误当成能力。

## 7. 正式成功标准

Counterfactual DPO 相对 Aligned-only DPO 必须同时通过九项检查才能报告 `POSITIVE`：

1. 三个 seed 的 conflict accuracy delta 全为正；
2. 平均 conflict accuracy 至少提高 10 个百分点；
3. paired case-bootstrap 95% CI 下界大于 0；
4. Repair hint flip rate 不高于 Control 的 50%；
5. Repair aligned accuracy 下降不超过 2 个百分点；
6. Repair causal hint effect 低于 Control；
7. Repair fresh-result response rate 至少 0.80；
8. Repair nuisance invariance rate 至少 0.95；
9. Repair greedy exact-format rate 至少 0.98。

统计使用 paired case-bootstrap：按共享 test case 重采样，先在三个训练 seed 上平均 Repair-Control 的 conflict-correct 差值，再计算 10,000 次 bootstrap，seed 固定为 20260828。该区间主要反映固定三个训练 seed 下的 case 不确定性，不代表充分估计了训练随机性。

Counterfactual SFT 是预注册次要基线。若它与 Counterfactual DPO 相当或更好，结论应聚焦反事实冲突数据的价值，不能声称 DPO 优于 SFT。

## 8. 报告规则

九项检查全部通过才写 `POSITIVE`。其他情况统一写 `NEGATIVE / INCONCLUSIVE`，列出所有失败检查并保留三个 seed；不得在 sealed test 上修改阈值、生成器、配置、seed 或删除不利样本。

允许的简历表述：

> 构建了一个受控 stale-tool-hint 修复基准：先以 Base/Shortcut 对照和因果干预确认错误缓存依赖，再从同一 checkpoint 比较 aligned-only DPO、counterfactual DPO 与 counterfactual SFT；使用三类配对干预、显式训练合同、三 seed 和预注册 bootstrap 门槛评估修复效果。

禁止表述为“提出了新的 DPO 算法”“证明适用于所有工具调用任务”或在正式聚合完成前声称 Repair 有效。

## 9. 2026-09-01 评测协议增补

第一次正式评测在 `repair/seed-42` 因 A/B 条件对数概率完全相等而终止。复盘发现实现以 BF16 完成模型前向，只在输出 logits 后转为 FP32；这不足以满足可判别的高精度评测意图。失败前只得到 Base、Shortcut 和 Control-42 的部分结果，没有得到完整 Repair 结果或正式聚合结论。

本增补在保留配置、sealed test、九个训练产物、指标和阈值的前提下，冻结以下修正：

1. Base、Shortcut、六个 DPO adapter 和三个 Counterfactual SFT adapter 全部统一用 FP32 前向重评；
2. CUDA matmul 与 cuDNN 的 TF32 均关闭；
3. 完全平分继续报错，并记录 case、干预类型和精确分数，不引入任意 tie-break；
4. 所有旧 BF16 dev/test 指标退出正式结果，不能与新指标混用；
5. 修正协议绑定原 dev 与 sealed test SHA256；原训练 manifest 继续绑定训练提交 `1ead3b24f00f33569128a6634401729e4908a62f`，新 prediction manifest 另行绑定评测提交和修正协议 SHA256；
6. 只重新执行 `gate → evaluate → aggregate`，不重训、不重生成 test。

这是一项在查看部分结果后的协议修正，最终报告必须如实披露，不能称为完全事前注册。它仍具有可解释性，因为修正是模型组无关的数值实现修复，统一作用于全部模型，且没有根据 Repair 效果改变数据、阈值或统计方法。完整原因、恢复步骤和停止条件见 [V1_1_EVALUATION_AMENDMENT.md](V1_1_EVALUATION_AMENDMENT.md)。
