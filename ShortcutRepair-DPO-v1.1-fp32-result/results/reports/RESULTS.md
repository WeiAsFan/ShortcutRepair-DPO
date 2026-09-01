# ShortcutRepair-DPO 结果

**结论：NEGATIVE / INCONCLUSIVE**

| 指标 | Control | Repair | Repair - Control |
|---|---:|---:|---:|
| aligned accuracy | 1.0000 | 0.8989 | -0.1011 |
| conflict accuracy | 0.0000 | 0.6900 | +0.6900 |
| hint flip rate | 1.0000 | 0.2089 | -0.7911 |
| causal hint effect | 138.9198 | 14.5801 | -124.3398 |
| fresh-result response | 0.0000 | 0.5856 | +0.5856 |
| nuisance invariance | 1.0000 | 0.8900 | -0.1100 |
| greedy exact format | 1.0000 | 1.0000 | +0.0000 |

配对 case-bootstrap 的 conflict delta 95% CI：`[0.6411, 0.7389]`。

## 全部模型组

| 模型 | aligned | conflict | fresh response | nuisance invariance | exact format |
|---|---:|---:|---:|---:|---:|
| Base | 0.7500 | 0.7500 | 0.5000 | 1.0000 | 1.0000 |
| Shortcut SFT | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Aligned-only DPO | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Counterfactual DPO | 0.8989 | 0.6900 | 0.5856 | 0.8900 | 1.0000 |
| Counterfactual SFT | 0.7533 | 0.7522 | 0.5111 | 0.9944 | 1.0000 |


## 预注册检查

| 检查 | 结果 |
|---|---|
| all_seed_conflict_deltas_positive | PASS |
| conflict_delta_at_least_10pp | PASS |
| conflict_ci_lower_positive | PASS |
| hint_flip_halved | PASS |
| aligned_drop_within_2pp | FAIL |
| causal_hint_effect_reduced | PASS |
| fresh_result_response_high | FAIL |
| nuisance_invariance_high | FAIL |
| greedy_exact_format_high | PASS |

## 运行身份与评测修正

- 训练代码 Git：`1ead3b24f00f33569128a6634401729e4908a62f`
- 评测代码 Git：`91dc2b63fe3a07b417c27392e5dde2e8f39138d6`
- 评测修正协议：`shortcut-repair-v1.1-fp32-evaluation-amendment`
- 修正协议 SHA256：`8eb695c77a319800fc3fea4c0525388a38fbaff84ccc329f219404ab02ff1f6a`
- 评测前向精度：`float32`；
  TF32：`False`
- 平分策略：`reject_with_context`

训练产物来自冻结训练提交；最终指标由修正后的统一 FP32 协议重新评测全部模型得到。
此前的 BF16 部分评测不进入正式结论。


本基准只测量对受控诱导 stale-hint 依赖的修复，不能证明同一 shortcut 会自然出现在生产模型中。
