# v1.3 Pilot 判定

选定候选：`sft_dpo_anchor`。格式锚定通过全部固定门槛，且没有明显损害规则能力。

| 检查 | 判定 |
|---|---|
| aligned_retained | 通过 |
| validity_retained | 通过 |
| score_conflict_retained | 通过 |
| score_fresh_retained | 通过 |
| score_nuisance_retained | 通过 |
| exact_format | 通过 |
| format_gain | 通过 |
| no_core_regression | 通过 |

锚定后减锚定前的核心差值：

- `aligned_accuracy`：+0.000000
- `validity_conflict_accuracy`：+0.000000
- `score_conflict_accuracy`：+0.000000
- `score_fresh_response`：+0.000000
- `score_nuisance_invariance`：-0.007812
- `greedy_exact_format_rate`：+0.045573

仅使用 seed 42 和既有 dev；没有生成或读取 test。
没有候选搜索、自动补充轮或阈值调整。
