# ShortcutRepair-DPO v1.3 正式结果

结论：`POSITIVE`

正式路径：`sft_dpo_anchor`。先执行 Score-aware SFT → DPO，再在同一 DPO adapter 上进行低学习率格式锚定 SFT。

## 同一 sealed test 上的比较

| 模型 | overall aligned | score conflict | score fresh response | validity conflict | score nuisance | 格式 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.7500 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Shortcut | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| v1.1 Control | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| v1.1 Repair | 0.8979 | 0.3937 | 0.1896 | 0.9938 | 0.7833 | 1.0000 |
| Score-aware SFT | 1.0000 | 0.0000 | 0.0000 | 0.3375 | 1.0000 | 1.0000 |
| SFT → DPO（锚定前） | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 0.9979 | 0.9646 |
| 选定 DPO | 0.9990 | 0.9979 | 0.9896 | 1.0000 | 0.9917 | 1.0000 |

## 五项预注册检查

- 通过：score fresh response ≥ 0.70，且每 seed 高于同 test 的 v1.1 Repair。
- 通过：score conflict accuracy ≥ 0.70。
- 通过：overall aligned ≥ 0.90 且 validity conflict ≥ 0.95。
- 通过：score nuisance invariance ≥ 0.95。
- 通过：greedy exact-format rate ≥ 0.98。

## 格式锚定前后差值

锚定后 − 锚定前：exact-format +0.0354，overall aligned -0.0010，score conflict -0.0021，score fresh response -0.0062，score nuisance -0.0063，validity conflict +0.0000。

该差值用于判断格式锚定是否损害规则能力，不改变五项正式成功标准。

## 不确定性和 SFT 对照

score fresh response 相对同 test 的 v1.1 Repair：Δ=0.8000，95% 配对 case-bootstrap CI=[0.75, 0.8479166666666667]。先对三个训练 seed 的配对差值取均值，再重采样 case；CI 不覆盖训练 seed 总体不确定性。

逐 seed 差值：`{42: 0.825, 43: 0.7937500000000001, 44: 0.78125}`。

选定 DPO 相对 Score-aware SFT 的 score fresh response 差值为 0.9896。DPO 达标不等于优于 SFT；差值为零或负数时不能声称 DPO 有额外收益。

完整的七项主指标、附加诊断、两种决策切片和逐 seed 数值见 metrics.csv/results.json。

## 计算成本与边界

训练分布为 75% score / 25% validity，dev/test 为 50/50。SFT 有明显分差 warm-up，DPO 只有普通分差；不是同数据、同计算量的纯损失函数消融。
SFT → DPO 包含两个训练阶段；v1.3 的格式锚定路径包含三个阶段，不能包装为与单阶段 DPO 或 SFT 等预算。这是受控合成二选一实验，不外推为生产工具调用或通用数值推理能力。

运行明细见 costs.csv。SFT 中间结果被复用，不重复计入总训练开销。

| 范围 | 阶段数 | 已记录阶段秒数 | 峰值显存 GiB |
|---|---:|---:|---:|
| pilot | 1 | 323.4 | 4.79 |
| formal | 9 | 4231.7 | 6.73 |

耗时包含加载、训练、保存和已捕获的失败尝试；强杀或断电造成的未记录耗时标为下界，不能当作精确总成本。峰值显存为已记录尝试的最大值。
