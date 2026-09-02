# ShortcutRepair-DPO v1.2 pilot 结果分析

> - 分析日期：2026-09-02
> - 结果阶段：dev pilot
> - 运行代码提交：`7047d067bf464e1ffbb4896a7f27103471bdec3b`
> - 结果归档：`artifacts/v1.2/shortcut-repair-v1.2-pilot-7047d06.tar.gz`
> - 归档 SHA256：`0fc73c6e1852086b6b60501427ac78ca6f32e32210daf26860611281dfaa54d6`
> - 协议判定：`STOP / NO FORMAL`

## 1. 结论

这次运行没有发生训练程序故障。数据准备、Shortcut sanity、五条候选训练、FP32 dev 评测和 pilot 选择均完整结束，记录的指标可以由原始预测逐项复算。

pilot 本身未通过：没有任何一条 DPO 路径同时满足 aligned、validity 和精确输出格式三项能力保持门槛，因此 `selected=null`。流水线按预注册协议停止，没有生成 test，也没有执行三个 seed 的正式实验。

科学上得到的是一个有意义的负 pilot，而不是正式负结果：标准学习率的 SFT → DPO 已经在条件 A/B 比较上同时学会 fresh score、validity 和 nuisance 不变性，但自由贪心生成的精确格式率只有 0.9544。核心推理信号已经出现，开放词表生成与格式合同尚未成立，所以不能报告 v1.2 正结果。

## 2. 证据完整性审计

独立审计对归档中的原始文件重新计算了以下内容，全部一致：

- 归档 SHA256 与伴随校验文件一致；
- 配置 SHA256 为 `ec4b234b3e2372c7faa6b5dc0ec11a0ed4e1db3f697648a56e15ffd01623c344`；
- 运行、预测和 pilot decision 中的 Git SHA 均指向 `7047d067...`；
- `sft.jsonl` 2560 行、`dpo.jsonl` 1920 行、`dev.jsonl` 1536 行，行数和 SHA256 均与 prepare manifest 一致；
- 五组预测各为 1536 行，metrics 可由 `predictions.jsonl` 完整复算；
- prediction manifest 记录的 run-manifest SHA256 与实际文件一致；
- 五个 run 均为 `status=complete`，optimizer step 达到合同值，loss 有限；
- 用冻结的 `select_pilot` 重新选择仍得到五项 `eligible=false` 和 `selected=null`；
- 归档中不存在 `test.jsonl` 或 freeze manifest，说明 test 没有被提前打开。

因此，这次未入选不是哈希误配、断点恢复混淆、指标聚合错误或 test 泄漏造成的。

## 3. 数据与 Shortcut 前提

训练数据包含 640 个底层 case，其中 75% 为 `score_decisive`、25% 为 `validity_decisive`；dev 包含 256 个底层 case，两种决策各占 50%。A/B gold、historical score 和 display rank 方向均精确平衡，case/request ID 唯一，train/dev case 不重叠。

原 Shortcut 模型在新 dev 上得到：

| 检查 | 数值 | 门槛 | 判定 |
|---|---:|---:|---|
| aligned accuracy | 1.0000 | ≥ 0.80 | 通过 |
| conflict accuracy | 0.0000 | ≤ 0.20 | 通过 |
| hint flip rate | 1.0000 | ≥ 0.80 | 通过 |
| causal hint effect | 30.1201 | ≥ 1.00 | 通过 |

两个 `decision_type` 切片都呈现同样的 shortcut 机制。v1.2 面对的是一个确实存在、且在新数据上可测的 stale-hint 依赖问题。

## 4. 训练运行状态

| 候选 | 阶段 | optimizer step | loss | 用时（秒） | 峰值显存（GiB） |
|---|---:|---:|---:|---:|---:|
| `direct_dpo` | 1 | 180/180 | 4.05 | 775.8 | 6.73 |
| `score_sft` | 1 | 80/80 | 0.92 | 271.6 | 4.79 |
| `sft_dpo` 的 DPO 阶段 | 2 | 180/180 | 2.51 | 790.1 | 6.73 |
| `direct_dpo_lr_half` | 1 | 180/180 | 5.25 | 810.0 | 6.73 |
| `sft_dpo_lr_half` 的 DPO 阶段 | 2 | 180/180 | 4.07 | 779.4 | 6.73 |

总记录训练时间为 3426.9 秒，约 57.1 分钟。SFT 中间模型被两条链式路径复用，没有重复训练。日志中没有 traceback、OOM、NaN 或 Inf；问题属于模型行为和选型门槛，而不是运行稳定性。

## 5. pilot 指标

pilot 的硬保留条件为：

1. overall aligned accuracy ≥ 0.90；
2. `validity_decisive` conflict accuracy ≥ 0.95；
3. greedy exact-format rate ≥ 0.98。

只有先通过三项硬条件的 DPO 才能按 `score_decisive` fresh response、nuisance invariance 和阶段数排序。

| 候选 | aligned | validity conflict | score conflict | score fresh | score nuisance | 精确格式 | 不合格原因 |
|---|---:|---:|---:|---:|---:|---:|---|
| `direct_dpo` | 0.9414 | 0.5703 | 0.9062 | 0.8906 | 0.8750 | 1.0000 | validity 丢失；nuisance 也低于正式目标 |
| `score_sft` | 1.0000 | 0.4219 | 0.0000 | 0.0000 | 1.0000 | 未学会 score；且只是 SFT 基线 |
| `sft_dpo` | 1.0000 | 1.0000 | 1.0000 | 0.9844 | 1.0000 | 精确格式 0.9544 < 0.98 |
| `direct_dpo_lr_half` | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 未学会 score，validity 不足 |
| `sft_dpo_lr_half` | 0.7500 | 0.5000 | 0.9922 | 0.9844 | 0.9922 | aligned 和 validity 丢失 |

### 5.1 Direct DPO

标准学习率 Direct DPO 确实获得了较强的 score 比较信号，但 validity conflict 只有 0.5703，score nuisance 也只有 0.8750。它把模型从“跟随 shortcut”推向了另一种不完整策略，没有同时保留过滤规则。

学习率减半后，aligned 和格式完全保留，但 score conflict/fresh 都降到 0。这个对照说明 Direct DPO 的改善不是稳定的统一规则学习。

### 5.2 Score-aware SFT

单独的短 SFT 保持了 aligned 和输出格式，却没有建立 score conflict 能力，validity conflict 也只有 0.4219。它不能单独完成修复；但作为 SFT → DPO 的中间起点，它明显改变了后续 DPO 的可学习性。

### 5.3 SFT → DPO

标准学习率链式路径在 teacher-forced A/B 条件评分上达到：aligned=1、validity conflict=1、score conflict=1、score fresh=0.9844、score nuisance=1，且 causal hint effect 接近 0。它是 v1.2 研究假设获得局部支持的核心证据。

但同一模型的贪心生成并未满足接口合同：

| 异常 | 数量 |
|---|---:|
| 非严格 A/B 输出 | 70 / 1536（4.5573%） |
| 其中 `score_decisive` | 48 |
| 其中 `validity_decisive` | 22 |
| 格式正确但答案错误 | 2 |

70 条非严格输出的文本分布为：`.getBcd` 41 条、`.getB` 20 条、`.getBirds` 7 条、`.getBussinesCard` 2 条。这不是空白字符归一化问题；模型没有先输出严格的 `B` 再附加后缀，而是在答案位置进入了包含 B 字符的代码样式生成路径。

学习率减半的链式路径恢复了精确格式，却把 aligned 降到 0.75、validity conflict 降到 0.50。它表明当前方法对学习率敏感，不能把标准学习率的近成功包装成稳健结果。

## 6. 已确认事实与根因假设

已确认的是：

- 条件 A/B log-prob 决策与自由贪心生成之间存在明确分离；
- `sft_dpo` 已学到对正确 A/B 的大幅条件 margin，但约 4.6% 的开放词表生成没有选择严格的单字符答案路径；
- 失败集中为包含 B 字符的代码样式片段，不是空白差异或随机乱码；
- 现有归档足以定位症状，但不包含权重，不能在本机继续做 token 级诊断。

尚未确认的是根因。当前最强的候选解释是训练目标与评测接口没有完全对齐：DPO 直接比较 chosen `A`/`B` 与 rejected `B`/`A` 的相对序列概率；当前条件指标也只累加强制 A/B completion token 的 log-prob，不比较第三条路径或答案后的 EOS。二者都不直接保证 A/B 是开放词表中的贪心最优路径。`.get...` 可能在首 token 上胜出，即使正确答案相对错误 A/B 已有很大 margin；EOS/停止行为可能是第二层问题。

代码中 SFT target 显式附加 EOS，而 DPO 路径把裸 `A`/`B` completion 交给固定版本的 TRL。第三路径是否真的在首 token 胜出、TRL 是否附加 EOS、A/B 各自如何分词、答案后的 EOS 概率，都需要在服务器现有权重上直接记录，不能仅凭结果反推。

## 7. 对项目方向的判断

v1.2 的方向仍然合适，并且这次结果是有意义的：decision-type 拆分成功把“会过滤无效候选”和“会比较 fresh score”分开，候选对照也揭示了 SFT warm-up、DPO 和学习率之间的不同作用。尤其是标准 SFT → DPO 的结果证明，目标规则并非当前模型和数据无法学习。

但 v1.2 尚不能报告正式成功，也不能声称方法稳健优于 v1.1：当前只有 seed 42 的 dev pilot，没有 sealed test、三个 seed、正式基线重评或五项正式检查。最准确的表述是：

> v1.2 在 dev pilot 上找到了能同时恢复 score 与 validity 条件决策的链式路径，但其自由生成格式未达到预注册门槛，因此按协议停止，正式效果仍未验证。

这反而能形成清晰的面试叙事：先用切片定位 v1.1 的机制缺口，再用受控消融找到近成功路径，同时用 exact-format 揭示“二选一条件排名”与“开放词表实际动作”之间的差异，并拒绝为了得到正结果而放宽门槛。需要避免把 1.0 的条件指标说成最终任务准确率。

## 8. 下一步

不要修改 v1.2 的阈值，也不要打开 v1.2 test。下一版本只做以下最小闭环：

1. 在服务器保留的 `sft_dpo` dev 权重上做只读诊断：记录首 token top-k、A/B 的 token ID/排名/概率、单 token 与四 token 贪心结果、A/B 后 EOS 概率，以及 DPO chosen/rejected 的实际 token 序列；不训练、不看 test。
2. 若第三条首 token 路径压过 A/B，需要在新协议中二选一：增加最小的格式保持训练信号，使 A/B 在开放词表中胜出；或把任务明确建模为固定二选一路由接口，采用受约束 A/B decoding。若首 token 已是 A/B 而只在后续失控，则修正 completion/EOS 合同。三种情形不能混做一轮，也不能追溯改写 v1.2 指标。
3. 新版本只冻结一种修正，用同一 seed 42 和 dev 重新做最小 pilot；仍保留 aligned、validity 和格式三项硬门槛。
4. 只有 pilot 合格后，才生成全新的 sealed test，并运行 seeds 42/43/44 与正式五项检查。

不增加 reward model、PPO/GRPO、Web 服务或大规模网格搜索。下一步的核心不是扩大工程，而是让训练目标、条件决策和实际输出合同一致。
