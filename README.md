# ShortcutRepair-DPO v1.3

ShortcutRepair-DPO 是一个面向 AI 算法工程师面试的小型、受控后训练实验。项目先诱导
Qwen2.5-1.5B-Instruct 依赖可能过期的 `cached_recommendation`，再用反事实数据研究如何让
模型重新服从 fresh tool result，并用严格的开发集门槛和封存测试区分“学会规则”与“碰巧答对”。

项目不提出新的 DPO 损失，也不把合成二选一任务外推为通用工具调用能力。它的价值在于展示：
如何定义 shortcut、构造可证伪的数据干预、组合 SFT 与 DPO、定位训练目标和真实生成行为之间的
偏差，并用最小实验修正它。

## v1.3 研究问题

v1.2 的 `SFT → DPO` 已在 dev 上学会 fresh-score、validity 和 nuisance 不变性，但开放词表
贪心生成的严格 A/B 格式率只有 0.9544：1,536 条中有 70 条生成 `.getBcd` 等额外字符串，
因此按协议得到 `selected=null`，没有生成 test。

v1.3 只回答：

> 能否在保留已学规则能力的同时，用一次低学习率 SFT continuation，让同一 DPO policy
> 恢复严格的 A/B+EOS 动作合同？

## 固定方法

- 逐字节复用 v1.2 的 train/dev、Score-aware SFT merged 模型和 seed-42 DPO adapter；
- 在原 DPO adapter 上以 `is_trainable=True` 继续训练，不创建第三个 LoRA；
- anchor 数据与原 SFT 数据相同，固定 learning rate `2e-6`、1 epoch、80 optimizer steps；
- 保持四 token、开放词表 greedy generation；不把 `max_new_tokens` 缩为 1，也不用约束解码代替模型修复；
- pilot 只运行一个候选、一个 seed，不搜索超参数；
- 只有八项检查全部通过，才生成新 seed `13023` 的 sealed test。

完整假设、阈值与可证伪结论见 [v1.3 设计文档](docs/V1_3_DESIGN.md)，实现里程碑见
[v1.3 执行计划](docs/V1_3_EXECUTION_PLAN.md)。

## 当前状态

v1.3 的代码与 CPU 合同测试已经实现，尚未在服务器上运行真实 GPU pilot。CPU 模拟仅证明：

- 配置、数据哈希和 v1.2 权重来源会被核对；
- anchor 确实继续训练原 LoRA，且预算固定；
- 格式改善且核心能力保持时得到 `selected=sft_dpo_anchor`；
- 格式未改善或能力回退时得到 `selected=null`，并且不会创建 `test.jsonl`；
- pilot 通过后，正式流程会先完成三 seed 的九个训练阶段，再开始 17 次 test 评测。

这不代表真实 pilot 已通过。下一步必须在原 Linux 训练设备上复用 v1.2 权重，实际运行
`prepare → pilot`；只有 `reports/v1.3/pilot_decision.json` 中的 `selected` 为
`sft_dpo_anchor`，才可继续 `freeze → formal → report`。

## 远程运行

```bash
git fetch origin
git switch codex/v1.3
python -m pip install -r requirements-dev.txt
python -m pip install -e .

python -m pytest -q
python -m ruff check src tests

bash scripts/run_v1_3.sh prepare
bash scripts/run_v1_3.sh pilot
```

检查 `reports/v1.3/pilot_decision.md`。若 `selected=null`，立即停止，不得执行 freeze。
若 pilot 通过：

```bash
bash scripts/run_v1_3.sh freeze
bash scripts/run_v1_3.sh formal
bash scripts/run_v1_3.sh report
```

每个阶段都可安全重跑：身份相同且已完成的训练/评测会复用，身份不一致则停止而不是覆盖。

## 目录

```text
configs/v1_3.yaml                   # v1.3 冻结配置与 v1.2 来源哈希
src/shortcut_repair/v13.py          # prepare/pilot/freeze/formal/report 入口
src/shortcut_repair/v13_data.py     # 复用数据、anchor 和新 sealed test
src/shortcut_repair/v13_runtime.py  # 原 LoRA 上的 format-anchor SFT
src/shortcut_repair/v13_analysis.py # 八项 pilot 检查与锚定前后对照
scripts/run_v1_3.sh                 # Linux 五阶段入口
tests/test_v13_*.py                 # 不加载真实模型的 CPU 合同测试
docs/V1_3_DESIGN.md                 # 冻结研究设计
docs/V1_3_EXECUTION_PLAN.md         # 实现与服务器执行计划
```

历史版本的结果与说明分别保留在其远端分支；v1.3 只保留当前版本文档。底层 v1.0–v1.2
代码、配置和回归测试仍作为可追溯依赖保留，但不复制历史实验结果。
