# ShortcutRepair-DPO

ShortcutRepair-DPO 是一个小型、受控、可复现的后训练项目。它先通过 SFT **受控诱导**小模型依赖可能过期的 `cached_recommendation`，确认该错误机制确实存在后，再比较等预算的 Aligned-only DPO 和 Counterfactual Repair DPO。

这不是新的 DPO 损失，也不声称模型会在所有真实系统里自然形成同类捷径。项目的小创新是：把“模型是否真的使用了 shortcut”从假设变成前置因果门控，并用同一新鲜工具结果的 hint-flip 反事实偏好对进行修复。

## 要解决的问题

历史行为数据可能来自一个依赖缓存建议的旧策略。即使新系统已经拿到权威的 fresh tool result，后训练模型仍可能复制旧建议。普通 aligned 数据里缓存和正确答案一致，因此无法告诉模型两者发生冲突时应该信谁。

v1.1 的主流程是：

```text
Base Qwen2.5-1.5B
        │
        ▼
Shortcut SFT（target 跟随 hint）
        │
        ▼
Dev 因果门控（固定工具结果，只翻转 hint）
        │ pass
        ├───────────────┐
        ▼               ▼
Aligned-only DPO   Counterfactual Repair DPO
        └─────── sealed test ───────┘
```

## v1.1 公平对比

| 条件 | 每个底层 case 的两条 DPO 行 | Chosen |
|---|---|---|
| Aligned-only Control | aligned prompt 重复两次 | fresh tool gold |
| Counterfactual Repair | aligned + conflict hint | fresh tool gold |

两组使用相同的 600 个 case、1,200 行、shortcut 起点、LoRA 配置、训练步数和 seeds 42/43/44。唯一变量是是否加入同观察、反转 hint 的 conflict 偏好行。

## v1.1 预注册结果标准

只有九项同时满足才报告 `POSITIVE`：三个 seed 的 conflict delta 全为正、平均 conflict accuracy 至少提高 10pp、paired bootstrap 95% CI 下界大于 0、hint flip rate 至少减半、aligned accuracy 下降不超过 2pp、causal hint effect 下降、fresh-result response rate 至少 0.80、nuisance invariance rate 至少 0.95、greedy exact-format rate 至少 0.98。否则报告 `NEGATIVE / INCONCLUSIVE`。

## 当前状态

v1.1 的训练和统一 FP32 正式评测已经完成。第一次 BF16 评测因数值协议缺口终止，属于无效评测；随后在不重训、不改数据、不改阈值的前提下，对全部模型统一执行 FP32 `gate → evaluate → aggregate`，得到正式结论 `NEGATIVE / INCONCLUSIVE`。

Counterfactual DPO 相对 Aligned-only DPO 将 conflict accuracy 从 0 提升到 0.69，将 hint flip rate 从 1.00 降到 0.2089，但 aligned accuracy 降至 0.8989、fresh-result response 仅为 0.5856、nuisance invariance 为 0.89。九项预注册检查通过六项、失败三项，因此只能报告“明显削弱 shortcut，但没有完成可靠修复”。

按 `decision_type` 的诊断进一步表明，Repair 在 `validity_decisive` 上接近完全正确，但在 `score_decisive` 上的 conflict accuracy 约为 0.389、fresh-result response 约为 0.178。正式分析工具现会把该切片写入 `results.json`、`decision_type_metrics.csv` 和 `RESULTS.md`；它用于解释结果，不追溯改变 v1.1 的九项判定。

原始正式结果见 [ShortcutRepair-DPO-v1.1-fp32-result](ShortcutRepair-DPO-v1.1-fp32-result/results/reports/RESULTS.md)。v1.1 至此冻结，不再修改训练、数据、sealed test、阈值或正式结论；后续改进另立 v1.2。

v1.2 代码现已实现，真实服务器 pilot 和正式训练尚未执行。新版本使用 75/25 的 score-aware 训练分布、hint-neutral preservation 和短 SFT warm-up，在 dev 上选择直接 DPO 或 SFT → DPO；随后用三个 seed、新 sealed test 和五项预注册标准验证能力保持。SFT 既是独立能力基线，也可作为链式路径的中间结果，不能预设 DPO 一定优于它。

## 本地 CPU 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check src tests
python -m shortcut_repair.v12 --help
```

CPU 验证不加载模型，也不能替代 GPU 训练。

## A6000 执行

v1.1 的服务器手册、完整复跑指南和 FP32 恢复指南均作为历史证据保留，不应再次执行。冻结口径和最终结论见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

下一轮实验使用独立的 [v1.2 设计](docs/V1_2_DESIGN.md)、[执行计划](docs/V1_2_EXECUTION_PLAN.md)和[远程操作指南](docs/V1_2_REMOTE_EXECUTION_GUIDE.md)。复用服务器上现有的 v1.1 模型和环境，不重新诱导，不独立 smoke，不重复全量权重哈希。

在服务器的仓库根目录、已激活的 v1.1 环境中依次执行：

```bash
bash scripts/run_v1_2.sh prepare
bash scripts/run_v1_2.sh pilot
# 先阅读 reports/v1.2/pilot_decision.md；仅在选出合格 DPO 时继续。
bash scripts/run_v1_2.sh freeze
bash scripts/run_v1_2.sh formal
bash scripts/run_v1_2.sh report
```

数据、训练、评测和报告分别隔离在 `data/v1.2/`、`runs/v1.2/`、`results/v1.2/`、`reports/v1.2/`，不会覆盖旧结果。中断后重跑同一阶段即可，已完成项自动跳过。正式负结果正常输出报告，不作为运行故障。

## 目录

```text
configs/experiment.yaml          # 冻结的 v1.1 配置
configs/v1_2.yaml                # 独立 v1.2 配置
configs/evaluation_amendment.yaml # 冻结的纯评测协议增补
src/shortcut_repair/data.py      # 确定性 oracle 与数据
src/shortcut_repair/train.py     # SFT merge 与 LoRA-DPO
src/shortcut_repair/evaluate.py  # A/B 条件概率与审计
src/shortcut_repair/analysis.py  # 因果指标、bootstrap、报告
src/shortcut_repair/cli.py       # 阶段命令
src/shortcut_repair/v12.py       # v1.2 五阶段入口
src/shortcut_repair/v12_data.py  # score-aware 与中性 hint 数据
src/shortcut_repair/v12_runtime.py # 轻量训练与复用 FP32 评测
src/shortcut_repair/v12_analysis.py # 五项检查、切片、SFT 对照与成本
scripts/                         # preflight、编排、脱敏打包
tests/                           # CPU 合同测试
docs/V1_2_DESIGN.md              # v1.2 研究设计与轻量实验边界
docs/V1_2_EXECUTION_PLAN.md      # v1.2 分阶段实施计划
docs/V1_2_REMOTE_EXECUTION_GUIDE.md # v1.2 远程五阶段操作
```
