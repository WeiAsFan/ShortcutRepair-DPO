# ShortcutRepair-DPO

ShortcutRepair-DPO 是一个小型、受控、可复现的后训练项目。它先通过 SFT **受控诱导**小模型依赖可能过期的 `cached_recommendation`，确认该错误机制确实存在后，再比较等预算的 Aligned-only DPO 和 Counterfactual Repair DPO。

这不是新的 DPO 损失，也不声称模型会在所有真实系统里自然形成同类捷径。项目的小创新是：把“模型是否真的使用了 shortcut”从假设变成前置因果门控，并用同一新鲜工具结果的 hint-flip 反事实偏好对进行修复。

## 要解决的问题

历史行为数据可能来自一个依赖缓存建议的旧策略。即使新系统已经拿到权威的 fresh tool result，后训练模型仍可能复制旧建议。普通 aligned 数据里缓存和正确答案一致，因此无法告诉模型两者发生冲突时应该信谁。

本项目的流程是：

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

## 公平对比

| 条件 | 每个底层 case 的两条 DPO 行 | Chosen |
|---|---|---|
| Aligned-only Control | aligned prompt 重复两次 | fresh tool gold |
| Counterfactual Repair | aligned + conflict hint | fresh tool gold |

两组使用相同的 600 个 case、1,200 行、shortcut 起点、LoRA 配置、训练步数和 seeds 42/43/44。唯一变量是是否加入同观察、反转 hint 的 conflict 偏好行。

## 预注册结果标准

只有九项同时满足才报告 `POSITIVE`：三个 seed 的 conflict delta 全为正、平均 conflict accuracy 至少提高 10pp、paired bootstrap 95% CI 下界大于 0、hint flip rate 至少减半、aligned accuracy 下降不超过 2pp、causal hint effect 下降、fresh-result response rate 至少 0.80、nuisance invariance rate 至少 0.95、greedy exact-format rate 至少 0.98。否则报告 `NEGATIVE / INCONCLUSIVE`。

## 当前状态

v1.1 已在 A6000 上完成 Shortcut、六个 DPO adapter 和三个 Counterfactual SFT adapter 的正式训练。第一次正式评测在 `repair/seed-42` 暴露出“BF16 前向后才转 FP32”的数值协议缺口并终止，因此这次运行是无效评测，不是 Repair 的负结果。仓库已冻结纯评测修正，等待在原训练产物上统一执行 FP32 `gate → evaluate → aggregate`；在新的 `reports/RESULTS.md` 生成前仍没有效果结论。

## 本地 CPU 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
python -m ruff check src tests
python -m shortcut_repair.cli generate --config configs/experiment.yaml --stage train-dev
python -m shortcut_repair.cli train-shortcut --config configs/experiment.yaml --dry-run
python -m shortcut_repair.cli train-dpo --config configs/experiment.yaml --method control --seed 42 --dry-run
python -m shortcut_repair.cli train-dpo --config configs/experiment.yaml --method repair --seed 42 --dry-run
```

CPU 验证不加载模型，也不能替代 GPU 训练。

## A6000 执行

当前已有训练产物的 FP32 评测恢复步骤见 [docs/V1_1_EVALUATION_AMENDMENT.md](docs/V1_1_EVALUATION_AMENDMENT.md)。从另一台 Linux 设备经 SSH 登录服务器、从头执行 v1.1 的完整指南见 [docs/V1_1_REMOTE_EXECUTION_GUIDE.md](docs/V1_1_REMOTE_EXECUTION_GUIDE.md)。通用服务器手册见 [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)，冻结实验口径与透明增补见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

原始完整实验的编排入口仍保留为：

```bash
bash scripts/run_experiment.sh all
```

但当前分支的评测增补已绑定训练提交 `1ead3b24...` 和现有九个 run，因此禁止在当前恢复中执行 `all`，即使运行目录为空也不能从当前评测提交重训。只按评测修正文档执行三个阶段；新的完整实验应另立 v1.2 协议和身份。

如果 shortcut 机制门控失败，`all` 会在生成 test 之前退出；这时正确结论是“shortcut induction 未建立”，而不是“Repair DPO 无效”。

## 目录

```text
configs/experiment.yaml          # 唯一正式配置
configs/evaluation_amendment.yaml # 冻结的纯评测协议增补
src/shortcut_repair/data.py      # 确定性 oracle 与数据
src/shortcut_repair/train.py     # SFT merge 与 LoRA-DPO
src/shortcut_repair/evaluate.py  # A/B 条件概率与审计
src/shortcut_repair/analysis.py  # 因果指标、bootstrap、报告
src/shortcut_repair/cli.py       # 阶段命令
scripts/                         # preflight、编排、脱敏打包
tests/                           # CPU 合同测试
```
