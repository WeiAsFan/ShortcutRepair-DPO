# ShortcutRepair-DPO v1.3 实现与执行计划

> - 状态：设计已冻结，等待实现
> - 日期：2026-09-03
> - 设计依据：[V1_3_DESIGN.md](V1_3_DESIGN.md)
> - 流程：`prepare → pilot → freeze → formal → report`

## 1. 完成定义

v1.3 的实现完成不等于真实 pilot 已通过。代码阶段必须做到：

- 能验证并复用 v1.2 seed-42 SFT → DPO 权重；
- 能在同一 LoRA adapter 上继续低学习率 SFT；
- 能生成锚定前/后的同 dev 对照并执行八项 pilot 判定；
- pilot 未通过时绝不生成 test；
- pilot 通过后能运行三 seed 完整链、sealed test 和正式报告；
- CPU 合同测试与 Ruff 全部通过。

真实 GPU pilot 是否通过，只能由服务器运行结果决定；不得用模拟测试宣称模型效果已经实现。

## 2. M0：版本隔离与文档

- [x] 从 `codex/v1.2` 当前提交建立 `codex/v1.3`；
- [x] 冻结 v1.3 设计，不修改 v1.2 结论或阈值；
- [x] 写明唯一变量、pilot 阈值、停止条件和正式对照；
- [ ] README 切换为 v1.3 当前状态；
- [ ] 从 v1.3 当前快照移除 v1.2 结果归档和 v1.2 文档，保留其远端分支与 Git 历史。

验收：v1.3 分支只维护自身文档，不复制 v1.2 结果。

## 3. M1：配置与数据身份

- [ ] 新增 `configs/v1_3.yaml`，固定 anchor learning rate `2e-6`、1 epoch；
- [ ] 固定 v1.2 SFT/DPO run manifest SHA 和原始运行 Git SHA；
- [ ] 新增轻量 v1.3 配置校验；
- [ ] 在 `data/v1.3` 生成与 v1.2 相同的 train/dev 文件；
- [ ] 生成与 SFT 文件逐字节相同的 `anchor.jsonl`；
- [ ] 验证 SFT、DPO、dev 的公开 SHA；
- [ ] prepare 只检查模型存在、manifest 身份和数据，不重复 Shortcut sanity 或旧评测。

验收：prepare 不加载 GPU 训练，且没有 `test.jsonl`。

## 4. M2：Format-anchor SFT 运行时

- [ ] 扩展训练合同，支持 `anchor` stage 的 2560 行、80 optimizer steps；
- [ ] 从原 SFT merged 模型加载 v1.2 DPO adapter，并以 `is_trainable=True` 继续训练；
- [ ] 复用现有 A/B+EOS SFT tokenization 和有限 loss 检查；
- [ ] 输出新的 final adapter，不修改原 v1.2 adapter；
- [ ] manifest 记录 starting model、starting adapter、二者 manifest SHA、实际步数、loss、耗时和显存；
- [ ] 支持当前 anchor run 的断点恢复和完成项跳过。

验收：CPU 接线测试证明没有新建第三个 LoRA adapter，训练预算与配置一致。

## 5. M3：单候选 Pilot

- [ ] 锚定前重新执行一次 FP32 dev 评测；
- [ ] 训练唯一 `sft_dpo_anchor`；
- [ ] 锚定后执行同协议 FP32 dev 评测；
- [ ] 输出固定阈值、前后 delta、八项检查和 `selected`；
- [ ] 不提供自动调参或第二候选入口；
- [ ] 重跑阶段时复用完成结果。

验收：模拟正例得到 `selected=sft_dpo_anchor`；格式未改善或核心指标回退的模拟负例均得到 `selected=null`，且没有 test。

服务器命令：

```bash
bash scripts/run_v1_3.sh prepare
bash scripts/run_v1_3.sh pilot
```

查看 `reports/v1.3/pilot_decision.md`。只有 `selected=sft_dpo_anchor` 才能继续。

## 6. M4：Freeze 与 Formal

- [ ] freeze 要求 Git 工作树 clean，并绑定 config/data/source manifests/pilot decision；
- [ ] 只在 pilot 通过后以 seed `13023` 生成 test；
- [ ] formal 对三个 seed 依次训练 SFT、DPO、anchor，共九个训练阶段；
- [ ] 全部训练完成后才进行 test；
- [ ] 同时评测锚定前和锚定后模型；
- [ ] 保留 v1.1 对照，只评测、不重训。

验收：test 被篡改、源 adapter 不匹配或任一训练未完成时，在报告前失败。

服务器命令：

```bash
bash scripts/run_v1_3.sh freeze
bash scripts/run_v1_3.sh formal
```

## 7. M5：报告与结果包

- [ ] 正式报告沿用五项成功检查；
- [ ] 增加锚定前/后 delta；
- [ ] 报告三阶段训练成本，不包装成等预算 DPO 对比；
- [ ] 生成 `RESULTS.md`、`results.json`、`metrics.csv`、`costs.csv` 和一张对比图；
- [ ] 结果包只包含配置、短 manifest、指标和 predictions，不包含权重。

服务器命令：

```bash
bash scripts/run_v1_3.sh report
```

## 8. M6：本地验收与交付

- [ ] `python -m pytest -q` 全部通过；
- [ ] `python -m ruff check src tests` 通过；
- [ ] `python -m shortcut_repair.v13 --help` 可用；
- [ ] README、设计和计划状态一致；
- [ ] 提交并推送 `codex/v1.3`。

实现阶段不新增独立 smoke、全量模型权重哈希、人工九步恢复脚本或与问题无关的工程功能。
