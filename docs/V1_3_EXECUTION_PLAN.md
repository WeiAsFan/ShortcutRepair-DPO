# ShortcutRepair-DPO v1.3 实现与执行计划

> - 状态：全部阶段完成，正式结论已冻结为 `POSITIVE`
> - 日期：2026-09-03
> - 设计依据：[V1_3_DESIGN.md](V1_3_DESIGN.md)
> - 结果分析：[V1_3_RESULT_ANALYSIS.md](V1_3_RESULT_ANALYSIS.md)
> - 流程：`prepare → pilot → freeze → formal → report`

## 1. 完成定义

本计划将代码完成与真实实验通过区分开；两者目前均已完成。代码阶段必须做到：

本计划遵守面试导向的小型项目边界，只实现回答研究问题所需的最小训练、判定和报告链。

- 能验证并复用 v1.2 seed-42 SFT → DPO 权重；
- 能在同一 LoRA adapter 上继续低学习率 SFT；
- 能生成锚定前/后的同 dev 对照并执行八项 pilot 判定；
- pilot 未通过时绝不生成 test；
- pilot 通过后能运行三 seed 完整链、sealed test 和正式报告；
- CPU 合同测试与 Ruff 全部通过。

真实 GPU pilot 已由服务器结果确认通过；该结论来自运行产物，不来自 CPU 模拟测试。

## 2. M0：版本隔离与文档

- [x] 从 `codex/v1.2` 当前提交建立 `codex/v1.3`；
- [x] 冻结 v1.3 设计，不修改 v1.2 结论或阈值；
- [x] 写明唯一变量、pilot 阈值、停止条件和正式对照；
- [x] README 切换为 v1.3 当前状态；
- [x] 从 v1.3 当前快照移除 v1.2 结果归档和 v1.2 文档，保留其远端分支与 Git 历史。

验收：v1.3 分支只维护自身文档，不复制 v1.2 结果。

## 3. M1：配置与数据身份

- [x] 新增 `configs/v1_3.yaml`，固定 anchor learning rate `2e-6`、1 epoch；
- [x] 固定 v1.2 SFT/DPO run manifest SHA 和原始运行 Git SHA；
- [x] 新增轻量 v1.3 配置校验；
- [x] 在 `data/v1.3` 生成与 v1.2 相同的 train/dev 文件；
- [x] 生成与 SFT 文件逐字节相同的 `anchor.jsonl`；
- [x] 验证 SFT、DPO、dev 的公开 SHA；
- [x] prepare 只检查模型存在、manifest 身份和数据，不重复 Shortcut sanity 或旧评测。

验收：prepare 不加载 GPU 训练，且没有 `test.jsonl`。

## 4. M2：Format-anchor SFT 运行时

- [x] 扩展训练合同，支持 `anchor` stage 的 2560 行、80 optimizer steps；
- [x] 从原 SFT merged 模型加载 v1.2 DPO adapter，并以 `is_trainable=True` 继续训练；
- [x] 复用现有 A/B+EOS SFT tokenization 和有限 loss 检查；
- [x] 输出新的 final adapter，不修改原 v1.2 adapter；
- [x] manifest 记录 starting model、starting adapter、二者 manifest SHA、实际步数、loss、耗时和显存；
- [x] 支持当前 anchor run 的断点恢复和完成项跳过。

验收：CPU 接线测试证明没有新建第三个 LoRA adapter，训练预算与配置一致。

## 5. M3：单候选 Pilot

- [x] 锚定前重新执行一次 FP32 dev 评测；
- [x] 训练唯一 `sft_dpo_anchor`；
- [x] 锚定后执行同协议 FP32 dev 评测；
- [x] 输出固定阈值、前后 delta、八项检查和 `selected`；
- [x] 不提供自动调参或第二候选入口；
- [x] 重跑阶段时复用完成结果。

验收：模拟正例得到 `selected=sft_dpo_anchor`；格式未改善或核心指标回退的模拟负例均得到 `selected=null`，且没有 test。

实际执行命令（已完成）：

```bash
bash scripts/run_v1_3.sh prepare
bash scripts/run_v1_3.sh pilot
```

查看 `reports/v1.3/pilot_decision.md`。只有 `selected=sft_dpo_anchor` 才能继续。

## 6. M4：Freeze 与 Formal

- [x] freeze 要求 Git 工作树 clean，并绑定 config/data/source manifests/pilot decision；
- [x] 只在 pilot 通过后以 seed `13023` 生成 test；
- [x] formal 对三个 seed 依次训练 SFT、DPO、anchor，共九个训练阶段；
- [x] 全部训练完成后才进行 test；
- [x] 同时评测锚定前和锚定后模型；
- [x] 保留 v1.1 对照，只评测、不重训。

验收：test 被篡改、源 adapter 不匹配或任一训练未完成时，在报告前失败。

实际执行命令（已完成）：

```bash
bash scripts/run_v1_3.sh freeze
bash scripts/run_v1_3.sh formal
```

## 7. M5：报告与结果包

- [x] 正式报告沿用五项成功检查；
- [x] 增加锚定前/后 delta；
- [x] 报告三阶段训练成本，不包装成等预算 DPO 对比；
- [x] 生成 `RESULTS.md`、`results.json`、`metrics.csv`、`costs.csv` 和一张对比图；
- [x] 结果包只包含配置、短 manifest、指标和 predictions，不包含权重。

实际执行命令（已完成）：

```bash
bash scripts/run_v1_3.sh report
```

## 8. M6：本地验收与交付

- [x] `python -m pytest -q` 全部通过；
- [x] `python -m ruff check src tests` 通过；
- [x] `python -m shortcut_repair.v13 --help` 可用；
- [x] README、设计和计划状态一致；
- [x] 提交并推送 `codex/v1.3`。

实现阶段不新增独立 smoke、全量模型权重哈希、人工九步恢复脚本或与问题无关的工程功能。

## 9. M7：真实实验完成记录

- [x] Pilot 八项检查全部通过，`selected=sft_dpo_anchor`；
- [x] Pilot 通过后才生成 seed 13023 的 sealed test；
- [x] 三 seed 共九个正式训练阶段全部完成；
- [x] 正式五项检查全部通过，结论为 `POSITIVE`；
- [x] 独立复算 predictions、metrics、聚合结果、运行 manifest 和结果归档；
- [x] 更新 README、设计状态和 [v1.3 结果分析](V1_3_RESULT_ANALYSIS.md)。

项目主实验至此冻结。不得根据已经查看的 sealed test 继续调整 v1.3 模型或阈值。
