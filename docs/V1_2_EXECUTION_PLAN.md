# ShortcutRepair-DPO v1.2 执行计划

> - 状态：M1 prepare 与 M2 dev pilot 已完成；没有合格 DPO，按协议停止，M3–M5 未执行
> - 日期：2026-09-02
> - 设计依据：[V1_2_DESIGN.md](V1_2_DESIGN.md)
> - 总流程：`prepare → pilot → freeze → formal → report`

## 1. 计划目标

本计划把 v1.2 控制在一个面试中能够完整说清的小型实验内。实施目标是：

1. 让正式分析按 `score_decisive` 和 `validity_decisive` 分层；
2. 构造 score-aware 与 hint-neutral 训练数据；
3. 在 dev 上用最少候选选择直接 DPO 或 SFT → DPO；
4. 在三个 seed 和一次 sealed test 上判断 fresh-result response、aligned/validity 保持和 nuisance invariance；
5. 用一份短报告形成可辩护的面试叙事。

不以增加流程、manifest 或哈希数量作为完成度。

代码入口为 `python -m shortcut_repair.v12`，薄脚本为 `scripts/run_v1_2.sh`。远程操作记录见 [V1_2_REMOTE_EXECUTION_GUIDE.md](V1_2_REMOTE_EXECUTION_GUIDE.md)，实际 pilot 复盘见 [V1_2_PILOT_ANALYSIS.md](V1_2_PILOT_ANALYSIS.md)。本地模拟通过不代表真实 GPU 训练完成；当前真实 pilot 完成也不等于得到正式正结果。

## 2. 工作量上限

- Pilot：默认三个 seed-42 候选；
- 调整：只有当全部候选未满足最低条件时，最多一轮、最多两个补充 run；
- Formal：最多三个 Score-aware SFT 和三个选定 DPO；
- 基础模型与 v1.1 adapter：只评测，不重训；
- 正式 test：只生成、评测和查看一次。

超过上限时必须先缩小问题，而不是继续堆实验。

## 3. M0：关闭 v1.1

- [x] 更新 README 为已完成的 FP32 正式结果；
- [x] 在 v1.1 协议中记录冻结结论和失败项；
- [x] 将 `decision_type` 切片加入分析结果、CSV 和 Markdown 报告；
- [x] 明确切片不改变 v1.1 九项预注册判定；
- [x] 将旧服务器指南标记为历史记录；
- [x] 已同步结果提交 `70618ae`，将文档和分析工具更新建立在该提交之上。
- [x] 文档和分析工具已以 `c2addff` 推送到 `codex/v1.1-repair`，另建 `codex/v1.2` 实现后续改动。

验收：v1.1 的状态、结果和后续边界在 README、协议和正式报告中一致。

## 4. M1：prepare

### 4.1 最小代码改动

- [x] 新建独立 v1.2 配置，不修改 v1.1 配置和校验文件；
- [x] 扩展数据生成器，支持 75/25 的 score/validity 训练比例；
- [x] 生成明显分差 SFT 数据和 hint-neutral preservation 数据；
- [x] 保持 dev/test 50/50，保留三类干预；
- [x] 让现有分析输出直接用于 v1.2 五项成功标准；
- [x] 增加薄的 v1.2 运行脚本，只提供五个阶段。

### 4.2 只写必要测试

- [x] 数据比例和 A/B 平衡；
- [x] hint-neutral 样本的 gold 仍由 Oracle 决定；
- [x] historical score/display rank 与 gold 不相关；
- [x] decision_type 切片行数完整；
- [x] 五项成功判定的正、负各一个测试；
- [x] 现有 v1.1 测试继续通过。

还以隔离的模拟模型验证两条完整路径、训练中断恢复、已完成项跳过、先训练后 test、最多两次补充运行和无权重打包。测试使用单独的测试 seed，不启动真实模型训练。

不为每个 shell 分支、日志格式或重复 checksum 写组合测试。

### 4.3 一次性准备检查

本地执行一次：

```bash
python -m pytest -q
python -m ruff check src tests
```

服务器第一次进入 v1.2 时执行一次：

```bash
python -m pip check
nvidia-smi
python -m shortcut_repair.v12 --help
```

随后生成 train/dev，运行数据审计，并在 v1.2 dev 上对复用的 Shortcut checkpoint 做一次 sanity check。这里不设置独立 smoke 阶段，seed-42 pilot 本身承担端到端验证作用。

```bash
bash scripts/run_v1_2.sh prepare
```

验收：

- 数据审计通过；
- Shortcut 在新 dev 上仍明显表现为 aligned 高、conflict 低和 hint 敏感；
- 没有正式 test 文件。

实际结果：以上三项均通过。Shortcut sanity 四项检查通过，train/dev 审计通过，且 prepare 后没有 test 文件。

## 5. M2：pilot

仅使用 seed 42，按相同 dev 评测：

1. Score-aware DPO；
2. Score-aware SFT；
3. Score-aware SFT → DPO。

- [x] 三个候选从同一 Shortcut checkpoint 开始；
- [x] 记录数据量、训练步数、运行时间和峰值显存；
- [x] 输出 overall 与 decision_type 指标；
- [x] 按设计文档中的最低条件和选择顺序执行候选择优；
- [x] 将选择理由写入简短的 `pilot_decision.md`。

如果三个候选全部不满足 aligned/validity/格式最低条件，只允许一次调整：

- 已固定为 DPO learning rate 减半；
- 只补跑 direct DPO、SFT → DPO 各一次；
- 不改 beta、epoch 或数据，不重训 SFT；
- 脚本自动执行这一轮，最多两个补充 run。

禁止同时修改数据、学习率、beta 和 epoch 后声称找到原因。

预期验收：确定一个正式 DPO 路径，并能用一段话解释为什么选择它。

实际结果：默认三条路径均未通过能力保持，脚本按冻结规则补跑两条半学习率路径；五条路径最终均为 `eligible=false`，`selected=null`。其中标准 SFT → DPO 只因精确格式率 0.9544 未达到 0.98 而失败。M2 未通过预期验收，并正确触发停止条件。

```bash
bash scripts/run_v1_2.sh pilot
```

查看 `reports/v1.2/pilot_decision.md`。若没有合格 DPO 路径，停在这里；不得继续冻结 test。SFT 单独达标不等于 DPO 达标。

## 6. M3：freeze

Pilot 完成后才进入正式冻结。

当前状态：未启动。M2 没有合格 DPO，因此不得执行本阶段；没有生成 test 或 freeze manifest。

- [ ] 固定选中的训练路径和全部超参数；
- [ ] 固定 train/dev 数据和新 test seed；
- [ ] 写入一份 `freeze_manifest.json`，只记录：
  - Git commit；
  - 配置文件 SHA256；
  - train/dev/test 文件 SHA256；
  - 基础模型 revision；
  - 复用的 v1.1 Shortcut manifest 身份；
  - pilot 选择及旧 Control/Repair 的短 manifest 身份；
  - 正式 seeds；
- [ ] 在正式冻结时要求 Git 工作树 clean；
- [ ] 生成 sealed test；
- [ ] 确认 test 未被训练或 pilot 读取。

不重新哈希完整 Base、Shortcut 或每个 adapter 权重；正式 run 只需记录产物存在、训练完成和配置身份。

验收：一个短 manifest 足以回答“用的哪份代码、配置和数据”，且没有重复审计链。

```bash
bash scripts/run_v1_2.sh freeze
```

冻结后不再改源代码或配置；小文件身份检查由阶段入口自动完成。不要打开 test 做人工选型。

## 7. M4：formal

当前状态：未启动，原因同 M3。

按 seeds 42/43/44 执行：

- [ ] 训练三个 Score-aware SFT；
- [ ] 训练三个选定 DPO；
- [ ] 若选择 SFT → DPO，直接复用同 seed 的 SFT checkpoint；
- [ ] 每个 run 记录实际 optimizer steps、有限 training loss、耗时和峰值显存；
- [ ] 失败时只恢复该 run，不重跑已完成模型；
- [ ] 在全部训练完成后统一评测一次 test。

同时在新 test 上评测但不重训：

- Base；
- Shortcut；
- v1.1 Control；
- v1.1 Repair。

必要停止条件只有：

- 数据或冻结配置身份不一致；
- OOM、NaN/Inf、CUDA error；
- 实际训练步数与当前配置不一致；
- 模型产物缺失或无法加载；
- sealed test 在冻结后被修改。

除此之外的指标不好看不属于运行故障，不应中止或重跑。

```bash
bash scripts/run_v1_2.sh formal
```

SFT 默认为 80 步，DPO 为 180 步。先完成六个训练阶段，再执行 14 个模型实例的统一 test 评测。发生中断时重跑相同命令，已完成项不会再次占用 GPU。

## 8. M5：report

当前状态：未启动；现有 [pilot 分析](V1_2_PILOT_ANALYSIS.md)不属于正式 test 报告。

- [ ] 聚合三个 seed；
- [ ] 报告全部模型的 overall 和 decision_type 指标；
- [ ] 应用五项冻结成功标准；
- [ ] 报告一组主要 delta 的 paired bootstrap CI；
- [ ] 对比 Score-aware SFT 与选定 DPO；
- [ ] 明确训练阶段数和额外计算成本；
- [ ] 生成 `RESULTS.md`、`results.json`、核心 CSV 和一张图；
- [ ] 打包必要配置、短 manifest、指标和 predictions，不包含 checkpoint。

正式结果不得：

- 修改阈值；
- 删除不利 seed；
- 根据 test 再选择 direct DPO 或 SFT → DPO；
- 把 SFT → DPO 的额外计算包装成等预算比较；
- 把受控合成结论外推到生产工具系统。

```bash
bash scripts/run_v1_2.sh report
```

产物为 `reports/v1.2/RESULTS.md`、`results.json`、`metrics.csv`（含全部模型、切片、逐 seed）、`costs.csv` 和 `comparison.png`。结果包写入 `artifacts/v1.2/`，不包含 checkpoint；报告阶段不加载 GPU 模型。

## 9. 最终验收

v1.2 完成时应能回答：

1. fresh-score reasoning 是否从 v1.1 的 17.8% 明显提高；
2. 改善来自 SFT 能力建立、DPO 冲突修复还是二者组合；
3. aligned 和 validity 是否保留；
4. nuisance invariance 是否达到要求；
5. 三个 seed 是否方向一致；
6. 额外训练阶段是否值得；
7. 该结论在面试中应如何准确表述。

如果最终仍为 `NEGATIVE / INCONCLUSIVE`，只要运行有效并能定位原因，项目仍然完成；不再扩展出 v1.2.1 的无穷调参循环。
