# ShortcutRepair-DPO 修复与优化执行计划

> - 状态：已批准，待执行
> - 起草日期：2026-08-31
> - 当前本地代码基线：`b69be45`
> - 首次结果提交：`6f9d8599588d7e5fb1754037494b0ce71d351a2d`
> - 目标实验版本：`v1.1`

## 1. 文档目的

本文是 ShortcutRepair-DPO 从首次失败实验走向可信正式结果的执行指南。它规定修改顺序、实验冻结方式、验收标准、停止条件和最终交付物，避免在修复过程中同时改变过多变量，也避免为了得到正结果而在 test 上反复调参。

本计划适用于项目 `v1.1`。现有实现和 2026-08-28 的首次 A6000 运行视为 `v1.0` 基线。`v1.0` 的失败结果必须保留，不覆盖、不删除；它将作为训练系统调试和实验复盘证据。

## 2. 当前已知事实

### 2.1 已确认的工程故障

- 当前正式配置包含 1,200 条 SFT 数据、micro batch 4、梯度累积 8、5 epochs，预期 optimizer step 为 190。
- `transformers==4.48.3` 在每个 epoch 的微批次数不能被梯度累积步数整除时，会错误地用向下取整值推导 `max_steps`。
- 首次运行因此在 185 step、epoch 4.88 处结束，随后被项目的 190-step 合同断言拦截。
- 同一问题会让正式 DPO 在 111 step 结束，而项目合同要求 114 step。
- 这是训练调度问题，不是 CUDA、显存、模型下载或数据生成失败。

参考：[Transformers issue #36297](https://github.com/huggingface/transformers/issues/36297)。

### 2.2 已确认的实验有效性风险

- 当前数据中，`B.historical_score == 100` 可以 100% 预测 gold=A；模型可能依赖这个无关旁路，而不是 fresh tool result。
- 当前生成数据里，gold 的 `fresh_score` 始终更高，因此 `is_valid` 从未真正改变 oracle 答案。
- `request_id` 包含 `induction`、`dpo`、`dev` 或 `test`，引入了不必要的 split 标记。
- 当前评测只翻转 cached hint，能够测量 hint 依赖，但不能证明修复后的模型真正依赖 fresh tool result。
- 当前没有 Counterfactual SFT baseline，因此无法回答“反事实数据有效”与“DPO 本身必要”之间的区别。

### 2.3 首次运行的训练信号

- SFT loss 在约第 15 step 已接近 0，第 20 step 后日志 loss 基本为 0。
- 这说明 shortcut induction 任务很容易，5 epochs 很可能过量。
- `v1.1` 应缩短 SFT 预算，但只能在新协议冻结、test 生成之前修改。

## 3. 总体原则

后续工作必须遵守以下原则：

1. 一次提交只解决一个可描述的问题，不把依赖升级、数据重构和算法扩展混在同一提交中。
2. 先修训练合同，再修数据有效性，再冻结协议，最后才消耗正式 GPU 预算。
3. 配置、数据生成器或模型起点发生变化后，禁止恢复旧 checkpoint。
4. dev 用于机制门控和开发；test 只有在协议冻结、gate 通过后才能生成。
5. test 一经 sealed，不得修改配置、生成器、阈值、seed 或删除不利样本。
6. 正结果和负结果都必须按预注册规则报告；不得通过补跑并挑选 seed 改变结论。
7. 项目核心保持为“小型受控后训练实验”，在核心结果完成前不加入 PPO、GRPO、reward model、RAG、多工具、大模型或 Web UI。

## 4. 里程碑概览

| 里程碑 | 目标 | 主要验收证据 |
|---|---|---|
| M0 | 同步仓库并保全 `v1.0` 失败证据 | Git SHA、失败日志、checkpoint 备份清单 |
| M1 | 修复 Trainer 步数合同 | 显式 `max_steps`、回归测试、实际 global step |
| M2 | 消除数据旁路并覆盖真实决策规则 | 数据审计 manifest、生成器测试 |
| M3 | 补齐 baseline、因果评测并冻结 `v1.1` 协议 | 评测测试、配置 SHA、协议文档、全部 dry-run 合同 |
| M4 | 重新训练并通过 shortcut 机制门控 | merged checkpoint、Base/Shortcut dev gate JSON |
| M5 | 完成正式多 seed 训练与 sealed test | 全部 run/prediction manifest |
| M6 | 聚合可信结果并形成面试交付物 | RESULTS、图表、失败复盘、README |

只有前一个里程碑通过，才能进入下一个里程碑。

## 5. M0：同步仓库并保全首次失败证据

### 5.1 操作

- [ ] 网络可用后执行 `git fetch origin`。
- [ ] 确认远端 `main` 包含结果提交 `6f9d8599588d7e5fb1754037494b0ce71d351a2d`。
- [ ] 工作树干净时执行 `git pull --ff-only`，禁止使用会覆盖本地修改的 reset/checkout 操作。
- [ ] 在服务器上记录当前代码 SHA、配置 SHA、数据 manifest SHA 和模型 revision。
- [ ] 将服务器原始 `runs/shortcut`、`experiment.log` 和环境信息复制到独立的只读失败归档；不要把模型权重提交到 Git。
- [ ] 新建修复分支，例如 `codex/v1.1-repair`。
- [ ] 新增中文失败复盘 `docs/FAILURE_ANALYSIS_2026-08-28.md`，说明症状、根因、影响范围和后续修复，不把失败描述为 DPO 效果失败。

### 5.2 验收标准

- 本地能够追溯 `v1.0` 源码 SHA 和结果提交 SHA。
- 原始 checkpoint 和日志在服务器或外部归档中可恢复。
- 仓库中没有模型权重、认证信息或包含个人路径的未脱敏公开日志。
- `v1.1` 修改与 `v1.0` 结果位于清晰分开的提交或分支中。

### 5.3 停止条件

如果无法确认原始结果对应的代码/config/data SHA，不进入正式结果对比；先完成来源核对。

## 6. M1：修复训练步数合同

### 6.1 代码修改

主要文件：

- `src/shortcut_repair/train.py`
- `tests/test_train.py`
- `configs/experiment.yaml`
- `docs/SERVER_RUNBOOK.md`

要求：

- [ ] SFT 的 `TrainingArguments` 显式设置 `max_steps=contract["optimizer_steps"]`。
- [ ] DPO 的 `DPOConfig` 统一设置 `max_steps=contract["optimizer_steps"]`；smoke 合同返回 2，formal 合同返回正式预算。
- [ ] 将 optimizer step 设为唯一的停止预算；epoch 只用于计算合同和描述数据暴露量，不再作为 Trainer 的实际停止来源。
- [ ] 保留训练结束后的 `trainer.state.global_step` 合同检查。
- [ ] manifest 增加 `actual_epoch`、显式 max step 来源、Git SHA 和恢复来源。
- [ ] `--resume` 前校验 checkpoint 对应的 config SHA、data SHA、训练阶段和预算；任一不一致就拒绝恢复。
- [ ] 不通过简单把 190/114 改成 185/111 来掩盖未跑满的 epoch。

### 6.2 测试

- [ ] 为“微批次数不能整除梯度累积步数”增加回归用例。
- [ ] 验证 SFT dry-run 的显式预算与实际 TrainingArguments 来源一致。
- [ ] 验证 DPO smoke/formal 都使用合同中的 `optimizer_steps`。
- [ ] 验证 config/data SHA 变化后恢复请求被拒绝。
- [ ] 保留 control/repair 同预算检查。

### 6.3 验收标准

在保留 `v1.0` 五 epoch 配置的专用验证中：

```text
SFT global_step = 190
SFT actual_epoch 已记录，且余数批次造成的偏差已解释
DPO formal budget = 114
DPO smoke global_step = 2
```

正式 `v1.1` 若将 SFT 改为一 epoch，则最终合同应相应变为 38 step，并由显式 `max_steps` 执行。由于正的 `max_steps` 会覆盖 Trainer 的 epoch 停止逻辑，验收以 global step 为准，同时保留 `actual_epoch` 作为可审计诊断值。

### 6.4 停止条件

出现以下任一情况时停止：

- actual step 与合同不同；
- epoch 明显偏离预期；
- resume 使用了不同 config/data 的 checkpoint；
- loss、grad norm 出现 NaN/Inf；
- manifest 没有写明实际训练预算。

## 7. M2：重构数据并建立自动审计

### 7.1 数据设计

主要文件：

- `src/shortcut_repair/data.py`
- `tests/test_data.py`
- `configs/experiment.yaml`

要求：

- [ ] A/B gold 严格平衡，但不直接由公开的 index 奇偶模式决定。
- [ ] `historical_score` 在 gold、case 类型和 hint 变体各分层下保持相同分布；预注册的“选高分/选低分”等启发式均不能有效预测 gold。
- [ ] `display_rank` 同样在各分层下平衡，不与 gold、fresh score 或 validity 形成确定关系。
- [ ] `request_id` 使用不透明、确定性的哈希 ID，不暴露 split 名称。
- [ ] 同一 case 的干预版本中，模型可见输入除被操纵字段外字节一致；oracle 标签允许随 `fresh_flip` 改变。

正式 case 至少包含两类：

| case 类型 | 构造 | 目的 |
|---|---|---|
| score-decisive | 两个候选均有效，gold 的 fresh score 更高 | 检查是否比较新鲜分数 |
| validity-decisive | wrong 无效但 fresh score 更高，gold 有效但分数更低 | 检查是否先过滤无效候选 |

两类在每个正式 split 中应尽量各占 50%，并在 A/B gold 下分别平衡。

### 7.2 manifest 自动审计

训练前 manifest 至少输出并验证：

```text
gold_A_fraction = 0.50
score_decisive_fraction = 0.50
validity_decisive_fraction = 0.50
historical_only_accuracy <= 0.55
display_rank_only_accuracy <= 0.55
constant_A_accuracy = 0.50
constant_B_accuracy = 0.50
split_marker_count = 0
request_id_unique_across_cases = true
request_id_disjoint_across_splits = true
control_repair_case_multiset_equal = true
```

其中 nuisance-only accuracy 取预注册简单启发式中的最高值，而不是只选择一个有利方向报告。

### 7.3 测试

- [ ] 测试实际生成数据中存在“无效但高分”的候选，而不只测试 oracle 函数的手写样例。
- [ ] 测试 nuisance-only 启发式不能预测 gold。
- [ ] 测试 request ID 不泄漏 split。
- [ ] 测试每个 split 的 gold、case 类型和提示变体平衡。
- [ ] 测试重新生成仍然字节确定。

### 7.4 验收标准

全部数据审计通过，且人工抽查至少 20 个 case 后，才能冻结协议。任何单个无关字段能高准确率预测 gold 时，都不得进入 GPU 正式训练。

## 8. M3：补齐 baseline、因果评测并冻结 `v1.1` 协议

### 8.1 正式模型组

`v1.1` 至少包含：

| 组别 | 训练数据/方法 | 回答的问题 |
|---|---|---|
| Base | 原始 Qwen，不训练 | 原模型初始依赖什么 |
| Shortcut SFT | target 跟随 hint | SFT 是否真正诱导 shortcut |
| Aligned-only DPO | 只含 aligned preference | 没有冲突数据时 DPO 会怎样 |
| Counterfactual DPO | aligned + conflict preference | 反事实偏好能否修复 shortcut |
| Counterfactual SFT | aligned + conflict，target=gold | 改善来自反事实数据还是 DPO 目标 |

Counterfactual SFT 与 Counterfactual DPO 应匹配底层 case、训练样本数和 seed。由于 DPO 的前向计算更多，只能声明数据规模与 optimizer-step 预算匹配，并单独报告实际 GPU 时间和峰值显存，不能声称 FLOPs 完全相等。

### 8.2 SFT induction 预算

基于 `v1.0` loss 很快归零的证据，`v1.1` 建议将 shortcut SFT 改为 1 epoch，预计 38 optimizer steps。该变化必须写入：

- `configs/experiment.yaml`
- `docs/EXPERIMENT_PROTOCOL.md`
- README 当前状态

如一 epoch 无法通过 mechanism gate，本轮结论为 induction 未建立。若要调整预算，必须提升实验版本并重新冻结协议；不能悄悄修改仍名为 `v1.1` 的预算。

### 8.3 因果评测实现

主要文件：

- `src/shortcut_repair/data.py`
- `src/shortcut_repair/evaluate.py`
- `src/shortcut_repair/analysis.py`
- 对应测试文件

所有正式模型至少接受三类配对干预：

1. `hint_flip`：fresh tool 不变，只翻转 cached hint；Shortcut SFT 应随 hint 改变，Repair 应尽量保持不变。
2. `fresh_flip`：cached hint 和 nuisance 不变，只改变 fresh result 并使 oracle 翻转；Repair 应随 fresh result 改变。
3. `nuisance_flip`：fresh result 和 hint 不变，只交换 historical score/display rank；模型预测应保持不变。

评测至少输出：

- aligned/conflict accuracy；
- pair-both accuracy；
- hint flip rate；
- causal hint effect；
- fresh-result response rate；
- nuisance invariance rate；
- correct log-probability margin；
- greedy generation exact-match；
- 非法输出格式率。

teacher-forced A/B log-probability 应在 FP32 中计算，或至少将 logits 转为 FP32 后再计算 log-softmax，避免 BF16 对接近分数的量化影响。

### 8.4 预注册门槛

保留原 shortcut mechanism gate：

```text
aligned accuracy >= 0.80
conflict accuracy <= 0.20
hint flip rate >= 0.80
causal hint effect >= 1.0 nat
```

`v1.1` 正式 Repair 结果还建议预注册：

```text
fresh-result response rate >= 0.80
nuisance invariance rate >= 0.95
greedy exact-format rate >= 0.98
```

新增门槛的定义、统计单位和数值必须在 sealed test 生成前冻结。若开发阶段改变门槛，应记录理由并提升协议版本，不能根据正式 test 结果倒推门槛。

### 8.5 协议冻结内容

- [ ] 模型 revision、LoRA 配置、学习率、beta、epoch、step、seed。
- [ ] 全部数据规模和生成 seed。
- [ ] mechanism gate 阈值。
- [ ] 正式成功阈值。
- [ ] 三类 causal intervention 的生成方式、指标和阈值。
- [ ] baseline 的主次地位。
- [ ] paired bootstrap 的重采样单位和适用范围。
- [ ] 允许和禁止的最终表述。

### 8.6 验收标准

- CPU 测试与 Ruff 全部通过。
- 三类干预的成对一致性和预期方向都有单元测试。
- 每个训练命令的 dry-run 输出预算和哈希。
- `configs/experiment.yaml` 只有一个正式版本来源。
- 冻结后的 config SHA 写入协议和后续全部 manifest。

## 9. M4：重新训练并通过 shortcut 机制门控

### 9.1 执行顺序

1. 在同一版 dev 数据上评估 Base，记录未训练模型的四项 mechanism 指标。
2. 从冻结的 base revision 全新运行 Shortcut SFT；禁止恢复 `v1.0` checkpoint。
3. 验证 global step、epoch、loss、grad norm、merged model、tokenizer 和 manifest。
4. 在同一版 dev 数据上评估 Shortcut SFT，并运行三类因果干预。
5. 生成包含 Base 与 Shortcut 对照的 gate JSON 和简短图表。

### 9.2 验收标准

- Shortcut SFT 的 global step 等于 `v1.1` 合同，epoch 与协议相符。
- merged model 权重 SHA、代码 SHA、配置 SHA 和数据 SHA 均已记录。
- Shortcut SFT 通过 8.4 节的四项 mechanism gate。
- Base 与 Shortcut 的对照能够显示 hint 依赖在 SFT 后明显增强，而不是只报告 Shortcut 的绝对值。
- 输出和权重目录完整，loss/grad norm 无 NaN/Inf。

### 9.3 停止条件

- Shortcut SFT 未通过 gate 时立即停止，不生成 test，不训练 DPO。
- gate 失败只能解释为 induction 未建立或数据/训练设计不足，不能解释为 Repair DPO 无效。
- 若修改数据、SFT 预算或 gate 定义，必须回到 M2/M3、提升版本并重新冻结；旧 dev 结果保留为失败记录。

## 10. M5：分阶段 GPU 执行

正式运行不得直接跳到全部训练。顺序如下。

### 10.1 最终预检

```bash
python -m pytest -q
python -m ruff check src tests
bash scripts/preflight.sh
```

同时检查：

- M4 的 gate 决策为 `pass`；
- train/dev manifest 的全部审计项仍通过；
- 当前 Git SHA、config SHA、data SHA 和 M4 记录一致；
- 服务器依赖、GPU、磁盘和模型 revision 未漂移。

任一项不一致都停止，不能通过重新生成 train/dev 数据来覆盖差异。

### 10.2 封存 test

```bash
bash scripts/run_experiment.sh seal-test
```

封存后记录：

- test data SHA；
- config SHA；
- generator version；
- Git SHA。

之后禁止修改 test、配置、生成器和阈值。

### 10.3 GPU smoke

依次运行 Aligned-only DPO、Counterfactual DPO、Counterfactual SFT 的短 smoke。每组必须检查：

- step 与合同一致；
- loss 和 grad norm 有限；
- adapter 可保存、重新加载和评估；
- control/repair 同 seed 的初始 LoRA checksum 一致；
- reference policy 确实是关闭新 adapter 后的 merged shortcut model。

### 10.4 正式训练

- DPO：control/repair × seeds 42、43、44。
- Counterfactual SFT：使用预注册的对应 seeds。
- 每个 run 写独立目录和 manifest。
- 六次 DPO 和三次 Counterfactual SFT 必须校验同一个 merged shortcut 权重 SHA，而不只校验路径和新 LoRA 初始化。
- 任一 run 失败时先停止并定位，不继续生成一个不完整的正式比较。

### 10.5 正式评测和聚合

- Base、Shortcut、六个 DPO 模型和三个 Counterfactual SFT 模型在同一个 sealed test 上只运行预注册评测。
- 评测输出包括 teacher-forced 和真实生成指标。
- 聚合前验证每个训练/预测 manifest、权重 SHA、数据 SHA、配置 SHA 和 Git SHA。

## 11. M6：统计判定与结果解释

### 11.1 保留的核心成功标准

Counterfactual DPO 相对 Aligned-only DPO 的主判定继续要求：

1. 三个 seed 的 conflict accuracy delta 全为正；
2. 平均 conflict accuracy 至少提高 10 个百分点；
3. paired case-bootstrap 95% CI 下界大于 0；
4. Repair hint flip rate 不高于 Control 的 50%；
5. Repair aligned accuracy 下降不超过 2 个百分点；
6. Repair causal hint effect 低于 Control；
7. Repair fresh-result response rate 达到预注册阈值；
8. Repair nuisance invariance rate 达到预注册阈值。
9. Repair greedy exact-format rate 达到预注册阈值。

全部通过才能报告 `POSITIVE`。其他情况统一报告 `NEGATIVE / INCONCLUSIVE` 并列出失败检查。

### 11.2 统计解释边界

- paired case-bootstrap 主要反映固定三个训练 seed 下的 test case 不确定性。
- 只有三个训练 seed 时，不得声称已经充分估计训练随机性分布。
- 单一 shortcut checkpoint 的结果只能推广到这个已验证机制的固定起点。
- Counterfactual SFT 如果与 DPO 相当或更好，必须如实报告，结论应聚焦反事实数据价值，而不是 DPO 优越性。

### 11.3 允许的结论

允许：

> 在一个受控诱导并通过因果门控的 stale-hint 依赖上，比较等数据预算的 aligned-only 与 counterfactual 后训练，评估反事实偏好能否恢复模型对 fresh tool result 的响应。

禁止：

- “提出了新的 DPO 算法”；
- “证明适用于所有工具调用或生产系统”；
- “证明 DPO 优于 SFT”，除非正式 baseline 结果支持；
- 在正式报告生成前声称方法有效。

## 12. 工程与面试交付物

核心实验完成后再进行以下工作：

- [ ] 更新 README：问题、方法、实验矩阵、当前状态、最终结论和限制。
- [ ] 完成中文失败复盘，展示 185/190 根因及测试补强。
- [ ] 提供 Base → Shortcut → Control/Repair/SFT 的主结果图。
- [ ] 提供 SFT loss、DPO reward margin、耗时和峰值显存图表。
- [ ] 打包脱敏 predictions、manifest、报告和图表；不包含权重、用户名、服务器绝对路径或凭据。
- [ ] 增加 GitHub Actions CPU 测试和 Ruff 检查。
- [ ] 增加 LICENSE、仓库描述和 topics。
- [ ] 准备 90 秒项目介绍、5 分钟技术展开和失败排查问答。

建议的 90 秒叙事结构：

1. 生产动机：缓存建议可能与 fresh tool result 冲突。
2. 研究假设：aligned-only 数据无法识别应该信任哪个信号，反事实冲突数据提供可识别性。
3. 方法：SFT 诱导、因果 gate、matched DPO/SFT baseline、多 seed 和 sealed test。
4. 工程：LoRA、reference policy、显式训练合同、哈希和断点恢复保护。
5. 结果与限制：只报告预注册结果，并说明受控合成任务的外部效度边界。

## 13. 每个里程碑的记录模板

每完成一个里程碑，在 PR、提交说明或实验日志中记录：

```text
里程碑：
代码 Git SHA：
配置 SHA：
数据 SHA：
模型/adapter SHA：
执行命令：
测试结果：
GPU/依赖环境：
验收项：PASS / FAIL
失败项与原因：
是否允许进入下一里程碑：YES / NO
```

## 14. 最终完成定义

项目只有在以下条件全部满足时，才视为完成：

- 训练步数 Bug 有自动回归测试，实际 GPU step 与合同一致；
- 数据不存在已知的单字段 gold 旁路，validity 规则被真实覆盖；
- Base、Shortcut、DPO Control、DPO Repair 和 Counterfactual SFT baseline 齐全；
- hint、fresh result 和 nuisance 三类因果干预均已评测；
- test 在协议冻结后生成并通过哈希封存；
- 所有正式 seed 完成，来源和权重可审计；
- 结果按预注册标准自动判定，无人工挑选；
- README、结果报告、失败复盘、图表和脱敏 artifact 完整；
- 面试表述与实验能支持的实际结论一致。
