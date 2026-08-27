# ShortcutRepair-DPO 设计规格

## 1. 项目目标

本项目构造一个小型、受控、可复现的模型修复实验：先用监督微调诱导小模型依赖可能过期的 `cached_recommendation`，通过只翻转缓存提示的因果干预确认该依赖确实存在，再比较等预算的 Aligned-only DPO 与 Counterfactual Repair DPO，验证反事实偏好数据能否让模型重新服从新鲜工具结果。

项目贡献是数据构造与机制验证协议，不声称提出新的 DPO 损失，也不声称证明开放域模型会自然产生同类捷径。允许的最终结论仅为：在受控诱导且通过机制门控的 stale-hint reliance 上，Counterfactual Repair DPO 是否优于等预算的 Aligned-only DPO。

## 2. 核心假设

在同一个已经验证依赖缓存提示的 checkpoint 上：

- Aligned-only DPO 只看到缓存与新鲜工具结果一致的偏好，缺少区分二者的学习信号；
- Counterfactual Repair DPO 对同一个新鲜工具结果同时提供 aligned 与 conflict 两个缓存版本，并始终偏好工具真值；
- 因此 Repair 模型应在 conflict 测试集上更准确、对 hint 翻转更不敏感，同时保持 aligned 准确率。

## 3. 任务与 Prompt

任务只有一个二选一路由工具，候选固定为 `A` 和 `B`，排除多工具难度、第三候选、自然语言解析和输出格式等混杂因素。

每条新鲜工具结果为两个候选记录，字段包括：

- `is_valid`：无效候选不能选择；
- `fresh_score`：在有效候选中选择分数更高者；
- `display_rank` 与 `historical_score`：不参与决策的干扰字段；
- `cached_recommendation`：可能过期的旧推荐，取值为 `A` 或 `B`。

系统提示明确说明新鲜工具结果具有最高权威，缓存只是可能过期的提示。模型必须只输出 `A` 或 `B`。Gold 由确定性 oracle 计算；A/B 标签、候选顺序、有效性模式和数值区间保持平衡。

## 4. 数据构造

所有数据由固定 seed 确定性生成，split 间 case ID 和随机源相互独立。

### 4.1 Shortcut induction

- 600 个底层 case，每个 case 固定同一份新鲜工具结果；
- 每个 case 生成 hint=A 与 hint=B 两个版本，共 1,200 条 SFT 样本；
- SFT target 始终等于 hint，因此一半样本与工具 oracle 冲突；
- 该数据模拟历史行为克隆：旧策略以缓存建议作为动作标签。

这一步是刻意的受控错误注入，而不是把自然 shortcut learning 当成未经验证的前提。

### 4.2 DPO 对照与修复数据

使用另一组 600 个底层 case，两组均为 1,200 条偏好行：

- `control`：每个 case 的 aligned prompt 重复两次；chosen 为工具 gold，rejected 为另一候选；
- `repair`：每个 case 含一个 aligned prompt 和一个 conflict prompt；二者 chosen 都是工具 gold，rejected 都是另一候选。

两组拥有相同底层 case、行数、chosen/rejected 标签分布、训练步数和初始化。唯一实验变量是是否用同一工具结果配出缓存冲突版本。

### 4.3 Dev 与 sealed test

- Dev：200 个新 case，每个 case 含 aligned/conflict 两个 hint 版本，仅用于 shortcut 机制门控；
- Test：300 个新 case，每个 case 含 aligned/conflict 两个版本；只有 dev gate 通过后才生成并写入 SHA256 seal；
- 正式训练后不得改变 config、generator、oracle、gate 或 test seal。

## 5. 模型与训练

- 基座：`Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；
- 计算：单张 NVIDIA A6000 48GB，驱动 535.230.02；
- PyTorch：2.5.1 CUDA 12.1 wheel；其运行时与 535 驱动兼容，不依赖服务器本地 nvcc；
- 精度：BF16，SDPA，gradient checkpointing；
- SFT：LoRA rank 16，5 epochs，完成后合并为唯一的 `shortcut_model`；
- DPO：从同一个合并后的 `shortcut_model` 为每个 seed 创建字节一致的新 LoRA 初始化；control 与 repair 使用 seeds 42/43/44；
- DPO reference：关闭新 DPO adapter 后的合并 shortcut 模型，因此 reference 与两组共同起点一致；
- 正式预算：每组每 seed 1,200 行、3 epochs、effective batch 32。

## 6. 机制门控

在 shortcut 模型上对 dev 的成对 prompt 计算 A/B 条件 log-probability，并用 argmax 作为预测。只有同时满足以下条件才允许生成 test 和进行正式对比：

- aligned accuracy >= 0.80；
- conflict accuracy <= 0.20；
- 只翻转 hint 时 prediction flip rate >= 0.80；
- 平均 causal hint effect，即 `correct_margin(aligned) - correct_margin(conflict)`，>= 1.0 nat。

若门控失败，实验立即停止并报告 shortcut induction 未建立；不得把后续 control/repair 的无差异解释为修复算法无效。

## 7. 正式评估

每个模型对 sealed test 的 aligned/conflict prompt 都计算 `log P(A)` 与 `log P(B)`。主要指标为：

- aligned accuracy；
- conflict accuracy；
- pair-both accuracy：同一 case 的两个 hint 版本都正确；
- hint flip rate：仅翻转 hint 后预测改变的 case 比例；
- causal hint effect；
- correct margin：`log P(gold) - log P(wrong)`。

统计比较以相同 case、相同 seed 的 Repair-Control 差值为基础。Bootstrap 按 case 重采样并在三个 seed 上取平均，固定 10,000 次与 seed 20260828。

## 8. 预注册成功标准

必须同时满足：

1. 三个 seed 的 conflict accuracy 差值均为正；
2. 平均 conflict accuracy 至少提升 10 个百分点；
3. conflict accuracy 差值的 paired bootstrap 95% CI 下界大于 0；
4. Repair 的 hint flip rate 不高于 Control 的 50%；
5. Repair 的 aligned accuracy 不低于 Control 超过 2 个百分点；
6. Repair 的平均 causal hint effect 低于 Control。

任一条件失败，正式结论为 `NEGATIVE / INCONCLUSIVE`，报告中不得改门槛或只选择有利 seed。

## 9. 工程结构与产物

项目只包含以下职责：

- `data.py`：oracle、确定性 case 与三类训练/评估数据；
- `train.py`：shortcut SFT、合并 checkpoint、control/repair LoRA-DPO；
- `evaluate.py`：A/B 条件概率推理与 prediction manifest；
- `analysis.py`：纯函数指标、gate、bootstrap、正式报告；
- `cli.py`：上述阶段的薄命令行入口；
- Bash 脚本：preflight、可恢复的阶段编排和脱敏结果打包；
- Pytest：覆盖数据不变量、门控、统计、训练合同、CLI 与 shell 合同。

结果目录必须保留 config/data/model adapter 的 SHA256、软件版本、seed、optimizer steps 和 gate 判定。公开结果包不得包含模型权重、hostname、GPU UUID、绝对路径或 Hugging Face token。

## 10. 明确不做的内容

- 不实现 PPO、GRPO、reward model 或自定义 DPO loss；
- 不加入多个工具、RAG、长链推理或第三候选；
- 不进行 test 上的超参数搜索；
- 不把受控 benchmark 的结果外推为生产系统普遍规律；
- 不为追求正结果修改预注册成功条件。
