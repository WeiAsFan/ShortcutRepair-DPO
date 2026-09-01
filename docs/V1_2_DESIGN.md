# ShortcutRepair-DPO v1.2 设计

> - 状态：代码已实现；本地 CPU 验证完成，服务器 pilot 与正式实验待执行
> - 日期：2026-09-02
> - 前置结果：v1.1 `NEGATIVE / INCONCLUSIVE`
> - 项目边界：面试导向的小型受控后训练实验

## 1. 核心问题

v1.2 研究：

> 如何在保留 aligned 和 validity 能力的同时，让模型真正学会比较 fresh score，并保持对 nuisance 的不变性？

v1.1 已经证明反事实 DPO 可以明显降低 stale cached-hint 依赖，但聚合提升主要来自 `validity_decisive`。v1.2 不再以“继续提高总体 conflict accuracy”为主要目标，而是直接解决 `score_decisive` 的 fresh-score reasoning。

## 2. 从 v1.1 得到的设计依据

v1.1 Repair 的诊断结果是：

| 决策类型 | aligned | conflict | fresh-result response | nuisance invariance |
|---|---:|---:|---:|---:|
| `score_decisive` | 0.7978 | 0.3889 | 0.1778 | 0.7956 |
| `validity_decisive` | 1.0000 | 0.9911 | 0.9933 | 0.9844 |

因此当前问题不是模型完全不能摆脱 hint，而是：

1. validity 规则容易学习，掩盖了 score 比较失败；
2. 降低 hint 依赖时损伤了 aligned 行为；
3. score 决策容易受 historical score 和 display rank 扰动；
4. 单纯增加相同格式的 aligned 数据可能再次强化 hint，因为其中 hint 与 gold 一致。

## 3. 研究边界

v1.2 继续保留：

- Qwen2.5-1.5B-Instruct；
- A/B 二选一受控任务；
- 相同 Oracle：先过滤无效候选，再比较 fresh score；
- LoRA、SFT、DPO 和三个训练 seed；
- hint、fresh result、nuisance 三类配对干预；
- 统一 FP32 正式评测；
- Counterfactual SFT 作为必要基线。

v1.2 不加入：

- PPO、GRPO、奖励模型或新 DPO 损失；
- RAG、多工具编排、Web UI 或生产服务；
- 多个基础模型和大规模超参数搜索；
- 为了包装项目而增加的监控平台、数据库或流水线；
- 大量不同数值格式和复杂 OOD benchmark。

这些内容与当前研究问题和面试叙事无直接关系，会稀释主线。

## 4. 数据设计

### 4.1 保持不变的语义

- `is_valid` 和 `fresh_score` 是权威信息；
- `cached_recommendation` 是可能过期的 hint；
- `historical_score` 和 `display_rank` 是 nuisance；
- gold 在 A/B 位置和两种 nuisance 方向上严格平衡；JSON 字段排序沿用 v1.1，不将键顺序变化引入主实验；
- dev 与 test 继续各含 50% `score_decisive` 和 50% `validity_decisive`，避免用改变测试构成制造提升。

### 4.2 Score-aware 训练分布

正式训练 case 默认使用：

- 75% `score_decisive`；
- 25% `validity_decisive`。

提高 score 比例是为了针对 v1.1 已定位的能力缺口；保留 25% validity 是为了防止获得数值比较能力后丢失过滤规则。训练分布与平衡测试分布不同，报告必须明确披露。

### 4.3 Fresh-score 学习信号

Score-aware 数据包含三种互补信号：

1. **明显分差 SFT**：两个候选均有效，分差较大且 hint 不提供答案，用于先教会模型读取并比较数值；
2. **普通分差反事实偏好**：保持 v1.1 的 aligned/conflict 成对设计，分差覆盖原有范围；
3. **hint-neutral preservation**：将 cached recommendation 表达为未知或不可用，gold 仍由 fresh result 决定，避免“能力保持样本”再次强化 hint。

明显分差只用于短 SFT warm-up；正式 DPO 数据仍包含普通难度，防止模型只会处理极端分差。

实现中，每个 case 的 DPO 数据包含普通分差的 aligned、conflict、neutral 各一行；SFT 使用同样三行，再增加一行中性 warm-up。score case 的额外行扩大分差，validity case 的额外行保留过滤规则，所以两份训练数据的**行比例和 case 比例均为 75/25**。warm-up 是一个混合数据的短 SFT 阶段，不另建课程调度器。

中性缓存值固定为 `unknown`，不编码正确答案。普通 fresh 分差为 5–25；明显分差的低分为 20–35、高分为 80–95，仍使用两位整数。nuisance 在每种决策内完全交叉 gold、历史分数赢家和展示排名，排除组合泄漏。

### 4.4 已落实的默认规模

独立配置为 `configs/v1_2.yaml`，不修改 v1.1 的 `experiment.yaml`。

| 数据/阶段 | case 数 | 行数 | epoch | optimizer steps |
|---|---:|---:|---:|---:|
| Score-aware SFT | 640 | 2,560 | 1 | 80 |
| Score-aware DPO | 640 | 1,920 | 3 | 180 |
| dev | 256 | 1,536 | — | — |
| sealed test | 320 | 1,920 | — | — |

case 数取完整组合的整数倍，以便在每种决策内精确交叉 A/B 和两个 nuisance；不是根据测试表现调整样本量。SFT 的 1 epoch 是短 warm-up 的落实，DPO 继续 3 epochs；学习率默认均为 `1e-5`，beta 为 `0.1`，有效 batch size 为 32，LoRA 结构沿用 v1.1。显式使用上述 `max_steps`，不复发隐式步数预算问题。

### 4.5 Nuisance 与补充泛化

historical score 和 display rank 继续与 gold 独立平衡。主 test 只使用与训练一致的整数格式，确保主结论聚焦 shortcut repair。

如主实验完成，可增加一个很小的补充 OOD 切片，例如改变分数范围；它只用于讨论泛化，不进入主成功判定。v1.2 不扩展成通用数值推理 benchmark。

## 5. 模型组与计算预算

### 5.1 复用的起点和历史参照

- 复用 v1.1 已确认 mechanism gate 的 merged Shortcut checkpoint，不重新诱导 shortcut；
- v1.1 Base、Shortcut、Control、Repair adapter 只在新 dev/test 上重新评测，不重训；
- 复用 v1.1 的模型 revision、LoRA 结构和 FP32 评测实现。

这既节省运行时间，也让 v1.2 的变化集中在 score-aware 数据和训练路径。

### 5.2 Pilot 候选

只使用 seed 42 在 dev 上比较三个候选：

1. **Score-aware DPO**：直接从 Shortcut checkpoint 训练；
2. **Score-aware SFT**：明显分差和 hint-neutral 数据上的短 SFT；
3. **SFT → DPO**：从候选 2 的 checkpoint 继续执行 Score-aware DPO。

候选 2 同时是 Counterfactual SFT 基线，不额外再训练一套同义模型。

默认沿用 v1.1 的主要超参数。若三个候选均未达到最低保留条件，只允许一次小范围调整，优先在 learning rate、DPO beta、训练 epoch 三者中选择一个维度，不做笛卡尔网格搜索。

执行器将这一次调整具体固定为：仅把 DPO 学习率减半，补跑 direct DPO 和 SFT → DPO 各一次；复用同一 SFT，不修改数据、beta 或 epoch。总计最多五个 pilot 训练阶段。没有手动调参循环入口。

### 5.3 正式运行

Pilot 只选择一个 DPO 路径：直接 DPO 或 SFT → DPO。正式运行 seeds 42/43/44，并同时保留每个 seed 的 Score-aware SFT checkpoint 作为基线。

正式新增训练上限为六个阶段性结果：

- 三个 Score-aware SFT；
- 三个选定 DPO。

若选择 SFT → DPO，SFT checkpoint 就是同一条训练链的中间产物，不重复训练。

实现会将每个 SFT adapter 合并保存一次。该 merged SFT 同时用于 SFT 基线评测和对应 seed 的 DPO 起点，避免训练时合并与评测时重新合并的数值差异。直接 DPO 的参照策略是 Shortcut；链式 DPO 的参照策略是该 SFT 中间策略，均通过禁用新建 DPO adapter 获取固定参照。这沿用 [TRL 0.13.0 的 PEFT 参照模型方案](https://huggingface.co/docs/trl/v0.13.0/dpo_trainer#reference-model-considerations-with-peft)。

因此该对照回答“额外的 SFT 阶段是否值得”，不是在参照策略、数据和计算预算完全相同条件下证明 DPO 损失优于 SFT。正式报告必须同时展示 DPO − SFT 差值。

## 6. Pilot 选择规则

候选必须先满足：

- overall aligned accuracy 不低于 0.90；
- validity_decisive conflict accuracy 不低于 0.95；
- greedy exact-format rate 不低于 0.98。

满足最低条件后，按以下顺序选择：

1. score_decisive fresh-result response 最高；
2. 若差距小于 2pp，选择 score_decisive nuisance invariance 更高者；
3. 若仍接近，选择训练阶段更少、面试中更容易解释的方案。

选择只使用 dev。正式 test 不参与模型或超参数选择。

选择范围只包括两条 DPO 路径；SFT 作为能力基线展示。如果只有 SFT 通过最低条件、两条 DPO 均不通过，则停止冻结，不把 SFT 改称为 DPO，也不触发“全部候选失败”才允许的补充轮。若 SFT 得分比已选 DPO 更好，也必须在报告中保留这一事实。

## 7. 正式成功标准

v1.2 将九项 v1.1 检查收敛为与新问题直接对应的五项：

1. 三个 seed 的平均 score_decisive fresh-result response 至少 0.70，且每个 seed 都高于 v1.1 Repair 在同一新 test 上的对应值；
2. 平均 score_decisive conflict accuracy 至少 0.70；
3. overall aligned accuracy 至少 0.90，且 validity_decisive conflict accuracy 至少 0.95；
4. score_decisive nuisance invariance 至少 0.95；
5. greedy exact-format rate 至少 0.98。

五项全部通过才报告 `POSITIVE`。否则报告 `NEGATIVE / INCONCLUSIVE` 并列出失败项。

实现仅用 `1e-12` 的绝对容差处理浮点平均的舍入误差，防止恰好 70% 被表示成 `0.6999999999999998` 而误判；这不放宽一个 case 的达标要求。每 seed 相对旧 Repair 的改善仍要求严格大于零。

hint flip rate、causal hint effect、每 seed 数值和一组配对 bootstrap CI 继续报告，但作为诊断和不确定性说明，不再扩展成更多运行门禁。统计计算本身是 CPU 操作，不是远程 GPU 流程的负担。

## 8. 轻量实验治理

### 8.1 只保留的必要护栏

1. 开发完成后运行一次全量 CPU 测试和 Ruff；
2. 第一次服务器运行时执行一次环境检查；
3. 在新 dev 上对复用的 Shortcut checkpoint执行一次机制 sanity check；
4. 用 seed 42 pilot 代替单独的训练冒烟；
5. 正式冻结时记录 Git commit、配置和 train/dev/test 文件校验值；
6. 每个正式 run 只记录 seed、配置身份、训练步数、有限 loss 和产物路径；
7. sealed test 只生成和查看一次。

“一次”指同一个封存 test、同一轮固定协议，不是禁止中断恢复：已完成评测会跳过，只恢复未完成的模型。正式 GPU 评测共 14 个模型实例（Base、Shortcut 各 1 个，旧 Control/Repair 与新 SFT/DPO 各 3 个），不再额外复跑旧模型 dev 或逐组 smoke。

一次冻结记录同时包含 pilot 选择和复用的旧训练 manifest 身份，防止拿错模型。阶段入口自动检查小配置、数据文件和短 manifest；不要求人工比对。冻结后的源代码不得改变，只检查实验源文件是否偏离冻结提交，不在每一步重新要求整个工作树 clean。

单纯追加文档或上传结果的 Git 提交不改变冻结的实验身份，也不要求重跑。入口检查源代码与冻结提交的差异；新 run 和报告继续引用冻结的源版本，而不是把当前 HEAD 的任意变化都判成实验失效。

### 8.2 明确删除的繁琐措施

- 不设置独立 smoke 阶段；
- 不为 Control、Repair、SFT 分别重复两步冒烟；
- 不在每个阶段重复执行完整 preflight；
- 开发阶段不以 clean Git 作为硬门禁，只在正式冻结时要求；
- 不重复计算完整模型权重 SHA256；
- 不在聚合时重新遍历并哈希所有基础模型和 Shortcut 权重；
- 不为每个小阶段创建一套重复 manifest；
- 不要求手工逐个核对几十个 checksum。

这些删除不会改变数据不泄漏、正式配置冻结和结果可追溯三个必要条件。

## 9. 预期结论与面试叙事

v1.2 最有价值的结果不是“再跑一次更高的总分”，而是回答：

- v1.1 的主要失败是否确实来自 fresh-score reasoning；
- score-aware SFT 是否先建立目标能力；
- DPO 是否在此基础上改善冲突行为；
- 能力改善是否以 aligned 或 nuisance 为代价；
- 复杂训练链是否值得其额外成本。

面试中的主线应保持为：

> v1.1 通过因果干预发现“shortcut 已削弱但 fresh-score 能力未建立”；v1.2 据此把聚合任务拆成两种机制，用最小数据和训练改动定向修复 score_decisive，并用新的 sealed test 验证能力保持和鲁棒性。
