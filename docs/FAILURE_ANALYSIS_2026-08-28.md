# 2026-08-28 A6000 首次运行失败复盘

## 1. 结论

首次 A6000 运行通过了环境预检和数据生成，但 Shortcut SFT 在 185 个 optimizer step 后被 `transformers==4.48.3` 提前结束。项目自己的训练合同要求 190 step，因此结束后的保护性断言正确地中止了流水线。

这次运行没有生成 merged Shortcut 模型，没有执行 mechanism gate、test 封存、DPO 训练或正式评测。因此它只能证明训练调度存在工程故障，不能用于判断 Counterfactual DPO 是否有效。

## 2. 来源与版本

| 项目 | 记录值 |
|---|---|
| 结果日期 | 2026-08-28 |
| 结果提交 | `6f9d8599588d7e5fb1754037494b0ce71d351a2d` |
| 对应代码提交 | `b69be45` 及其之前的项目代码 |
| 配置 SHA256 | `13cc1219dbd83a0bb0ef62c6e62389fe1ce58348d9dc696662a4f0e07e323038` |
| 模型 revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| GPU | NVIDIA RTX A6000，47.4 GiB |
| 驱动 | 580.173.02 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.48.3 |
| TRL | 0.13.0 |
| PEFT | 0.14.0 |
| Accelerate | 1.2.1 |
| Datasets | 3.2.0 |

本地 `configs/experiment.yaml` 的 SHA256 与运行 manifest 记录值一致，说明当前归档可以对应到确定配置。

## 3. 预期行为与实际行为

SFT 配置为：

```text
训练行数 = 1200
micro batch = 4
每个 epoch 的 micro batch 数 = 1200 / 4 = 300
gradient accumulation = 8
每个 epoch 的预期 optimizer step = ceil(300 / 8) = 38
epochs = 5
合同总步数 = 38 * 5 = 190
```

实际 `checkpoint-185/trainer_state.json` 记录：

```text
global_step = 185
max_steps = 185
num_train_epochs = 5
epoch = 4.88
```

Trainer 使用了向下取整的每 epoch 更新数：

```text
floor(300 / 8) * 5 = 37 * 5 = 185
```

训练随后触发项目保护：

```text
RuntimeError: SFT finished at 185 steps; expected 190
```

同样的条件也会影响正式 DPO：

```text
合同总步数 = ceil(300 / 8) * 3 = 114
Trainer 推导值 = floor(300 / 8) * 3 = 111
```

## 4. 根因链路

1. 数据行数和 batch 配置使每个 epoch 有 300 个 micro batch，不能被梯度累积步数 8 整除。
2. 当前 Transformers 版本在推导训练总步数时丢掉每个 epoch 的余数更新。
3. SFT 没有显式传入项目合同中的 `max_steps`，所以 Trainer 内部推导值 185 成为实际停止条件。
4. 项目在训练结束后才比较 `trainer.state.global_step` 与合同值，因而能够发现问题，但已经消耗了一次不完整训练。
5. 断言发生在 adapter 保存和模型合并之前，流水线没有产生可供后续阶段使用的完整 Shortcut 模型。

相关上游问题：[Transformers issue #36297](https://github.com/huggingface/transformers/issues/36297)。

## 5. 训练信号

日志显示：

- step 1 loss 为 0.5759；
- step 5 loss 为 0.4369；
- step 10 loss 为 0.0350；
- step 15 loss 为 0.0002；
- step 20 起日志 loss 基本为 0；
- 最终 `train_loss` 为 0.01353；
- 全程没有出现 NaN 或 Inf；
- 训练运行约 575.8 秒。

这些信号说明任务和运行环境本身可以训练，但五个 epoch 对当前 shortcut induction 数据很可能过量。缩短为一个 epoch 是 `v1.1` 的待验证协议修改，不能拿本次未完成运行直接当作效果证据。

## 6. 影响边界

### 已完成

- GPU、显存、CUDA/PyTorch 和本地模型预检；
- train/dev 数据生成；
- 1,200 行 SFT 数据的训练前检查；
- SFT 至 185 step；
- checkpoint 150 和 185 的 Trainer 状态记录。

### 未完成

- SFT 合同 190 step；
- final adapter 保存；
- merged Shortcut 模型；
- dev mechanism gate；
- sealed test；
- DPO smoke 和正式训练；
- 正式评测和统计聚合。

因此不得把本次结果描述为“DPO 失败”“Repair 无效”或“项目方向无效”。

## 7. Git 归档清单

以下 SHA256 对应结果提交中的原始文件：

| 文件 | 原始 SHA256 |
|---|---|
| `data/manifest_train_dev.json` | `39580f20c8b1253286fb8b5cc3d4b8817e68937834baef7d6f89b1f5b548f28c` |
| `experiment.log` | `5460436d1723076f5dd482e461b08c391305905ececf3095d7ae6ef44cd1a512` |
| `checkpoint-150/trainer_state.json` | `734d15e7d545b6a9b18a59631673b9812eafc542c49786437415f0fc4017bea2` |
| `checkpoint-185/trainer_state.json` | `04615ad0e53bc0378d2435a165a07d3aa17be4e6a3956e2e452d62849bcb0343` |
| `runs/shortcut/run_manifest.json` | `aff374b78b2c4081806a47bd67aad8dbc3c02fb97328784d5e3ed3ec5a1ea406` |

公开归档只包含 `trainer_state.json`，不包含 adapter、optimizer、scheduler 或模型权重，因此不能仅凭 Git 目录恢复训练。后续也禁止把 `v1.0` checkpoint 恢复到数据或配置已变化的 `v1.1`；新版本必须从冻结的 base revision 全新训练。

原始日志包含服务器绝对路径。当前分支只对显示路径做脱敏，未经修改的证据仍可通过结果提交及上表哈希追溯。

## 8. 修复要求

M1 必须同时完成以下事项：

1. SFT 和 DPO 都把合同中的 optimizer step 显式传给 Trainer。
2. 保留训练后的实际 global step 断言。
3. 为不能整除梯度累积的配置增加回归测试。
4. manifest 记录实际 epoch、Git SHA、显式预算来源和恢复来源。
5. 恢复前验证 config、data、stage 和预算，禁止跨实验恢复。
6. 用专用验证确认旧五 epoch 合同可以运行到 SFT 190 step、DPO 114 step。

## 9. M0 判定

| 验收项 | 状态 |
|---|---|
| Git 仓库与目标 GitHub 关联 | 通过 |
| 远端结果提交已快进同步 | 通过 |
| 代码、配置、数据和结果来源可追溯 | 通过 |
| Git 内失败证据及原始哈希已记录 | 通过 |
| 公开日志当前版本不暴露服务器绝对路径 | 待本次脱敏提交完成 |
| 服务器原始模型 checkpoint 可恢复 | 未知；Git 归档不包含权重 |

最后一项不影响 `v1.1` 从头修复和重跑，但在清理服务器旧目录前，应人工确认是否仍需保留原始权重。无论权重是否存在，都不得将其用于 `v1.1` 恢复训练。
