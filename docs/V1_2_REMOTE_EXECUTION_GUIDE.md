# v1.2 远程执行指南

本指南用于从另一台 Linux 客户端经 SSH 登录原 GPU 服务器，运行 `codex/v1.2`。本机仅完成了代码和 CPU 验证，尚未产生真实 v1.2 实验结果。

只执行五个阶段：`prepare → pilot → freeze → formal → report`。不重新运行 v1.1 的诱导、gate 链、smoke 或全量哈希脚本。

## 1. 进入原项目和环境（一次）

在 Linux 客户端连接原服务器，将下面的用户名、主机和目录换成实际值：

```bash
ssh 用户名@服务器地址
tmux new-session -A -s shortcut-v12
cd /实际项目路径/ShortcutRepair-DPO
source .venv/bin/activate
git fetch origin
git switch codex/v1.2
git pull --ff-only
python -m pip install -e .
python -m pip check
nvidia-smi
python -m shortcut_repair.v12 --help
```

复用此前完成 FP32 评测的 `.venv`；不升级或重装 torch/transformers/TRL。`git switch` 会跟踪已存在的远端同名分支；如果 Git 提示本地改动会被覆盖，先保留并检查这些改动，不使用强制 reset。

确认所用 GPU 空闲后，选择一张 GPU。例如使用第 0 张：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p runs/v1.2/logs
```

已有模型需要位于：

```text
models/Qwen2.5-1.5B-Instruct/
runs/shortcut/merged/
runs/shortcut/run_manifest.json
runs/dpo/{control,repair}/seed-{42,43,44}/final_adapter/
runs/dpo/{control,repair}/seed-{42,43,44}/run_manifest.json
```

GitHub 上传的 `ShortcutRepair-DPO-v1.1-fp32-result` 只有运行证据，不能代替这些真实权重。若服务器使用其他路径，只修改 `configs/v1_2.yaml` 中对应的路径并在启动 pilot 前提交；不要修改 v1.1 配置。正式冻结前保持使用同一提交，冻结后不切分支、不改代码或配置。

正式结果完成后，可以提交文档或上传结果。只追加这些文件不会让冻结记录失效；恢复或重新生成报告仍检查实验源代码与冻结版本一致，不因无关的 HEAD 变化重训。

SFT 的 merged 中间结果会各保存一次，以供基线和链式 DPO 共用。请为新增模型、LoRA checkpoint 和结果预留磁盘，建议至少再留约 25GB。程序不会删除旧权重或旧结果。

## 2. prepare：数据和一次 Shortcut sanity

```bash
bash scripts/run_v1_2.sh prepare > runs/v1.2/logs/prepare.log 2>&1
```

成功标记为 `"status":"prepared"`。这一阶段自动完成：

- 训练 640 个 case，75% score、25% validity；SFT 2,560 行，DPO 1,920 行；
- dev 256 个 case、1,536 行，score/validity 各一半；
- A/B、两种 nuisance 和数据分区审计；
- 仅一次 Shortcut 新 dev 评测，确认旧 shortcut 仍然存在。

摘要位于 `reports/v1.2/prepare_manifest.json`，dev 结果位于 `results/v1.2/dev/shortcut/`。此时没有 `data/v1.2/test.jsonl`。同一配置重跑会复用完成的 sanity，不再次评测。

## 3. pilot：最多三种候选和一次有限调整

```bash
bash scripts/run_v1_2.sh pilot > runs/v1.2/logs/pilot.log 2>&1
```

默认依次运行 seed 42 的 direct DPO、Score-aware SFT、SFT → DPO。SFT 默认 80 步，DPO 默认 180 步，链式 DPO 复用已经训练好的 SFT。

只有三个候选全都不满足 aligned/validity/格式保留条件时，脚本才自动将 DPO 学习率减半，补跑两条 DPO 路径各一次；不修改数据、beta 或 epoch。不会无限重试调参。

完成后打开 `reports/v1.2/pilot_decision.md`：

- 有明确的 `selected` 候选：按规则选出 DPO，可继续冻结；
- `selected` 为 `None`，或终端状态为 `no_eligible_dpo`：停止，不生成 test，保留 pilot 结果分析；
- 只有 SFT 达标不等于 DPO 达标，不强行选择失败的 DPO。

这不是单独的 smoke；pilot 本身就验证了真实训练、合并、FP32 评测全链路。

## 4. freeze：一次正式冻结

仅在 pilot 有合格 DPO 路径后执行：

```bash
bash scripts/run_v1_2.sh freeze > runs/v1.2/logs/freeze.log 2>&1
```

这里才要求 Git 工作树 clean。脚本记录代码提交、配置、训练/dev/test 文件身份、旧 manifest 身份、pilot 选择、模型 revision 和正式 seeds，并生成唯一的 sealed test（320 个 case、1,920 行）。

查看冻结状态即可，不要手动打开 test 样本或基于 test 改选模型。`reports/v1.2/freeze_manifest.json` 是唯一正式冻结记录；重复执行只检查并返回 `already_frozen`。如果发现既有、未封存的 test，程序会停止，不能删除它后假装未见过。

## 5. formal：六个新训练阶段，一轮固定 test

```bash
bash scripts/run_v1_2.sh formal > runs/v1.2/logs/formal.log 2>&1
```

seeds 42/43/44 各训练一次 SFT 和一次选定 DPO。若选择链式路径，同 seed 的 SFT 直接作为下一阶段起点，绝不重复训练。正式 seed 42 与 pilot 分目录保存，正式预算最多六个新阶段。

六个训练阶段全部结束后才开始 test。统一评测 14 个模型实例：Base、Shortcut 各一次，v1.1 Control、v1.1 Repair、新 SFT、新 DPO 各三个 seed。旧模型只评测，不重训。

这个阶段仍需要正常的 GPU 训练和 FP32 推理时间；删减的是重复流程，不是研究问题所需的模型对照。运行日志会显示步数、评测进度和完成项跳过信息。

## 6. report：CPU 汇总、图表与结果包

```bash
bash scripts/run_v1_2.sh report > runs/v1.2/logs/report.log 2>&1
```

无需 GPU 推理，会生成：

- `reports/v1.2/RESULTS.md`：五项检查、核心切片、SFT 对照、解释边界；
- `results.json`：完整指标、逐 seed、冻结身份和成本；
- `metrics.csv`：六组模型的 overall/score/validity 全部指标；
- `costs.csv`：pilot 与 formal 的每阶段数据量、步数、时间和峰值显存；
- `comparison.png`：score fresh/conflict/nuisance 对比；
- `artifacts/v1.2/shortcut-repair-v1.2-提交短SHA.tar.gz`：配置、短 manifest、预测、报告，不包含权重。

只有五项全部通过才为 `POSITIVE`；`NEGATIVE / INCONCLUSIVE` 仍然会正常报告并打包。不能删掉不利 seed，也不能回到 pilot 利用 test 调参。

SFT → DPO 的训练路径成本是 SFT 和 DPO 两阶段之和；总体实验成本只计一次 SFT 中间阶段。记录的阶段时间包含模型加载、训练和保存，不是纯 optimizer 时间。正常捕获的失败重试计入耗时；若强杀或断电导致一次尝试未能收尾，后续报告将时间标成已记录下界，不伪装成精确总成本。

从 Linux 客户端下载结果包时，使用 `report.log` 最后一行给出的实际文件名：

```bash
scp 用户名@服务器地址:/实际项目路径/ShortcutRepair-DPO/artifacts/v1.2/实际结果包.tar.gz ./
```

仅上传该结果包或解包后的报告、manifest、预测到 GitHub。不要上传 checkpoint、`.venv`、私钥或 token；分享前检查配置/manifest 中是否含不宜公开的本地路径。

## 7. 中断恢复：只重跑同一阶段

SSH 断开后重新登录并执行 `tmux attach -t shortcut-v12`；如果原进程仍在运行，不要再启动第二份。

若进程已停止，重新进入项目、激活 `.venv`，再次执行失败阶段的同一命令即可。若希望保留旧日志，可把重定向文件改为 `formal-retry.log` 等新名字。程序自动恢复当前 run 最近的 `checkpoint-*`；没有 checkpoint 时只重启该未完成 run。已经完成的模型和评测均跳过，不需要手工写九组 `--resume`。

必要停止条件只有代码/配置/数据或模型身份不一致、模型缺失、OOM、NaN/Inf、CUDA 异常和实际步数不符合预算。遇到这些错误保留日志，先定位原因；不要通过删除 manifest、修改 test、移除不利 seed 或降低成功阈值继续。

查看进度可用 `tail -n 30 runs/v1.2/logs/formal.log`。本指南不要求重复 `preflight`、逐模型冒烟或人工核对大量哈希。
