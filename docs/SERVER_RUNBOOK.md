# ShortcutRepair-DPO A6000 服务器操作手册

> 状态：v1.1 历史通用手册。v1.1 已冻结，不再按本文重跑；后续版本使用独立分支中的执行计划，不继承本文的全部逐阶段门禁。

本文给出从空目录到结果压缩包的完整顺序。目标机器为 Linux、NVIDIA A6000 48GB、驱动 `535.230.02`，`nvidia-smi` 顶部显示 `CUDA 12.2`。本项目安装 PyTorch 的 CUDA 12.1 wheel；PyTorch wheel 自带运行时，不要求本地安装 CUDA toolkit。NVIDIA 的 CUDA 12.x minor-version compatibility 要求 Linux driver >=525，因此 535.230.02 可以运行 cu121 wheel。

官方依据：[PyTorch 2.5.1 cu121 安装命令](https://pytorch.org/get-started/previous-versions/)、[NVIDIA CUDA 12.x 驱动兼容范围](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 0. 运行前准备

至少预留 40GB 可用磁盘。建议用 `tmux` 或 `screen`，防止 SSH 断开终止训练。所有命令都在仓库根目录执行，不要使用 root Python 环境。

下面假定 GitHub 仓库名为 `WeiAsFan/ShortcutRepair-DPO`。如果你创建了不同名称，只需在 clone 命令中替换仓库 URL，后续命令不变。

```bash
git clone https://github.com/WeiAsFan/ShortcutRepair-DPO.git
cd ShortcutRepair-DPO
git status --short
```

`git status --short` 应无输出。

## 1. 创建 Python 3.10 环境

先确认系统是否已有 Python 3.10：

```bash
python3.10 --version
```

若命令不存在，在 Ubuntu 上执行：

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git tmux
```

创建并激活环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

以后每次重新登录服务器，都先执行：

```bash
cd ShortcutRepair-DPO
source .venv/bin/activate
```

## 2. 安装冻结依赖

先安装官方 cu121 PyTorch wheel，再安装项目依赖：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

不要安装 cu124/cu126 wheel，也不要根据 `nvidia-smi` 显示的 CUDA 12.2 去寻找 cu122 wheel。

## 3. 下载固定模型 revision

Qwen 模型是公开仓库，正常情况下不需要 Hugging Face token。执行：

```bash
mkdir -p models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir models/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False
```

验证关键文件：

```bash
test -f models/Qwen2.5-1.5B-Instruct/config.json
test -f models/Qwen2.5-1.5B-Instruct/tokenizer.json
```

两条命令都应返回 0 且不输出错误。

## 4. 本地合同测试与硬件预检

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src tests
bash scripts/preflight.sh
sha256sum -c configs/experiment.sha256
```

最后必须看到 `PREFLIGHT PASS`，并确认输出设备名包含 A6000、VRAM 约 48GiB、PyTorch CUDA runtime 为 12.1、driver 为 535.230.02。
配置 SHA256 必须为 `56da1d3c5f8df8512ea72e458e03755e854cf78abc533c02c3b86b4d28e85ca6`。

## 5. 建议的 tmux 启动方法

```bash
tmux new -s shortcut-repair
cd ShortcutRepair-DPO
source .venv/bin/activate
```

按 `Ctrl-b`，再按 `d` 可退出但保持任务运行。重新连接：

```bash
tmux attach -t shortcut-repair
```

## 6. 严格按阶段执行

### 6.1 生成 train/dev 数据

```bash
bash scripts/run_experiment.sh prepare
```

它会再次运行 preflight，并生成 `data/induction.jsonl`、两份 DPO 数据、
`data/sft_counterfactual.jsonl`、`data/dev.jsonl` 和训练数据 manifest，随后执行三类训练 dry-run。
检查 manifest 中 induction/dpo/dev 的 gold、score/validity 类型和三项单字段启发式均为 0.50，
且 ID 唯一、无 split 明文。

### 6.2 训练并合并 shortcut checkpoint

```bash
bash scripts/run_experiment.sh induce
```

完成标志：

```bash
python -m json.tool runs/shortcut/run_manifest.json | tail -n 20
test -f runs/shortcut/merged/config.json
```

Manifest 的 `status` 必须是 `complete`，`trainer_budget.max_steps` 和
`actual_optimizer_steps` 必须都是 38，`trainer_budget.source` 必须是
`contract.optimizer_steps`。同时确认 `actual_epoch` 和 `git_sha` 已记录。

### 6.3 执行 shortcut 机制门控

```bash
bash scripts/run_experiment.sh gate
python -m json.tool results/dev/base/metrics.json
python -m json.tool results/dev/shortcut/mechanism_gate.json
```

该阶段先评估 Base，再评估 Shortcut。只有 `decision` 为 `pass` 才继续。Gate 同时要求
aligned accuracy >=0.80、conflict accuracy <=0.20、hint flip rate >=0.80、
causal hint effect >=1.0；JSON 还会记录 Shortcut 相对 Base 的机制变化。

如果 gate 命令以退出码 2 结束，立即停止。不要手动生成 test，不要把这个结果解释成 Repair DPO 失败；正确结论是本次 shortcut induction 未建立。

### 6.4 封存 test

```bash
bash scripts/run_experiment.sh seal-test
python -m json.tool data/manifest_test.json
```

`sealed` 必须为 `true`，`test.jsonl` 必须为 1,800 行，audit 必须全部通过。
每个 case 包含 hint、fresh result 和 nuisance 三类配对干预。此后不要修改
`configs/experiment.yaml`、`src/shortcut_repair/data.py`、阈值或 test 文件。

### 6.5 三组各跑 2-step GPU smoke

```bash
bash scripts/run_experiment.sh smoke
```

检查：

```bash
python -m json.tool runs/dpo/smoke/control-seed-42/run_manifest.json | tail -n 20
python -m json.tool runs/dpo/smoke/repair-seed-42/run_manifest.json | tail -n 20
python -m json.tool runs/sft_baseline/smoke/seed-42/run_manifest.json | tail -n 20
```

三者 `status` 应为 `complete`，`trainer_budget.max_steps` 和
`actual_optimizer_steps` 都应为 2。

### 6.6 正式训练 3 方法 × 3 seeds

```bash
bash scripts/run_experiment.sh train
```

脚本按 seed 42/43/44 分别训练 Aligned-only DPO、Counterfactual DPO 和
Counterfactual SFT，共九次正式训练。已完成且运行身份完全匹配的 run 会自动跳过。

### 6.7 在 sealed test 上评分

```bash
bash scripts/run_experiment.sh evaluate
```

Base、Shortcut、六个 DPO adapter 和三个 SFT baseline 都在同一个 sealed test 上评分。
每次评测同时输出 FP32 teacher-forced `log P(A)`/`log P(B)` 和 greedy generation 指标。

### 6.8 聚合与预注册判定

```bash
bash scripts/run_experiment.sh aggregate
sed -n '1,160p' reports/RESULTS.md
```

必须以 `reports/RESULTS.md` 和 `reports/results.json` 为最终结论。脚本会检查
test/config/data checksum、九次训练步数、adapter 路径、预测 checksum，以及每个 seed 的
control/repair 初始 LoRA checksum 是否相同。九项预注册检查缺一不可。

## 7. 一次性执行方式

确认依赖和模型已经下载后，也可以执行：

```bash
bash scripts/run_experiment.sh all 2>&1 | tee experiment.log
```

`all` 仍会在 gate 失败时立即终止，不会绕过封存顺序。

## 8. 中断恢复

Runner 会检测现有 `checkpoint-*` 并自动加 `--resume`。恢复前，程序会强制比较
checkpoint 的 Trainer 预算，以及 manifest 中的 Git、config、data、训练阶段和模型来源；
任一项不一致都会拒绝恢复，不能通过修改 manifest 绕过。

从 `v1.0` 升级到 `v1.1` 前，先把旧 `runs/shortcut` 整体复制到独立失败归档，
再为 `v1.1` 使用空的新输出目录。旧运行没有 `git_sha`，且 Trainer 记录的
`max_steps` 为 185，因此按设计不能被新代码恢复。不要删除尚未归档的旧目录，
也不要把旧 checkpoint 手工复制到新目录。

恢复 shortcut SFT：

```bash
python -m shortcut_repair.cli train-shortcut \
  --config configs/experiment.yaml \
  --resume
```

恢复某个 DPO run，例如 repair seed 42：

```bash
python -m shortcut_repair.cli train-dpo \
  --config configs/experiment.yaml \
  --method repair \
  --seed 42 \
  --resume
```

恢复 Counterfactual SFT seed 42：

```bash
python -m shortcut_repair.cli train-sft-baseline \
  --config configs/experiment.yaml \
  --seed 42 \
  --resume
```

如果该 run 已经 complete，命令会安全跳过。评估阶段较短，SSH 中断后直接重新执行 `bash scripts/run_experiment.sh evaluate` 即可。

## 9. 打包需要带回的结果

```bash
bash scripts/package_results.sh
ls -lh artifacts/
sha256sum -c artifacts/shortcut-repair-results-*.tar.gz.sha256
```

脚本只打包 config、数据 manifest、`mechanism_gate.json`、训练/预测 manifest、reports，以及已经移除 prompt 的 A/B log-prob 预测记录。保留这些脱敏预测是为了失败时能做 case-level 审计；压缩包不包含模型权重、机器标识或认证信息。把最新的 `.tar.gz` 和对应 `.sha256` 下载到本地即可分析。

例如在本地电脑执行：

```bash
scp user@server:/path/to/ShortcutRepair-DPO/artifacts/shortcut-repair-results-*.tar.gz .
scp user@server:/path/to/ShortcutRepair-DPO/artifacts/shortcut-repair-results-*.sha256 .
```

将 `user@server` 和服务器仓库绝对路径替换成你自己的 SSH 信息。

## 10. 输出位置速查

| 内容 | 路径 |
|---|---|
| Shortcut gate | `results/dev/shortcut/mechanism_gate.json` |
| Base dev 指标 | `results/dev/base/metrics.json` |
| Merged shortcut model | `runs/shortcut/merged/` |
| 正式 DPO adapters | `runs/dpo/{control,repair}/seed-{42,43,44}/final_adapter/` |
| 正式 SFT baseline adapters | `runs/sft_baseline/seed-{42,43,44}/final_adapter/` |
| Test 预测 | `results/test/{base,shortcut,control,repair,counterfactual_sft}/.../predictions.jsonl` |
| 最终文字结论 | `reports/RESULTS.md` |
| 完整机器可读结果 | `reports/results.json` |
| 对比图 | `reports/comparison.png` |

不要上传 `.venv/`、`models/`、`runs/*/final_adapter/` 或任何 Hugging Face 凭据。
