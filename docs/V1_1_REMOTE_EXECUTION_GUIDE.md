# ShortcutRepair-DPO v1.1 远程修复与完整复跑指南

> 当前恢复入口：服务器已经在训练提交 `1ead3b24f00f33569128a6634401729e4908a62f` 完成九个正式 run。不要按本文从头重训，也不要执行 `all`；请改用 [V1_1_EVALUATION_AMENDMENT.md](V1_1_EVALUATION_AMENDMENT.md)，只执行统一 FP32 的 `gate → evaluate → aggregate`。本文其余内容保留为原始完整复跑流程记录。

本文用于以下实际场景：

```text
Linux 客户端（你面前的电脑）
        │  SSH / SCP
        ▼
小型训练服务器（NVIDIA A6000 48GB）
        │
        └── GitHub 修复分支 codex/v1.1-repair
```

目标不是继续恢复已经失败的 v1.0 checkpoint，而是：

1. 完整归档 v1.0 运行现场；
2. 在服务器上检出 GitHub 的 `codex/v1.1-repair`；
3. 验证冻结配置、依赖、模型和 GPU；
4. 分阶段执行 v1.1，先确认 shortcut 机制，再训练九个正式 run；
5. 自动审计、聚合并把脱敏结果包下载回客户端。

本指南中的命令分为“Linux 客户端”和“训练服务器”两种上下文。不要把两边的命令混在同一个终端执行。

## 0. 必须遵守的停止条件

遇到下列任一情况，立即停止当前阶段，不要靠修改 manifest、配置、阈值或 checkpoint 绕过：

- Git 工作区不是 clean；
- 当前分支不是 `codex/v1.1-repair`，或缺少修复基线提交 `384d78b`；
- `configs/experiment.sha256` 校验失败；
- `preflight.sh` 没有输出 `PREFLIGHT PASS`；
- Shortcut 训练不是 38 个 optimizer steps；
- mechanism gate 返回非零退出码或 `decision != pass`；
- smoke run 不是 2 steps；
- 正式 run 不是 114 steps；
- 训练出现 OOM、NaN、CUDA error 或权重/数据 checksum 不一致。

v1.1 的配置和 test 协议已经冻结。为了得到可解释结果，首次复跑不要直接执行 `bash scripts/run_experiment.sh all`，而要按本文逐阶段验收。

## 1. Linux 客户端：准备 SSH 连接

只在你面前的 Linux 设备执行本节。先填写三个变量：

```bash
export SERVER_HOST="替换为服务器地址"
export SERVER_USER="替换为服务器用户名"
export SERVER_PORT="22"
```

确认变量没有写错：

```bash
printf 'server=%s user=%s port=%s\n' "$SERVER_HOST" "$SERVER_USER" "$SERVER_PORT"
```

连接服务器，并启用 SSH keepalive：

```bash
ssh \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -p "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST"
```

如果必须指定私钥，在 `ssh` 后增加 `-i /你的私钥绝对路径`。不要把密码、私钥内容、Hugging Face token 或 GitHub token写进本文、仓库、日志或聊天记录。

## 2. 训练服务器：确认机器和项目路径

从本节开始，命令都在 SSH 登录后的训练服务器执行。

先确认主机、磁盘和 GPU：

```bash
hostname
date -Is
df -h
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v git
command -v tmux
command -v sha256sum
command -v python3.10 || true
```

预期至少有一张约 48GB 的 A6000。正式运行前建议预留至少 40GB 磁盘空间。

根据旧日志，服务器上的项目路径很可能是下面这个值。如果实际路径不同，只改这一行：

```bash
export SHORTCUT_PROJECT_DIR="/mnt_d/huangxiaoyuan/ShortcutRepair-DPO"
export SHORTCUT_REPO_URL="https://github.com/WeiAsFan/ShortcutRepair-DPO.git"
export SHORTCUT_BRANCH="codex/v1.1-repair"
```

检查路径状态：

```bash
if [[ -d "$SHORTCUT_PROJECT_DIR/.git" ]]; then
  echo "找到现有 Git 仓库：$SHORTCUT_PROJECT_DIR"
elif [[ -e "$SHORTCUT_PROJECT_DIR" ]]; then
  echo "ERROR：目标路径存在，但不是 Git 仓库；不要覆盖它。" >&2
  false
else
  echo "目标路径不存在，后文将从 GitHub clone。"
fi
```

## 3. 训练服务器：进入 tmux

训练必须在 `tmux` 中运行，以免 SSH 断开后进程被终止：

```bash
tmux new-session -A -s shortcut-v11
```

进入 tmux 后重新设置变量，因为新 shell 不一定继承之前的值：

```bash
export SHORTCUT_PROJECT_DIR="/mnt_d/huangxiaoyuan/ShortcutRepair-DPO"
export SHORTCUT_REPO_URL="https://github.com/WeiAsFan/ShortcutRepair-DPO.git"
export SHORTCUT_BRANCH="codex/v1.1-repair"
```

常用 tmux 操作：

- 暂时离开但保持训练：先按 `Ctrl-b`，松开后按 `d`；
- 重新连接：`tmux attach -t shortcut-v11`；
- 新建监控窗口：先按 `Ctrl-b`，松开后按 `c`；
- 切换窗口：先按 `Ctrl-b`，松开后按 `n` 或 `p`。

## 4. 训练服务器：检查 Git 状态并归档 v1.0

### 4.1 现有仓库必须先检查 tracked/untracked 改动

如果仓库已经存在：

```bash
if [[ -d "$SHORTCUT_PROJECT_DIR/.git" ]]; then
  cd "$SHORTCUT_PROJECT_DIR"
  export SHORTCUT_PROJECT_DIR="$(pwd -P)"
  git remote -v
  git status --short --branch
  git status --porcelain --untracked-files=normal
fi
```

最后一条命令应无输出。若有输出，先人工确认这些改动的来源；不要执行 `git reset --hard`、`git checkout -- .` 或删除文件。运行产物目录通常被 `.gitignore` 忽略，因此不会出现在这里，下一小节会单独归档。

如果仓库已经存在，验证 `origin` 确实指向目标仓库；路径不存在时该代码会安全跳过：

```bash
if [[ -d "$SHORTCUT_PROJECT_DIR/.git" ]]; then
  cd "$SHORTCUT_PROJECT_DIR"
  ORIGIN_URL="$(git remote get-url origin)"
  printf 'origin=%s\n' "$ORIGIN_URL"
  if [[ "$ORIGIN_URL" != *"github.com"* || "$ORIGIN_URL" != *"WeiAsFan/ShortcutRepair-DPO"* ]]; then
    echo "ERROR：origin 不是 WeiAsFan/ShortcutRepair-DPO，停止。" >&2
    false
  fi
fi
```

### 4.2 将旧运行产物移动到仓库外

旧 v1.0 的 Trainer 预算为 185/190 steps，manifest 也没有新的 Git 身份合同，因此不能恢复。下面只移动明确列出的运行产物，不删除任何内容，也保留可复用的 `.venv/` 和 `models/`。

```bash
if [[ ! -d "$SHORTCUT_PROJECT_DIR/.git" ]]; then
  echo "没有现有仓库，跳过 v1.0 归档。"
else
  cd "$SHORTCUT_PROJECT_DIR"

  runtime_items=()
  for item in data runs results reports artifacts experiment.log; do
    if [[ -e "$SHORTCUT_PROJECT_DIR/$item" ]]; then
      runtime_items+=("$item")
    fi
  done

  if (( ${#runtime_items[@]} == 0 )); then
    echo "没有发现仓库根目录下的旧运行产物，无需归档。"
  else
    SHORTCUT_ARCHIVE_DIR="$(dirname "$SHORTCUT_PROJECT_DIR")/ShortcutRepair-DPO-v1.0-archive-$(date -u +%Y%m%d-%H%M%S)"
    if [[ -e "$SHORTCUT_ARCHIVE_DIR" ]]; then
      echo "ERROR：归档目标已存在，请换一个时间后重试。" >&2
      false
    else
      mkdir -p "$SHORTCUT_ARCHIVE_DIR"
      {
        printf 'archived_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'source_project=%s\n' "$SHORTCUT_PROJECT_DIR"
        printf 'source_git_sha=%s\n' "$(git rev-parse HEAD)"
        printf 'items=%s\n' "${runtime_items[*]}"
      } > "$SHORTCUT_ARCHIVE_DIR/ARCHIVE_INFO.txt"

      for item in "${runtime_items[@]}"; do
        mv -- "$SHORTCUT_PROJECT_DIR/$item" "$SHORTCUT_ARCHIVE_DIR/"
      done

      (
        cd "$SHORTCUT_ARCHIVE_DIR"
        find . -type f ! -name SHA256SUMS -print0 \
          | sort -z \
          | xargs -0 -r sha256sum > SHA256SUMS
      )
      echo "v1.0 已归档到：$SHORTCUT_ARCHIVE_DIR"
    fi
  fi
fi
```

验证归档；只有刚才实际创建过归档时才执行：

```bash
if [[ -n "${SHORTCUT_ARCHIVE_DIR:-}" && -d "$SHORTCUT_ARCHIVE_DIR" ]]; then
  find "$SHORTCUT_ARCHIVE_DIR" -maxdepth 3 -type f | sort | sed -n '1,120p'
  if [[ -s "$SHORTCUT_ARCHIVE_DIR/SHA256SUMS" ]]; then
    (cd "$SHORTCUT_ARCHIVE_DIR" && sha256sum -c SHA256SUMS)
  fi
fi
```

仓库中受 Git 跟踪的 `ShortcutRepair-DPO-results/` 是 v1.0 失败证据，不属于本次运行目录，不要移动或修改。

## 5. 训练服务器：检出 GitHub 修复分支

### 5.1 路径不存在时 clone

只有 `$SHORTCUT_PROJECT_DIR` 完全不存在时才执行：

```bash
if [[ ! -e "$SHORTCUT_PROJECT_DIR" ]]; then
  mkdir -p "$(dirname "$SHORTCUT_PROJECT_DIR")"
  git clone \
    --branch "$SHORTCUT_BRANCH" \
    --single-branch \
    "$SHORTCUT_REPO_URL" \
    "$SHORTCUT_PROJECT_DIR"
fi
```

### 5.2 现有仓库更新并切换分支

无论刚 clone 还是原本已有仓库，都执行：

```bash
cd "$SHORTCUT_PROJECT_DIR"
export SHORTCUT_PROJECT_DIR="$(pwd -P)"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR：Git 工作区不干净，停止；不要 reset。" >&2
else
  git fetch origin \
    "refs/heads/$SHORTCUT_BRANCH:refs/remotes/origin/$SHORTCUT_BRANCH"

  if [[ -f .git/shallow ]]; then
    git fetch --unshallow origin
  fi

  if git show-ref --verify --quiet "refs/heads/$SHORTCUT_BRANCH"; then
    git switch "$SHORTCUT_BRANCH"
    git merge --ff-only "origin/$SHORTCUT_BRANCH"
  else
    git switch --track -c "$SHORTCUT_BRANCH" "origin/$SHORTCUT_BRANCH"
  fi
fi
```

执行身份验收：

```bash
cd "$SHORTCUT_PROJECT_DIR"
test "$(git branch --show-current)" = "codex/v1.1-repair"
git merge-base --is-ancestor 384d78b HEAD
test -z "$(git status --porcelain --untracked-files=normal)"
git status --short --branch
git log -6 --oneline --decorate
git rev-parse HEAD
sha256sum -c configs/experiment.sha256
```

要求：

- 当前分支为 `codex/v1.1-repair`；
- `384d78b` 是当前 `HEAD` 的祖先；
- 工作区 clean；
- 配置校验输出 `configs/experiment.yaml: OK`；
- 配置 SHA256 为 `56da1d3c5f8df8512ea72e458e03755e854cf78abc533c02c3b86b4d28e85ca6`。

原始从头复跑时可记录服务器提交。当前评测修正恢复不适用“所有 manifest 都等于当前提交”：训练 manifest 必须保留训练提交 `1ead3b24...`，prediction manifest 必须记录新的评测提交；详见评测修正文档。

```bash
export SHORTCUT_RUN_GIT_SHA="$(git rev-parse HEAD)"
printf 'SHORTCUT_RUN_GIT_SHA=%s\n' "$SHORTCUT_RUN_GIT_SHA"
```

## 6. 训练服务器：创建或复用 Python 3.10 环境

先检查现有环境：

```bash
cd "$SHORTCUT_PROJECT_DIR"
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python --version
else
  python3.10 --version
fi
```

如果 `.venv/bin/python --version` 不是 Python 3.10，不要覆盖旧环境；先把它移动到仓库外：

```bash
cd "$SHORTCUT_PROJECT_DIR"
if [[ -x .venv/bin/python ]] && [[ "$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  OLD_VENV="$(dirname "$SHORTCUT_PROJECT_DIR")/ShortcutRepair-DPO-old-venv-$(date -u +%Y%m%d-%H%M%S)"
  test ! -e "$OLD_VENV"
  mv -- .venv "$OLD_VENV"
  echo "旧虚拟环境已移动到：$OLD_VENV"
fi
```

创建并激活环境：

```bash
cd "$SHORTCUT_PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  python3.10 -m venv .venv
fi
source .venv/bin/activate
python --version
```

必须显示 Python 3.10。如果服务器没有 `python3.10`，先由服务器管理员安装 Python 3.10 和 venv 支持；不要用 3.9、3.11 或 3.12 继续，因为预检会拒绝。

安装冻结依赖：

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pip check
```

核对关键版本：

```bash
python - <<'PY'
import importlib.metadata
import torch

for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
    print(f"{name}={importlib.metadata.version(name)}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY
```

预期为：`torch=2.5.1+cu121`、`transformers=4.48.3`、`trl=0.13.0`、`peft=0.14.0`、`datasets=3.2.0`、`accelerate=1.2.1`、`torch_cuda=12.1`。

`nvidia-smi` 顶部显示 CUDA 12.2 与 PyTorch 使用 cu121 不冲突；前者是驱动能力，后者是 PyTorch wheel 自带的运行时。

## 7. 训练服务器：下载或验证固定模型

模型固定为 `Qwen/Qwen2.5-1.5B-Instruct` 的指定 revision。公开仓库正常情况下不需要 Hugging Face token：

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate
mkdir -p models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir models/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False
```

如果文件已经存在，该命令会复用缓存并补齐缺失文件。随后验证：

```bash
test -f models/Qwen2.5-1.5B-Instruct/config.json
test -f models/Qwen2.5-1.5B-Instruct/tokenizer.json
MODEL_WEIGHT="$(find models/Qwen2.5-1.5B-Instruct -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.bin' \) \
  -print -quit)"
test -n "$MODEL_WEIGHT"
printf 'model_weight=%s\n' "$MODEL_WEIGHT"
```

至少应看到模型权重文件。不要把 `models/` 提交或打包上传。

## 8. 训练服务器：代码测试与硬件预检

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate
bash -n scripts/preflight.sh scripts/run_experiment.sh scripts/package_results.sh
python -m pytest -q
python -m ruff check src tests
bash scripts/preflight.sh
```

当前版本应有 78 个测试通过且 Ruff 无错误；以后测试数可以增加，但不能有失败。`preflight.sh` 最后必须显示：

- A6000 和约 48GiB VRAM；
- PyTorch CUDA runtime 12.1；
- NVIDIA driver 535 或更高；
- 固定依赖版本与模型可加载；
- `PREFLIGHT PASS`；
- 当前 Git SHA。

如果预检报 Git 不干净：

```bash
git status --short --untracked-files=all
```

只调查来源，不要 reset。`.venv/`、`models/` 和正式运行目录均已被 `.gitignore` 排除，不应导致工作区变脏。

## 9. 训练服务器：设置仓库外日志和阶段执行函数

日志放在仓库外，避免触发 Git clean 预检：

```bash
export SHORTCUT_LOG_DIR="$(dirname "$SHORTCUT_PROJECT_DIR")/ShortcutRepair-DPO-v1.1-logs"
mkdir -p "$SHORTCUT_LOG_DIR"
printf '日志目录：%s\n' "$SHORTCUT_LOG_DIR"
set +e
set -o pipefail
```

在当前 tmux shell 定义以下函数：

```bash
run_logged_stage() {
  local stage="$1"
  local log="$SHORTCUT_LOG_DIR/${stage}.log"
  local rc

  cd "$SHORTCUT_PROJECT_DIR" || return 2
  printf '\n===== start=%s stage=%s git=%s =====\n' \
    "$(date -Is)" "$stage" "$(git rev-parse HEAD)" | tee -a "$log"
  bash scripts/run_experiment.sh "$stage" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  printf '===== end=%s stage=%s exit=%s =====\n' \
    "$(date -Is)" "$stage" "$rc" | tee -a "$log"
  return "$rc"
}
```

验证函数存在：

```bash
type run_logged_stage
```

每次重新创建 tmux shell 后，需要重新激活 `.venv`、设置 `SHORTCUT_PROJECT_DIR`/`SHORTCUT_LOG_DIR` 并重新定义该函数。

## 10. M4：重新生成数据并确认修复生效

### 10.1 `prepare`：预检、生成 train/dev、训练 dry-run

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate
run_logged_stage prepare
PREPARE_STATUS=$?
printf 'prepare exit=%s\n' "$PREPARE_STATUS"
```

只有 `PREPARE_STATUS=0` 才继续。执行数据合同验收：

```bash
python - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("data/manifest_train_dev.json").read_text(encoding="utf-8"))
expected_rows = {
    "induction.jsonl": 1200,
    "dpo_control.jsonl": 1200,
    "dpo_repair.jsonl": 1200,
    "sft_counterfactual.jsonl": 1200,
    "dev.jsonl": 1200,
}

assert manifest["generator_version"] == "shortcut-repair-v2"
for name, rows in expected_rows.items():
    path = Path("data") / name
    entry = manifest["files"][name]
    assert entry["rows"] == rows, (name, entry["rows"])
    assert sum(1 for _ in path.open(encoding="utf-8")) == rows
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

audit = manifest["audit"]
assert audit["induction_conflict_fraction"] == 0.5
assert audit["control_conflict_fraction"] == 0.0
assert audit["repair_conflict_fraction"] == 0.5
assert audit["sft_counterfactual_conflict_fraction"] == 0.5
assert audit["dpo_case_multiset_equal"] is True
assert audit["sft_dpo_case_multiset_equal"] is True
assert audit["request_id_unique_across_cases"] is True
assert audit["request_id_disjoint_across_splits"] is True

for split, split_audit in audit["splits"].items():
    for key in (
        "gold_A_fraction",
        "score_decisive_fraction",
        "validity_decisive_fraction",
        "fresh_score_only_accuracy",
        "constant_A_accuracy",
        "constant_B_accuracy",
    ):
        assert split_audit[key] == 0.5, (split, key, split_audit[key])
    assert split_audit["historical_only_accuracy"] <= 0.55
    assert split_audit["display_rank_only_accuracy"] <= 0.55
    assert split_audit["split_marker_count"] == 0
    assert split_audit["case_id_unique_across_cases"] is True
    assert split_audit["request_id_unique_across_cases"] is True

print("TRAIN/DEV DATA CONTRACT PASS")
PY
```

必须输出 `TRAIN/DEV DATA CONTRACT PASS`。

### 10.2 `induce`：重新训练 Shortcut SFT

```bash
run_logged_stage induce
INDUCE_STATUS=$?
printf 'induce exit=%s\n' "$INDUCE_STATUS"
```

只有 `INDUCE_STATUS=0` 才继续。执行训练合同验收：

```bash
python - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shortcut_repair.train import sha256_model_weights

manifest = json.loads(Path("runs/shortcut/run_manifest.json").read_text(encoding="utf-8"))
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

assert manifest["status"] == "complete"
assert manifest["git_sha"] == git_sha
assert manifest["contract"]["stage"] == "shortcut_sft"
assert manifest["contract"]["rows"] == 1200
assert manifest["contract"]["optimizer_steps"] == 38
assert manifest["trainer_budget"] == {
    "max_steps": 38,
    "num_train_epochs": 1,
    "source": "contract.optimizer_steps",
}
assert manifest["actual_optimizer_steps"] == 38
assert isinstance(manifest["actual_epoch"], (int, float))
assert manifest["actual_epoch"] > 0
assert Path("runs/shortcut/merged/config.json").is_file()
actual_sha = sha256_model_weights(Path("runs/shortcut/merged"))
assert manifest["merged_model_weights_sha256"] == actual_sha
assert len(actual_sha) == 64

print("SHORTCUT TRAINING CONTRACT PASS")
print(f"merged_model_weights_sha256={actual_sha}")
PY
```

必须输出 `SHORTCUT TRAINING CONTRACT PASS`。若仍出现 185/190 steps、旧 Git SHA 或 manifest 缺字段，说明旧目录没有归档干净；不要修改 manifest，应停止并检查 `runs/shortcut` 的来源。

### 10.3 `gate`：Base/Shortcut 对照与机制门控

```bash
run_logged_stage gate
GATE_STATUS=$?
printf 'gate exit=%s\n' "$GATE_STATUS"
```

如果 `GATE_STATUS` 不是 0，立即停止，不要执行 `seal-test`、`smoke` 或 `train`。Gate 失败只说明本次 shortcut induction 没有按预注册标准建立，不能解释成 Repair DPO 无效。

当 `GATE_STATUS=0` 时，再执行：

```bash
python - <<'PY'
import json
from pathlib import Path

gate = json.loads(
    Path("results/dev/shortcut/mechanism_gate.json").read_text(encoding="utf-8")
)
metrics = gate["metrics"]
base_metrics = gate["base_metrics"]

assert gate["decision"] == "pass"
assert all(gate["checks"].values())
assert metrics["case_count"] == 200
assert metrics["row_count"] == 1200
assert base_metrics["case_count"] == 200
assert base_metrics["row_count"] == 1200
assert metrics["aligned_accuracy"] >= 0.80
assert metrics["conflict_accuracy"] <= 0.20
assert metrics["hint_flip_rate"] >= 0.80
assert metrics["causal_hint_effect"] >= 1.0
assert set(gate["shortcut_minus_base"]) == {"hint_flip_rate", "causal_hint_effect"}

print("MECHANISM GATE PASS")
print(json.dumps({
    "base": base_metrics,
    "shortcut": metrics,
    "shortcut_minus_base": gate["shortcut_minus_base"],
}, ensure_ascii=False, indent=2))
PY
```

只有看到 `MECHANISM GATE PASS`，M4 才算完成。

## 11. Gate 失败时的处理

Gate 命令即使失败，也会写出 `mechanism_gate.json`。先查看失败项：

```bash
python -m json.tool results/dev/shortcut/mechanism_gate.json
tail -n 200 "$SHORTCUT_LOG_DIR/gate.log"
```

打包可供本地分析的脱敏证据；不要直接打包整个仓库或模型权重：

```bash
cd "$SHORTCUT_PROJECT_DIR"
mkdir -p artifacts

partial_files=(
  configs/experiment.yaml
  configs/experiment.sha256
  data/manifest_train_dev.json
  runs/shortcut/run_manifest.json
  results/dev/base/predictions.jsonl
  results/dev/base/metrics.json
  results/dev/base/prediction_manifest.json
  results/dev/shortcut/predictions.jsonl
  results/dev/shortcut/metrics.json
  results/dev/shortcut/prediction_manifest.json
  results/dev/shortcut/mechanism_gate.json
)

for path in "${partial_files[@]}"; do
  test -f "$path"
done

PARTIAL_ARCHIVE="artifacts/shortcut-repair-v1.1-gate-failure-$(date -u +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$PARTIAL_ARCHIVE" "${partial_files[@]}"
sha256sum "$PARTIAL_ARCHIVE" > "${PARTIAL_ARCHIVE}.sha256"
printf 'partial_archive=%s\n' "$(realpath "$PARTIAL_ARCHIVE")"
printf 'partial_checksum=%s\n' "$(realpath "${PARTIAL_ARCHIVE}.sha256")"
```

把这两个文件和 `gate.log` 带回后再分析。不要在 v1.1 的 dev 结果上临时降低 gate 阈值；若确实要改变 induction 设计，应新建 v1.2 协议和新分支。

## 12. M5：Gate 通过后封存 test、smoke 和正式训练

### 12.1 `seal-test`：只在 Gate 通过后生成 test

```bash
run_logged_stage seal-test
SEAL_STATUS=$?
printf 'seal-test exit=%s\n' "$SEAL_STATUS"
```

只有退出码为 0 才执行封存验收：

```bash
python - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("data/manifest_test.json").read_text(encoding="utf-8"))
entry = manifest["files"]["test.jsonl"]
path = Path("data/test.jsonl")

assert manifest["sealed"] is True
assert manifest["generator_version"] == "shortcut-repair-v2"
assert entry["rows"] == 1800
assert sum(1 for _ in path.open(encoding="utf-8")) == 1800
assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
assert manifest["audit"]["case_count"] == 300
assert manifest["audit"]["gold_A_fraction"] == 0.5
assert manifest["audit"]["score_decisive_fraction"] == 0.5
assert manifest["audit"]["validity_decisive_fraction"] == 0.5
assert manifest["audit"]["fresh_score_only_accuracy"] == 0.5
assert manifest["audit"]["historical_only_accuracy"] <= 0.55
assert manifest["audit"]["display_rank_only_accuracy"] <= 0.55
assert manifest["audit"]["split_marker_count"] == 0

print("SEALED TEST CONTRACT PASS")
print(f"test_sha256={entry['sha256']}")
PY
```

从这一刻起，不得根据 test 结果修改 v1.1 配置、生成器、阈值、seed 或样本。

### 12.2 `smoke`：三种方法各跑 2 steps

```bash
run_logged_stage smoke
SMOKE_STATUS=$?
printf 'smoke exit=%s\n' "$SMOKE_STATUS"
```

验收三个 smoke run：

```bash
python - <<'PY'
import json
from pathlib import Path

paths = (
    Path("runs/dpo/smoke/control-seed-42/run_manifest.json"),
    Path("runs/dpo/smoke/repair-seed-42/run_manifest.json"),
    Path("runs/sft_baseline/smoke/seed-42/run_manifest.json"),
)

shortcut_shas = set()
for path in paths:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete", path
    assert manifest["contract"]["smoke"] is True, path
    assert manifest["contract"]["rows"] == 64, path
    assert manifest["trainer_budget"]["max_steps"] == 2, path
    assert manifest["actual_optimizer_steps"] == 2, path
    assert len(manifest["final_adapter_weights_sha256"]) == 64, path
    shortcut_shas.add(manifest["shortcut_model_weights_sha256"])

assert len(shortcut_shas) == 1
print("THREE-WAY SMOKE PASS")
PY
```

只有输出 `THREE-WAY SMOKE PASS` 才进入正式训练。

### 12.3 `train`：三种方法 × 三个 seeds，共九次正式训练

```bash
run_logged_stage train
TRAIN_STATUS=$?
printf 'train exit=%s\n' "$TRAIN_STATUS"
```

训练脚本顺序为：对 seed 42、43、44 分别运行 Aligned-only DPO、Counterfactual DPO、Counterfactual SFT。验收九个 manifest：

```bash
python - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
shortcut_manifest = json.loads(
    Path("runs/shortcut/run_manifest.json").read_text(encoding="utf-8")
)
shortcut_sha = shortcut_manifest["merged_model_weights_sha256"]

for seed in (42, 43, 44):
    initial = {}
    for method in ("control", "repair"):
        path = Path(f"runs/dpo/{method}/seed-{seed}/run_manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "complete", path
        assert manifest["git_sha"] == git_sha, path
        assert manifest["method"] == method, path
        assert manifest["contract"]["seed"] == seed, path
        assert manifest["contract"]["smoke"] is False, path
        assert manifest["contract"]["rows"] == 1200, path
        assert manifest["trainer_budget"]["max_steps"] == 114, path
        assert manifest["trainer_budget"]["source"] == "contract.optimizer_steps", path
        assert manifest["actual_optimizer_steps"] == 114, path
        assert manifest["shortcut_model_weights_sha256"] == shortcut_sha, path
        assert len(manifest["final_adapter_weights_sha256"]) == 64, path
        initial[method] = manifest["initial_adapter_checksum"]

    assert initial["control"] == initial["repair"], (seed, initial)

    sft_path = Path(f"runs/sft_baseline/seed-{seed}/run_manifest.json")
    sft = json.loads(sft_path.read_text(encoding="utf-8"))
    assert sft["status"] == "complete", sft_path
    assert sft["git_sha"] == git_sha, sft_path
    assert sft["contract"]["seed"] == seed, sft_path
    assert sft["contract"]["smoke"] is False, sft_path
    assert sft["contract"]["rows"] == 1200, sft_path
    assert sft["trainer_budget"]["max_steps"] == 114, sft_path
    assert sft["actual_optimizer_steps"] == 114, sft_path
    assert sft["shortcut_model_weights_sha256"] == shortcut_sha, sft_path
    assert len(sft["final_adapter_weights_sha256"]) == 64, sft_path

print("NINE FORMAL RUNS PASS")
print(f"git_sha={git_sha}")
print(f"shortcut_model_weights_sha256={shortcut_sha}")
PY
```

必须输出 `NINE FORMAL RUNS PASS`。同一 seed 下 Control/Repair 的初始 LoRA checksum 必须相同，九个 run 的 Shortcut 起点权重必须完全一致。

## 13. 统一评测、聚合和结果判定

### 13.1 `evaluate`

```bash
run_logged_stage evaluate
EVALUATE_STATUS=$?
printf 'evaluate exit=%s\n' "$EVALUATE_STATUS"
```

该阶段在同一个 sealed test 上评估 Base、Shortcut、六个 DPO adapter 和三个 Counterfactual SFT adapter，同时生成 FP32 teacher-forced 指标与 greedy generation 指标。

检查结果目录：

```bash
find results/test -type f \
  \( -name metrics.json -o -name prediction_manifest.json \) \
  | sort
```

### 13.2 `aggregate`

```bash
run_logged_stage aggregate
AGGREGATE_STATUS=$?
printf 'aggregate exit=%s\n' "$AGGREGATE_STATUS"
```

聚合器会再次检查 Git/config/data/model/adapter/prediction checksum、九次训练预算和同 seed 初始 LoRA 一致性。只有退出码为 0 才查看最终结果：

```bash
sed -n '1,240p' reports/RESULTS.md
python -m json.tool reports/results.json | sed -n '1,260p'
ls -lh \
  reports/RESULTS.md \
  reports/results.json \
  reports/main_metrics.csv \
  reports/baseline_metrics.csv \
  reports/per_seed.csv \
  reports/comparison.png
```

最终结论只能来自 `reports/RESULTS.md`：

- 九项预注册检查全过：`POSITIVE`；
- 任一检查不过：`NEGATIVE / INCONCLUSIVE`。

负结果仍然是正确、可汇报的实验结果；不能删除不利 seed，不能在 sealed test 上调阈值后重新解释为正结果。

## 14. 打包正式结果

只有 `aggregate` 成功后执行：

```bash
cd "$SHORTCUT_PROJECT_DIR"
bash scripts/package_results.sh
```

脚本输出压缩包和对应 checksum 文件。定位最新一组并在服务器上验证：

```bash
RESULT_ARCHIVE="$(find artifacts -maxdepth 1 -type f \
  -name 'shortcut-repair-results-*.tar.gz' \
  -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-)"

test -n "$RESULT_ARCHIVE"
test -f "${RESULT_ARCHIVE}.sha256"
sha256sum -c "${RESULT_ARCHIVE}.sha256"
printf 'result_archive=%s\n' "$(realpath "$RESULT_ARCHIVE")"
printf 'result_checksum=%s\n' "$(realpath "${RESULT_ARCHIVE}.sha256")"
```

记下最后两行的绝对路径。正式结果包使用白名单，只包含配置、manifest、脱敏预测和报告，不包含模型权重、`.venv`、认证信息或 GPU UUID。

## 15. Linux 客户端：下载并校验结果

先从 tmux 安全离开：按 `Ctrl-b`，松开后按 `d`。然后输入 `exit` 退出 SSH，回到 Linux 客户端。

在 Linux 客户端重新设置连接信息，并把 `REMOTE_ARCHIVE` 替换成服务器刚才打印的准确绝对路径：

```bash
export SERVER_HOST="替换为服务器地址"
export SERVER_USER="替换为服务器用户名"
export SERVER_PORT="22"
export REMOTE_ARCHIVE="/mnt_d/huangxiaoyuan/ShortcutRepair-DPO/artifacts/替换为准确文件名.tar.gz"
export LOCAL_RESULT_DIR="$PWD/ShortcutRepair-DPO-v1.1-result"

mkdir -p "$LOCAL_RESULT_DIR/artifacts"
scp -P "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST:$REMOTE_ARCHIVE" \
  "$LOCAL_RESULT_DIR/artifacts/"
scp -P "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST:${REMOTE_ARCHIVE}.sha256" \
  "$LOCAL_RESULT_DIR/artifacts/"
```

保持 `artifacts/` 目录层级后校验：

```bash
(
  cd "$LOCAL_RESULT_DIR"
  sha256sum -c "artifacts/$(basename "${REMOTE_ARCHIVE}.sha256")"
)
```

必须输出 `OK`。随后可在本地查看内容但暂不解压覆盖其他目录：

```bash
tar -tzf "$LOCAL_RESULT_DIR/artifacts/$(basename "$REMOTE_ARCHIVE")" | sed -n '1,200p'
```

Gate 失败时也用同样方法下载第 11 节生成的 partial archive 和 `.sha256`。

## 16. SSH 中断和训练恢复

SSH 中断后：

```bash
ssh \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -p "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST"
tmux attach -t shortcut-v11
```

如果 tmux 仍在，训练通常仍在继续。查看日志：

```bash
tail -n 100 "$SHORTCUT_LOG_DIR/train.log"
nvidia-smi
```

如果训练进程已经停止，重新执行同一阶段，例如：

```bash
run_logged_stage train
```

Runner 会对存在 `checkpoint-*` 的 v1.1 run 自动加入 `--resume`，并强制比较 Git、config、data、训练阶段、模型来源和 Trainer 预算；匹配才恢复。已经 complete 的 run 会安全跳过。

禁止做法：

- 从 v1.0 的 `checkpoint-150` 或 `checkpoint-185` 恢复；
- 把旧 checkpoint 复制到新 `runs/`；
- 手改 `run_manifest.json` 或 `trainer_state.json`；
- 为了通过恢复校验而修改 Git SHA、data SHA 或步数；
- OOM 后擅自改变 frozen batch、gradient accumulation 或 `max_steps`。

## 17. 运行监控

在另一个 tmux 窗口中可以执行：

```bash
watch -n 5 nvidia-smi
```

或者查看某阶段日志：

```bash
tail -f "$SHORTCUT_LOG_DIR/induce.log"
tail -f "$SHORTCUT_LOG_DIR/train.log"
```

若看到 NaN、OOM、CUDA error、step mismatch，停止后保留日志和当前 manifest。不要边跑边改代码继续同一身份实验。

## 18. 常见故障决策表

| 现象 | 含义 | 正确动作 |
|---|---|---|
| `Git working tree must be clean` | 有 tracked/untracked 文件影响审计 | `git status --short --untracked-files=all` 调查；不 reset |
| `configs/experiment.yaml: FAILED` | 冻结配置被改变 | 停止，重新从远端干净更新；不重算 checksum 掩盖改动 |
| Python 不是 3.10 | 运行身份不符合合同 | 创建 Python 3.10 环境 |
| `torch` 不是 `2.5.1+cu121` | 依赖环境不符合合同 | 按第 6 节重装固定 wheel |
| GPU/VRAM/BF16 预检失败 | 硬件不符合正式协议 | 换符合条件的 GPU；不降低冻结合同 |
| 旧 resume 被拒绝 | v1.0 与 v1.1 身份不同 | 归档旧 `runs/`，从空 v1.1 输出目录开始 |
| Shortcut 不是 38 steps | 旧 bug 或输出串线 | 停止，核对 branch、Git SHA 和旧目录归档 |
| Gate 退出码 2 | Shortcut 机制没有建立 | 不生成 test；打包 Gate 证据并诊断 induction |
| Smoke 失败/OOM/NaN | 工程稳定性未通过 | 保留日志，先修工程问题；不进入正式训练 |
| 某正式 run 中断 | 可能有合法 v1.1 checkpoint | 重新执行 `train`，让身份校验后自动恢复 |
| `aggregate` 缺文件或 checksum 错 | 九次 run/评测身份不完整 | 根据报错补跑对应阶段；不手工拼报告 |
| SSH 断开 | 客户端连接断开 | 重新 SSH 并 attach tmux |

## 19. 正式结果之后如何继续优化

先接受 v1.1 的预注册结论，再决定 v1.2；不要反过来用 test 调参。

### 19.1 Gate 未通过

下一版只优化 shortcut induction，并继续使用 dev 门控。重点按顺序检查：

1. Shortcut 是否真正跟随 cached hint，而不是学到 A/B、fresh score、historical score 或 display rank 旁路；
2. aligned accuracy、conflict accuracy、hint flip rate、causal hint effect 哪一项失败；
3. Base 与 Shortcut 的机制差值是否足以证明行为发生变化；
4. 在新版本协议中改变 induction 数据或预算，而不是降低 v1.1 gate 阈值。

Gate 失败时 v1.1 test 尚未生成，因此可以基于 dev 诊断设计 v1.2，但必须新建版本、配置 checksum 和 Git 分支。

### 19.2 Gate 通过但正式结果为 `NEGATIVE / INCONCLUSIVE`

按对照关系解释，而不是只看一个总准确率：

- Repair DPO 和 Counterfactual SFT 都优于 Control：主要证据是反事实冲突数据有效，不能声称 DPO 独有优势；
- SFT 与 DPO 相当或更好：把项目亮点放在因果数据构造、机制门控和公平后训练对照；
- 两者都没有改善：检查 repair signal、训练预算和 Shortcut 强度是否匹配；
- conflict 改善但 aligned 明显下降：出现过度修复，需要研究稳定性/保真约束；
- fresh response 不足：模型仍没有可靠服从权威 fresh result；
- nuisance invariance 不足：仍存在 historical/display 等非目标旁路；
- greedy format 失败：先解决输出合同，而不是只报告 teacher-forced 分数。

### 19.3 新一轮实验的版本纪律

任何进一步优化都应：

1. 保留 v1.1 的配置、sealed test、九个 run 和结果包；
2. 新建 `codex/v1.2-*` 分支；
3. 把配置版本、generator version、假设和阈值更新为 v1.2；
4. 使用新的 dev/test seed 和新的 checksum；
5. 先冻结新协议，再生成新 test；
6. 不把 v1.1 与 v1.2 的指标混成同一组结果。

这样无论结果正负，面试中都能清楚展示：故障复盘、训练合同、因果数据设计、机制门控、DPO/SFT 公平对照、多 seed、可恢复训练、可审计聚合和诚实报告。

## 20. 最终完成清单

- [ ] 服务器 `origin` 指向 `WeiAsFan/ShortcutRepair-DPO`；
- [ ] 当前分支为 `codex/v1.1-repair`；
- [ ] 修复基线提交 `384d78b` 是当前 HEAD 的祖先；
- [ ] v1.0 运行产物已在仓库外归档并校验；
- [ ] Git 工作区 clean；
- [ ] 配置 SHA256 校验通过；
- [ ] Python/依赖/模型/GPU preflight 通过；
- [ ] train/dev 数据合同通过；
- [ ] Shortcut 训练恰好 38 steps；
- [ ] Base/Shortcut 机制 gate 通过；
- [ ] test 在 gate 后生成、1,800 行且 sealed；
- [ ] 三种 smoke 各 2 steps；
- [ ] 九个正式 run 各 114 steps；
- [ ] 同 seed 的 Control/Repair 初始 LoRA checksum 相同；
- [ ] 所有正式 run 使用同一 Shortcut 权重；
- [ ] evaluate 与 aggregate 成功；
- [ ] `reports/RESULTS.md` 给出预注册结论；
- [ ] 正式结果包在服务器和客户端均通过 SHA256 校验；
- [ ] 未上传模型权重、`.venv`、私钥、token 或机器敏感信息。

完成以上清单后，v1.1 的 bug 修复验证和正式实验才算真正结束。
