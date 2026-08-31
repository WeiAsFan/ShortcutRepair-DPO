# v1.1 评测协议修正与远程恢复指南

本文只处理 2026-09-01 发现的评测失败。服务器上的 Shortcut、六个 DPO adapter 和三个 Counterfactual SFT adapter 已完成训练，不应重训。

## 1. 事故与修正边界

旧评测把模型以 BF16 加载并完成前向，只在得到 logits 后才转成 FP32 计算 log-softmax。`repair/seed-42` 的一个样本因此出现完全相等的 A/B 条件对数概率，严格平分检查按设计终止了评测。

这次修正只改变评测数值协议：

- 所有模型统一以 FP32 加载并执行前向；
- 关闭 CUDA matmul 和 cuDNN 的 TF32，并使用最高 FP32 matmul 精度；
- A/B logits 仍在 FP32 中计算 log-softmax；
- 完全平分仍然报错，不按 A、B 或随机规则强行判定；错误中增加 case 和干预上下文；
- Base、Shortcut、六个 DPO adapter、三个 SFT adapter 的 dev/test 结果按同一协议重评；
- 训练 Git 与评测 Git 分开记录；
- predictions、metrics 和完成 manifest 原子写入，重试时只跳过身份和 checksum 均完整匹配的模型。

以下内容不变：`configs/experiment.yaml`、train/dev/test 数据、sealed test、九个训练产物、模型权重、seed、训练预算、指标、bootstrap 和成功阈值。

机器可读的冻结增补是 `configs/evaluation_amendment.yaml`。它绑定：

```text
training_git_sha = 1ead3b24f00f33569128a6634401729e4908a62f
experiment_config_sha256 = 56da1d3c5f8df8512ea72e458e03755e854cf78abc533c02c3b86b4d28e85ca6
dev_data_sha256 = c33e24f5f93cc97185438042b4dfc3e14eab9002c3906cd098883fbf050e58f7
sealed_test_sha256 = 22dcd320143f2070e1e5cc928f84a83692f6636d55303b95fc63d5c490c76e42
```

这是看到部分 Base、Shortcut、Control-42 结果和 Repair-42 失败后的透明协议增补，不应伪装成事前注册。由于修正与模型组无关、全部模型统一重评、成功阈值未改且失败前没有得到 Repair 的完整结果，修正后的比较仍可作为有效的受控实验；正式报告必须保留这项披露。

## 2. 严格禁止的操作

- 不执行 `prepare`、`induce`、`seal-test`、`smoke`、`train` 或 `all`；
- 不删除或改写 `runs/`、`data/test.jsonl`、`data/manifest_test.json`；
- 不修改 `configs/experiment.yaml`、成功阈值或 test 样本；
- 不手工修改任何 run/prediction manifest；
- 不用旧 BF16 部分指标和新 FP32 指标拼接报告；
- 不为消除平分而加 epsilon、固定选 A/B 或随机打破平分；
- 不因显存问题擅自修改冻结 batch size。

## 3. Linux 客户端登录服务器

在本地 Linux 设备执行：

```bash
export SERVER_HOST="替换为服务器地址"
export SERVER_USER="替换为服务器用户名"
export SERVER_PORT="22"

ssh \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -p "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST"
```

登录后进入 tmux：

```bash
tmux new-session -A -s shortcut-v11-eval-fix
```

## 4. 更新修复分支

以下命令都在训练服务器的 tmux 中执行。若实际路径不同，只修改第一行：

```bash
export SHORTCUT_PROJECT_DIR="/mnt_d/huangxiaoyuan/ShortcutRepair-DPO"
export SHORTCUT_BRANCH="codex/v1.1-repair"

cd "$SHORTCUT_PROJECT_DIR"
git remote -v
git status --short --branch
test -z "$(git status --porcelain --untracked-files=normal)"

git fetch origin \
  "refs/heads/$SHORTCUT_BRANCH:refs/remotes/origin/$SHORTCUT_BRANCH"
git switch "$SHORTCUT_BRANCH"
git merge --ff-only "origin/$SHORTCUT_BRANCH"

test "$(git branch --show-current)" = "$SHORTCUT_BRANCH"
git merge-base --is-ancestor 1ead3b24f00f33569128a6634401729e4908a62f HEAD
test -z "$(git status --porcelain --untracked-files=normal)"
git log -8 --oneline --decorate
```

这里的当前 `HEAD` 是新的评测代码身份，不应再要求训练 manifest 的 Git SHA 等于当前 `HEAD`。训练 manifest 应继续指向 `1ead3b24...`；新 prediction manifest 应指向当前 `HEAD`。

## 5. 激活环境并验证代码

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate
python -m pip install -e .
python -m pip check

bash -n scripts/preflight.sh scripts/run_experiment.sh scripts/package_results.sh
python -m pytest -q
python -m ruff check src tests
bash scripts/preflight.sh
```

所有测试必须通过、Ruff 必须无错误、预检必须以 `PREFLIGHT PASS` 结束。不要只运行新增测试后就直接评测。

## 6. 验证冻结增补和已有训练产物

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from shortcut_repair.train import sha256_model_weights

training_git_sha = "1ead3b24f00f33569128a6634401729e4908a62f"
config_sha = "56da1d3c5f8df8512ea72e458e03755e854cf78abc533c02c3b86b4d28e85ca6"
test_sha = "22dcd320143f2070e1e5cc928f84a83692f6636d55303b95fc63d5c490c76e42"
dev_sha = "c33e24f5f93cc97185438042b4dfc3e14eab9002c3906cd098883fbf050e58f7"

def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

amendment_path = Path("configs/evaluation_amendment.yaml")
amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
assert amendment["status"] == "frozen"
assert amendment["training_git_sha"] == training_git_sha
assert amendment["experiment_config_sha256"] == config_sha
assert amendment["dev_data_sha256"] == dev_sha
assert amendment["sealed_test_sha256"] == test_sha
assert amendment["evaluation"] == {
    "model_dtype": "float32",
    "allow_tf32": False,
    "tie_policy": "reject_with_context",
    "rerun_scope": "all_dev_and_test_models",
}
assert file_sha(Path("configs/experiment.yaml")) == config_sha
assert file_sha(Path("data/dev.jsonl")) == dev_sha
assert file_sha(Path("data/test.jsonl")) == test_sha

shortcut = load_json(Path("runs/shortcut/run_manifest.json"))
assert shortcut["status"] == "complete"
assert shortcut["git_sha"] == training_git_sha
assert shortcut["actual_optimizer_steps"] == 38
assert shortcut["merged_model_weights_sha256"] == sha256_model_weights(
    Path("runs/shortcut/merged")
)

initial_checksums = {}
for seed in (42, 43, 44):
    initial_checksums[seed] = {}
    for method in ("control", "repair"):
        run_dir = Path(f"runs/dpo/{method}/seed-{seed}")
        manifest = load_json(run_dir / "run_manifest.json")
        adapter_dir = run_dir / "final_adapter"
        assert manifest["status"] == "complete"
        assert manifest["git_sha"] == training_git_sha
        assert manifest["actual_optimizer_steps"] == 114
        assert manifest["final_adapter_weights_sha256"] == sha256_model_weights(
            adapter_dir
        )
        initial_checksums[seed][method] = manifest["initial_adapter_checksum"]
    assert initial_checksums[seed]["control"] == initial_checksums[seed]["repair"]

    sft_dir = Path(f"runs/sft_baseline/seed-{seed}")
    sft = load_json(sft_dir / "run_manifest.json")
    assert sft["status"] == "complete"
    assert sft["git_sha"] == training_git_sha
    assert sft["actual_optimizer_steps"] == 114
    assert sft["final_adapter_weights_sha256"] == sha256_model_weights(
        sft_dir / "final_adapter"
    )

evaluation_git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
print("FROZEN TRAINING ARTIFACTS PASS")
print(f"training_git_sha={training_git_sha}")
print(f"evaluation_git_sha={evaluation_git_sha}")
print(f"evaluation_amendment_sha256={file_sha(amendment_path)}")
PY
```

必须输出 `FROZEN TRAINING ARTIFACTS PASS`。任何断言失败都先调查文件来源，不要重训或修改 manifest 来绕过。

## 7. 备份旧的部分评测现场

仓库中的 `ShortcutRepair-DPO-v1.1-evaluate-failure/` 已保留公开失败证据。再把服务器根目录下将被重评覆盖的运行结果复制到仓库外：

```bash
cd "$SHORTCUT_PROJECT_DIR"
export SHORTCUT_EVAL_BACKUP="$(dirname "$SHORTCUT_PROJECT_DIR")/ShortcutRepair-DPO-v1.1-bf16-eval-backup-$(date -u +%Y%m%d-%H%M%S)"
test ! -e "$SHORTCUT_EVAL_BACKUP"
mkdir -p "$SHORTCUT_EVAL_BACKUP"

for item in results reports; do
  if [[ -e "$SHORTCUT_PROJECT_DIR/$item" ]]; then
    cp -a -- "$SHORTCUT_PROJECT_DIR/$item" "$SHORTCUT_EVAL_BACKUP/"
  fi
done

find "$SHORTCUT_EVAL_BACKUP" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 -r sha256sum > "$SHORTCUT_EVAL_BACKUP/SHA256SUMS"
sha256sum -c "$SHORTCUT_EVAL_BACKUP/SHA256SUMS"
printf '旧评测备份：%s\n' "$SHORTCUT_EVAL_BACKUP"
```

这是复制而非删除。新评测的身份不匹配旧 manifest，因此会统一重算；若重试时发现完全匹配且 checksum 完整的新结果，则会安全跳过。

## 8. 设置日志函数

```bash
export SHORTCUT_LOG_DIR="$(dirname "$SHORTCUT_PROJECT_DIR")/ShortcutRepair-DPO-v1.1-fp32-eval-logs"
mkdir -p "$SHORTCUT_LOG_DIR"
set +e
set -o pipefail

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

type run_logged_stage
```

## 9. 只执行三阶段恢复

执行顺序严格为 `gate → evaluate → aggregate`，任何一步失败都停止。

### 9.1 重新计算 dev gate

```bash
run_logged_stage gate
GATE_STATUS=$?
printf 'gate exit=%s\n' "$GATE_STATUS"
test "$GATE_STATUS" -eq 0
```

这一步会用 FP32 统一重评 dev Base 和 Shortcut，再按原阈值重算 mechanism gate。不得复用旧 BF16 gate。若 gate 不再通过，停止，不执行 test 聚合，并把它报告为评测协议修正后的门控失败。

### 9.2 统一重评全部 test 模型

```bash
run_logged_stage evaluate
EVALUATE_STATUS=$?
printf 'evaluate exit=%s\n' "$EVALUATE_STATUS"
test "$EVALUATE_STATUS" -eq 0
```

该阶段依次评测 Base、Shortcut、Control/Repair 的三个 seed 和 Counterfactual SFT 的三个 seed，共 11 个 test 模型。命令失败后可以原样再次执行；已经由当前协议完整完成的模型会跳过，其余模型会重算。

### 9.3 聚合

```bash
run_logged_stage aggregate
AGGREGATE_STATUS=$?
printf 'aggregate exit=%s\n' "$AGGREGATE_STATUS"
test "$AGGREGATE_STATUS" -eq 0
```

聚合器会同时审计旧训练身份、新评测身份、修正协议 checksum、模型权重、数据、预测、指标和九个训练合同。不要手工拼接报告。

## 10. 验收新评测身份和报告

```bash
cd "$SHORTCUT_PROJECT_DIR"
source .venv/bin/activate

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

training_git_sha = "1ead3b24f00f33569128a6634401729e4908a62f"
evaluation_git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
amendment_path = Path("configs/evaluation_amendment.yaml")
amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
amendment_sha = hashlib.sha256(amendment_path.read_bytes()).hexdigest()

manifests = [
    Path("results/dev/base/prediction_manifest.json"),
    Path("results/dev/shortcut/prediction_manifest.json"),
    Path("results/test/base/prediction_manifest.json"),
    Path("results/test/shortcut/prediction_manifest.json"),
]
for seed in (42, 43, 44):
    manifests.extend(
        Path(f"results/test/{method}/seed-{seed}/prediction_manifest.json")
        for method in ("control", "repair")
    )
    manifests.append(
        Path(f"results/test/counterfactual_sft/seed-{seed}/prediction_manifest.json")
    )

assert len(manifests) == 13
for manifest_path in manifests:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_dir = manifest_path.parent
    assert manifest["status"] == "complete", manifest_path
    assert manifest["training_git_sha"] == training_git_sha, manifest_path
    assert manifest["evaluation_git_sha"] == evaluation_git_sha, manifest_path
    assert manifest["evaluation_amendment_sha256"] == amendment_sha, manifest_path
    assert manifest["evaluation_protocol_id"] == amendment["protocol_id"], manifest_path
    assert manifest["evaluation_model_dtype"] == "float32", manifest_path
    assert manifest["evaluation_allow_tf32"] is False, manifest_path
    assert manifest["tie_policy"] == "reject_with_context", manifest_path
    assert manifest["prediction_rows"] in (1200, 1800), manifest_path
    assert manifest["predictions_sha256"] == hashlib.sha256(
        (result_dir / "predictions.jsonl").read_bytes()
    ).hexdigest(), manifest_path
    assert manifest["metrics_sha256"] == hashlib.sha256(
        (result_dir / "metrics.json").read_bytes()
    ).hexdigest(), manifest_path

result = json.loads(Path("reports/results.json").read_text(encoding="utf-8"))
provenance = result["provenance"]
assert provenance["training_git_sha"] == training_git_sha
assert provenance["evaluation_git_sha"] == evaluation_git_sha
assert provenance["evaluation_amendment_sha256"] == amendment_sha
assert provenance["evaluation_model_dtype"] == "float32"
assert result["decision"] in {"POSITIVE", "NEGATIVE / INCONCLUSIVE"}

report = Path("reports/RESULTS.md").read_text(encoding="utf-8")
assert training_git_sha in report
assert evaluation_git_sha in report
assert "统一 FP32 协议" in report

print("FP32 EVALUATION AUDIT PASS")
print(f"decision={result['decision']}")
print(f"evaluation_git_sha={evaluation_git_sha}")
print(f"evaluation_amendment_sha256={amendment_sha}")
PY
```

必须输出 `FP32 EVALUATION AUDIT PASS`。随后查看正式报告：

```bash
sed -n '1,240p' reports/RESULTS.md
```

## 11. 打包和下载

服务器执行：

```bash
cd "$SHORTCUT_PROJECT_DIR"
bash scripts/package_results.sh

RESULT_ARCHIVE="$(find artifacts -maxdepth 1 -type f \
  -name 'shortcut-repair-results-*.tar.gz' \
  -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-)"
test -n "$RESULT_ARCHIVE"
sha256sum -c "${RESULT_ARCHIVE}.sha256"
printf 'result_archive=%s\n' "$(realpath "$RESULT_ARCHIVE")"
```

退出 SSH 后，在本地 Linux 设备把 `REMOTE_ARCHIVE` 替换为上一步的绝对路径：

```bash
export REMOTE_ARCHIVE="替换为服务器上的结果包绝对路径"
export LOCAL_RESULT_DIR="$PWD/ShortcutRepair-DPO-v1.1-fp32-result"
mkdir -p "$LOCAL_RESULT_DIR/artifacts"

scp -P "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST:$REMOTE_ARCHIVE" \
  "$LOCAL_RESULT_DIR/artifacts/"
scp -P "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST:${REMOTE_ARCHIVE}.sha256" \
  "$LOCAL_RESULT_DIR/artifacts/"

(
  cd "$LOCAL_RESULT_DIR"
  sha256sum -c "artifacts/$(basename "${REMOTE_ARCHIVE}.sha256")"
)
```

## 12. 停止条件

出现以下任一情况就停止并保留日志：

- 原配置或 sealed test checksum 不匹配；
- 任一训练 manifest 不是训练提交 `1ead3b24...`；
- 任一训练 run 步数、权重 checksum 或配对初始 LoRA checksum 不匹配；
- FP32 下仍出现完全平分；
- OOM、NaN、CUDA error 或评测模型仍含非 FP32 浮点参数；
- 新 prediction manifest 的评测 Git 不是当前 `HEAD`；
- aggregate 拒绝任一 provenance 或 checksum。

FP32 下仍平分不等于 Repair 无效，而表示当前二选一判定协议仍无法为该样本定义唯一预测。此时应保留错误中的 `case_id`、干预类型和精确分数，另立后续协议处理，不能在 v1.1 内临时改变判定规则。
