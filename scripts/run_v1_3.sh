#!/usr/bin/env bash
set -euo pipefail

# v1.3 仅封装五阶段入口；pilot 只训练一次格式锚定，不搜索超参数。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
python -m shortcut_repair.v13 "$@"
