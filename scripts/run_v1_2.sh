#!/usr/bin/env bash
set -euo pipefail

# 仅封装五阶段入口；不重复 preflight、冒烟或模型哈希。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
python -m shortcut_repair.v12 "$@"
