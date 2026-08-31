#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${script_dir}/.."
repo_root="$(pwd)"
cd "$repo_root"

shortcut_python="${SHORTCUT_PYTHON:-python}"
shortcut_config="${SHORTCUT_CONFIG:-configs/experiment.yaml}"
stage="${1:-}"

run_cli() {
  "$shortcut_python" -m shortcut_repair.cli "$@"
}

run_stage() {
  bash scripts/run_experiment.sh "$1"
}

resume_shortcut=()
if compgen -G "runs/shortcut/checkpoint-*" >/dev/null; then
  resume_shortcut=(--resume)
fi

case "$stage" in
  prepare)
    bash scripts/preflight.sh
    run_cli generate --config "$shortcut_config" --stage train-dev
    run_cli train-shortcut --config "$shortcut_config" --dry-run
    for method in control repair; do
      run_cli train-dpo \
        --config "$shortcut_config" \
        --method "$method" \
        --seed 42 \
        --dry-run
    done
    run_cli train-sft-baseline \
      --config "$shortcut_config" \
      --seed 42 \
      --dry-run
    ;;
  induce)
    run_cli train-shortcut \
      --config "$shortcut_config" \
      "${resume_shortcut[@]}"
    ;;
  gate)
    run_cli evaluate \
      --config "$shortcut_config" \
      --split dev \
      --model base \
      --output-dir results/dev/base
    run_cli evaluate \
      --config "$shortcut_config" \
      --split dev \
      --model shortcut \
      --output-dir results/dev/shortcut
    run_cli gate \
      --config "$shortcut_config" \
      --base-predictions results/dev/base/predictions.jsonl \
      --predictions results/dev/shortcut/predictions.jsonl
    ;;
  seal-test)
    run_cli generate --config "$shortcut_config" --stage test
    ;;
  smoke)
    for method in control repair; do
      smoke_resume=()
      if compgen -G "runs/dpo/smoke/${method}-seed-42/checkpoint-*" >/dev/null; then
        smoke_resume=(--resume)
      fi
      run_cli train-dpo \
        --config "$shortcut_config" \
        --method "$method" \
        --seed 42 \
        --smoke \
        "${smoke_resume[@]}"
    done
    sft_smoke_resume=()
    if compgen -G "runs/sft_baseline/smoke/seed-42/checkpoint-*" >/dev/null; then
      sft_smoke_resume=(--resume)
    fi
    run_cli train-sft-baseline \
      --config "$shortcut_config" \
      --seed 42 \
      --smoke \
      "${sft_smoke_resume[@]}"
    ;;
  train)
    for seed in 42 43 44; do
      for method in control repair; do
        formal_resume=()
        if compgen -G "runs/dpo/${method}/seed-${seed}/checkpoint-*" >/dev/null; then
          formal_resume=(--resume)
        fi
        run_cli train-dpo \
          --config "$shortcut_config" \
          --method "$method" \
          --seed "$seed" \
          "${formal_resume[@]}"
      done
      sft_resume=()
      if compgen -G "runs/sft_baseline/seed-${seed}/checkpoint-*" >/dev/null; then
        sft_resume=(--resume)
      fi
      run_cli train-sft-baseline \
        --config "$shortcut_config" \
        --seed "$seed" \
        "${sft_resume[@]}"
    done
    ;;
  evaluate)
    run_cli evaluate \
      --config "$shortcut_config" \
      --split test \
      --model base \
      --output-dir results/test/base
    run_cli evaluate \
      --config "$shortcut_config" \
      --split test \
      --model shortcut \
      --output-dir results/test/shortcut
    for seed in 42 43 44; do
      for method in control repair; do
        run_cli evaluate \
          --config "$shortcut_config" \
          --split test \
          --model adapter \
          --method "$method" \
          --seed "$seed" \
          --output-dir "results/test/${method}/seed-${seed}"
      done
      run_cli evaluate \
        --config "$shortcut_config" \
        --split test \
        --model sft-baseline \
        --seed "$seed" \
        --output-dir "results/test/counterfactual_sft/seed-${seed}"
    done
    ;;
  aggregate)
    run_cli aggregate --config "$shortcut_config" --output-dir reports
    ;;
  all)
    run_stage "prepare"
    run_stage "induce"
    run_stage "gate"
    run_stage "seal-test"
    run_stage "smoke"
    run_stage "train"
    run_stage "evaluate"
    run_stage "aggregate"
    ;;
  *)
    echo "Usage: bash scripts/run_experiment.sh {prepare|induce|gate|seal-test|smoke|train|evaluate|aggregate|all}" >&2
    exit 2
    ;;
esac
