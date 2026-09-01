#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${script_dir}/.."
repo_root="$(pwd)"
cd "$repo_root"

required_files=(
  "configs/experiment.yaml"
  "configs/experiment.sha256"
  "configs/evaluation_amendment.yaml"
  "data/manifest_train_dev.json"
  "data/manifest_test.json"
  "results/dev/shortcut/mechanism_gate.json"
  "results/dev/base/predictions.jsonl"
  "results/dev/base/metrics.json"
  "results/dev/base/prediction_manifest.json"
  "results/dev/shortcut/predictions.jsonl"
  "results/dev/shortcut/metrics.json"
  "results/dev/shortcut/prediction_manifest.json"
  "runs/shortcut/run_manifest.json"
  "results/test/base/predictions.jsonl"
  "results/test/base/metrics.json"
  "results/test/base/prediction_manifest.json"
  "results/test/shortcut/predictions.jsonl"
  "results/test/shortcut/metrics.json"
  "results/test/shortcut/prediction_manifest.json"
  "reports/RESULTS.md"
  "reports/results.json"
  "reports/main_metrics.csv"
  "reports/baseline_metrics.csv"
  "reports/decision_type_metrics.csv"
  "reports/per_seed.csv"
  "reports/comparison.png"
)

for method in control repair; do
  for seed in 42 43 44; do
    required_files+=("runs/dpo/${method}/seed-${seed}/run_manifest.json")
    required_files+=("results/test/${method}/seed-${seed}/predictions.jsonl")
    required_files+=("results/test/${method}/seed-${seed}/metrics.json")
    required_files+=("results/test/${method}/seed-${seed}/prediction_manifest.json")
  done
done

for seed in 42 43 44; do
  required_files+=("runs/sft_baseline/seed-${seed}/run_manifest.json")
  required_files+=("results/test/counterfactual_sft/seed-${seed}/predictions.jsonl")
  required_files+=("results/test/counterfactual_sft/seed-${seed}/metrics.json")
  required_files+=("results/test/counterfactual_sft/seed-${seed}/prediction_manifest.json")
done

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required public artifact is missing: $path" >&2
    exit 2
  fi
done

mkdir -p artifacts
timestamp="$(date -u +%Y%m%d-%H%M%S)"
archive="artifacts/shortcut-repair-results-${timestamp}.tar.gz"
tar -czf "$archive" "${required_files[@]}"
sha256sum "$archive" > "${archive}.sha256"

echo "$archive"
echo "${archive}.sha256"
