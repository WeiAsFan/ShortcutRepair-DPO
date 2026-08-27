#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${script_dir}/.."
repo_root="$(pwd)"
cd "$repo_root"

required_files=(
  "configs/experiment.yaml"
  "data/manifest_train_dev.json"
  "data/manifest_test.json"
  "results/dev/shortcut/mechanism_gate.json"
  "results/dev/shortcut/predictions.jsonl"
  "results/dev/shortcut/metrics.json"
  "runs/shortcut/run_manifest.json"
  "results/test/shortcut/predictions.jsonl"
  "results/test/shortcut/metrics.json"
  "results/test/shortcut/prediction_manifest.json"
  "reports/RESULTS.md"
  "reports/results.json"
  "reports/main_metrics.csv"
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
