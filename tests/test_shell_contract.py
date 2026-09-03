from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ("preflight.sh", "run_experiment.sh", "package_results.sh")
)
def test_shell_scripts_exist_use_strict_mode_and_lf(name):
    path = ROOT / "scripts" / name
    data = path.read_bytes()
    text = data.decode()

    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in data
    assert "set -euo pipefail" in text


def test_runner_exposes_every_stage_and_orders_gate_before_formal_work():
    text = (ROOT / "scripts/run_experiment.sh").read_text(encoding="utf-8")
    stages = (
        "prepare",
        "induce",
        "gate",
        "seal-test",
        "smoke",
        "train",
        "evaluate",
        "aggregate",
        "all",
    )
    for stage in stages:
        assert f"{stage})" in text
    all_block = text[text.index("all)") :]
    assert all_block.index('run_stage "gate"') < all_block.index('run_stage "seal-test"')
    assert all_block.index('run_stage "seal-test"') < all_block.index('run_stage "train"')


def test_runner_uses_all_three_seeds_for_both_methods():
    text = (ROOT / "scripts/run_experiment.sh").read_text(encoding="utf-8")

    assert 'for seed in 42 43 44' in text
    assert 'for method in control repair' in text
    assert "train-sft-baseline" in text
    assert "--model base" in text
    assert "--model sft-baseline" in text
    assert "--base-predictions" in text
    assert "shortcut_repair.cli" in text
    prepare_block = text[text.index("prepare)") : text.index("induce)")]
    assert prepare_block.count("--dry-run") == 3


def test_public_package_is_allowlist_only_and_keeps_sanitized_predictions():
    text = (ROOT / "scripts/package_results.sh").read_text(encoding="utf-8")

    assert "manifest_train_dev.json" in text
    assert "mechanism_gate.json" in text
    assert "RESULTS.md" in text
    assert "run_manifest.json" in text
    assert "prediction_manifest.json" in text
    assert "predictions.jsonl" in text
    assert "metrics.json" in text
    assert "sft_baseline" in text
    assert "counterfactual_sft" in text
    assert "results/test/base" in text
    assert "baseline_metrics.csv" in text
    assert "decision_type_metrics.csv" in text
    assert "configs/evaluation_amendment.yaml" in text
    assert "hostname" not in text
    assert "gpu_uuid" not in text.lower()


def test_runner_passes_the_frozen_evaluation_amendment_explicitly():
    text = (ROOT / "scripts/run_experiment.sh").read_text(encoding="utf-8")

    assert "SHORTCUT_EVALUATION_AMENDMENT" in text
    assert text.count('--evaluation-amendment "$shortcut_evaluation_amendment"') == 7


def test_preflight_checks_the_actual_a6000_contract():
    text = (ROOT / "scripts/preflight.sh").read_text(encoding="utf-8")

    assert "driver_version" in text
    assert "535" in text
    assert "torch.version.cuda" in text
    assert 'startswith("12.1")' in text
    assert "is_bf16_supported" in text
    assert "45 * 1024**3" in text
    assert "git rev-parse --verify HEAD" in text
    assert "git status --porcelain" in text
    assert "sha256sum -c configs/experiment.sha256" in text


def test_v13_runner_is_a_small_strict_five_stage_entrypoint():
    path = ROOT / "scripts/run_v1_3.sh"
    data = path.read_bytes()
    text = data.decode()

    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in data
    assert "set -euo pipefail" in text
    assert "shortcut_repair.v13" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "smoke" not in text
