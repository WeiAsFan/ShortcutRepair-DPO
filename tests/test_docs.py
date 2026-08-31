from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_the_controlled_claim_and_current_result_status():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "受控诱导" in text
    assert "不是新的 DPO" in text
    assert "尚未在 A6000" in text
    assert "NEGATIVE / INCONCLUSIVE" in text


def test_runbook_contains_exact_install_and_every_manual_stage():
    text = (ROOT / "docs/SERVER_RUNBOOK.md").read_text(encoding="utf-8")

    assert "535.230.02" in text
    assert "CUDA 12.2" in text
    assert "torch==2.5.1" in text
    assert "https://download.pytorch.org/whl/cu121" in text
    assert "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" in text
    for stage in (
        "prepare",
        "induce",
        "gate",
        "seal-test",
        "smoke",
        "train",
        "evaluate",
        "aggregate",
    ):
        assert f"run_experiment.sh {stage}" in text
    assert "--resume" in text
    assert "mechanism_gate.json" in text
    assert "predictions.jsonl" in text
    assert "RESULTS.md" in text


def test_protocol_freezes_hypothesis_metrics_and_success_checks():
    text = (ROOT / "docs/EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")

    assert "Aligned-only DPO" in text
    assert "Counterfactual DPO" in text
    assert "Counterfactual SFT" in text
    assert "fresh-result response rate" in text
    assert "nuisance invariance rate" in text
    assert "greedy exact-format rate" in text
    assert "38" in text
    assert "114" in text
    assert "conflict accuracy" in text
    assert "hint flip rate" in text
    assert "paired case-bootstrap" in text
    assert "10 个百分点" in text
    assert "三个 seed" in text
    config_sha = hashlib.sha256((ROOT / "configs/experiment.yaml").read_bytes()).hexdigest()
    assert config_sha in text
    checksum_file = (ROOT / "configs/experiment.sha256").read_text(encoding="utf-8")
    assert checksum_file == f"{config_sha}  configs/experiment.yaml\n"


def test_documented_commands_match_the_real_config_and_cli():
    runbook = (ROOT / "docs/SERVER_RUNBOOK.md").read_text(encoding="utf-8")
    config = (ROOT / "configs/experiment.yaml").read_text(encoding="utf-8")

    assert "Qwen/Qwen2.5-1.5B-Instruct" in runbook
    assert "Qwen/Qwen2.5-1.5B-Instruct" in config
    assert "configs/experiment.yaml" in runbook
    assert "python -m shortcut_repair.cli" in runbook
    assert "package_results.sh" in runbook
    assert "train-sft-baseline" in runbook
    assert "results/dev/base" in runbook
