from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_the_controlled_claim_and_current_result_status():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "受控诱导" in text
    assert "不是新的 DPO" in text
    assert "无效评测" in text
    assert "NEGATIVE / INCONCLUSIVE" in text
    assert "正式评测已经完成" in text
    assert "decision_type_metrics.csv" in text


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
    assert "正式结论：`NEGATIVE / INCONCLUSIVE`" in text
    assert "不改变九项预注册判定" in text
    config_sha = hashlib.sha256((ROOT / "configs/experiment.yaml").read_bytes()).hexdigest()
    assert config_sha in text
    checksum_file = (ROOT / "configs/experiment.sha256").read_text(encoding="utf-8")
    assert checksum_file == f"{config_sha}  configs/experiment.yaml\n"


def test_evaluation_amendment_discloses_the_incident_and_recovery_boundary():
    text = (ROOT / "docs/V1_1_EVALUATION_AMENDMENT.md").read_text(
        encoding="utf-8"
    )
    amendment = (ROOT / "configs/evaluation_amendment.yaml").read_text(
        encoding="utf-8"
    )

    assert "BF16" in text
    assert "FP32" in text
    assert "看到部分" in text
    assert "不应伪装成事前注册" in text
    assert "不应重训" in text
    assert "gate → evaluate → aggregate" in text
    assert "1ead3b24f00f33569128a6634401729e4908a62f" in text
    assert "tie_policy: reject_with_context" in amendment


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


def test_v1_2_documents_keep_the_project_small_and_the_runtime_flow_lightweight():
    design = (ROOT / "docs/V1_2_DESIGN.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs/V1_2_EXECUTION_PLAN.md").read_text(encoding="utf-8")

    for text in (design, plan):
        assert "score_decisive" in text
        assert "validity_decisive" in text
        assert "fresh-result response" in text
        assert "nuisance invariance" in text
        assert "面试" in text

    assert "不设置独立 smoke 阶段" in design
    assert "不重复计算完整模型权重 SHA256" in design
    assert "prepare → pilot → freeze → formal → report" in plan
    assert "正式冻结时" in plan and "Git 工作树" in plan
