from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_the_v1_2_claim_and_pilot_stop_boundary():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "受控" in text
    assert "不提出新的 DPO" in text
    assert "score_decisive" in text
    assert "validity_decisive" in text
    assert "selected" in text and "null" in text
    assert "没有生成 test" in text
    assert "不得继续执行 `freeze → formal → report`" in text
    assert "V1_2_PILOT_ANALYSIS.md" in text


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


def test_remote_guide_uses_the_five_registered_stages_and_stop_condition():
    text = (ROOT / "docs/V1_2_REMOTE_EXECUTION_GUIDE.md").read_text(
        encoding="utf-8"
    )

    assert "prepare → pilot → freeze → formal → report" in text
    assert "selected=null" in text
    assert "不要执行 `freeze`" in text
    assert "bash scripts/run_v1_2.sh pilot" in text


def test_pilot_analysis_records_the_audited_result_without_claiming_formal_success():
    text = (ROOT / "docs/V1_2_PILOT_ANALYSIS.md").read_text(encoding="utf-8")

    assert "STOP / NO FORMAL" in text
    assert "7047d067bf464e1ffbb4896a7f27103471bdec3b" in text
    assert "3426.9" in text
    assert "70 / 1536" in text
    assert ".getBcd" in text
    assert "没有生成 test" in text
    assert "不能报告 v1.2 正结果" in text


def test_v1_2_branch_only_keeps_v1_2_results_and_documents():
    for path in (
        "ShortcutRepair-DPO-results",
        "ShortcutRepair-DPO-v1.1-evaluate-failure",
        "ShortcutRepair-DPO-v1.1-fp32-result",
        "docs/EXPERIMENT_PROTOCOL.md",
        "docs/FAILURE_ANALYSIS_2026-08-28.md",
        "docs/PROJECT_EXECUTION_PLAN.md",
        "docs/SERVER_RUNBOOK.md",
        "docs/V1_1_EVALUATION_AMENDMENT.md",
        "docs/V1_1_REMOTE_EXECUTION_GUIDE.md",
        "docs/superpowers",
    ):
        assert not (ROOT / path).exists()

    expected_docs = {
        "V1_2_DESIGN.md",
        "V1_2_EXECUTION_PLAN.md",
        "V1_2_REMOTE_EXECUTION_GUIDE.md",
        "V1_2_PILOT_ANALYSIS.md",
    }
    assert {path.name for path in (ROOT / "docs").glob("*.md")} == expected_docs


def test_pilot_artifact_and_config_match_the_recorded_hashes():
    artifact = ROOT / "artifacts/v1.2/shortcut-repair-v1.2-pilot-7047d06.tar.gz"
    checksum = (
        ROOT / "artifacts/v1.2/shortcut-repair-v1.2-pilot-7047d06.tar.gz.sha256"
    ).read_text(encoding="utf-8")
    expected = "0fc73c6e1852086b6b60501427ac78ca6f32e32210daf26860611281dfaa54d6"

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    assert checksum == (
        f"{expected}  artifacts/v1.2/shortcut-repair-v1.2-pilot-7047d06.tar.gz\n"
    )
    config_sha = hashlib.sha256((ROOT / "configs/v1_2.yaml").read_bytes()).hexdigest()
    assert config_sha == "ec4b234b3e2372c7faa6b5dc0ec11a0ed4e1db3f697648a56e15ffd01623c344"
