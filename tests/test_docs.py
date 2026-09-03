from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_v13_claim_protocol_and_real_gpu_boundary():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for phrase in (
        "面试",
        "SFT → DPO",
        "A/B+EOS",
        "is_trainable=True",
        "2e-6",
        "开放词表",
        "selected=null",
        "没有生成 test",
        "真实 GPU pilot",
    ):
        assert phrase in text
    assert "不得执行 freeze" in text
    assert "V1_3_DESIGN.md" in text
    assert "V1_3_EXECUTION_PLAN.md" in text


def test_v13_documents_freeze_one_small_falsifiable_change():
    design = (ROOT / "docs/V1_3_DESIGN.md").read_text(encoding="utf-8")
    plan = (ROOT / "docs/V1_3_EXECUTION_PLAN.md").read_text(encoding="utf-8")

    for text in (design, plan):
        assert "format-anchor" in text or "Format-anchor" in text
        assert "面试" in text
        assert "prepare → pilot → freeze → formal → report" in text
    for phrase in (
        "不把 greedy exact-format 门槛从 0.98 下调",
        "不在 pilot 中搜索",
        "继续训练原 DPO adapter",
        "selected=null",
        "不生成 test",
    ):
        assert phrase in design
    assert "不新增独立 smoke" in plan
    assert "全量模型权重哈希" in plan


def test_v13_branch_only_keeps_v13_documents_and_no_previous_result_archives():
    expected_docs = {"V1_3_DESIGN.md", "V1_3_EXECUTION_PLAN.md"}
    assert {path.name for path in (ROOT / "docs").glob("*.md")} == expected_docs
    assert not (ROOT / "artifacts/v1.2").exists()
    assert not (ROOT / "ShortcutRepair-DPO-results").exists()


def test_v13_config_keeps_published_v12_data_hashes():
    text = (ROOT / "configs/v1_3.yaml").read_text(encoding="utf-8")
    for digest in (
        "52cf8f7f52df4961cea82df3c4c26b650f879219b4f586a7b383f4cb509ce7c3",
        "cc29ff24ed17399209c743272f97bac19f1476c4854d616690f483f028235325",
        "7223511e6da410e20b70adb168f5c1b0231b1357707f7f434dfcb00ffa237434",
    ):
        assert digest in text
    assert hashlib.sha256((ROOT / "configs/v1_2.yaml").read_bytes()).hexdigest() == (
        "ec4b234b3e2372c7faa6b5dc0ec11a0ed4e1db3f697648a56e15ffd01623c344"
    )
