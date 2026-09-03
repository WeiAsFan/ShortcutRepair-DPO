from __future__ import annotations

from pathlib import Path

import pytest

from shortcut_repair.v13_data import build_test, build_train_dev, load_v13_config, prepare_data

ROOT = Path(__file__).resolve().parents[1]


def test_v13_reuses_v12_data_and_adds_identical_anchor():
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    datasets, audit = build_train_dev(config)

    assert {name: len(rows) for name, rows in datasets.items()} == {
        "sft.jsonl": 2560,
        "dpo.jsonl": 1920,
        "dev.jsonl": 1536,
        "anchor.jsonl": 2560,
    }
    assert datasets["anchor.jsonl"] == datasets["sft.jsonl"]
    assert audit["split_case_ids_disjoint"]
    test_rows, test_audit = build_test(config)
    assert len(test_rows) == 1920
    assert test_audit["score_decisive_fraction"] == 0.5
    assert {row["case_id"] for row in test_rows}.isdisjoint(
        {row["case_id"] for rows in datasets.values() for row in rows}
    )


def test_prepare_checks_public_v12_hashes_and_never_creates_test(tmp_path):
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["paths"]["data_dir"] = str(tmp_path)

    first = prepare_data(config)
    assert prepare_data(config) == first
    assert first["files"]["anchor.jsonl"] == first["files"]["sft.jsonl"]
    assert not (tmp_path / "test.jsonl").exists()

    (tmp_path / "anchor.jsonl").write_text("被修改", encoding="utf-8")
    with pytest.raises(ValueError, match="已有数据"):
        prepare_data(config)


def test_v13_does_not_allow_format_gate_or_decoding_protocol_to_be_relaxed():
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["pilot"]["min_greedy_exact_format"] = 0.95
    with pytest.raises(ValueError, match="pilot 阈值"):
        load_v13_config_dict(config)

    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["anchor"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="训练参数"):
        load_v13_config_dict(config)


def load_v13_config_dict(config: dict) -> dict:
    """通过临时 YAML 走公开加载入口，避免绕过冻结配置校验。"""
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as file:
        yaml.safe_dump(config, file, allow_unicode=True)
        path = Path(file.name)
    try:
        return load_v13_config(path)
    finally:
        path.unlink()
