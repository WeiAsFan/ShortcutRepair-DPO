"""v1.3 复用 v1.2 train/dev，只改变后训练链。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shortcut_repair.data import canonical_json, load_config, sha256_file
from shortcut_repair.evaluate import _write_jsonl
from shortcut_repair.train import PINNED_MODEL_REVISION
from shortcut_repair.v12_data import build_test as build_v12_test
from shortcut_repair.v12_data import build_train_dev as build_v12_train_dev


def load_v13_config(path: Path | str) -> dict[str, Any]:
    config = load_config(path)
    if config["project"] != {
        "version": "1.3",
        "generator_version": "shortcut-repair-v12-reuse-v13-anchor",
    }:
        raise ValueError("请使用独立的 v1.3 配置")
    if config["model"]["revision"] != PINNED_MODEL_REVISION:
        raise ValueError("v1.3 必须复用冻结的 Qwen 模型 revision")
    if config["seeds"] != [42, 43, 44] or config["data"]["score_fraction"] != 0.75:
        raise ValueError("正式 seeds 必须为 42/43/44，训练 score 比例必须为 75%")
    expected_data_seeds = {"train": 12021, "dev": 12022, "test": 13023}
    if config["data"]["seeds"] != expected_data_seeds:
        raise ValueError("v1.3 必须复用 v1.2 train/dev，并使用冻结的新 test seed")
    for name, divisor in (("train_cases", 32), ("dev_cases", 16), ("test_cases", 16)):
        value = config["data"][name]
        if not isinstance(value, int) or value <= 0 or value % divisor:
            raise ValueError(f"{name} 必须为 {divisor} 的正整数倍")
    training = config["training"]
    if training["micro_batch_size"] * training["gradient_accumulation_steps"] != 32:
        raise ValueError("v1.3 保持单 GPU、有效 batch size 32")
    expected_stages = {
        "sft": {"learning_rate": 1e-5, "epochs": 1},
        "dpo": {
            "learning_rate": 1e-5,
            "epochs": 3,
            "beta": 0.1,
            "loss_type": "sigmoid",
        },
        "anchor": {"learning_rate": 2e-6, "epochs": 1},
    }
    if any(config[name] != expected for name, expected in expected_stages.items()):
        raise ValueError("SFT、DPO 与 format anchor 的训练参数必须保持冻结值")
    if config["evaluation"]["generation_max_new_tokens"] != 4:
        raise ValueError("不得用缩短生成长度隐藏 v1.2 的开放词表格式问题")
    expected_pilot = {
        "min_aligned_accuracy": 0.90,
        "min_validity_conflict_accuracy": 0.95,
        "min_score_conflict_accuracy": 0.95,
        "min_score_fresh_response": 0.95,
        "min_score_nuisance_invariance": 0.95,
        "min_greedy_exact_format": 0.98,
        "min_format_gain": 0.02,
        "max_core_regression": 0.02,
    }
    if config["pilot"] != expected_pilot:
        raise ValueError("v1.3 pilot 阈值已冻结，不得放宽")
    return config


def build_train_dev(config: dict) -> tuple[dict[str, list[dict]], dict]:
    datasets, audit = build_v12_train_dev(config)
    return {**datasets, "anchor.jsonl": datasets["sft.jsonl"]}, audit


def build_test(config: dict) -> tuple[list[dict], dict]:
    return build_v12_test(config)


def prepare_data(config: dict) -> dict:
    datasets, audit = build_train_dev(config)
    root = Path(config["paths"]["data_dir"])
    for name, rows in datasets.items():
        path = root / name
        expected = "".join(canonical_json(row) + "\n" for row in rows)
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"已有数据与当前配置不一致，不覆盖：{path}")
    for name, rows in datasets.items():
        path = root / name
        if not path.exists():
            _write_jsonl(path, rows)
    files = {
        name: {"rows": len(rows), "sha256": sha256_file(root / name)}
        for name, rows in datasets.items()
    }
    for name, expected_hash in config["v12_pilot"]["data_sha256"].items():
        if files[name]["sha256"] != expected_hash:
            raise ValueError(f"{name} 未与 v1.2 公开数据保持逐字节一致")
    if files["anchor.jsonl"] != files["sft.jsonl"]:
        raise AssertionError("anchor 数据必须与 SFT 数据逐字节一致")
    return {"files": files, "audit": audit}
