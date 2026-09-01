"""v1.2 定向数据；不改动已经冻结的 v1.1 生成器。"""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path
from typing import Any

from shortcut_repair.data import (
    ANSWERS,
    DECISION_TYPES,
    _base_row,
    _copy_case,
    _evaluation_rows,
    _other,
    audit_cases,
    canonical_json,
    load_config,
    make_cases,
    oracle,
    sha256_file,
)
from shortcut_repair.evaluate import _write_jsonl
from shortcut_repair.train import PINNED_MODEL_REVISION


def load_v12_config(path: Path | str) -> dict[str, Any]:
    config = load_config(path)
    if config["project"] != {"version": "1.2", "generator_version": "shortcut-repair-v12"}:
        raise ValueError("请使用独立的 v1.2 配置")
    if config["model"]["revision"] != PINNED_MODEL_REVISION:
        raise ValueError("v1.2 必须复用冻结的 Qwen 模型 revision")
    if config["seeds"] != [42, 43, 44] or config["data"]["score_fraction"] != 0.75:
        raise ValueError("正式 seeds 必须为 42/43/44，训练 score 比例必须为 75%")
    for name, divisor in (("train_cases", 32), ("dev_cases", 16), ("test_cases", 16)):
        value = config["data"][name]
        if not isinstance(value, int) or value <= 0 or value % divisor:
            raise ValueError(f"{name} 必须为 {divisor} 的正整数倍，以保证组合平衡")
    training = config["training"]
    if training["micro_batch_size"] * training["gradient_accumulation_steps"] != 32:
        raise ValueError("v1.2 保持单 GPU、有效 batch size 32")
    for name in ("sft", "dpo"):
        stage = config[name]
        if not isinstance(stage["epochs"], int) or stage["epochs"] <= 0:
            raise ValueError(f"{name}.epochs 必须为正整数")
        if stage["learning_rate"] <= 0:
            raise ValueError(f"{name}.learning_rate 必须为正数")
    if config["dpo"]["loss_type"] != "sigmoid" or config["dpo"]["beta"] <= 0:
        raise ValueError("v1.2 使用正 beta 的标准 sigmoid DPO")
    return config


def make_v12_cases(split: str, count: int, seed: int, score_fraction: float) -> list[dict]:
    """在每种决策内完全交叉 gold、历史分数赢家和展示顺序。"""
    score_count = int(count * score_fraction)
    sizes = (score_count, count - score_count)
    if min(sizes) <= 0 or any(size % 8 for size in sizes):
        raise ValueError("每种决策的 case 数必须能精确平衡 8 种 gold/nuisance 组合")
    assignments = [
        (decision_type, gold, historical, first)
        for decision_type, size in zip(DECISION_TYPES, sizes, strict=True)
        for gold, historical, first in product(ANSWERS, repeat=3)
        for _ in range(size // 8)
    ]
    random.Random(f"v12|{split}|{seed}").shuffle(assignments)
    cases = make_cases(f"v12-{split}", count, seed)
    for case, (decision_type, gold, historical, first) in zip(cases, assignments, strict=True):
        candidates = case["candidates"]
        low, high = sorted(c["fresh_score"] for c in candidates.values())
        hist_low, hist_high = sorted(c["historical_score"] for c in candidates.values())
        for key, fields in candidates.items():
            is_gold = key == gold
            score_decisive = decision_type == "score_decisive"
            fields["is_valid"] = score_decisive or is_gold
            fields["fresh_score"] = high if is_gold == score_decisive else low
            fields["historical_score"] = hist_high if key == historical else hist_low
            fields["display_rank"] = 1 if key == first else 2
        case.update(gold=gold, decision_type=decision_type, split=split)
        if oracle(case) != gold:
            raise AssertionError("v1.2 生成器与 Oracle 不一致")
    return cases


def _neutral_row(case: dict) -> dict:
    row = _base_row(case, case["gold"])
    payload = json.loads(row["prompt_messages"][1]["content"])
    payload["cached_recommendation"] = "unknown"
    row["prompt_messages"][1]["content"] = canonical_json(payload)
    return {**row, "hint": "unknown", "variant": "neutral", "signal": "neutral"}


def _training_rows(cases: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    sft, dpo = [], []
    rng = random.Random(seed)
    for case in cases:
        gold, wrong = case["gold"], _other(case["gold"])
        normal_rows = [
            {**_base_row(case, hint), "signal": "normal_gap"} for hint in (gold, wrong)
        ] + [_neutral_row(case)]
        for row in normal_rows:
            sft.append({**row, "target": gold})
            dpo.append({**row, "chosen": gold, "rejected": wrong})
        warmup = _copy_case(case)
        if case["decision_type"] == "score_decisive":
            warmup["candidates"][gold]["fresh_score"] = rng.randint(80, 95)
            warmup["candidates"][wrong]["fresh_score"] = rng.randint(20, 35)
        # validity 也保留一条中性样本，使 SFT 的行比例仍为 75/25。
        signal = "obvious_gap" if case["decision_type"] == "score_decisive" else "neutral"
        sft.append({**_neutral_row(warmup), "signal": signal, "target": gold})
    return sft, dpo


def build_train_dev(config: dict) -> tuple[dict[str, list[dict]], dict]:
    data = config["data"]
    train = make_v12_cases("train", data["train_cases"], data["seeds"]["train"], 0.75)
    dev = make_v12_cases("dev", data["dev_cases"], data["seeds"]["dev"], 0.5)
    sft, dpo = _training_rows(train, data["seeds"]["train"])
    disjoint = {c["case_id"] for c in train}.isdisjoint(c["case_id"] for c in dev)
    if not disjoint:
        raise AssertionError("train/dev 的 case_id 重叠")
    return {"sft.jsonl": sft, "dpo.jsonl": dpo, "dev.jsonl": _evaluation_rows(dev)}, {
        "train": audit_cases(train),
        "dev": audit_cases(dev),
        "split_case_ids_disjoint": disjoint,
    }


def build_test(config: dict) -> tuple[list[dict], dict]:
    data = config["data"]
    cases = make_v12_cases("test", data["test_cases"], data["seeds"]["test"], 0.5)
    return _evaluation_rows(cases), audit_cases(cases)


def prepare_data(config: dict) -> dict:
    """只生成 train/dev；重复执行时保留相同文件、拒绝覆盖不同数据。"""
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
    return {
        "files": {
            name: {"rows": len(rows), "sha256": sha256_file(root / name)}
            for name, rows in datasets.items()
        },
        "audit": audit,
    }
