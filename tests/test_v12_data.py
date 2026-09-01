from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from shortcut_repair.data import oracle
from shortcut_repair.v12_data import build_test, build_train_dev, load_v12_config, prepare_data

ROOT = Path(__file__).resolve().parents[1]


def test_training_balance_neutral_oracle_and_nuisance_independence():
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    datasets, audit = build_train_dev(config)
    assert {name: len(rows) for name, rows in datasets.items()} == {
        "sft.jsonl": 2560,
        "dpo.jsonl": 1920,
        "dev.jsonl": 1536,
    }
    for filename in ("sft.jsonl", "dpo.jsonl"):
        rows = datasets[filename]
        assert sum(row["decision_type"] == "score_decisive" for row in rows) / len(rows) == 0.75
        assert Counter(row["gold"] for row in rows) == {"A": len(rows) // 2, "B": len(rows) // 2}
        cells = Counter()
        for row in rows:
            payload = json.loads(row["prompt_messages"][1]["content"])
            case = payload["fresh_tool_result"]
            assert oracle(case) == row["gold"]
            assert row.get("target", row.get("chosen")) == row["gold"]
            if row["variant"] == "neutral":
                assert payload["cached_recommendation"] == "unknown"
            candidates = case["candidates"]
            historical = max(candidates, key=lambda key: candidates[key]["historical_score"])
            first = min(candidates, key=lambda key: candidates[key]["display_rank"])
            cells[row["decision_type"], row["gold"], historical, first] += 1
        for decision_type in ("score_decisive", "validity_decisive"):
            assert len({n for key, n in cells.items() if key[0] == decision_type}) == 1
    assert audit["train"]["score_decisive_fraction"] == 0.75
    assert audit["dev"]["score_decisive_fraction"] == 0.5
    assert audit["split_case_ids_disjoint"]


def test_obvious_gaps_are_only_in_sft_and_pairs_remain_complete():
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["data"]["seeds"]["test"] = 73
    datasets, _ = build_train_dev(config)
    for filename in ("sft.jsonl", "dpo.jsonl"):
        for row in datasets[filename]:
            candidates = json.loads(row["prompt_messages"][1]["content"])["fresh_tool_result"][
                "candidates"
            ]
            gap = abs(candidates["A"]["fresh_score"] - candidates["B"]["fresh_score"])
            if row["signal"] == "obvious_gap":
                assert filename == "sft.jsonl" and row["hint"] == "unknown" and gap >= 45
            else:
                assert 5 <= gap <= 25
    test_rows, _ = build_test(config)
    for rows in (datasets["dev.jsonl"], test_rows):
        assert set(Counter(row["case_id"] for row in rows).values()) == {6}
        assert Counter(row["decision_type"] for row in rows) == {
            "score_decisive": len(rows) // 2,
            "validity_decisive": len(rows) // 2,
        }
    assert {r["case_id"] for r in test_rows}.isdisjoint(
        {r["case_id"] for rows in datasets.values() for r in rows}
    )


def test_prepare_is_deterministic_preserves_existing_files_and_does_not_make_test(tmp_path):
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["paths"]["data_dir"] = str(tmp_path)
    first = prepare_data(config)
    assert prepare_data(config) == first
    assert not (tmp_path / "test.jsonl").exists()
    (tmp_path / "dpo.jsonl").write_text("用户保留的数据", encoding="utf-8")
    with pytest.raises(ValueError, match="已有数据"):
        prepare_data(config)
    assert (tmp_path / "dpo.jsonl").read_text(encoding="utf-8") == "用户保留的数据"
