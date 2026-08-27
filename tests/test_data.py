from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from conftest import read_jsonl, write_small_config

from shortcut_repair.data import (
    generate_sealed_test,
    generate_train_dev,
    make_cases,
    oracle,
    prompt_messages,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oracle_obeys_validity_before_fresh_score():
    case = {
        "candidates": {
            "A": {"is_valid": False, "fresh_score": 99},
            "B": {"is_valid": True, "fresh_score": 1},
        }
    }

    assert oracle(case) == "B"


def test_oracle_uses_highest_fresh_score_and_rejects_invalid_cases():
    assert oracle(
        {
            "candidates": {
                "A": {"is_valid": True, "fresh_score": 70},
                "B": {"is_valid": True, "fresh_score": 80},
            }
        }
    ) == "B"
    with pytest.raises(ValueError, match="valid"):
        oracle(
            {
                "candidates": {
                    "A": {"is_valid": False, "fresh_score": 70},
                    "B": {"is_valid": False, "fresh_score": 80},
                }
            }
        )


def test_cases_are_deterministic_balanced_and_oracle_consistent():
    first = make_cases("dev", count=20, seed=8303)
    again = make_cases("dev", count=20, seed=8303)

    assert first == again
    assert Counter(case["gold"] for case in first) == {"A": 10, "B": 10}
    assert all(oracle(case) == case["gold"] for case in first)
    assert len({case["case_id"] for case in first}) == 20


def test_prompt_intervention_changes_only_the_cached_hint():
    case = make_cases("dev", count=1, seed=8303)[0]
    prompt_a = prompt_messages(case, "A")
    prompt_b = prompt_messages(case, "B")
    user_a = json.loads(prompt_a[1]["content"])
    user_b = json.loads(prompt_b[1]["content"])

    assert prompt_a[0] == prompt_b[0]
    assert user_a["fresh_tool_result"] == user_b["fresh_tool_result"]
    assert user_a["decision_rule"] == user_b["decision_rule"]
    assert user_a["cached_recommendation"] == "A"
    assert user_b["cached_recommendation"] == "B"


def test_generated_conditions_have_matched_budget_and_cases(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, dev=4)
    generate_train_dev(config_path)
    data_dir = tmp_path / "data"
    induction = read_jsonl(data_dir / "induction.jsonl")
    control = read_jsonl(data_dir / "dpo_control.jsonl")
    repair = read_jsonl(data_dir / "dpo_repair.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")

    assert len(induction) == 8
    assert len(control) == len(repair) == 12
    assert len(dev) == 8
    assert Counter(row["case_id"] for row in control) == Counter(
        row["case_id"] for row in repair
    )
    assert {row["variant"] for row in control} == {"aligned"}
    assert Counter(row["variant"] for row in repair) == {"aligned": 6, "conflict": 6}
    assert all(row["chosen"] == row["gold"] for row in control + repair)
    assert all(row["chosen"] != row["rejected"] for row in control + repair)


def test_induction_targets_hint_with_exactly_half_oracle_conflicts(tmp_path):
    config_path = write_small_config(tmp_path, induction=10)
    generate_train_dev(config_path)
    rows = read_jsonl(tmp_path / "data/induction.jsonl")

    assert all(row["target"] == row["hint"] for row in rows)
    assert sum(row["target"] != row["gold"] for row in rows) == len(rows) // 2
    assert Counter(row["variant"] for row in rows) == {"aligned": 10, "conflict": 10}


def test_regeneration_is_byte_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_config = write_small_config(first_dir)
    second_config = write_small_config(second_dir)

    generate_train_dev(first_config)
    generate_train_dev(second_config)

    for name in ("induction.jsonl", "dpo_control.jsonl", "dpo_repair.jsonl", "dev.jsonl"):
        assert _sha256(first_dir / "data" / name) == _sha256(second_dir / "data" / name)


def test_test_split_is_sealed_and_tampering_is_refused(tmp_path):
    config_path = write_small_config(tmp_path, test=6)
    manifest = generate_sealed_test(config_path)
    test_path = tmp_path / "data/test.jsonl"

    assert manifest["sealed"] is True
    assert manifest["files"]["test.jsonl"]["rows"] == 12
    assert manifest["files"]["test.jsonl"]["sha256"] == _sha256(test_path)
    assert generate_sealed_test(config_path) == manifest

    test_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed"):
        generate_sealed_test(config_path)
