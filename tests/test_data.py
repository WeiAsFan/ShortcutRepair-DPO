from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from conftest import read_jsonl, write_small_config

from shortcut_repair.data import (
    audit_cases,
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
    assert Counter(case["decision_type"] for case in first) == {
        "score_decisive": 10,
        "validity_decisive": 10,
    }
    assert all(oracle(case) == case["gold"] for case in first)
    assert len({case["case_id"] for case in first}) == 20
    assert len({case["request_id"] for case in first}) == 20
    assert all("dev" not in case["case_id"] for case in first)
    assert all("dev" not in case["request_id"] for case in first)


def test_generated_cases_make_score_and_validity_decisive():
    cases = make_cases("dev", count=20, seed=8303)

    for case in cases:
        gold = case["gold"]
        wrong = "B" if gold == "A" else "A"
        candidates = case["candidates"]
        if case["decision_type"] == "score_decisive":
            assert candidates[gold]["is_valid"] is True
            assert candidates[wrong]["is_valid"] is True
            assert candidates[gold]["fresh_score"] > candidates[wrong]["fresh_score"]
        else:
            assert candidates[gold]["is_valid"] is True
            assert candidates[wrong]["is_valid"] is False
            assert candidates[gold]["fresh_score"] < candidates[wrong]["fresh_score"]


def test_nuisance_fields_and_fresh_score_alone_cannot_predict_gold():
    audit = audit_cases(make_cases("dpo", count=200, seed=8202))

    assert audit["gold_A_fraction"] == 0.5
    assert audit["score_decisive_fraction"] == 0.5
    assert audit["validity_decisive_fraction"] == 0.5
    assert audit["fresh_score_only_accuracy"] == 0.5
    assert audit["historical_only_accuracy"] == 0.5
    assert audit["display_rank_only_accuracy"] == 0.5
    assert audit["split_marker_count"] == 0
    assert audit["request_id_unique_across_cases"] is True


def test_opaque_ids_are_disjoint_across_splits():
    dev = make_cases("dev", count=20, seed=8303)
    test = make_cases("test", count=20, seed=9404)

    assert {case["case_id"] for case in dev}.isdisjoint(
        {case["case_id"] for case in test}
    )
    assert {case["request_id"] for case in dev}.isdisjoint(
        {case["request_id"] for case in test}
    )


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
    sft_baseline = read_jsonl(data_dir / "sft_counterfactual.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")

    assert len(induction) == 8
    assert len(control) == len(repair) == 12
    assert len(sft_baseline) == 12
    assert len(dev) == 24
    assert Counter(row["case_id"] for row in control) == Counter(
        row["case_id"] for row in repair
    )
    assert {row["variant"] for row in control} == {"aligned"}
    assert Counter(row["variant"] for row in repair) == {"aligned": 6, "conflict": 6}
    assert all(row["chosen"] == row["gold"] for row in control + repair)
    assert all(row["chosen"] != row["rejected"] for row in control + repair)
    assert all(row["target"] == row["gold"] for row in sft_baseline)
    assert Counter(row["case_id"] for row in sft_baseline) == Counter(
        row["case_id"] for row in repair
    )
    assert {row["decision_type"] for row in control + repair + dev} == {
        "score_decisive",
        "validity_decisive",
    }


def test_evaluation_rows_isolate_three_causal_interventions(tmp_path):
    config_path = write_small_config(tmp_path, dev=4)
    generate_train_dev(config_path)
    rows = read_jsonl(tmp_path / "data/dev.jsonl")
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], {}).setdefault(
            row["intervention"], []
        ).append(row)

    assert len(grouped) == 4
    for interventions in grouped.values():
        assert set(interventions) == {"hint_flip", "fresh_flip", "nuisance_flip"}
        for pair in interventions.values():
            assert {row["intervention_variant"] for row in pair} == {
                "original",
                "flipped",
            }

        hint_pair = sorted(
            interventions["hint_flip"], key=lambda row: row["intervention_variant"]
        )
        hint_payloads = [json.loads(row["prompt_messages"][1]["content"]) for row in hint_pair]
        assert hint_payloads[0]["fresh_tool_result"] == hint_payloads[1]["fresh_tool_result"]
        assert hint_payloads[0]["cached_recommendation"] != hint_payloads[1][
            "cached_recommendation"
        ]

        fresh_pair = sorted(
            interventions["fresh_flip"], key=lambda row: row["intervention_variant"]
        )
        fresh_payloads = [
            json.loads(row["prompt_messages"][1]["content"]) for row in fresh_pair
        ]
        assert fresh_pair[0]["gold"] != fresh_pair[1]["gold"]
        assert fresh_payloads[0]["cached_recommendation"] == fresh_payloads[1][
            "cached_recommendation"
        ]
        assert fresh_payloads[0]["fresh_tool_result"] != fresh_payloads[1][
            "fresh_tool_result"
        ]
        for candidate_id in ("A", "B"):
            left = fresh_payloads[0]["fresh_tool_result"]["candidates"][candidate_id]
            right = fresh_payloads[1]["fresh_tool_result"]["candidates"][candidate_id]
            assert left["historical_score"] == right["historical_score"]
            assert left["display_rank"] == right["display_rank"]

        nuisance_pair = sorted(
            interventions["nuisance_flip"],
            key=lambda row: row["intervention_variant"],
        )
        nuisance_payloads = [
            json.loads(row["prompt_messages"][1]["content"]) for row in nuisance_pair
        ]
        assert nuisance_pair[0]["gold"] == nuisance_pair[1]["gold"]
        assert nuisance_payloads[0]["cached_recommendation"] == nuisance_payloads[1][
            "cached_recommendation"
        ]
        for candidate_id in ("A", "B"):
            left = nuisance_payloads[0]["fresh_tool_result"]["candidates"][candidate_id]
            right = nuisance_payloads[1]["fresh_tool_result"]["candidates"][candidate_id]
            assert left["is_valid"] == right["is_valid"]
            assert left["fresh_score"] == right["fresh_score"]
            assert (left["historical_score"], left["display_rank"]) != (
                right["historical_score"],
                right["display_rank"],
            )


def test_train_dev_manifest_contains_and_enforces_data_audits(tmp_path):
    config_path = write_small_config(tmp_path, induction=20, dpo=20, dev=20)
    manifest = generate_train_dev(config_path)

    assert manifest["generator_version"] == "shortcut-repair-v2"
    assert manifest["audit"]["request_id_unique_across_cases"] is True
    assert manifest["audit"]["request_id_disjoint_across_splits"] is True
    assert manifest["audit"]["dpo_case_multiset_equal"] is True
    assert manifest["audit"]["sft_dpo_case_multiset_equal"] is True
    for split in ("induction", "dpo", "dev"):
        split_audit = manifest["audit"]["splits"][split]
        assert split_audit["gold_A_fraction"] == 0.5
        assert split_audit["score_decisive_fraction"] == 0.5
        assert split_audit["validity_decisive_fraction"] == 0.5
        assert split_audit["historical_only_accuracy"] <= 0.55
        assert split_audit["display_rank_only_accuracy"] <= 0.55
        assert split_audit["split_marker_count"] == 0


def test_generation_rejects_audit_threshold_that_data_cannot_meet(tmp_path):
    config_path = write_small_config(tmp_path, induction=20, dpo=20, dev=20)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["audit"]["max_nuisance_accuracy"] = 0.49
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="historical_only_accuracy"):
        generate_train_dev(config_path)


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

    for name in (
        "induction.jsonl",
        "dpo_control.jsonl",
        "dpo_repair.jsonl",
        "sft_counterfactual.jsonl",
        "dev.jsonl",
    ):
        assert _sha256(first_dir / "data" / name) == _sha256(second_dir / "data" / name)


def test_test_split_is_sealed_and_tampering_is_refused(tmp_path):
    config_path = write_small_config(tmp_path, test=6)
    manifest = generate_sealed_test(config_path)
    test_path = tmp_path / "data/test.jsonl"

    assert manifest["sealed"] is True
    assert manifest["generator_version"] == "shortcut-repair-v2"
    assert manifest["files"]["test.jsonl"]["rows"] == 36
    assert manifest["files"]["test.jsonl"]["sha256"] == _sha256(test_path)
    assert manifest["audit"]["gold_A_fraction"] == 0.5
    assert manifest["audit"]["validity_decisive_fraction"] == 0.5
    assert manifest["audit"]["historical_only_accuracy"] <= 0.55
    assert generate_sealed_test(config_path) == manifest

    test_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed"):
        generate_sealed_test(config_path)
