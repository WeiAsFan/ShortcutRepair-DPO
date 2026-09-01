from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from shortcut_repair.evaluate import build_prediction_record
from shortcut_repair.v12_analysis import aggregate_results, select_pilot, summarize, write_report
from shortcut_repair.v12_data import build_test, load_v12_config

ROOT = Path(__file__).resolve().parents[1]


def example_predictions(follow_hint=False, count=16):
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["data"]["test_cases"] = count
    config["data"]["seeds"]["test"] = 73
    rows, _ = build_test(config)
    predictions = []
    for row in rows:
        answer = row["hint"] if follow_hint else row["gold"]
        a, b = (-0.1, -2.0) if answer == "A" else (-2.0, -0.1)
        predictions.append(build_prediction_record(row, a, b, answer))
    return predictions


def formal_records(count=16):
    good, bad = example_predictions(count=count), example_predictions(True, count=count)
    return {
        "base": {"single": good},
        "shortcut": {"single": bad},
        "v1_1_control": {seed: bad for seed in (42, 43, 44)},
        "v1_1_repair": {seed: bad for seed in (42, 43, 44)},
        "score_sft": {seed: good for seed in (42, 43, 44)},
        "selected_dpo": {seed: good for seed in (42, 43, 44)},
    }


def test_five_checks_positive_negative_and_all_model_slices(tmp_path):
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["evaluation"]["bootstrap_samples"] = 100
    records = formal_records()
    result = aggregate_results(records, config)
    assert result["decision"] == "POSITIVE"
    assert len(result["checks"]) == 5 and all(result["checks"].values())
    assert result["bootstrap"]["mean_delta"] == 1
    assert result["bootstrap"]["ci95"] == [1, 1]
    assert len(result["models"]) == 6
    assert all(len(model["decision_type"]) == 2 for model in result["models"].values())
    write_report(result, tmp_path)
    for name in ("RESULTS.md", "results.json", "metrics.csv", "comparison.png"):
        assert (tmp_path / name).is_file()
    records["selected_dpo"] = {seed: example_predictions(True) for seed in (42, 43, 44)}
    negative = aggregate_results(records, config)
    assert negative["decision"] == "NEGATIVE / INCONCLUSIVE"
    assert not negative["checks"]["score_fresh_response"]
    assert not negative["checks"]["score_conflict_accuracy"]
    # 盲从 hint 可以保持 nuisance 不变，不能据此宣称完整修复。
    assert negative["checks"]["score_nuisance_invariance"]


def test_each_seed_must_improve_and_no_seed_may_be_dropped():
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    records = formal_records()
    records["v1_1_repair"][44] = example_predictions()
    result = aggregate_results(records, config)
    assert not result["checks"]["score_fresh_response"]
    del records["selected_dpo"][44]
    with pytest.raises(ValueError, match="seeds"):
        aggregate_results(records, config)


def test_exactly_seventy_percent_passes_despite_floating_mean_roundoff():
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["evaluation"]["bootstrap_samples"] = 100
    records = formal_records(count=320)
    rows = deepcopy(records["selected_dpo"][42])
    score_ids = sorted({row["case_id"] for row in rows if row["decision_type"] == "score_decisive"})
    failing_ids = set(score_ids[:48])  # 每 seed 为 112/160 = 70%。
    for index, row in enumerate(rows):
        if (
            row["case_id"] in failing_ids
            and row["intervention"] == "fresh_flip"
            and row["intervention_variant"] == "flipped"
        ):
            wrong = "B" if row["gold"] == "A" else "A"
            a, b = (-0.1, -2.0) if wrong == "A" else (-2.0, -0.1)
            rows[index] = build_prediction_record(row, a, b, wrong)
    records["selected_dpo"] = {seed: rows for seed in config["seeds"]}
    assert aggregate_results(records, config)["checks"]["score_fresh_response"]


def test_paired_comparisons_reject_different_cases():
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    records = deepcopy(formal_records())
    for row in records["base"]["single"]:
        row["case_id"] += "-different"
    with pytest.raises(ValueError, match="相同"):
        aggregate_results(records, config)


def test_pilot_uses_retention_then_fresh_then_nuisance_then_simplicity():
    rules = load_v12_config(ROOT / "configs/v1_2.yaml")["pilot"]
    good = summarize(example_predictions())
    candidates = {
        "direct_dpo": {"method": "direct_dpo", "metrics": deepcopy(good), "stages": 1},
        "sft_dpo": {"method": "sft_dpo", "metrics": deepcopy(good), "stages": 2},
        "score_sft": {"method": "score_sft", "metrics": good, "stages": 1},
    }
    assert select_pilot(candidates, rules)["selected"] == "direct_dpo"
    candidates["direct_dpo"]["metrics"]["decision_type"]["score_decisive"][
        "nuisance_invariance_rate"
    ] = 0.90
    assert select_pilot(candidates, rules)["selected"] == "sft_dpo"
    candidates["sft_dpo"]["metrics"]["overall"]["aligned_accuracy"] = 0.89
    assert select_pilot(candidates, rules)["selected"] == "direct_dpo"
    candidates["direct_dpo"]["metrics"]["overall"]["aligned_accuracy"] = 0.89
    decision = select_pilot(candidates, rules)
    assert decision["selected"] is None  # SFT 不能冒充已选定的 DPO 路径。
    assert not decision["all_failed_retention"]
