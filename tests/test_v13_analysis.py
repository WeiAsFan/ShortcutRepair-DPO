from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from shortcut_repair.evaluate import build_prediction_record
from shortcut_repair.v12_analysis import summarize
from shortcut_repair.v13_analysis import aggregate_v13, select_anchor_pilot
from shortcut_repair.v13_data import build_test, load_v13_config

ROOT = Path(__file__).resolve().parents[1]


def predictions(*, follow_hint: bool = False, malformed: int = 0) -> list[dict]:
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["data"]["test_cases"] = 16
    config["data"]["seeds"]["test"] = 73
    rows, _ = build_test(config)
    output = []
    for index, row in enumerate(rows):
        answer = row["hint"] if follow_hint else row["gold"]
        a, b = (-0.1, -2.0) if answer == "A" else (-2.0, -0.1)
        generated = ".getB" if index < malformed else answer
        output.append(build_prediction_record(row, a, b, generated))
    return output


def test_anchor_pilot_requires_format_gain_and_retains_core_metrics():
    rules = load_v13_config(ROOT / "configs/v1_3.yaml")["pilot"]
    before = summarize(predictions(malformed=6))
    after = summarize(predictions())
    decision = select_anchor_pilot(before, after, rules)

    assert decision["selected"] == "sft_dpo_anchor"
    assert len(decision["checks"]) == 8 and all(decision["checks"].values())
    assert decision["deltas"]["greedy_exact_format_rate"] > 0.02

    unchanged = select_anchor_pilot(after, after, rules)
    assert unchanged["selected"] is None
    assert not unchanged["checks"]["format_gain"]

    regressed = deepcopy(after)
    regressed["decision_type"]["score_decisive"]["conflict_accuracy"] = 0.90
    failed = select_anchor_pilot(before, regressed, rules)
    assert failed["selected"] is None
    assert not failed["checks"]["score_conflict_retained"]


def test_formal_report_contains_paired_pre_and_post_anchor_delta():
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["data"]["test_cases"] = 16
    config["data"]["seeds"]["test"] = 73
    config["evaluation"]["bootstrap_samples"] = 100
    good, old, pre = predictions(), predictions(follow_hint=True), predictions(malformed=6)
    records = {
        "base": {"single": good},
        "shortcut": {"single": old},
        "v1_1_control": {seed: old for seed in config["seeds"]},
        "v1_1_repair": {seed: old for seed in config["seeds"]},
        "score_sft": {seed: good for seed in config["seeds"]},
        "sft_dpo_pre_anchor": {seed: pre for seed in config["seeds"]},
        "selected_dpo": {seed: good for seed in config["seeds"]},
    }

    result = aggregate_v13(records, config)
    assert result["version"] == "1.3"
    assert result["decision"] == "POSITIVE"
    assert len(result["models"]) == 7
    assert result["anchor_minus_pre_anchor"]["overall"]["greedy_exact_format_rate"] > 0
