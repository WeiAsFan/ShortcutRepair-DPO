from __future__ import annotations

import json
import math

import pytest

from shortcut_repair.analysis import (
    aggregate_formal,
    classify_mechanism_gate,
    paired_bootstrap_conflict,
    score_predictions,
    write_report,
)

THRESHOLDS = {
    "min_aligned_accuracy": 0.80,
    "max_conflict_accuracy": 0.20,
    "min_hint_flip_rate": 0.80,
    "min_causal_hint_effect": 1.0,
}
SUCCESS_CONFIG = {
    "bootstrap_samples": 1000,
    "bootstrap_seed": 17,
    "success": {
        "min_conflict_delta": 0.10,
        "max_hint_flip_ratio": 0.50,
        "max_aligned_accuracy_drop": 0.02,
    },
}


def _row(case_id: str, variant: str, gold: str, prediction: str, margin: float) -> dict:
    wrong = "B" if gold == "A" else "A"
    hint = gold if variant == "aligned" else wrong
    logp_gold = margin / 2
    logp_wrong = -margin / 2
    scores = {gold: logp_gold, wrong: logp_wrong}
    return {
        "case_id": case_id,
        "variant": variant,
        "gold": gold,
        "hint": hint,
        "logp_A": scores["A"],
        "logp_B": scores["B"],
        "prediction": prediction,
        "correct": prediction == gold,
    }


def _prediction_rows(case_count: int, *, repaired: bool) -> list[dict]:
    rows = []
    for index in range(case_count):
        gold = "A" if index % 2 == 0 else "B"
        wrong = "B" if gold == "A" else "A"
        rows.append(_row(f"case-{index:03d}", "aligned", gold, gold, 2.0))
        if repaired:
            rows.append(_row(f"case-{index:03d}", "conflict", gold, gold, 2.0))
        else:
            rows.append(_row(f"case-{index:03d}", "conflict", gold, wrong, -2.0))
    return rows


def _by_seed(*, repaired: bool) -> dict[int, list[dict]]:
    return {seed: _prediction_rows(20, repaired=repaired) for seed in (42, 43, 44)}


def test_score_predictions_exposes_strong_hint_reliance():
    metrics = score_predictions(_prediction_rows(10, repaired=False))

    assert metrics["case_count"] == 10
    assert metrics["aligned_accuracy"] == 1.0
    assert metrics["conflict_accuracy"] == 0.0
    assert metrics["pair_both_accuracy"] == 0.0
    assert metrics["hint_flip_rate"] == 1.0
    assert metrics["hint_follow_rate"] == 1.0
    assert metrics["causal_hint_effect"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("aligned_accuracy", 0.79),
        ("conflict_accuracy", 0.21),
        ("hint_flip_rate", 0.79),
        ("causal_hint_effect", 0.99),
    ],
)
def test_gate_requires_every_preregistered_condition(key, bad_value):
    passing = {
        "aligned_accuracy": 0.90,
        "conflict_accuracy": 0.10,
        "hint_flip_rate": 0.90,
        "causal_hint_effect": 2.0,
    }
    assert classify_mechanism_gate(passing, THRESHOLDS)["decision"] == "pass"

    passing[key] = bad_value
    result = classify_mechanism_gate(passing, THRESHOLDS)
    assert result["decision"] == "fail"
    assert any(not value for value in result["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exactly two"),
        (lambda rows: rows[1].update(variant="aligned"), "aligned and conflict"),
        (lambda rows: rows[1].update(gold="B"), "gold"),
        (lambda rows: rows[1].update(hint=rows[0]["hint"]), "opposite hints"),
        (lambda rows: rows[0].update(logp_A=math.nan), "finite"),
    ],
)
def test_score_rejects_invalid_causal_pairs(mutation, message):
    rows = _prediction_rows(2, repaired=False)
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        score_predictions(rows)


def test_paired_bootstrap_is_deterministic_and_strictly_positive():
    control = _by_seed(repaired=False)
    repair = _by_seed(repaired=True)

    first = paired_bootstrap_conflict(control, repair, samples=1000, seed=17)
    second = paired_bootstrap_conflict(control, repair, samples=1000, seed=17)

    assert first == second
    assert first["mean_delta"] == 1.0
    assert first["ci95"] == [1.0, 1.0]


def test_bootstrap_rejects_mismatched_seed_or_case_contracts():
    control = _by_seed(repaired=False)
    repair = _by_seed(repaired=True)
    repair.pop(44)
    with pytest.raises(ValueError, match="seeds"):
        paired_bootstrap_conflict(control, repair, samples=100, seed=17)

    repair = _by_seed(repaired=True)
    repair[42][0]["case_id"] = "different"
    with pytest.raises(ValueError, match="case IDs"):
        paired_bootstrap_conflict(control, repair, samples=100, seed=17)


def test_formal_success_requires_clear_repair_without_clean_regression():
    result = aggregate_formal(
        _by_seed(repaired=False),
        _by_seed(repaired=True),
        SUCCESS_CONFIG,
    )

    assert result["decision"] == "POSITIVE"
    assert result["comparison"]["conflict_accuracy_delta"] == 1.0
    assert result["checks"] == {
        "all_seed_conflict_deltas_positive": True,
        "conflict_delta_at_least_10pp": True,
        "conflict_ci_lower_positive": True,
        "hint_flip_halved": True,
        "aligned_drop_within_2pp": True,
        "causal_hint_effect_reduced": True,
    }


def test_one_nonpositive_seed_forces_negative_or_inconclusive():
    control = _by_seed(repaired=False)
    repair = _by_seed(repaired=True)
    repair[44] = control[44]

    result = aggregate_formal(control, repair, SUCCESS_CONFIG)

    assert result["decision"] == "NEGATIVE / INCONCLUSIVE"
    assert result["checks"]["all_seed_conflict_deltas_positive"] is False


def test_report_writes_machine_and_interview_readable_artifacts(tmp_path):
    result = aggregate_formal(
        _by_seed(repaired=False),
        _by_seed(repaired=True),
        SUCCESS_CONFIG,
    )

    write_report(result, tmp_path)

    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))[
        "decision"
    ] == "POSITIVE"
    assert "POSITIVE" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "conflict_accuracy" in (tmp_path / "main_metrics.csv").read_text(encoding="utf-8")
    assert "seed" in (tmp_path / "per_seed.csv").read_text(encoding="utf-8")
    assert (tmp_path / "comparison.png").stat().st_size > 0
