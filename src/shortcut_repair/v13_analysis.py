"""v1.3 的格式锚定 pilot 判定和正式前后对照。"""

from __future__ import annotations

from statistics import fmean

from shortcut_repair.analysis import METRIC_NAMES
from shortcut_repair.data import DECISION_TYPES
from shortcut_repair.v12_analysis import _signature, aggregate_results, summarize

CORE_PILOT_METRICS = {
    "aligned_accuracy": ("overall", "aligned_accuracy"),
    "validity_conflict_accuracy": (
        "decision_type",
        "validity_decisive",
        "conflict_accuracy",
    ),
    "score_conflict_accuracy": ("decision_type", "score_decisive", "conflict_accuracy"),
    "score_fresh_response": (
        "decision_type",
        "score_decisive",
        "fresh_result_response_rate",
    ),
    "score_nuisance_invariance": (
        "decision_type",
        "score_decisive",
        "nuisance_invariance_rate",
    ),
}


def _value(metrics: dict, path: tuple[str, ...]) -> float:
    value = metrics
    for key in path:
        value = value[key]
    return float(value)


def select_anchor_pilot(before: dict, after: dict, rules: dict) -> dict:
    deltas = {
        name: _value(after, path) - _value(before, path)
        for name, path in CORE_PILOT_METRICS.items()
    }
    format_delta = (
        after["overall"]["greedy_exact_format_rate"]
        - before["overall"]["greedy_exact_format_rate"]
    )
    checks = {
        "aligned_retained": after["overall"]["aligned_accuracy"]
        >= rules["min_aligned_accuracy"],
        "validity_retained": after["decision_type"]["validity_decisive"][
            "conflict_accuracy"
        ]
        >= rules["min_validity_conflict_accuracy"],
        "score_conflict_retained": after["decision_type"]["score_decisive"][
            "conflict_accuracy"
        ]
        >= rules["min_score_conflict_accuracy"],
        "score_fresh_retained": after["decision_type"]["score_decisive"][
            "fresh_result_response_rate"
        ]
        >= rules["min_score_fresh_response"],
        "score_nuisance_retained": after["decision_type"]["score_decisive"][
            "nuisance_invariance_rate"
        ]
        >= rules["min_score_nuisance_invariance"],
        "exact_format": after["overall"]["greedy_exact_format_rate"]
        >= rules["min_greedy_exact_format"],
        "format_gain": format_delta >= rules["min_format_gain"],
        "no_core_regression": min(deltas.values()) >= -rules["max_core_regression"],
    }
    selected = "sft_dpo_anchor" if all(checks.values()) else None
    return {
        "selected": selected,
        "checks": checks,
        "deltas": {**deltas, "greedy_exact_format_rate": format_delta},
        "reason": (
            "格式锚定通过全部固定门槛，且没有明显损害规则能力。"
            if selected
            else "格式锚定未同时满足格式改善与能力保持；停止，不生成 test。"
        ),
    }


def _mean_model(runs: dict[int, list[dict]]) -> dict:
    per_seed = {seed: summarize(rows) for seed, rows in runs.items()}
    return {
        "overall": {
            metric: fmean(item["overall"][metric] for item in per_seed.values())
            for metric in METRIC_NAMES
        },
        "decision_type": {
            kind: {
                metric: fmean(item["decision_type"][kind][metric] for item in per_seed.values())
                for metric in METRIC_NAMES
            }
            for kind in DECISION_TYPES
        },
        "per_seed": per_seed,
    }


def aggregate_v13(records: dict[str, dict], config: dict) -> dict:
    expected = {
        "base",
        "shortcut",
        "v1_1_control",
        "v1_1_repair",
        "score_sft",
        "sft_dpo_pre_anchor",
        "selected_dpo",
    }
    if set(records) != expected:
        raise ValueError("v1.3 正式报告必须包含七组模型")
    pre_runs = records["sft_dpo_pre_anchor"]
    if set(pre_runs) != set(config["seeds"]):
        raise ValueError("锚定前 DPO 的 seeds 不完整")
    for seed in config["seeds"]:
        if _signature(pre_runs[seed]) != _signature(records["selected_dpo"][seed]):
            raise ValueError("锚定前后必须评测相同的配对 test")
    base_records = {name: runs for name, runs in records.items() if name != "sft_dpo_pre_anchor"}
    result = aggregate_results(base_records, config)
    result["version"] = "1.3"
    pre = _mean_model(pre_runs)
    ordered = {}
    for name, model in result["models"].items():
        if name == "selected_dpo":
            ordered["sft_dpo_pre_anchor"] = pre
        ordered[name] = model
    result["models"] = ordered
    selected = result["models"]["selected_dpo"]
    result["anchor_minus_pre_anchor"] = {
        "overall": {
            metric: selected["overall"][metric] - pre["overall"][metric]
            for metric in METRIC_NAMES
        },
        "decision_type": {
            kind: {
                metric: selected["decision_type"][kind][metric]
                - pre["decision_type"][kind][metric]
                for metric in METRIC_NAMES
            }
            for kind in DECISION_TYPES
        },
    }
    return result
