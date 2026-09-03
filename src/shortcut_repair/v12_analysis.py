"""v1.2 的 dev 选型、五项判定和按决策类型报告。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from shortcut_repair.analysis import (
    METRIC_NAMES,
    _validated_intervention_pairs,
    score_predictions,
    score_predictions_by_decision_type,
)
from shortcut_repair.data import DECISION_TYPES
from shortcut_repair.evaluate import _write_json

MODEL_LABELS = {
    "base": "Base",
    "shortcut": "Shortcut",
    "v1_1_control": "v1.1 Control",
    "v1_1_repair": "v1.1 Repair",
    "score_sft": "Score-aware SFT",
    "sft_dpo_pre_anchor": "SFT → DPO（锚定前）",
    "selected_dpo": "选定 DPO",
}
BASE_MODEL_NAMES = {
    "base",
    "shortcut",
    "v1_1_control",
    "v1_1_repair",
    "score_sft",
    "selected_dpo",
}
CHECK_LABELS = {
    "score_fresh_response": "score fresh response ≥ 0.70，且每 seed 高于同 test 的 v1.1 Repair",
    "score_conflict_accuracy": "score conflict accuracy ≥ 0.70",
    "capability_retention": "overall aligned ≥ 0.90 且 validity conflict ≥ 0.95",
    "score_nuisance_invariance": "score nuisance invariance ≥ 0.95",
    "exact_format": "greedy exact-format rate ≥ 0.98",
}


def summarize(rows: list[dict]) -> dict:
    return {
        "overall": score_predictions(rows),
        "decision_type": score_predictions_by_decision_type(rows),
    }


def _at_least(value: float, threshold: float) -> bool:
    # 只容忍浮点舍入误差，远小于单个 case 或训练 seed 对比例的影响。
    return value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)


def select_pilot(candidates: dict[str, dict], rules: dict) -> dict:
    """SFT 只作为基线；正式路径必须是达到保留条件的 DPO。"""
    eligible = {}
    for name, candidate in candidates.items():
        metrics = candidate["metrics"]
        eligible[name] = (
            _at_least(metrics["overall"]["aligned_accuracy"], rules["min_aligned_accuracy"])
            and _at_least(
                metrics["overall"]["greedy_exact_format_rate"], rules["min_greedy_exact_format"]
            )
            and _at_least(
                metrics["decision_type"]["validity_decisive"]["conflict_accuracy"],
                rules["min_validity_conflict_accuracy"],
            )
        )
    remaining = [n for n in candidates if eligible[n] and candidates[n]["method"] != "score_sft"]
    for metric in ("fresh_result_response_rate", "nuisance_invariance_rate"):
        if remaining:
            values = {
                n: candidates[n]["metrics"]["decision_type"]["score_decisive"][metric]
                for n in remaining
            }
            best = max(values.values())
            remaining = [n for n in remaining if best - values[n] < rules["tie_tolerance"]]
    selected = min(remaining, key=lambda n: (candidates[n]["stages"], n)) if remaining else None
    return {
        "selected": selected,
        "eligible": eligible,
        "all_failed_retention": not any(eligible.values()),
        "reason": (
            "先检查 aligned/validity/格式保持，再比较 score fresh response、nuisance 和阶段数。"
            if selected
            else "没有满足能力保持条件的 DPO 路径；不打开 test，也不强行选择 SFT。"
        ),
    }


def _signature(rows: list[dict]) -> list[tuple]:
    keys = (
        "case_id",
        "split",
        "decision_type",
        "intervention",
        "intervention_variant",
        "gold",
        "hint",
    )
    return sorted(tuple(row[key] for key in keys) for row in rows)


def _fresh_correct(rows: list[dict]) -> dict[str, float]:
    pairs = _validated_intervention_pairs(rows)["fresh_flip"]
    return {
        a["case_id"]: float(a["prediction"] == a["gold"] and b["prediction"] == b["gold"])
        for a, b in pairs
        if a["decision_type"] == "score_decisive"
    }


def _bootstrap_fresh(old: dict, new: dict, evaluation: dict) -> dict:
    differences = []
    for seed in sorted(old):
        before, after = _fresh_correct(old[seed]), _fresh_correct(new[seed])
        if before.keys() != after.keys():
            raise ValueError("配对 bootstrap 要求相同 case")
        differences.append([after[key] - before[key] for key in sorted(before)])
    case_delta = np.mean(differences, axis=0)
    rng = np.random.default_rng(evaluation["bootstrap_seed"])
    samples = evaluation["bootstrap_samples"]
    if samples <= 0:
        raise ValueError("bootstrap_samples 必须为正数")
    indices = rng.integers(0, len(case_delta), size=(samples, len(case_delta)))
    distribution = case_delta[indices].mean(axis=1)
    return {
        "metric": "score_decisive.fresh_result_response_rate",
        "comparison": "selected_dpo - v1_1_repair",
        "mean_delta": float(case_delta.mean()),
        "ci95": [float(v) for v in np.quantile(distribution, [0.025, 0.975])],
        "samples": samples,
        "seed": evaluation["bootstrap_seed"],
        "case_count": len(case_delta),
        "note": (
            "先对三个训练 seed 的配对差值取均值，再重采样 case；CI 不覆盖训练 seed 总体不确定性。"
        ),
    }


def aggregate_results(records: dict[str, dict[Any, list[dict]]], config: dict) -> dict:
    if set(records) != BASE_MODEL_NAMES:
        raise ValueError("正式报告必须包含全部六组模型")
    reference_signature = None
    models = {}
    for name, runs in records.items():
        expected = {"single"} if name in {"base", "shortcut"} else set(config["seeds"])
        if set(runs) != expected:
            raise ValueError(f"{name} 的 seeds 不完整或包含未注册 seed")
        per_seed = {}
        for seed, rows in runs.items():
            signature = _signature(rows)
            if reference_signature is None:
                reference_signature = signature
            if signature != reference_signature:
                raise ValueError("全部模型和 seed 必须评测相同的配对 test")
            per_seed[seed] = summarize(rows)
        models[name] = {
            "overall": {
                metric: float(np.mean([m["overall"][metric] for m in per_seed.values()]))
                for metric in METRIC_NAMES
            },
            "decision_type": {
                kind: {
                    metric: float(
                        np.mean([m["decision_type"][kind][metric] for m in per_seed.values()])
                    )
                    for metric in METRIC_NAMES
                }
                for kind in DECISION_TYPES
            },
            "per_seed": per_seed,
        }
    new, old, sft = (models[n] for n in ("selected_dpo", "v1_1_repair", "score_sft"))
    score, validity = (new["decision_type"][kind] for kind in DECISION_TYPES)
    thresholds = config["evaluation"]["success"]
    deltas = {
        seed: new["per_seed"][seed]["decision_type"]["score_decisive"]["fresh_result_response_rate"]
        - old["per_seed"][seed]["decision_type"]["score_decisive"]["fresh_result_response_rate"]
        for seed in config["seeds"]
    }
    checks = {
        "score_fresh_response": _at_least(
            score["fresh_result_response_rate"], thresholds["min_score_fresh_response"]
        )
        and all(v > 0 for v in deltas.values()),
        "score_conflict_accuracy": _at_least(
            score["conflict_accuracy"], thresholds["min_score_conflict_accuracy"]
        ),
        "capability_retention": _at_least(
            new["overall"]["aligned_accuracy"], thresholds["min_aligned_accuracy"]
        )
        and _at_least(validity["conflict_accuracy"], thresholds["min_validity_conflict_accuracy"]),
        "score_nuisance_invariance": _at_least(
            score["nuisance_invariance_rate"], thresholds["min_score_nuisance_invariance"]
        ),
        "exact_format": _at_least(
            new["overall"]["greedy_exact_format_rate"], thresholds["min_greedy_exact_format"]
        ),
    }
    return {
        "version": config["project"]["version"],
        "decision": "POSITIVE" if all(checks.values()) else "NEGATIVE / INCONCLUSIVE",
        "checks": checks,
        "success_thresholds": thresholds,
        "models": models,
        "per_seed_score_fresh_delta": deltas,
        "dpo_minus_sft": {
            kind: {
                metric: new["decision_type"][kind][metric] - sft["decision_type"][kind][metric]
                for metric in METRIC_NAMES
            }
            for kind in DECISION_TYPES
        },
        "bootstrap": _bootstrap_fresh(
            records["v1_1_repair"], records["selected_dpo"], config["evaluation"]
        ),
    }


def write_report(result: dict, output_dir: Path | str) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "results.json", result)
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "decision_type", "metric", "value"])
        for name, model in result["models"].items():
            for seed, metrics in {"mean": model, **model["per_seed"]}.items():
                for kind, values in {
                    "overall": metrics["overall"],
                    **metrics["decision_type"],
                }.items():
                    writer.writerows(
                        [name, seed, kind, key, value] for key, value in values.items()
                    )
    lines = [
        f"# ShortcutRepair-DPO v{result['version']} 正式结果",
        "",
        f"结论：`{result['decision']}`",
        "",
    ]
    if "provenance" in result:
        selected = result["provenance"]["selected"]
        if selected["method"] == "sft_dpo_anchor":
            lines += [
                "正式路径：`sft_dpo_anchor`。先执行 Score-aware SFT → DPO，再在同一 "
                "DPO adapter 上进行低学习率格式锚定 SFT。",
                "",
            ]
        else:
            reference = (
                "同 seed 的 SFT 中间策略" if selected["method"] == "sft_dpo" else "Shortcut"
            )
            lines += [
                f"正式路径：`{selected['method']}`；pilot 候选：`{selected['candidate']}`。"
                f"DPO 参照策略为{reference}，学习率 {selected['params']['learning_rate']:g}，"
                f"beta={selected['params']['beta']:g}。",
                "",
            ]
    lines += [
        "## 同一 sealed test 上的比较",
        "",
        "| 模型 | overall aligned | score conflict | score fresh response | "
        "validity conflict | score nuisance | 格式 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        score = model["decision_type"]["score_decisive"]
        validity = model["decision_type"]["validity_decisive"]
        values = (
            model["overall"]["aligned_accuracy"],
            score["conflict_accuracy"],
            score["fresh_result_response_rate"],
            validity["conflict_accuracy"],
            score["nuisance_invariance_rate"],
            model["overall"]["greedy_exact_format_rate"],
        )
        lines.append(f"| {MODEL_LABELS[name]} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 五项预注册检查", ""]
    lines += [
        f"- {'通过' if passed else '未通过'}：{CHECK_LABELS[key]}。"
        for key, passed in result["checks"].items()
    ]
    bootstrap = result["bootstrap"]
    if "anchor_minus_pre_anchor" in result:
        anchor_delta = result["anchor_minus_pre_anchor"]
        score_delta = anchor_delta["decision_type"]["score_decisive"]
        validity_delta = anchor_delta["decision_type"]["validity_decisive"]
        lines += [
            "",
            "## 格式锚定前后差值",
            "",
            "锚定后 − 锚定前："
            f"exact-format {anchor_delta['overall']['greedy_exact_format_rate']:+.4f}，"
            f"overall aligned {anchor_delta['overall']['aligned_accuracy']:+.4f}，"
            f"score conflict {score_delta['conflict_accuracy']:+.4f}，"
            f"score fresh response {score_delta['fresh_result_response_rate']:+.4f}，"
            f"score nuisance {score_delta['nuisance_invariance_rate']:+.4f}，"
            f"validity conflict {validity_delta['conflict_accuracy']:+.4f}。",
            "",
            "该差值用于判断格式锚定是否损害规则能力，不改变五项正式成功标准。",
        ]
    lines += [
        "",
        "## 不确定性和 SFT 对照",
        "",
        f"score fresh response 相对同 test 的 v1.1 Repair：Δ={bootstrap['mean_delta']:.4f}，"
        f"95% 配对 case-bootstrap CI={bootstrap['ci95']}。{bootstrap['note']}",
        "",
        f"逐 seed 差值：`{result['per_seed_score_fresh_delta']}`。",
        "",
        "选定 DPO 相对 Score-aware SFT 的 score fresh response 差值为 "
        f"{result['dpo_minus_sft']['score_decisive']['fresh_result_response_rate']:.4f}。"
        "DPO 达标不等于优于 SFT；差值为零或负数时不能声称 DPO 有额外收益。",
        "",
        "完整的七项主指标、附加诊断、两种决策切片和逐 seed 数值见 metrics.csv/results.json。",
        "",
        "## 计算成本与边界",
        "",
        "训练分布为 75% score / 25% validity，dev/test 为 50/50。"
        "SFT 有明显分差 warm-up，DPO 只有普通分差；不是同数据、同计算量的纯损失函数消融。",
        "SFT → DPO 包含两个训练阶段；v1.3 的格式锚定路径包含三个阶段，"
        "不能包装为与单阶段 DPO 或 SFT 等预算。"
        "这是受控合成二选一实验，不外推为生产工具调用或通用数值推理能力。",
        "",
    ]
    if "costs" in result:
        lines += [
            "运行明细见 costs.csv。SFT 中间结果被复用，不重复计入总训练开销。",
            "",
            "| 范围 | 阶段数 | 已记录阶段秒数 | 峰值显存 GiB |",
            "|---|---:|---:|---:|",
        ]
        for phase in ("pilot", "formal"):
            costs = [c for c in result["costs"] if c["phase"] == phase]
            seconds = sum(c["training_runtime_seconds"] for c in costs)
            peak = max((c["peak_gpu_memory_bytes"] for c in costs), default=0) / 2**30
            prefix = "≥ " if any(c.get("runtime_is_lower_bound") for c in costs) else ""
            lines.append(f"| {phase} | {len(costs)} | {prefix}{seconds:.1f} | {peak:.2f} |")
        lines += [
            "",
            "耗时包含加载、训练、保存和已捕获的失败尝试；强杀或断电造成的未记录耗时"
            "标为下界，不能当作精确总成本。峰值显存为已记录尝试的最大值。",
        ]
        with (root / "costs.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "phase",
                "candidate",
                "seed",
                "rows",
                "actual_optimizer_steps",
                "training_runtime_seconds",
                "peak_gpu_memory_bytes",
                "learning_rate",
                "attempts",
                "runtime_is_lower_bound",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["costs"])
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["models"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric, threshold in zip(
        axes,
        ("fresh_result_response_rate", "conflict_accuracy", "nuisance_invariance_rate"),
        (0.70, 0.70, 0.95),
        strict=True,
    ):
        axis.bar(
            range(len(names)),
            [result["models"][n]["decision_type"]["score_decisive"][metric] for n in names],
            color="#387d9e",
        )
        axis.axhline(threshold, color="#c15b3e", linestyle="--")
        axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(metric, fontsize=10)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
