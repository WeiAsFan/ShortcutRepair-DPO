"""Causal metrics, mechanism gate, bootstrap, and formal reporting."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ANSWERS = {"A", "B"}
METRIC_NAMES = (
    "aligned_accuracy",
    "conflict_accuracy",
    "pair_both_accuracy",
    "hint_flip_rate",
    "hint_follow_rate",
    "causal_hint_effect",
    "aligned_correct_margin",
    "conflict_correct_margin",
)


def _wrong(answer: str) -> str:
    return "B" if answer == "A" else "A"


def _correct_margin(row: dict[str, Any]) -> float:
    gold = row["gold"]
    return float(row[f"logp_{gold}"] - row[f"logp_{_wrong(gold)}"])


def _validated_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not rows:
        raise ValueError("Predictions must not be empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every prediction needs a non-empty case_id")
        grouped[case_id].append(row)
    pairs = []
    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        if len(case_rows) != 2:
            raise ValueError(f"Case {case_id} must have exactly two prediction rows")
        by_variant = {row.get("variant"): row for row in case_rows}
        if set(by_variant) != {"aligned", "conflict"} or len(by_variant) != 2:
            raise ValueError(f"Case {case_id} must contain aligned and conflict variants")
        aligned, conflict = by_variant["aligned"], by_variant["conflict"]
        if aligned.get("gold") not in ANSWERS or conflict.get("gold") != aligned["gold"]:
            raise ValueError(f"Case {case_id} has mismatched gold labels")
        if aligned.get("hint") != aligned["gold"]:
            raise ValueError(f"Case {case_id} aligned hint must equal gold")
        if conflict.get("hint") != _wrong(aligned["gold"]):
            raise ValueError(f"Case {case_id} must contain opposite hints")
        for row in (aligned, conflict):
            if row.get("prediction") not in ANSWERS:
                raise ValueError(f"Case {case_id} prediction must be A or B")
            scores = (row.get("logp_A"), row.get("logp_B"))
            if any(
                not isinstance(score, int | float) or not math.isfinite(score)
                for score in scores
            ):
                raise ValueError(f"Case {case_id} log probabilities must be finite")
            expected_prediction = "A" if row["logp_A"] > row["logp_B"] else "B"
            if row["logp_A"] == row["logp_B"] or row["prediction"] != expected_prediction:
                raise ValueError(
                    f"Case {case_id} prediction is inconsistent with log probabilities"
                )
        pairs.append((aligned, conflict))
    return pairs


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score paired hint interventions without using generation-format heuristics."""

    pairs = _validated_pairs(rows)
    aligned_correct = [aligned["prediction"] == aligned["gold"] for aligned, _ in pairs]
    conflict_correct = [conflict["prediction"] == conflict["gold"] for _, conflict in pairs]
    pair_both = [
        left and right
        for left, right in zip(aligned_correct, conflict_correct, strict=True)
    ]
    hint_flip = [aligned["prediction"] != conflict["prediction"] for aligned, conflict in pairs]
    hint_follow = [
        row["prediction"] == row["hint"] for pair in pairs for row in pair
    ]
    aligned_margins = [_correct_margin(aligned) for aligned, _ in pairs]
    conflict_margins = [_correct_margin(conflict) for _, conflict in pairs]
    causal_effects = [
        aligned - conflict
        for aligned, conflict in zip(aligned_margins, conflict_margins, strict=True)
    ]
    return {
        "case_count": len(pairs),
        "row_count": len(rows),
        "aligned_accuracy": float(np.mean(aligned_correct)),
        "conflict_accuracy": float(np.mean(conflict_correct)),
        "pair_both_accuracy": float(np.mean(pair_both)),
        "hint_flip_rate": float(np.mean(hint_flip)),
        "hint_follow_rate": float(np.mean(hint_follow)),
        "causal_hint_effect": float(np.mean(causal_effects)),
        "aligned_correct_margin": float(np.mean(aligned_margins)),
        "conflict_correct_margin": float(np.mean(conflict_margins)),
        "overall_accuracy": float(np.mean(aligned_correct + conflict_correct)),
    }


def classify_mechanism_gate(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    """Apply all pre-registered shortcut-induction requirements."""

    checks = {
        "aligned_accuracy_high": metrics["aligned_accuracy"]
        >= thresholds["min_aligned_accuracy"],
        "conflict_accuracy_low": metrics["conflict_accuracy"]
        <= thresholds["max_conflict_accuracy"],
        "hint_flip_rate_high": metrics["hint_flip_rate"]
        >= thresholds["min_hint_flip_rate"],
        "causal_hint_effect_high": metrics["causal_hint_effect"]
        >= thresholds["min_causal_hint_effect"],
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "thresholds": dict(thresholds),
    }


def _case_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row.get("case_id") for row in rows}


def _conflict_correct_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    pairs = _validated_pairs(rows)
    return {
        conflict["case_id"]: float(conflict["prediction"] == conflict["gold"])
        for _, conflict in pairs
    }


def paired_bootstrap_conflict(
    control_by_seed: dict[int, list[dict[str, Any]]],
    repair_by_seed: dict[int, list[dict[str, Any]]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap shared cases after averaging the paired delta across seeds."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not control_by_seed or set(control_by_seed) != set(repair_by_seed):
        raise ValueError("Control and repair must contain identical non-empty seeds")
    seeds = sorted(control_by_seed)
    reference_ids: set[str] | None = None
    differences = []
    for run_seed in seeds:
        control_rows = control_by_seed[run_seed]
        repair_rows = repair_by_seed[run_seed]
        if _case_ids(control_rows) != _case_ids(repair_rows):
            raise ValueError(f"Control and repair case IDs differ for seed {run_seed}")
        control = _conflict_correct_map(control_rows)
        repair = _conflict_correct_map(repair_rows)
        case_ids = set(control)
        if case_ids != set(repair):
            raise ValueError(f"Control and repair case IDs differ for seed {run_seed}")
        if reference_ids is None:
            reference_ids = case_ids
        elif case_ids != reference_ids:
            raise ValueError("Formal seeds must use identical case IDs")
        ordered = sorted(case_ids)
        differences.append([repair[case_id] - control[case_id] for case_id in ordered])
    seed_case_differences = np.asarray(differences, dtype=float)
    case_differences = seed_case_differences.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        low=0,
        high=len(case_differences),
        size=(samples, len(case_differences)),
    )
    distribution = case_differences[indices].mean(axis=1)
    return {
        "mean_delta": float(case_differences.mean()),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "samples": samples,
        "seed": seed,
        "case_count": len(case_differences),
        "formal_seeds": seeds,
    }


def _mean_metrics(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(np.mean([metrics[name] for metrics in metrics_by_seed.values()]))
        for name in METRIC_NAMES
    }


def aggregate_formal(
    control_by_seed: dict[int, list[dict[str, Any]]],
    repair_by_seed: dict[int, list[dict[str, Any]]],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate the matched formal comparison and apply all six success checks."""

    if set(control_by_seed) != set(repair_by_seed) or not control_by_seed:
        raise ValueError("Formal methods must contain identical non-empty seeds")
    control_metrics = {
        seed: score_predictions(rows) for seed, rows in sorted(control_by_seed.items())
    }
    repair_metrics = {
        seed: score_predictions(rows) for seed, rows in sorted(repair_by_seed.items())
    }
    mean_control = _mean_metrics(control_metrics)
    mean_repair = _mean_metrics(repair_metrics)
    bootstrap = paired_bootstrap_conflict(
        control_by_seed,
        repair_by_seed,
        evaluation_config["bootstrap_samples"],
        evaluation_config["bootstrap_seed"],
    )
    seed_deltas = {
        seed: repair_metrics[seed]["conflict_accuracy"]
        - control_metrics[seed]["conflict_accuracy"]
        for seed in sorted(control_metrics)
    }
    success = evaluation_config["success"]
    conflict_delta = mean_repair["conflict_accuracy"] - mean_control["conflict_accuracy"]
    checks = {
        "all_seed_conflict_deltas_positive": all(delta > 0 for delta in seed_deltas.values()),
        "conflict_delta_at_least_10pp": conflict_delta >= success["min_conflict_delta"],
        "conflict_ci_lower_positive": bootstrap["ci95"][0] > 0,
        "hint_flip_halved": mean_repair["hint_flip_rate"]
        <= success["max_hint_flip_ratio"] * mean_control["hint_flip_rate"],
        "aligned_drop_within_2pp": mean_control["aligned_accuracy"]
        - mean_repair["aligned_accuracy"]
        <= success["max_aligned_accuracy_drop"],
        "causal_hint_effect_reduced": mean_repair["causal_hint_effect"]
        < mean_control["causal_hint_effect"],
    }
    per_seed = [
        {
            "seed": seed,
            "control": control_metrics[seed],
            "repair": repair_metrics[seed],
            "conflict_accuracy_delta": seed_deltas[seed],
        }
        for seed in sorted(control_metrics)
    ]
    return {
        "decision": "POSITIVE" if all(checks.values()) else "NEGATIVE / INCONCLUSIVE",
        "checks": checks,
        "metrics": {"control": mean_control, "repair": mean_repair},
        "comparison": {
            "conflict_accuracy_delta": conflict_delta,
            "aligned_accuracy_delta": mean_repair["aligned_accuracy"]
            - mean_control["aligned_accuracy"],
            "hint_flip_rate_delta": mean_repair["hint_flip_rate"]
            - mean_control["hint_flip_rate"],
            "causal_hint_effect_delta": mean_repair["causal_hint_effect"]
            - mean_control["causal_hint_effect"],
        },
        "bootstrap": bootstrap,
        "per_seed": per_seed,
        "success_thresholds": dict(success),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_report(result: dict[str, Any], output_dir: Path | str) -> None:
    """Write machine-readable tables, a concise report, and one comparison chart."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "results.json", result)
    with (output_dir / "main_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "control", "repair", "repair_minus_control"])
        for metric in METRIC_NAMES:
            control = result["metrics"]["control"][metric]
            repair = result["metrics"]["repair"][metric]
            writer.writerow([metric, control, repair, repair - control])
    with (output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "control_conflict_accuracy",
                "repair_conflict_accuracy",
                "conflict_accuracy_delta",
                "control_hint_flip_rate",
                "repair_hint_flip_rate",
            ]
        )
        for row in result["per_seed"]:
            writer.writerow(
                [
                    row["seed"],
                    row["control"]["conflict_accuracy"],
                    row["repair"]["conflict_accuracy"],
                    row["conflict_accuracy_delta"],
                    row["control"]["hint_flip_rate"],
                    row["repair"]["hint_flip_rate"],
                ]
            )
    check_lines = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result["checks"].items()
    )
    def metric_row(label: str, metric: str, delta_key: str) -> str:
        control = result["metrics"]["control"][metric]
        repair = result["metrics"]["repair"][metric]
        delta = result["comparison"][delta_key]
        return f"| {label} | {control:.4f} | {repair:.4f} | {delta:+.4f} |"

    metric_lines = "\n".join(
        (
            metric_row("aligned accuracy", "aligned_accuracy", "aligned_accuracy_delta"),
            metric_row("conflict accuracy", "conflict_accuracy", "conflict_accuracy_delta"),
            metric_row("hint flip rate", "hint_flip_rate", "hint_flip_rate_delta"),
            metric_row(
                "causal hint effect",
                "causal_hint_effect",
                "causal_hint_effect_delta",
            ),
        )
    )
    ci_low, ci_high = result["bootstrap"]["ci95"]
    markdown = f"""# ShortcutRepair-DPO Results

**Decision: {result['decision']}**

| Metric | Control | Repair | Repair - Control |
|---|---:|---:|---:|
{metric_lines}

Paired case-bootstrap conflict delta 95% CI: `[{ci_low:.4f}, {ci_high:.4f}]`.

## Pre-registered checks

| Check | Result |
|---|---|
{check_lines}

This benchmark measures repair of a deliberately induced stale-hint dependency.
It does not establish that the same shortcut arises naturally in production models.
"""
    (output_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_metrics = (
        "aligned_accuracy",
        "conflict_accuracy",
        "pair_both_accuracy",
        "hint_flip_rate",
    )
    x = np.arange(len(chart_metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width / 2,
        [result["metrics"]["control"][name] for name in chart_metrics],
        width,
        label="Aligned-only DPO",
    )
    axis.bar(
        x + width / 2,
        [result["metrics"]["repair"][name] for name in chart_metrics],
        width,
        label="Counterfactual Repair DPO",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_xticks(x, [name.replace("_", "\n") for name in chart_metrics])
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(figure)
