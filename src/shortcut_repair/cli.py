"""Command-line interface for the staged ShortcutRepair-DPO experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from shortcut_repair.analysis import classify_mechanism_gate, score_predictions
from shortcut_repair.data import (
    canonical_json,
    generate_sealed_test,
    generate_train_dev,
    load_config,
)
from shortcut_repair.evaluate import (
    DEFAULT_EVALUATION_AMENDMENT,
    aggregate_from_artifacts,
    evaluate_checkpoint,
)
from shortcut_repair.train import train_dpo, train_sft_baseline, train_shortcut


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _default_gate_path(config: dict) -> Path:
    return Path(config["paths"]["results_dir"]) / "dev/shortcut/mechanism_gate.json"


def _require_passing_gate(config: dict) -> dict:
    gate_path = _default_gate_path(config)
    if not gate_path.is_file():
        raise RuntimeError(f"Mechanism gate artifact is missing: {gate_path}")
    result = json.loads(gate_path.read_text(encoding="utf-8"))
    if result.get("decision") != "pass":
        raise RuntimeError("Mechanism gate did not pass; test generation is blocked")
    return result


def _run_generate(args: argparse.Namespace) -> None:
    if args.stage == "train-dev":
        result = generate_train_dev(args.config)
    else:
        config = load_config(args.config)
        _require_passing_gate(config)
        result = generate_sealed_test(args.config)
    print(canonical_json(result))


def _run_gate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    base_metrics = score_predictions(_read_jsonl(args.base_predictions))
    metrics = score_predictions(_read_jsonl(args.predictions))
    result = {
        "metrics": metrics,
        "base_metrics": base_metrics,
        "shortcut_minus_base": {
            "hint_flip_rate": metrics["hint_flip_rate"] - base_metrics["hint_flip_rate"],
            "causal_hint_effect": metrics["causal_hint_effect"]
            - base_metrics["causal_hint_effect"],
        },
        **classify_mechanism_gate(
            metrics, config["evaluation"]["mechanism_gate"]
        ),
    }
    output = args.output or _default_gate_path(config)
    _write_json(output, result)
    print(canonical_json(result))
    if result["decision"] != "pass":
        raise SystemExit(2)


def _run_aggregate(args: argparse.Namespace) -> None:
    result = aggregate_from_artifacts(
        args.config, args.output_dir, args.evaluation_amendment
    )
    print(
        canonical_json(
            {
                "status": "complete",
                "decision": result["decision"],
                "output_dir": str(
                    args.output_dir or load_config(args.config)["paths"]["reports_dir"]
                ),
            }
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate deterministic data")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--stage", choices=("train-dev", "test"), required=True)
    generate.set_defaults(func=_run_generate)

    shortcut = subparsers.add_parser("train-shortcut", help="Train and merge shortcut SFT")
    shortcut.add_argument("--config", type=Path, required=True)
    shortcut.add_argument("--model-path")
    shortcut.add_argument("--output-dir", type=Path)
    shortcut.add_argument("--resume", action="store_true")
    shortcut.add_argument("--dry-run", action="store_true")
    shortcut.set_defaults(func=train_shortcut)

    dpo = subparsers.add_parser("train-dpo", help="Train one matched DPO adapter")
    dpo.add_argument("--config", type=Path, required=True)
    dpo.add_argument("--method", choices=("control", "repair"), required=True)
    dpo.add_argument("--seed", type=int, required=True)
    dpo.add_argument("--model-path")
    dpo.add_argument("--output-dir", type=Path)
    dpo.add_argument("--smoke", action="store_true")
    dpo.add_argument("--resume", action="store_true")
    dpo.add_argument("--dry-run", action="store_true")
    dpo.set_defaults(func=train_dpo)

    sft_baseline = subparsers.add_parser(
        "train-sft-baseline", help="Train one matched Counterfactual SFT adapter"
    )
    sft_baseline.add_argument("--config", type=Path, required=True)
    sft_baseline.add_argument("--seed", type=int, required=True)
    sft_baseline.add_argument("--model-path")
    sft_baseline.add_argument("--output-dir", type=Path)
    sft_baseline.add_argument("--smoke", action="store_true")
    sft_baseline.add_argument("--resume", action="store_true")
    sft_baseline.add_argument("--dry-run", action="store_true")
    sft_baseline.set_defaults(func=train_sft_baseline)

    evaluate = subparsers.add_parser("evaluate", help="Score A/B conditional probabilities")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument(
        "--evaluation-amendment",
        type=Path,
        default=DEFAULT_EVALUATION_AMENDMENT,
    )
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument(
        "--model",
        choices=("base", "shortcut", "adapter", "sft-baseline"),
        required=True,
    )
    evaluate.add_argument("--method", choices=("control", "repair"))
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--model-path")
    evaluate.add_argument("--adapter-path", type=Path)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(func=evaluate_checkpoint)

    gate = subparsers.add_parser("gate", help="Apply the pre-registered dev mechanism gate")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--base-predictions", type=Path, required=True)
    gate.add_argument("--predictions", type=Path, required=True)
    gate.add_argument("--output", type=Path)
    gate.set_defaults(func=_run_gate)

    aggregate = subparsers.add_parser("aggregate", help="Audit and aggregate nine formal runs")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument(
        "--evaluation-amendment",
        type=Path,
        default=DEFAULT_EVALUATION_AMENDMENT,
    )
    aggregate.add_argument("--output-dir", type=Path)
    aggregate.set_defaults(func=_run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
