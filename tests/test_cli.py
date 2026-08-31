from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import read_jsonl, write_small_config

from shortcut_repair.cli import _build_parser, main
from shortcut_repair.evaluate import build_prediction_record


def _shortcut_predictions(dev_rows: list[dict], follows_hint: bool) -> list[dict]:
    predictions = []
    for row in dev_rows:
        gold = row["gold"]
        wrong = "B" if gold == "A" else "A"
        prediction = row["hint"] if follows_hint else gold
        margin = 2.0 if prediction == gold else -2.0
        scores = {gold: margin / 2, wrong: -margin / 2}
        predictions.append(build_prediction_record(row, scores["A"], scores["B"]))
    return predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_parser_exposes_all_staged_commands():
    parser = _build_parser()

    generate = parser.parse_args(
        ["generate", "--config", "c.yaml", "--stage", "train-dev"]
    )
    shortcut = parser.parse_args(
        ["train-shortcut", "--config", "c.yaml", "--dry-run"]
    )
    assert generate.command == "generate"
    assert shortcut.command == "train-shortcut"
    assert parser.parse_args(
        [
            "train-dpo",
            "--config",
            "c.yaml",
            "--method",
            "repair",
            "--seed",
            "42",
            "--dry-run",
        ]
    ).method == "repair"
    assert parser.parse_args(
        [
            "train-sft-baseline",
            "--config",
            "c.yaml",
            "--seed",
            "42",
            "--dry-run",
        ]
    ).command == "train-sft-baseline"
    evaluation = parser.parse_args(
        [
            "evaluate",
            "--config",
            "c.yaml",
            "--split",
            "test",
            "--model",
            "adapter",
            "--method",
            "control",
            "--seed",
            "42",
            "--output-dir",
            "out",
        ]
    )
    assert evaluation.command == "evaluate"
    assert evaluation.evaluation_amendment == Path("configs/evaluation_amendment.yaml")
    gate = parser.parse_args(
        [
            "gate",
            "--config",
            "c.yaml",
            "--base-predictions",
            "base.jsonl",
            "--predictions",
            "p.jsonl",
        ]
    )
    assert gate.command == "gate"
    aggregate = parser.parse_args(["aggregate", "--config", "c.yaml"])
    assert aggregate.command == "aggregate"
    assert aggregate.evaluation_amendment == Path("configs/evaluation_amendment.yaml")


def test_gate_failure_blocks_test_generation_and_pass_allows_it(tmp_path):
    config_path = write_small_config(tmp_path, dev=10, test=10)
    assert main(["generate", "--config", str(config_path), "--stage", "train-dev"]) == 0
    dev_rows = read_jsonl(tmp_path / "data/dev.jsonl")
    base_prediction_path = tmp_path / "base_predictions.jsonl"
    prediction_path = tmp_path / "shortcut_predictions.jsonl"
    _write_jsonl(base_prediction_path, _shortcut_predictions(dev_rows, follows_hint=False))

    _write_jsonl(prediction_path, _shortcut_predictions(dev_rows, follows_hint=False))
    with pytest.raises(SystemExit) as error:
        main(
            [
                "gate",
                "--config",
                str(config_path),
                "--base-predictions",
                str(base_prediction_path),
                "--predictions",
                str(prediction_path),
            ]
        )
    assert error.value.code == 2
    with pytest.raises(RuntimeError, match="gate"):
        main(["generate", "--config", str(config_path), "--stage", "test"])

    _write_jsonl(prediction_path, _shortcut_predictions(dev_rows, follows_hint=True))
    assert main(
        [
            "gate",
            "--config",
            str(config_path),
            "--base-predictions",
            str(base_prediction_path),
            "--predictions",
            str(prediction_path),
        ]
    ) == 0
    assert main(["generate", "--config", str(config_path), "--stage", "test"]) == 0
    assert (tmp_path / "data/manifest_test.json").is_file()


def test_cli_training_dry_runs_are_cpu_only(tmp_path, capsys):
    config_path = write_small_config(tmp_path, induction=4, dpo=32)
    main(["generate", "--config", str(config_path), "--stage", "train-dev"])
    capsys.readouterr()

    assert main(["train-shortcut", "--config", str(config_path), "--dry-run"]) == 0
    shortcut = json.loads(capsys.readouterr().out)
    assert shortcut["contract"]["stage"] == "shortcut_sft"

    for method in ("control", "repair"):
        assert main(
            [
                "train-dpo",
                "--config",
                str(config_path),
                "--method",
                method,
                "--seed",
                "42",
                "--dry-run",
            ]
        ) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["contract"]["method"] == method

    assert main(
        [
            "train-sft-baseline",
            "--config",
            str(config_path),
            "--seed",
            "42",
            "--dry-run",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract"]["stage"] == "counterfactual_sft"
