from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from conftest import read_jsonl, write_small_config

from shortcut_repair.data import generate_sealed_test, generate_train_dev, sha256_file
from shortcut_repair.evaluate import (
    aggregate_from_artifacts,
    build_prediction_record,
    prediction_from_scores,
    validate_test_seal,
)
from shortcut_repair.train import validate_dpo_contract


def _row(case_id: str, variant: str, gold: str, prediction: str, margin: float) -> dict:
    wrong = "B" if gold == "A" else "A"
    hint = gold if variant == "aligned" else wrong
    scores = {gold: margin / 2, wrong: -margin / 2}
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


def _predictions_from_test(rows: list[dict], repaired: bool) -> list[dict]:
    predictions = []
    for row in rows:
        gold = row["gold"]
        wrong = "B" if gold == "A" else "A"
        if row["variant"] == "aligned" or repaired:
            predictions.append(_row(row["case_id"], row["variant"], gold, gold, 2.0))
        else:
            predictions.append(_row(row["case_id"], row["variant"], gold, wrong, -2.0))
    return predictions


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prediction_uses_conditional_logprob_not_generation_format():
    row = {"case_id": "c1", "variant": "conflict", "gold": "A", "hint": "B"}

    record = build_prediction_record(row, logp_a=-0.2, logp_b=-1.7)

    assert record["prediction"] == "A"
    assert record["correct"] is True
    assert record["correct_margin"] == pytest.approx(1.5)
    assert "prompt_messages" not in record


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (-1.0, -1.0, "equal"),
        (math.nan, -1.0, "finite"),
        (-1.0, math.inf, "finite"),
    ],
)
def test_invalid_candidate_scores_are_rejected(left, right, message):
    with pytest.raises(ValueError, match=message):
        prediction_from_scores(left, right)


def test_test_seal_detects_config_or_data_tampering(tmp_path):
    config_path = write_small_config(tmp_path, test=4)
    generate_sealed_test(config_path)

    seal = validate_test_seal(config_path)
    assert seal["sealed"] is True

    test_path = tmp_path / "data/test.jsonl"
    original = test_path.read_text(encoding="utf-8")
    test_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_test_seal(config_path)
    test_path.write_text(original, encoding="utf-8")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["project"]["version"] = "changed"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        validate_test_seal(config_path)


def _create_formal_artifacts(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = Path(config["paths"]["data_dir"])
    test_rows = read_jsonl(data_dir / "test.jsonl")
    test_sha = sha256_file(data_dir / "test.jsonl")
    config_sha = sha256_file(config_path)
    for method, repaired in (("control", False), ("repair", True)):
        train_sha = sha256_file(data_dir / f"dpo_{method}.jsonl")
        predictions = _predictions_from_test(test_rows, repaired)
        for seed in config["dpo"]["seeds"]:
            run_dir = Path(config["paths"]["dpo_runs_dir"]) / method / f"seed-{seed}"
            adapter_dir = run_dir / "final_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            contract = validate_dpo_contract(
                config,
                method,
                seed,
                rows=config["dpo"]["expected_rows"],
                smoke=False,
            )
            _write_json(
                run_dir / "run_manifest.json",
                {
                    "status": "complete",
                    "method": method,
                    "data_sha256": train_sha,
                    "config_sha256": config_sha,
                    "initial_adapter_checksum": f"init-{seed}",
                    "actual_optimizer_steps": contract["optimizer_steps"],
                    "final_adapter": str(adapter_dir),
                    "contract": contract,
                },
            )
            result_dir = (
                Path(config["paths"]["results_dir"])
                / "test"
                / method
                / f"seed-{seed}"
            )
            prediction_path = result_dir / "predictions.jsonl"
            _write_jsonl(prediction_path, predictions)
            _write_json(
                result_dir / "prediction_manifest.json",
                {
                    "status": "complete",
                    "split": "test",
                    "method": method,
                    "seed": seed,
                    "adapter_path": str(adapter_dir),
                    "data_sha256": test_sha,
                    "config_sha256": config_sha,
                    "predictions_sha256": sha256_file(prediction_path),
                    "prediction_rows": len(predictions),
                },
            )


def test_aggregate_validates_manifests_and_writes_positive_report(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, test=10)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    _create_formal_artifacts(config_path)

    result = aggregate_from_artifacts(config_path, tmp_path / "reports")

    assert result["decision"] == "POSITIVE"
    assert result["provenance"]["formal_training_runs"] == 6
    assert (tmp_path / "reports/RESULTS.md").is_file()
    assert (tmp_path / "reports/comparison.png").is_file()


def test_aggregate_rejects_prediction_or_initialization_tampering(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, test=10)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    _create_formal_artifacts(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = Path(config["paths"]["results_dir"])
    runs_dir = Path(config["paths"]["dpo_runs_dir"])

    prediction_path = results_dir / "test/control/seed-42/predictions.jsonl"
    prediction_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction checksum"):
        aggregate_from_artifacts(config_path, tmp_path / "reports")

    _create_formal_artifacts(config_path)
    manifest_path = runs_dir / "repair/seed-42/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["initial_adapter_checksum"] = "different"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="initial adapter"):
        aggregate_from_artifacts(config_path, tmp_path / "reports")
