from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from conftest import read_jsonl, write_small_config

from shortcut_repair.data import generate_sealed_test, generate_train_dev, sha256_file
from shortcut_repair.evaluate import (
    _fp32_log_softmax,
    aggregate_from_artifacts,
    build_prediction_record,
    prediction_from_scores,
    validate_test_seal,
)
from shortcut_repair.train import (
    _git_sha,
    _trainer_budget,
    sha256_model_weights,
    validate_counterfactual_sft_contract,
    validate_dpo_contract,
    validate_sft_contract,
)


def _predictions_from_test(rows: list[dict], repaired: bool) -> list[dict]:
    predictions = []
    for row in rows:
        gold = row["gold"]
        wrong = "B" if gold == "A" else "A"
        if row["intervention"] == "fresh_flip":
            prediction = gold if repaired else row["hint"]
        elif row["intervention"] == "nuisance_flip":
            prediction = gold if repaired else row["hint"]
        elif row["variant"] == "aligned" or repaired:
            prediction = gold
        else:
            prediction = wrong
        margin = 2.0 if prediction == gold else -2.0
        scores = {gold: margin / 2, wrong: -margin / 2}
        predictions.append(
            build_prediction_record(
                row,
                scores["A"],
                scores["B"],
                generated_text=prediction,
            )
        )
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


def test_prediction_record_keeps_intervention_and_generation_audit_fields():
    row = {
        "case_id": "c1",
        "decision_type": "validity_decisive",
        "intervention": "fresh_flip",
        "intervention_variant": "flipped",
        "variant": "conflict",
        "gold": "B",
        "hint": "A",
    }

    record = build_prediction_record(row, logp_a=-2.0, logp_b=-0.1, generated_text="B")

    assert record["decision_type"] == "validity_decisive"
    assert record["intervention"] == "fresh_flip"
    assert record["generation_prediction"] == "B"
    assert record["generation_exact_format"] is True
    assert record["generation_correct"] is True


def test_conditional_log_softmax_is_computed_in_fp32():
    calls = []

    class Logits:
        def float(self):
            calls.append("float")
            return "fp32-logits"

    class TorchModule:
        @staticmethod
        def log_softmax(tensor, dim):
            calls.append((tensor, dim))
            return "log-probabilities"

    result = _fp32_log_softmax(Logits(), TorchModule())

    assert result == "log-probabilities"
    assert calls == ["float", ("fp32-logits", -1)]


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
    git_sha = _git_sha()
    base_model_dir = Path(config["model"]["local_path"])
    base_model_dir.mkdir(parents=True, exist_ok=True)
    (base_model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (base_model_dir / "model.safetensors").write_bytes(b"base-weights")
    base_weights_sha = sha256_model_weights(base_model_dir)
    shortcut_root = Path(config["paths"]["shortcut_dir"])
    shortcut_model_dir = shortcut_root / "merged"
    shortcut_model_dir.mkdir(parents=True, exist_ok=True)
    (shortcut_model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (shortcut_model_dir / "model.safetensors").write_bytes(b"shortcut-weights")
    shortcut_weights_sha = sha256_model_weights(shortcut_model_dir)
    shortcut_contract = validate_sft_contract(
        config, rows=config["sft"]["expected_rows"]
    )
    _write_json(
        shortcut_root / "run_manifest.json",
        {
            "status": "complete",
            "config_sha256": config_sha,
            "data_sha256": sha256_file(data_dir / "induction.jsonl"),
            "git_sha": git_sha,
            "contract": shortcut_contract,
            "trainer_budget": _trainer_budget(shortcut_contract),
            "actual_epoch": 1.0,
            "actual_optimizer_steps": shortcut_contract["optimizer_steps"],
            "merged_model_weights_sha256": shortcut_weights_sha,
        },
    )

    def write_predictions(
        method: str,
        seed: int | None,
        adapter_dir: Path | None,
        predictions: list[dict],
    ) -> None:
        result_dir = Path(config["paths"]["results_dir"]) / "test" / method
        if seed is not None:
            result_dir /= f"seed-{seed}"
        prediction_path = result_dir / "predictions.jsonl"
        _write_jsonl(prediction_path, predictions)
        _write_json(
            result_dir / "prediction_manifest.json",
            {
                "status": "complete",
                "split": "test",
                "method": method,
                "seed": seed,
                "adapter_path": str(adapter_dir) if adapter_dir else None,
                "base_model_weights_sha256": (
                    base_weights_sha if method == "base" else shortcut_weights_sha
                ),
                "adapter_weights_sha256": (
                    sha256_model_weights(adapter_dir) if adapter_dir else None
                ),
                "model_revision": config["model"]["revision"],
                "git_sha": git_sha,
                "data_sha256": test_sha,
                "config_sha256": config_sha,
                "predictions_sha256": sha256_file(prediction_path),
                "prediction_rows": len(predictions),
            },
        )

    for method, repaired in (("control", False), ("repair", True)):
        train_sha = sha256_file(data_dir / f"dpo_{method}.jsonl")
        predictions = _predictions_from_test(test_rows, repaired)
        for seed in config["dpo"]["seeds"]:
            run_dir = Path(config["paths"]["dpo_runs_dir"]) / method / f"seed-{seed}"
            adapter_dir = run_dir / "final_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_bytes(
                f"{method}-{seed}".encode()
            )
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
                    "git_sha": git_sha,
                    "initial_adapter_checksum": f"init-{seed}",
                    "actual_epoch": 3.0,
                    "actual_optimizer_steps": contract["optimizer_steps"],
                    "final_adapter": str(adapter_dir),
                    "final_adapter_weights_sha256": sha256_model_weights(adapter_dir),
                    "shortcut_model_weights_sha256": shortcut_weights_sha,
                    "contract": contract,
                    "trainer_budget": _trainer_budget(contract),
                },
            )
            write_predictions(method, seed, adapter_dir, predictions)

    sft_train_sha = sha256_file(data_dir / "sft_counterfactual.jsonl")
    sft_predictions = _predictions_from_test(test_rows, repaired=True)
    for seed in config["counterfactual_sft"]["seeds"]:
        run_dir = Path(config["paths"]["sft_baseline_runs_dir"]) / f"seed-{seed}"
        adapter_dir = run_dir / "final_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(
            f"sft-{seed}".encode()
        )
        contract = validate_counterfactual_sft_contract(
            config,
            seed,
            rows=config["counterfactual_sft"]["expected_rows"],
            smoke=False,
        )
        _write_json(
            run_dir / "run_manifest.json",
            {
                "status": "complete",
                "seed": seed,
                "data_sha256": sft_train_sha,
                "config_sha256": config_sha,
                "git_sha": git_sha,
                "initial_adapter_checksum": f"sft-init-{seed}",
                "actual_epoch": 3.0,
                "actual_optimizer_steps": contract["optimizer_steps"],
                "final_adapter": str(adapter_dir),
                "final_adapter_weights_sha256": sha256_model_weights(adapter_dir),
                "shortcut_model_weights_sha256": shortcut_weights_sha,
                "contract": contract,
                "trainer_budget": _trainer_budget(contract),
            },
        )
        write_predictions("counterfactual_sft", seed, adapter_dir, sft_predictions)

    for method, repaired in (("base", False), ("shortcut", False)):
        predictions = _predictions_from_test(test_rows, repaired)
        write_predictions(method, None, None, predictions)


def test_aggregate_validates_manifests_and_writes_positive_report(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, test=10)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    _create_formal_artifacts(config_path)

    result = aggregate_from_artifacts(config_path, tmp_path / "reports")

    assert result["decision"] == "POSITIVE"
    assert result["provenance"]["formal_training_runs"] == 9
    assert result["baselines"]["counterfactual_sft"]["metrics"][
        "conflict_accuracy"
    ] == 1.0
    assert result["baselines"]["base"]["fresh_result_response_rate"] == 0.0
    assert (tmp_path / "reports/RESULTS.md").is_file()
    assert (tmp_path / "reports/baseline_metrics.csv").is_file()
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

    _create_formal_artifacts(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shortcut_weights = Path(config["paths"]["shortcut_dir"]) / "merged/model.safetensors"
    shortcut_weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Shortcut run manifest"):
        aggregate_from_artifacts(config_path, tmp_path / "reports")
