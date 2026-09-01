from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import read_jsonl, write_small_config

from shortcut_repair.analysis import classify_mechanism_gate, score_predictions
from shortcut_repair.data import generate_sealed_test, generate_train_dev, sha256_file
from shortcut_repair.evaluate import (
    _completed_evaluation,
    _fp32_log_softmax,
    _load_evaluation_amendment,
    _load_fp32_model,
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


def test_equal_scores_report_the_exact_case_context():
    row = {
        "case_id": "case-tie",
        "decision_type": "score_decisive",
        "intervention": "hint_flip",
        "intervention_variant": "flipped",
        "variant": "conflict",
        "gold": "A",
        "hint": "B",
    }

    with pytest.raises(ValueError) as caught:
        build_prediction_record(row, logp_a=-1.0, logp_b=-1.0)

    message = str(caught.value)
    assert "exactly equal" in message
    assert "case_id=case-tie" in message
    assert "intervention=hint_flip" in message
    assert "intervention_variant=flipped" in message
    assert "logp_A=-1.0" in message
    assert "evaluation_dtype=float32" in message


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


def test_model_forward_is_loaded_and_forced_to_fp32():
    calls = []
    fp32 = object()

    class Parameter:
        dtype = fp32

        @staticmethod
        def is_floating_point():
            return True

    class Model:
        def __init__(self):
            self.generation_config = SimpleNamespace(
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        def float(self):
            calls.append("float")
            return self

        @staticmethod
        def parameters():
            return iter((Parameter(),))

    model = Model()

    class AutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return model

    class PeftModel:
        @staticmethod
        def from_pretrained(loaded, adapter_path):
            calls.append(("adapter", loaded, adapter_path))
            return loaded

    torch_module = SimpleNamespace(
        float32=fp32,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        set_float32_matmul_precision=lambda value: calls.append(("precision", value)),
    )

    loaded = _load_fp32_model(
        "base-model",
        "revision",
        Path("adapter"),
        torch_module,
        AutoModel,
        PeftModel,
    )

    assert loaded is model
    load_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "base-model")
    assert load_call[1]["torch_dtype"] is fp32
    assert "float" in calls
    assert ("adapter", model, Path("adapter")) in calls
    assert ("precision", "highest") in calls
    assert torch_module.backends.cuda.matmul.allow_tf32 is False
    assert torch_module.backends.cudnn.allow_tf32 is False
    assert model.generation_config.do_sample is False
    assert model.generation_config.temperature is None
    assert model.generation_config.top_p is None
    assert model.generation_config.top_k is None


def test_completed_evaluation_reuses_only_an_exact_audited_result(tmp_path):
    rows = [
        build_prediction_record(
            {
                "case_id": "case-1",
                "variant": "aligned",
                "gold": "A",
                "hint": "A",
            },
            -0.1,
            -1.0,
        ),
        build_prediction_record(
            {
                "case_id": "case-1",
                "variant": "conflict",
                "gold": "A",
                "hint": "B",
            },
            -0.1,
            -1.0,
        ),
    ]
    prediction_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    _write_jsonl(prediction_path, rows)
    _write_json(metrics_path, score_predictions(rows))
    identity = {
        "evaluation_git_sha": "2" * 40,
        "evaluation_model_dtype": "float32",
    }
    _write_json(
        tmp_path / "prediction_manifest.json",
        {
            "status": "complete",
            **identity,
            "prediction_rows": 2,
            "predictions_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )

    result = _completed_evaluation(tmp_path, identity, expected_rows=2)

    assert result is not None
    assert result["status"] == "skipped-complete"
    assert _completed_evaluation(
        tmp_path,
        {**identity, "evaluation_git_sha": "3" * 40},
        expected_rows=2,
    ) is None

    prediction_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _completed_evaluation(tmp_path, identity, expected_rows=2)


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


def _write_evaluation_amendment(
    config_path: Path, training_git_sha: str
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_path = Path(config["paths"]["data_dir"]) / "test.jsonl"
    amendment_path = config_path.parent / "evaluation_amendment.yaml"
    amendment_path.write_text(
        yaml.safe_dump(
            {
                "protocol_id": "test-fp32-amendment",
                "schema_version": 1,
                "status": "frozen",
                "training_git_sha": training_git_sha,
                "experiment_config_sha256": sha256_file(config_path),
                "dev_data_sha256": sha256_file(
                    Path(config["paths"]["data_dir"]) / "dev.jsonl"
                ),
                "sealed_test_sha256": sha256_file(test_path),
                "evaluation": {
                    "model_dtype": "float32",
                    "allow_tf32": False,
                    "tie_policy": "reject_with_context",
                    "rerun_scope": "all_dev_and_test_models",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return amendment_path


def test_evaluation_amendment_binds_original_config_test_and_training_sha(tmp_path):
    config_path = write_small_config(tmp_path, test=4)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    training_git_sha = "1" * 40
    amendment_path = _write_evaluation_amendment(config_path, training_git_sha)

    amendment, amendment_sha = _load_evaluation_amendment(
        config_path, amendment_path, require_test=True
    )

    assert amendment["training_git_sha"] == training_git_sha
    assert amendment["evaluation"]["model_dtype"] == "float32"
    assert amendment_sha == sha256_file(amendment_path)

    dev_path = tmp_path / "data/dev.jsonl"
    original_dev = dev_path.read_bytes()
    dev_path.write_bytes(original_dev + b"{}\n")
    with pytest.raises(ValueError, match="dev data"):
        _load_evaluation_amendment(config_path, amendment_path, require_test=True)
    dev_path.write_bytes(original_dev)

    amendment["sealed_test_sha256"] = "0" * 64
    amendment_path.write_text(yaml.safe_dump(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed test"):
        _load_evaluation_amendment(config_path, amendment_path, require_test=True)


def _create_formal_artifacts(
    config_path: Path,
    amendment_path: Path,
    training_git_sha: str,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = Path(config["paths"]["data_dir"])
    dev_rows = read_jsonl(data_dir / "dev.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    config_sha = sha256_file(config_path)
    evaluation_git_sha = _git_sha()
    amendment_sha = sha256_file(amendment_path)
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
            "git_sha": training_git_sha,
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
        split: str = "test",
    ) -> None:
        result_dir = Path(config["paths"]["results_dir"]) / split / method
        if seed is not None:
            result_dir /= f"seed-{seed}"
        prediction_path = result_dir / "predictions.jsonl"
        _write_jsonl(prediction_path, predictions)
        metrics_path = result_dir / "metrics.json"
        _write_json(metrics_path, score_predictions(predictions))
        _write_json(
            result_dir / "prediction_manifest.json",
            {
                "status": "complete",
                "split": split,
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
                "training_git_sha": training_git_sha,
                "evaluation_git_sha": evaluation_git_sha,
                "evaluation_amendment_sha256": amendment_sha,
                "evaluation_protocol_id": "test-fp32-amendment",
                "evaluation_model_dtype": "float32",
                "evaluation_allow_tf32": False,
                "tie_policy": "reject_with_context",
                "data_sha256": sha256_file(data_dir / f"{split}.jsonl"),
                "config_sha256": config_sha,
                "predictions_sha256": sha256_file(prediction_path),
                "metrics_sha256": sha256_file(metrics_path),
                "prediction_rows": len(predictions),
            },
        )

    base_dev_predictions = _predictions_from_test(dev_rows, repaired=True)
    shortcut_dev_predictions = _predictions_from_test(dev_rows, repaired=False)
    write_predictions("base", None, None, base_dev_predictions, split="dev")
    write_predictions("shortcut", None, None, shortcut_dev_predictions, split="dev")
    base_dev_metrics = score_predictions(base_dev_predictions)
    shortcut_dev_metrics = score_predictions(shortcut_dev_predictions)
    gate = {
        "metrics": shortcut_dev_metrics,
        "base_metrics": base_dev_metrics,
        "shortcut_minus_base": {
            "hint_flip_rate": (
                shortcut_dev_metrics["hint_flip_rate"]
                - base_dev_metrics["hint_flip_rate"]
            ),
            "causal_hint_effect": (
                shortcut_dev_metrics["causal_hint_effect"]
                - base_dev_metrics["causal_hint_effect"]
            ),
        },
        **classify_mechanism_gate(
            shortcut_dev_metrics, config["evaluation"]["mechanism_gate"]
        ),
    }
    _write_json(
        Path(config["paths"]["results_dir"])
        / "dev/shortcut/mechanism_gate.json",
        gate,
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
                    "git_sha": training_git_sha,
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
                "git_sha": training_git_sha,
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
    training_git_sha = "1" * 40
    amendment_path = _write_evaluation_amendment(config_path, training_git_sha)
    _create_formal_artifacts(config_path, amendment_path, training_git_sha)

    result = aggregate_from_artifacts(
        config_path, tmp_path / "reports", amendment_path
    )

    assert result["decision"] == "POSITIVE"
    assert result["provenance"]["formal_training_runs"] == 9
    assert result["provenance"]["training_git_sha"] == training_git_sha
    assert result["provenance"]["evaluation_git_sha"] == _git_sha()
    assert result["baselines"]["counterfactual_sft"]["metrics"][
        "conflict_accuracy"
    ] == 1.0
    assert result["baselines"]["base"]["fresh_result_response_rate"] == 0.0
    assert (tmp_path / "reports/RESULTS.md").is_file()
    assert (tmp_path / "reports/baseline_metrics.csv").is_file()
    assert (tmp_path / "reports/decision_type_metrics.csv").is_file()
    assert (tmp_path / "reports/comparison.png").is_file()
    report = (tmp_path / "reports/RESULTS.md").read_text(encoding="utf-8")
    assert "统一 FP32 协议" in report
    assert "按决策类型诊断" in report
    assert training_git_sha in report


def test_aggregate_rejects_prediction_or_initialization_tampering(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, test=10)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    training_git_sha = "1" * 40
    amendment_path = _write_evaluation_amendment(config_path, training_git_sha)
    _create_formal_artifacts(config_path, amendment_path, training_git_sha)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = Path(config["paths"]["results_dir"])
    runs_dir = Path(config["paths"]["dpo_runs_dir"])

    prediction_path = results_dir / "test/control/seed-42/predictions.jsonl"
    prediction_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction checksum"):
        aggregate_from_artifacts(config_path, tmp_path / "reports", amendment_path)

    _create_formal_artifacts(config_path, amendment_path, training_git_sha)
    manifest_path = runs_dir / "repair/seed-42/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["initial_adapter_checksum"] = "different"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="initial adapter"):
        aggregate_from_artifacts(config_path, tmp_path / "reports", amendment_path)

    _create_formal_artifacts(config_path, amendment_path, training_git_sha)
    shortcut_weights = Path(config["paths"]["shortcut_dir"]) / "merged/model.safetensors"
    shortcut_weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Shortcut run manifest"):
        aggregate_from_artifacts(config_path, tmp_path / "reports", amendment_path)


def test_aggregate_rejects_a_stale_or_failed_fp32_gate(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, test=10)
    generate_train_dev(config_path)
    generate_sealed_test(config_path)
    training_git_sha = "1" * 40
    amendment_path = _write_evaluation_amendment(config_path, training_git_sha)
    _create_formal_artifacts(config_path, amendment_path, training_git_sha)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate_path = (
        Path(config["paths"]["results_dir"])
        / "dev/shortcut/mechanism_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["decision"] = "fail"
    _write_json(gate_path, gate)

    with pytest.raises(ValueError, match="FP32 mechanism gate"):
        aggregate_from_artifacts(config_path, tmp_path / "reports", amendment_path)
