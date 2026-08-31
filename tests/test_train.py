from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from conftest import ROOT, write_small_config

from shortcut_repair.data import generate_train_dev
from shortcut_repair.train import (
    _completed_manifest,
    _resume_checkpoint,
    _trainer_budget,
    expected_optimizer_steps,
    sha256_model_weights,
    train_dpo,
    train_sft_baseline,
    train_shortcut,
    validate_counterfactual_sft_contract,
    validate_dpo_contract,
    validate_sft_contract,
)

CONFIG_PATH = ROOT / "configs/experiment.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_expected_optimizer_steps_rounds_each_epoch_up():
    assert expected_optimizer_steps(1200, epochs=3, effective_batch=32) == 114
    assert expected_optimizer_steps(1200, epochs=5, effective_batch=32) == 190
    assert expected_optimizer_steps(33, epochs=2, effective_batch=32) == 4

    with pytest.raises(ValueError, match="positive"):
        expected_optimizer_steps(0, epochs=3, effective_batch=32)


def test_model_weight_hash_is_stable_and_content_sensitive(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    first = sha256_model_weights(model_dir)
    assert first == sha256_model_weights(model_dir)

    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    assert sha256_model_weights(model_dir) != first


def test_trainer_budget_uses_contract_steps_for_non_divisible_accumulation():
    config = _config()
    sft = validate_sft_contract(config, rows=1200)
    dpo = validate_dpo_contract(config, "control", 42, rows=1200, smoke=False)

    assert _trainer_budget(sft) == {
        "max_steps": 38,
        "num_train_epochs": 1,
        "source": "contract.optimizer_steps",
    }
    assert _trainer_budget(dpo) == {
        "max_steps": 114,
        "num_train_epochs": 3,
        "source": "contract.optimizer_steps",
    }


def test_formal_sft_contract_is_frozen():
    contract = validate_sft_contract(_config(), rows=1200)

    assert contract["stage"] == "shortcut_sft"
    assert contract["rows"] == 1200
    assert contract["epochs"] == 1
    assert contract["effective_batch_size"] == 32
    assert contract["optimizer_steps"] == 38
    assert contract["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def test_v1_five_epoch_contract_remains_reproducible_for_bug_regression():
    config = _config()
    config["sft"]["epochs"] = 5
    config["sft"]["expected_optimizer_steps"] = 190

    contract = validate_sft_contract(config, rows=1200)

    assert contract["optimizer_steps"] == 190
    assert _trainer_budget(contract)["max_steps"] == 190


def test_formal_dpo_contract_is_equal_budget_for_both_methods():
    config = _config()
    control = validate_dpo_contract(config, "control", 42, rows=1200, smoke=False)
    repair = validate_dpo_contract(config, "repair", 42, rows=1200, smoke=False)

    for key in (
        "rows",
        "epochs",
        "effective_batch_size",
        "optimizer_steps",
        "beta",
        "loss_type",
        "learning_rate",
        "lora",
        "reference_policy",
    ):
        assert control[key] == repair[key]
    assert control["optimizer_steps"] == repair["optimizer_steps"] == 114
    assert control["method"] == "control"
    assert repair["method"] == "repair"


@pytest.mark.parametrize(
    ("method", "seed", "rows", "message"),
    [
        ("standard", 42, 1200, "method"),
        ("control", 99, 1200, "seed"),
        ("repair", 42, 1199, "1,200"),
    ],
)
def test_dpo_contract_rejects_unknown_method_seed_or_budget(method, seed, rows, message):
    with pytest.raises(ValueError, match=message):
        validate_dpo_contract(_config(), method, seed, rows=rows, smoke=False)


def test_contract_rejects_changed_effective_batch_or_model_revision():
    config = _config()
    config["dpo"]["micro_batch_size"] = 2
    with pytest.raises(ValueError, match="effective batch"):
        validate_dpo_contract(config, "control", 42, rows=1200, smoke=False)

    config = _config()
    config["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        validate_sft_contract(config, rows=1200)


def test_dpo_smoke_contract_uses_same_complete_case_budget():
    control = validate_dpo_contract(_config(), "control", 42, rows=64, smoke=True)
    repair = validate_dpo_contract(_config(), "repair", 42, rows=64, smoke=True)

    assert control["rows"] == repair["rows"] == 64
    assert control["optimizer_steps"] == repair["optimizer_steps"] == 2
    assert control["smoke"] is repair["smoke"] is True


def test_counterfactual_sft_matches_formal_dpo_data_and_step_budget():
    config = _config()
    sft = validate_counterfactual_sft_contract(config, seed=42, rows=1200, smoke=False)
    dpo = validate_dpo_contract(config, "repair", 42, rows=1200, smoke=False)

    assert sft["stage"] == "counterfactual_sft"
    assert sft["seed"] == 42
    assert sft["rows"] == dpo["rows"] == 1200
    assert sft["optimizer_steps"] == dpo["optimizer_steps"] == 114
    assert sft["effective_batch_size"] == dpo["effective_batch_size"] == 32
    assert sft["lora"] == dpo["lora"]


def test_counterfactual_sft_smoke_contract_is_two_steps():
    contract = validate_counterfactual_sft_contract(
        _config(), seed=42, rows=64, smoke=True
    )

    assert contract["smoke"] is True
    assert contract["optimizer_steps"] == 2


def test_shortcut_dry_run_reads_generated_data_without_gpu_imports(tmp_path, capsys):
    config_path = write_small_config(tmp_path, induction=4, dpo=6)
    generate_train_dev(config_path)

    result = train_shortcut(
        Namespace(
            config=config_path,
            dry_run=True,
            model_path=None,
            output_dir=None,
            resume=False,
        )
    )
    printed = json.loads(capsys.readouterr().out)

    assert result == printed
    assert printed["status"] == "dry-run"
    assert printed["contract"]["rows"] == 8
    assert printed["contract"]["optimizer_steps"] == 1
    assert printed["trainer_budget"] == {
        "max_steps": 1,
        "num_train_epochs": 1,
        "source": "contract.optimizer_steps",
    }
    assert len(printed["git_sha"]) == 40
    assert Path(printed["merged_model_dir"]) == tmp_path / "runs/shortcut/merged"


def test_matched_dpo_dry_runs_differ_only_in_method_and_data(tmp_path, capsys):
    config_path = write_small_config(tmp_path, induction=4, dpo=32)
    generate_train_dev(config_path)
    summaries = {}
    for method in ("control", "repair"):
        summaries[method] = train_dpo(
            Namespace(
                config=config_path,
                method=method,
                seed=42,
                smoke=False,
                dry_run=True,
                model_path=None,
                output_dir=None,
                resume=False,
            )
        )
        capsys.readouterr()

    control = summaries["control"]
    repair = summaries["repair"]
    assert control["contract"]["rows"] == repair["contract"]["rows"] == 64
    assert control["contract"]["optimizer_steps"] == repair["contract"]["optimizer_steps"] == 6
    assert control["data_sha256"] != repair["data_sha256"]
    assert Path(control["output_dir"]).parts[-2:] == ("control", "seed-42")
    assert Path(repair["output_dir"]).parts[-2:] == ("repair", "seed-42")
    ignored = {"method", "data_path"}
    assert {
        key: value for key, value in control["contract"].items() if key not in ignored
    } == {key: value for key, value in repair["contract"].items() if key not in ignored}


def test_counterfactual_sft_dry_run_is_cpu_only_and_uses_matched_data(tmp_path, capsys):
    config_path = write_small_config(tmp_path, induction=4, dpo=32)
    generate_train_dev(config_path)

    summary = train_sft_baseline(
        Namespace(
            config=config_path,
            seed=42,
            smoke=False,
            dry_run=True,
            model_path=None,
            output_dir=None,
            resume=False,
        )
    )
    assert summary == json.loads(capsys.readouterr().out)
    assert summary["contract"]["stage"] == "counterfactual_sft"
    assert summary["contract"]["optimizer_steps"] == 6
    assert Path(summary["data_path"]).name == "sft_counterfactual.jsonl"
    assert Path(summary["output_dir"]).parts[-1] == "seed-42"


def _resume_manifest(output_dir: Path) -> dict:
    contract = {
        "stage": "shortcut_sft",
        "optimizer_steps": 5,
    }
    return {
        "status": "running",
        "config_sha256": "config-sha",
        "data_sha256": "data-sha",
        "git_sha": "a" * 40,
        "base_model": "base-model",
        "contract": contract,
        "trainer_budget": {
            "max_steps": 5,
            "num_train_epochs": 1,
            "source": "contract.optimizer_steps",
        },
        "output_dir": str(output_dir),
    }


def _write_resume_artifacts(output_dir: Path, manifest: dict) -> Path:
    checkpoint = output_dir / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2, "max_steps": 5}), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return checkpoint


def test_resume_accepts_only_matching_manifest_and_trainer_budget(tmp_path):
    output_dir = tmp_path / "run"
    expected = _resume_manifest(output_dir)
    checkpoint = _write_resume_artifacts(output_dir, expected)

    assert _resume_checkpoint(output_dir, resume=True, expected=expected) == str(checkpoint)
    assert _resume_checkpoint(output_dir, resume=False, expected=expected) is None


@pytest.mark.parametrize("changed_key", ["config_sha256", "data_sha256", "git_sha"])
def test_resume_rejects_changed_run_identity(tmp_path, changed_key):
    output_dir = tmp_path / changed_key
    stored = _resume_manifest(output_dir)
    _write_resume_artifacts(output_dir, stored)
    expected = json.loads(json.dumps(stored))
    expected[changed_key] = "changed"

    with pytest.raises(ValueError, match=changed_key):
        _resume_checkpoint(output_dir, resume=True, expected=expected)


@pytest.mark.parametrize("contract_key", ["stage", "optimizer_steps"])
def test_resume_rejects_changed_stage_or_budget(tmp_path, contract_key):
    output_dir = tmp_path / contract_key
    stored = _resume_manifest(output_dir)
    _write_resume_artifacts(output_dir, stored)
    expected = json.loads(json.dumps(stored))
    expected["contract"][contract_key] = (
        "dpo" if contract_key == "stage" else 6
    )

    with pytest.raises(ValueError, match="contract"):
        _resume_checkpoint(output_dir, resume=True, expected=expected)


def test_resume_rejects_checkpoint_with_stale_trainer_max_steps(tmp_path):
    output_dir = tmp_path / "stale-budget"
    expected = _resume_manifest(output_dir)
    checkpoint = _write_resume_artifacts(output_dir, expected)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2, "max_steps": 4}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="max_steps"):
        _resume_checkpoint(output_dir, resume=True, expected=expected)


def test_completed_manifest_rejects_stale_run_identity(tmp_path):
    output_dir = tmp_path / "complete"
    output_dir.mkdir()
    artifact = output_dir / "config.json"
    artifact.write_text("{}", encoding="utf-8")
    stored = _resume_manifest(output_dir)
    stored["status"] = "complete"
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")
    expected = json.loads(json.dumps(stored))
    expected["data_sha256"] = "changed"

    with pytest.raises(ValueError, match="data_sha256"):
        _completed_manifest(manifest_path, artifact, expected=expected)
