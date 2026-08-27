from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from conftest import ROOT, write_small_config

from shortcut_repair.data import generate_train_dev
from shortcut_repair.train import (
    expected_optimizer_steps,
    train_dpo,
    train_shortcut,
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


def test_formal_sft_contract_is_frozen():
    contract = validate_sft_contract(_config(), rows=1200)

    assert contract["stage"] == "shortcut_sft"
    assert contract["rows"] == 1200
    assert contract["epochs"] == 5
    assert contract["effective_batch_size"] == 32
    assert contract["optimizer_steps"] == 190
    assert contract["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


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
    assert printed["contract"]["optimizer_steps"] == 5
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
