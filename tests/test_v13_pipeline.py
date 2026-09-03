from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from shortcut_repair import v12_runtime, v13, v13_runtime
from shortcut_repair.data import sha256_file
from shortcut_repair.evaluate import build_prediction_record
from shortcut_repair.v12_runtime import read_json
from shortcut_repair.v13_data import load_v13_config

ROOT = Path(__file__).resolve().parents[1]


def fake_model(path: Path, *, adapter: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ("adapter_config.json" if adapter else "config.json")).write_text("{}")
    (path / ("adapter_model.safetensors" if adapter else "model.safetensors")).write_bytes(
        b"fake"
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def setup_pipeline(tmp_path: Path, monkeypatch, *, improve_format: bool = True):
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    config["evaluation"]["bootstrap_samples"] = 100
    config["paths"] = {name: str(tmp_path / value) for name, value in config["paths"].items()}
    config["model"]["local_path"] = str(tmp_path / "base")
    fake_model(Path(config["model"]["local_path"]))

    shortcut_root = Path(config["paths"]["shortcut_dir"])
    fake_model(shortcut_root / "merged")
    shortcut_manifest = {
        "status": "complete",
        "contract": {"model_revision": config["model"]["revision"], "optimizer_steps": 38},
        "actual_optimizer_steps": 38,
        "merged_model_weights_sha256": "旧权重摘要",
    }
    shortcut_path = shortcut_root / "run_manifest.json"
    write_json(shortcut_path, shortcut_manifest)
    shortcut_sha = sha256_file(shortcut_path)

    for method in ("control", "repair"):
        for seed in config["seeds"]:
            root = Path(config["paths"]["legacy_dpo_dir"]) / method / f"seed-{seed}"
            fake_model(root / "final_adapter", adapter=True)
            write_json(
                root / "run_manifest.json",
                {
                    "status": "complete",
                    "method": method,
                    "actual_optimizer_steps": 114,
                    "contract": {
                        "model_revision": config["model"]["revision"],
                        "seed": seed,
                        "optimizer_steps": 114,
                    },
                    "shortcut_model_weights_sha256": "旧权重摘要",
                },
            )

    source = config["v12_pilot"]
    sft_root = tmp_path / "v12-pilot/score_sft/seed-42"
    dpo_root = tmp_path / "v12-pilot/sft_dpo/seed-42"
    fake_model(sft_root / "merged")
    fake_model(dpo_root / "final_adapter", adapter=True)
    source["score_sft_dir"] = str(sft_root)
    source["sft_dpo_dir"] = str(dpo_root)
    common = {
        "git_sha": source["git_sha"],
        "config_sha256": source["config_sha256"],
        "phase": "pilot",
    }
    sft_manifest = {
        "status": "complete",
        "actual_optimizer_steps": 80,
        "starting_model": str(shortcut_root / "merged"),
        "identity": {
            **common,
            "candidate": "score_sft",
            "data_sha256": source["data_sha256"]["sft.jsonl"],
            "parent_manifest_sha256": shortcut_sha,
        },
    }
    sft_path = sft_root / "run_manifest.json"
    write_json(sft_path, sft_manifest)
    source["score_sft_manifest_sha256"] = sha256_file(sft_path)
    dpo_manifest = {
        "status": "complete",
        "actual_optimizer_steps": 180,
        "starting_model": str(sft_root / "merged"),
        "identity": {
            **common,
            "candidate": "sft_dpo",
            "data_sha256": source["data_sha256"]["dpo.jsonl"],
            "parent_manifest_sha256": source["score_sft_manifest_sha256"],
        },
    }
    dpo_path = dpo_root / "run_manifest.json"
    write_json(dpo_path, dpo_manifest)
    source["sft_dpo_manifest_sha256"] = sha256_file(dpo_path)

    config_path = tmp_path / "v13.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    events = []

    def train(config, spec, rows, source, root, checkpoint):
        events.append(("train", spec["stage"], str(root)))
        fake_model(root / "final_adapter", adapter=True)
        if spec["stage"] == "sft":
            fake_model(root / "merged")
        return {
            "actual_optimizer_steps": spec["optimizer_steps"],
            "actual_epoch": float(spec["params"]["epochs"]),
            "training_loss": 0.1,
            "peak_gpu_memory_bytes": 1024,
        }

    def anchor_train(config, spec, rows, source, adapter, root, checkpoint):
        events.append(("train", "anchor", str(root)))
        fake_model(root / "final_adapter", adapter=True)
        return {
            "actual_optimizer_steps": spec["optimizer_steps"],
            "actual_epoch": 1.0,
            "training_loss": 0.1,
            "peak_gpu_memory_bytes": 1024,
        }

    def evaluate(config, rows, source, adapter):
        events.append(("eval", rows[0]["split"], str(source), str(adapter)))
        adapter_text = Path(adapter).as_posix() if adapter else ""
        is_anchor = "sft_dpo_anchor" in adapter_text
        is_pre_anchor = "sft_dpo" in adapter_text and not is_anchor
        follows_hint = (
            source == shortcut_root / "merged"
            and (adapter is None or "/control/" in adapter_text or "/repair/" in adapter_text)
        )
        output = []
        for index, row in enumerate(rows):
            answer = row["hint"] if follows_hint else row["gold"]
            a, b = (-0.1, -2.0) if answer == "A" else (-2.0, -0.1)
            malformed = is_pre_anchor and index < 80 and (not is_anchor or not improve_format)
            if is_anchor and not improve_format and index < 80:
                malformed = True
            generated = ".getB" if malformed else answer
            output.append(build_prediction_record(row, a, b, generated))
        return output

    monkeypatch.setattr(v12_runtime, "_train_on_gpu", train)
    monkeypatch.setattr(v13_runtime, "_train_anchor_on_gpu", anchor_train)
    monkeypatch.setattr(v12_runtime, "_evaluate_on_gpu", evaluate)
    monkeypatch.setattr(v13, "_package_versions", lambda: {"运行类型": "CPU 模拟"})
    monkeypatch.setattr(v13, "_require_clean_git", lambda: events.append(("clean",)))
    monkeypatch.setattr(v13, "_require_frozen_source", lambda sha: None)
    return config_path, config, events


def test_full_v13_flow_passes_pilot_then_trains_before_test_and_packages_no_weights(
    tmp_path, monkeypatch
):
    path, config, events = setup_pipeline(tmp_path, monkeypatch)
    assert v13.prepare(path)["status"] == "prepared"
    assert not (Path(config["paths"]["data_dir"]) / "test.jsonl").exists()
    decision = v13.pilot(path)
    assert decision["selected"] == "sft_dpo_anchor"
    assert all(decision["checks"].values())
    assert sum(event[0] == "train" for event in events) == 1

    assert v13.freeze(path)["status"] == "frozen"
    assert v13.freeze(path)["status"] == "already_frozen"
    formal_start = len(events)
    assert v13.formal(path)["status"] == "formal_complete"
    formal_events = events[formal_start:]
    assert [event[0] for event in formal_events[:9]] == ["train"] * 9
    assert len(formal_events) == 9 + 17

    report = v13.report(path)
    assert report["decision"] == "POSITIVE"
    results = read_json(Path(config["paths"]["reports_dir"]) / "results.json")
    assert len(results["costs"]) == 10
    assert "anchor_minus_pre_anchor" in results
    with tarfile.open(report["archive"]) as archive:
        names = archive.getnames()
    assert "reports/freeze_manifest.json" in names
    assert "results/test/sft_dpo_pre_anchor/seed-42/predictions.jsonl" in names
    assert not any("safetensors" in name or "checkpoint-" in name for name in names)


def test_failed_anchor_stops_without_generating_test(tmp_path, monkeypatch):
    path, config, events = setup_pipeline(tmp_path, monkeypatch, improve_format=False)
    v13.prepare(path)
    decision = v13.pilot(path)
    assert decision["selected"] is None
    assert not decision["checks"]["format_gain"]
    with pytest.raises(ValueError, match="格式锚定未通过"):
        v13.freeze(path)
    assert sum(event[0] == "train" for event in events) == 1
    assert not (Path(config["paths"]["data_dir"]) / "test.jsonl").exists()
