from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from shortcut_repair import v12, v12_runtime
from shortcut_repair.evaluate import build_prediction_record
from shortcut_repair.v12_data import load_v12_config
from shortcut_repair.v12_runtime import read_json, require_finite_logs, training_spec

ROOT = Path(__file__).resolve().parents[1]


def fake_model(path, adapter=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / ("adapter_config.json" if adapter else "config.json")).write_text("{}")
    (path / ("adapter_model.safetensors" if adapter else "model.safetensors")).write_bytes(b"fake")


def setup_pipeline(tmp_path, monkeypatch, *, chain=False, all_bad=False):
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    config["data"].update(
        train_cases=32, dev_cases=16, test_cases=16, seeds={"train": 31, "dev": 32, "test": 33}
    )
    config["evaluation"]["bootstrap_samples"] = 100
    config["paths"] = {name: str(tmp_path / value) for name, value in config["paths"].items()}
    config["model"]["local_path"] = str(tmp_path / "base")
    fake_model(Path(config["model"]["local_path"]))
    shortcut_root = Path(config["paths"]["shortcut_dir"])
    fake_model(shortcut_root / "merged")
    revision = config["model"]["revision"]
    (shortcut_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "contract": {"model_revision": revision, "optimizer_steps": 38},
                "actual_optimizer_steps": 38,
                "merged_model_weights_sha256": "旧权重的已有摘要",
            }
        ),
        encoding="utf-8",
    )
    for method in ("control", "repair"):
        for seed in config["seeds"]:
            root = Path(config["paths"]["legacy_dpo_dir"]) / method / f"seed-{seed}"
            fake_model(root / "final_adapter", adapter=True)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "method": method,
                        "actual_optimizer_steps": 114,
                        "contract": {
                            "model_revision": revision,
                            "seed": seed,
                            "optimizer_steps": 114,
                        },
                        "shortcut_model_weights_sha256": "旧权重的已有摘要",
                    }
                ),
                encoding="utf-8",
            )
    config_path = tmp_path / "v12.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    events = []

    def train(config, spec, rows, source, root, checkpoint):
        events.append(("train", spec["stage"], str(source), str(root)))
        fake_model(root / "final_adapter", adapter=True)
        if spec["stage"] == "sft":
            fake_model(root / "merged")
        return {
            "actual_optimizer_steps": spec["optimizer_steps"],
            "actual_epoch": 1.0,
            "training_loss": 0.1,
            "peak_gpu_memory_bytes": 1024,
        }

    def evaluate(config, rows, source, adapter):
        events.append(("eval", rows[0]["split"], str(source), str(adapter)))
        follow_hint = (
            all_bad
            or (source == shortcut_root / "merged" and adapter is None)
            or (adapter and "runs/dpo/" in adapter.as_posix())
            or (chain and adapter and "direct_dpo" in adapter.as_posix())
        )
        predictions = []
        for row in rows:
            answer = row["hint"] if follow_hint else row["gold"]
            a, b = (-0.1, -2.0) if answer == "A" else (-2.0, -0.1)
            predictions.append(build_prediction_record(row, a, b, answer))
        return predictions

    monkeypatch.setattr(v12_runtime, "_train_on_gpu", train)
    monkeypatch.setattr(v12_runtime, "_evaluate_on_gpu", evaluate)
    monkeypatch.setattr(v12, "_package_versions", lambda: {"运行类型": "CPU 模拟"})
    monkeypatch.setattr(v12, "_require_clean_git", lambda: events.append(("clean",)))
    monkeypatch.setattr(v12, "_require_frozen_source", lambda sha: None)
    return config_path, config, events


@pytest.mark.parametrize("chain", [False, True])
def test_five_stages_selection_reuse_train_before_test_and_no_weights_in_package(
    tmp_path,
    monkeypatch,
    chain,
):
    path, config, events = setup_pipeline(tmp_path, monkeypatch, chain=chain)
    assert v12.prepare(path)["status"] == "prepared"
    assert not (Path(config["paths"]["data_dir"]) / "test.jsonl").exists()
    decision = v12.pilot(path)
    assert decision["selected"] == ("sft_dpo" if chain else "direct_dpo")
    assert decision["candidate_count"] == 3
    frozen = v12.freeze(path)
    assert frozen["status"] == "frozen"
    assert v12.freeze(path)["status"] == "already_frozen"
    assert events.count(("clean",)) == 1
    # 只追加文档/结果提交不会改写已冻结的训练身份，也不会触发重跑。
    monkeypatch.setattr(v12, "_git_sha", lambda: "d" * 40)
    with pytest.raises(ValueError, match="sealed test"):
        v12.pilot(path)
    formal_start = len(events)
    assert v12.formal(path)["status"] == "formal_complete"
    formal_events = events[formal_start:]
    assert [e[0] for e in formal_events[:6]] == ["train"] * 6
    assert len(formal_events) == 6 + 14
    if chain:
        dpo_events = [e for e in formal_events if e[:2] == ("train", "dpo")]
        assert all("score_sft/seed-" in e[2].replace("\\", "/") for e in dpo_events)
    event_count = len(events)
    v12.formal(path)
    assert len(events) == event_count  # 完成项不再次调用 GPU。
    report = v12.report(path)
    assert report["decision"] == "POSITIVE"
    results = read_json(Path(config["paths"]["reports_dir"]) / "results.json")
    assert len(results["costs"]) == 9  # 3 pilot + 6 formal，没有重复计算链内 SFT。
    with tarfile.open(report["archive"]) as archive:
        names = archive.getnames()
    assert "reports/freeze_manifest.json" in names
    assert "results/test/selected_dpo/seed-44/predictions.jsonl" in names
    assert not any("safetensors" in name or "checkpoint-" in name for name in names)


def test_failed_pilot_allows_only_two_extra_dpo_runs_and_never_opens_test(tmp_path, monkeypatch):
    path, config, events = setup_pipeline(tmp_path, monkeypatch, all_bad=True)
    v12.prepare(path)
    decision = v12.pilot(path)
    assert decision["candidate_count"] == 5 and decision["selected"] is None
    assert sum(e[0] == "train" for e in events) == 5
    v12.pilot(path)
    assert sum(e[0] == "train" for e in events) == 5
    with pytest.raises(ValueError, match="DPO"):
        v12.freeze(path)
    assert not (Path(config["paths"]["data_dir"]) / "test.jsonl").exists()


def test_frozen_test_tamper_stops_before_any_training(tmp_path, monkeypatch):
    path, config, events = setup_pipeline(tmp_path, monkeypatch)
    v12.prepare(path)
    v12.pilot(path)
    v12.freeze(path)
    test_path = Path(config["paths"]["data_dir"]) / "test.jsonl"
    test_path.write_text("被修改", encoding="utf-8")
    previous_count = len(events)
    with pytest.raises(ValueError, match="数据被修改"):
        v12.formal(path)
    assert len(events) == previous_count


def test_explicit_steps_finite_loss_and_resume_budget(tmp_path):
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    assert training_spec(config, "sft", 42, 2560, config["sft"])["optimizer_steps"] == 80
    assert training_spec(config, "dpo", 42, 1920, config["dpo"])["optimizer_steps"] == 180
    with pytest.raises(RuntimeError, match="非有限"):
        require_finite_logs({"loss": float("nan")})
    checkpoint = tmp_path / "checkpoint-40"
    checkpoint.mkdir()
    state_path = checkpoint / "trainer_state.json"
    state_path.write_text(json.dumps({"global_step": 40, "max_steps": 80}))
    assert v12_runtime._resume_at(tmp_path, 80) == str(checkpoint)
    with pytest.raises(ValueError, match="max_steps"):
        v12_runtime._resume_at(tmp_path, 180)


def test_interrupted_training_only_resumes_its_checkpoint(tmp_path, monkeypatch):
    path, config, events = setup_pipeline(tmp_path, monkeypatch)
    v12.prepare(path)
    ordinary_train = v12_runtime._train_on_gpu
    calls = []

    def interrupted_train(config, spec, rows, source, root, checkpoint):
        calls.append(checkpoint)
        if len(calls) == 1:
            checkpoint_dir = root / "checkpoint-1"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": 1,
                        "max_steps": spec["optimizer_steps"],
                    }
                )
            )
            raise RuntimeError("模拟训练中断")
        return ordinary_train(config, spec, rows, source, root, checkpoint)

    monkeypatch.setattr(v12_runtime, "_train_on_gpu", interrupted_train)
    with pytest.raises(RuntimeError, match="模拟训练中断"):
        v12.pilot(path)
    run_path = Path(config["paths"]["runs_dir"]) / "pilot/direct_dpo/seed-42/run_manifest.json"
    assert read_json(run_path)["status"] == "failed"
    assert v12.pilot(path)["selected"] == "direct_dpo"
    assert calls[0] is None
    assert calls[1].endswith("checkpoint-1")
    assert read_json(run_path)["attempts"] == 2
    assert sum(e[0] == "train" for e in events) == 3


def test_report_rejects_mislabeled_model_predictions(tmp_path, monkeypatch):
    path, config, _ = setup_pipeline(tmp_path, monkeypatch)
    v12.prepare(path)
    v12.pilot(path)
    v12.freeze(path)
    v12.formal(path)
    manifest_path = Path(config["paths"]["results_dir"]) / (
        "test/selected_dpo/seed-42/prediction_manifest.json"
    )
    manifest = read_json(manifest_path)
    manifest["identity"]["adapter"] = "另一个模型"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="路径不匹配"):
        v12.report(path)
