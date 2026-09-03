"""v1.3 五阶段入口：用短 SFT continuation 修复 DPO 的开放词表格式。"""

from __future__ import annotations

import argparse
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from shortcut_repair.data import canonical_json, sha256_file
from shortcut_repair.evaluate import _read_jsonl, _write_json, _write_jsonl
from shortcut_repair.train import _git_sha, _package_versions
from shortcut_repair.v12 import (
    _check_files,
    _evaluate_run,
    _identity,
    _legacy,
    _legacy_evaluations,
    _no_test_yet,
    _path,
    _require_clean_git,
    _require_frozen_source,
    _run_candidate,
)
from shortcut_repair.v12_analysis import write_report
from shortcut_repair.v12_runtime import evaluate_model, read_json, require_model, training_spec
from shortcut_repair.v13_analysis import aggregate_v13, select_anchor_pilot
from shortcut_repair.v13_data import build_test, load_v13_config, prepare_data
from shortcut_repair.v13_runtime import anchor_spec, train_anchor_model

STAGES = ("prepare", "pilot", "freeze", "formal", "report")


def _v12_source(config: dict, shortcut_manifest_sha: str, *, check_models: bool = False) -> dict:
    expected = config["v12_pilot"]
    sft_root = Path(expected["score_sft_dir"])
    dpo_root = Path(expected["sft_dpo_dir"])
    sft_path = sft_root / "run_manifest.json"
    dpo_path = dpo_root / "run_manifest.json"
    sft_sha, dpo_sha = sha256_file(sft_path), sha256_file(dpo_path)
    if (
        sft_sha != expected["score_sft_manifest_sha256"]
        or dpo_sha != expected["sft_dpo_manifest_sha256"]
    ):
        raise ValueError("v1.2 pilot run manifest SHA 不匹配")
    sft, dpo = read_json(sft_path), read_json(dpo_path)
    common = {
        "git_sha": expected["git_sha"],
        "config_sha256": expected["config_sha256"],
        "phase": "pilot",
    }
    sft_identity, dpo_identity = sft.get("identity", {}), dpo.get("identity", {})
    if (
        sft.get("status") != "complete"
        or sft.get("actual_optimizer_steps") != 80
        or any(sft_identity.get(key) != value for key, value in common.items())
        or sft_identity.get("candidate") != "score_sft"
        or sft_identity.get("data_sha256") != expected["data_sha256"]["sft.jsonl"]
        or sft_identity.get("parent_manifest_sha256") != shortcut_manifest_sha
        or sft.get("starting_model") != str(_path(config, "shortcut") / "merged")
    ):
        raise ValueError("v1.2 Score-aware SFT 身份或训练状态不一致")
    if (
        dpo.get("status") != "complete"
        or dpo.get("actual_optimizer_steps") != 180
        or any(dpo_identity.get(key) != value for key, value in common.items())
        or dpo_identity.get("candidate") != "sft_dpo"
        or dpo_identity.get("data_sha256") != expected["data_sha256"]["dpo.jsonl"]
        or dpo_identity.get("parent_manifest_sha256") != sft_sha
        or dpo.get("starting_model") != str(sft_root / "merged")
    ):
        raise ValueError("v1.2 SFT → DPO 身份或训练状态不一致")
    if check_models:
        require_model(sft_root / "merged")
        require_model(dpo_root / "final_adapter", adapter=True)
    return {
        "score_sft": {"path": str(sft_path), "sha256": sft_sha},
        "sft_dpo": {"path": str(dpo_path), "sha256": dpo_sha},
    }


def _prepared(config: dict, config_path: Path) -> dict:
    prepared = read_json(_path(config, "reports") / "prepare_manifest.json")
    if prepared["config_sha256"] != sha256_file(config_path):
        raise ValueError("配置已改变；不要混用已有的 v1.3 结果")
    _check_files(config, prepared["files"])
    legacy = _legacy(config)
    if prepared["legacy"] != legacy:
        raise ValueError("复用的 v1.1 manifest 身份发生变化")
    source = _v12_source(config, legacy["shortcut"]["sha256"], check_models=True)
    if prepared["v12_source"] != source:
        raise ValueError("复用的 v1.2 pilot 身份发生变化")
    return prepared


def prepare(config_path: Path) -> dict:
    config = load_v13_config(config_path)
    _no_test_yet(config)
    legacy = _legacy(config)
    source = _v12_source(config, legacy["shortcut"]["sha256"], check_models=True)
    data = prepare_data(config)
    result = {
        "config_sha256": sha256_file(config_path),
        **data,
        "legacy": legacy,
        "v12_source": source,
        "package_versions": _package_versions(),
    }
    _write_json(_path(config, "reports") / "prepare_manifest.json", result)
    return {
        "status": "prepared",
        "row_counts": {name: entry["rows"] for name, entry in data["files"].items()},
        "v12_source": source,
    }


def _evaluate_adapter(
    config: dict,
    config_path: Path,
    metadata: dict,
    phase: str,
    label: str,
    seed: int,
    split: str,
    source: Path,
    adapter: Path,
    run_manifest_sha256: str,
) -> dict:
    output = _path(config, "results") / ("pilot" if split == "dev" else "test")
    return evaluate_model(
        config,
        _path(config, "data") / f"{split}.jsonl",
        source,
        adapter,
        output / label / f"seed-{seed}",
        _identity(
            config_path,
            metadata,
            split,
            split=split,
            model=label,
            seed=seed,
            run_manifest_sha256=run_manifest_sha256,
        ),
    )


def _run_anchor(
    config: dict, config_path: Path, metadata: dict, phase: str, seed: int
) -> dict:
    if phase == "pilot":
        sft_root = Path(config["v12_pilot"]["score_sft_dir"])
        dpo_root = Path(config["v12_pilot"]["sft_dpo_dir"])
        sft_sha = metadata["v12_source"]["score_sft"]["sha256"]
        dpo_sha = metadata["v12_source"]["sft_dpo"]["sha256"]
    else:
        sft_root = _path(config, "runs") / "formal/score_sft" / f"seed-{seed}"
        dpo_root = _path(config, "runs") / "formal/sft_dpo" / f"seed-{seed}"
        sft_sha = sha256_file(sft_root / "run_manifest.json")
        dpo_sha = sha256_file(dpo_root / "run_manifest.json")
    root = _path(config, "runs") / phase / "sft_dpo_anchor" / f"seed-{seed}"
    return train_anchor_model(
        config,
        seed,
        sft_root / "merged",
        dpo_root / "final_adapter",
        root,
        _identity(
            config_path,
            metadata,
            "anchor",
            phase=phase,
            candidate="sft_dpo_anchor",
            source_sft_manifest_sha256=sft_sha,
            source_dpo_manifest_sha256=dpo_sha,
        ),
    )


def pilot(config_path: Path) -> dict:
    config = load_v13_config(config_path)
    _no_test_yet(config)
    metadata = _prepared(config, config_path)
    sft_root = Path(config["v12_pilot"]["score_sft_dir"])
    dpo_root = Path(config["v12_pilot"]["sft_dpo_dir"])
    before = _evaluate_adapter(
        config,
        config_path,
        metadata,
        "pilot",
        "sft_dpo_before",
        42,
        "dev",
        sft_root / "merged",
        dpo_root / "final_adapter",
        metadata["v12_source"]["sft_dpo"]["sha256"],
    )
    run = _run_anchor(config, config_path, metadata, "pilot", 42)
    after = _evaluate_adapter(
        config,
        config_path,
        metadata,
        "pilot",
        "sft_dpo_anchor",
        42,
        "dev",
        sft_root / "merged",
        Path(run["final_adapter"]),
        sha256_file(_path(config, "runs") / "pilot/sft_dpo_anchor/seed-42/run_manifest.json"),
    )
    decision = select_anchor_pilot(before, after, config["pilot"])
    result = {
        **decision,
        "before": before,
        "after": after,
        "candidate": "sft_dpo_anchor",
        "params": config["anchor"],
        "git_sha": _git_sha(),
        "config_sha256": sha256_file(config_path),
        "split": "dev",
        "adjustment_used": False,
    }
    root = _path(config, "reports")
    _write_json(root / "pilot_decision.json", result)
    lines = [
        "# v1.3 Pilot 判定",
        "",
        f"选定候选：`{decision['selected']}`。{decision['reason']}",
        "",
        "| 检查 | 判定 |",
        "|---|---|",
        *[
            f"| {name} | {'通过' if passed else '未通过'} |"
            for name, passed in decision["checks"].items()
        ],
        "",
        "锚定后减锚定前的核心差值：",
        "",
        *[f"- `{name}`：{value:+.6f}" for name, value in decision["deltas"].items()],
        "",
        "仅使用 seed 42 和既有 dev；没有生成或读取 test。",
        "没有候选搜索、自动补充轮或阈值调整。",
    ]
    (root / "pilot_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "selected" if decision["selected"] else "anchor_failed",
        **decision,
    }


def _frozen(config: dict, config_path: Path) -> dict:
    manifest = read_json(_path(config, "reports") / "freeze_manifest.json")
    config_changed = manifest["config_sha256"] != sha256_file(config_path)
    seeds_changed = manifest["seeds"] != config["seeds"]
    if config_changed or seeds_changed:
        raise ValueError("冻结后的配置或 seeds 身份不一致")
    _require_frozen_source(manifest["git_sha"])
    _check_files(config, manifest["files"])
    if manifest["legacy"] != _legacy(config):
        raise ValueError("冻结后复用的 v1.1 训练身份被修改")
    if manifest["v12_source"] != _v12_source(
        config, manifest["legacy"]["shortcut"]["sha256"]
    ):
        raise ValueError("冻结后复用的 v1.2 pilot 身份被修改")
    if manifest["pilot_decision_sha256"] != sha256_file(
        _path(config, "reports") / "pilot_decision.json"
    ):
        raise ValueError("冻结后的 pilot 决策被修改")
    return manifest


def freeze(config_path: Path) -> dict:
    config = load_v13_config(config_path)
    root = _path(config, "reports")
    if (root / "freeze_manifest.json").exists():
        manifest = _frozen(config, config_path)
        return {"status": "already_frozen", "selected": manifest["selected"]}
    metadata = _prepared(config, config_path)
    decision = read_json(root / "pilot_decision.json")
    if decision["config_sha256"] != sha256_file(config_path) or decision.get("split") != "dev":
        raise ValueError("Pilot 与待冻结的代码、配置或数据分区不一致")
    expected = select_anchor_pilot(decision["before"], decision["after"], config["pilot"])
    if expected["selected"] != "sft_dpo_anchor" or decision["selected"] != expected["selected"]:
        raise ValueError("格式锚定未通过固定门槛；停止，不生成 test")
    _require_clean_git()
    _require_frozen_source(decision["git_sha"])
    test_path = _path(config, "data") / "test.jsonl"
    if test_path.exists():
        raise ValueError("发现未封存的既有 test，不覆盖")
    rows, audit = build_test(config)
    _write_jsonl(test_path, rows)
    manifest = {
        "version": "1.3",
        "git_sha": _git_sha(),
        "config_sha256": sha256_file(config_path),
        "model_revision": config["model"]["revision"],
        "legacy": metadata["legacy"],
        "v12_source": metadata["v12_source"],
        "seeds": config["seeds"],
        "selected": {
            "candidate": "sft_dpo_anchor",
            "method": "sft_dpo_anchor",
            "params": {
                "sft": config["sft"],
                "dpo": config["dpo"],
                "anchor": config["anchor"],
            },
        },
        "files": {
            **metadata["files"],
            "test.jsonl": {"sha256": sha256_file(test_path), "rows": len(rows)},
        },
        "test_audit": audit,
        "pilot_decision_sha256": sha256_file(root / "pilot_decision.json"),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "freeze_manifest.json", manifest)
    return {"status": "frozen", "selected": manifest["selected"], "test_rows": len(rows)}


def formal(config_path: Path) -> dict:
    config = load_v13_config(config_path)
    metadata = _frozen(config, config_path)
    for seed in config["seeds"]:
        _run_candidate(
            config, config_path, metadata, "formal", "score_sft", "score_sft", seed, config["sft"]
        )
        _run_candidate(
            config, config_path, metadata, "formal", "sft_dpo", "sft_dpo", seed, config["dpo"]
        )
        _run_anchor(config, config_path, metadata, "formal", seed)
    for seed in config["seeds"]:
        _evaluate_run(
            config, config_path, metadata, "formal", "score_sft", "score_sft", seed, "test"
        )
        sft_root = _path(config, "runs") / "formal/score_sft" / f"seed-{seed}"
        dpo_root = _path(config, "runs") / "formal/sft_dpo" / f"seed-{seed}"
        anchor_root = _path(config, "runs") / "formal/sft_dpo_anchor" / f"seed-{seed}"
        _evaluate_adapter(
            config,
            config_path,
            metadata,
            "formal",
            "sft_dpo_pre_anchor",
            seed,
            "test",
            sft_root / "merged",
            dpo_root / "final_adapter",
            sha256_file(dpo_root / "run_manifest.json"),
        )
        _evaluate_adapter(
            config,
            config_path,
            metadata,
            "formal",
            "selected_dpo",
            seed,
            "test",
            sft_root / "merged",
            anchor_root / "final_adapter",
            sha256_file(anchor_root / "run_manifest.json"),
        )
    _legacy_evaluations(config, config_path, metadata)
    return {"status": "formal_complete", "new_training_stages": 9, "test_evaluations": 17}


def _read_evaluation(
    root: Path, config: dict, config_path: Path, metadata: dict, label: str, seed
) -> list[dict]:
    manifest = read_json(root / "prediction_manifest.json")
    identity = manifest.get("identity", {})
    if (
        manifest.get("status") != "complete"
        or identity.get("split") != "test"
        or identity.get("data_sha256") != metadata["files"]["test.jsonl"]["sha256"]
        or identity.get("config_sha256") != sha256_file(config_path)
        or identity.get("git_sha") != metadata["git_sha"]
        or identity.get("model") != label
        or identity.get("seed") != seed
        or identity.get("model_dtype") != "float32"
        or identity.get("allow_tf32") is not False
        or identity.get("tie_policy") != "reject_with_context"
    ):
        raise ValueError(f"正式评测身份不一致：{root}")
    if label in {"score_sft", "sft_dpo_pre_anchor", "selected_dpo"}:
        candidates = {
            "score_sft": ("score_sft", "sft"),
            "sft_dpo_pre_anchor": ("sft_dpo", "dpo"),
            "selected_dpo": ("sft_dpo_anchor", "anchor"),
        }
        candidate, stage = candidates[label]
        run_root = _path(config, "runs") / "formal" / candidate / f"seed-{seed}"
        run_path = run_root / "run_manifest.json"
        run = read_json(run_path)
        data_name = stage
        expected_steps = (
            anchor_spec(config, seed, metadata["files"]["anchor.jsonl"]["rows"])[
                "optimizer_steps"
            ]
            if stage == "anchor"
            else training_spec(
                config,
                stage,
                seed,
                metadata["files"][f"{stage}.jsonl"]["rows"],
                config[stage],
            )["optimizer_steps"]
        )
        run_identity = run.get("identity", {})
        if (
            run.get("status") != "complete"
            or run.get("actual_optimizer_steps") != expected_steps
            or run_identity.get("git_sha") != metadata["git_sha"]
            or run_identity.get("config_sha256") != sha256_file(config_path)
            or run_identity.get("data_sha256") != metadata["files"][f"{data_name}.jsonl"][
                "sha256"
            ]
            or run_identity.get("phase") != "formal"
            or run_identity.get("candidate") != candidate
            or identity.get("run_manifest_sha256") != sha256_file(run_path)
        ):
            raise ValueError(f"正式预测与训练结果不匹配：{run_root}")
        sft_root = _path(config, "runs") / "formal/score_sft" / f"seed-{seed}"
        expected_source = str(run_root / "merged") if stage == "sft" else str(sft_root / "merged")
        expected_adapter = None if stage == "sft" else str(run_root / "final_adapter")
    else:
        expected_source = (
            str(Path(config["model"]["local_path"]))
            if label == "base"
            else str(_path(config, "shortcut") / "merged")
        )
        expected_adapter = None
        legacy_key = "shortcut" if label == "shortcut" else None
        if label.startswith("v1_1_"):
            method = label.removeprefix("v1_1_")
            legacy_key = f"{method}/{seed}"
            expected_adapter = str(
                _path(config, "legacy_dpo") / method / f"seed-{seed}" / "final_adapter"
            )
        expected_sha = metadata["legacy"][legacy_key]["sha256"] if legacy_key else None
        if identity.get("source_manifest_sha256") != expected_sha:
            raise ValueError(f"旧模型评测的来源不匹配：{root}")
    if identity.get("source") != expected_source or identity.get("adapter") != expected_adapter:
        raise ValueError(f"评测加载的模型或 adapter 路径不匹配：{root}")
    rows = _read_jsonl(root / "predictions.jsonl")
    if len(rows) != metadata["files"]["test.jsonl"]["rows"]:
        raise ValueError(f"正式评测缺行：{root}")
    return rows


def report(config_path: Path) -> dict:
    config = load_v13_config(config_path)
    metadata = _frozen(config, config_path)
    records = {}
    labels = (
        "base",
        "shortcut",
        "v1_1_control",
        "v1_1_repair",
        "score_sft",
        "sft_dpo_pre_anchor",
        "selected_dpo",
    )
    for label in labels:
        root = _path(config, "results") / "test" / label
        seeds = [None] if label in {"base", "shortcut"} else config["seeds"]
        records[label] = {
            seed if seed is not None else "single": _read_evaluation(
                root / f"seed-{seed}" if seed is not None else root,
                config,
                config_path,
                metadata,
                label,
                seed,
            )
            for seed in seeds
        }
    result = aggregate_v13(records, config)
    costs = []
    for path in sorted(_path(config, "runs").glob("*/*/seed-*/run_manifest.json")):
        run = read_json(path)
        if run["status"] != "complete":
            raise ValueError(f"存在未完成的训练，不能省略成本：{path}")
        costs.append(
            {
                **run,
                "phase": run["identity"]["phase"],
                "candidate": run["identity"]["candidate"],
                "learning_rate": run["params"]["learning_rate"],
            }
        )
    result.update(provenance=metadata, costs=costs)
    write_report(result, _path(config, "reports"))
    archive = _package(config, config_path, metadata)
    return {
        "status": "reported",
        "decision": result["decision"],
        "checks": result["checks"],
        "report": str(_path(config, "reports") / "RESULTS.md"),
        "archive": str(archive),
    }


def _package(config: dict, config_path: Path, metadata: dict) -> Path:
    root = _path(config, "artifacts")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"shortcut-repair-v1.3-{metadata['git_sha'][:12]}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(config_path, arcname="configs/v1_3.yaml")
        for name in ("V1_3_DESIGN.md", "V1_3_EXECUTION_PLAN.md"):
            tar.add(Path("docs") / name, arcname=f"docs/{name}")
        for name in (
            "prepare_manifest.json",
            "pilot_decision.json",
            "pilot_decision.md",
            "freeze_manifest.json",
            "RESULTS.md",
            "results.json",
            "metrics.csv",
            "costs.csv",
            "comparison.png",
        ):
            tar.add(_path(config, "reports") / name, arcname=f"reports/{name}")
        for source_name in ("runs", "results"):
            source = _path(config, source_name)
            allowed = {
                "run_manifest.json",
                "prediction_manifest.json",
                "metrics.json",
                "predictions.jsonl",
            }
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.name in allowed:
                    tar.add(path, arcname=f"{source_name}/{path.relative_to(source).as_posix()}")
        for label, entry in metadata["legacy"].items():
            tar.add(entry["path"], arcname=f"legacy/{label}/run_manifest.json")
        for label, entry in metadata["v12_source"].items():
            tar.add(entry["path"], arcname=f"v12_source/{label}/run_manifest.json")
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v1.3：在 v1.2 SFT → DPO 后追加一次低学习率格式锚定 SFT"
    )
    parser.add_argument("stage", choices=STAGES, help="执行阶段")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/v1_3.yaml"), help="v1.3 配置路径"
    )
    args = parser.parse_args(argv)
    result = {
        "prepare": prepare,
        "pilot": pilot,
        "freeze": freeze,
        "formal": formal,
        "report": report,
    }[args.stage](args.config)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
