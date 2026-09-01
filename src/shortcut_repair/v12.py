"""五阶段入口：prepare → pilot → freeze → formal → report。"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from shortcut_repair.analysis import classify_mechanism_gate
from shortcut_repair.data import canonical_json, sha256_file
from shortcut_repair.evaluate import _read_jsonl, _write_json, _write_jsonl
from shortcut_repair.train import REPOSITORY_ROOT, _git_sha, _package_versions
from shortcut_repair.v12_analysis import aggregate_results, select_pilot, write_report
from shortcut_repair.v12_data import build_test, load_v12_config, prepare_data
from shortcut_repair.v12_runtime import (
    evaluate_model,
    read_json,
    require_model,
    train_model,
    training_spec,
)

STAGES = ("prepare", "pilot", "freeze", "formal", "report")


def _path(config: dict, name: str) -> Path:
    return Path(config["paths"][f"{name}_dir"])


def _legacy(config: dict, *, check_models: bool = False) -> dict:
    """复用旧 manifest 身份和已有权重；不重新诱导或重算权重 SHA。"""
    if check_models:
        require_model(config["model"]["local_path"])
    shortcut_path = _path(config, "shortcut") / "run_manifest.json"
    shortcut = read_json(shortcut_path)
    if (
        shortcut.get("status") != "complete"
        or shortcut.get("contract", {}).get("model_revision") != config["model"]["revision"]
        or shortcut.get("actual_optimizer_steps") != shortcut["contract"]["optimizer_steps"]
        or not shortcut.get("merged_model_weights_sha256")
    ):
        raise ValueError("v1.1 Shortcut manifest 未完成或模型 revision 不符")
    if check_models:
        require_model(_path(config, "shortcut") / "merged")
    manifests = {"shortcut": {"path": str(shortcut_path), "sha256": sha256_file(shortcut_path)}}
    for method in ("control", "repair"):
        for seed in config["seeds"]:
            root = _path(config, "legacy_dpo") / method / f"seed-{seed}"
            path = root / "run_manifest.json"
            run = read_json(path)
            contract = run.get("contract", {})
            if (
                run.get("status") != "complete"
                or run.get("method") != method
                or contract.get("seed") != seed
                or contract.get("model_revision") != config["model"]["revision"]
                or run.get("actual_optimizer_steps") != contract.get("optimizer_steps")
                or run.get("shortcut_model_weights_sha256")
                != shortcut.get("merged_model_weights_sha256")
            ):
                raise ValueError(f"v1.1 {method} seed {seed} 的训练身份不一致")
            if check_models:
                require_model(root / "final_adapter", adapter=True)
            manifests[f"{method}/{seed}"] = {"path": str(path), "sha256": sha256_file(path)}
    return manifests


def _check_files(config: dict, entries: dict) -> None:
    for name, entry in entries.items():
        path = _path(config, "data") / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"数据被修改或缺失：{path}")


def _prepared(config: dict, config_path: Path) -> dict:
    prepared = read_json(_path(config, "reports") / "prepare_manifest.json")
    if prepared["config_sha256"] != sha256_file(config_path):
        raise ValueError("配置已改变；不要混用已有的 pilot 或冻结结果")
    _check_files(config, prepared["files"])
    if prepared["legacy"] != _legacy(config):
        raise ValueError("复用的 v1.1 manifest 身份发生变化")
    if prepared["shortcut_sanity"]["decision"] != "pass":
        raise ValueError("Shortcut dev sanity 未通过，不能进入 pilot")
    return prepared


def _identity(config_path: Path, metadata: dict, data_name: str, **extra) -> dict:
    return {
        "git_sha": metadata.get("git_sha") or _git_sha(),
        "config_sha256": sha256_file(config_path),
        "data_sha256": metadata["files"][f"{data_name}.jsonl"]["sha256"],
        **extra,
    }


def _no_test_yet(config: dict) -> None:
    if (_path(config, "data") / "test.jsonl").exists() or (
        _path(config, "reports") / "freeze_manifest.json"
    ).exists():
        raise ValueError("已进入 sealed test 阶段；禁止重新 prepare/pilot 或据此调参")


def prepare(config_path: Path) -> dict:
    config = load_v12_config(config_path)
    _no_test_yet(config)
    legacy = _legacy(config, check_models=True)
    data = prepare_data(config)
    metrics = evaluate_model(
        config,
        _path(config, "data") / "dev.jsonl",
        _path(config, "shortcut") / "merged",
        None,
        _path(config, "results") / "dev/shortcut",
        _identity(
            config_path,
            data,
            "dev",
            split="dev",
            model="shortcut",
            source_manifest_sha256=legacy["shortcut"]["sha256"],
        ),
    )
    sanity = classify_mechanism_gate(metrics["overall"], config["evaluation"]["sanity"])
    result = {
        "config_sha256": sha256_file(config_path),
        **data,
        "legacy": legacy,
        "shortcut_sanity": sanity,
        "shortcut_metrics": metrics,
        "package_versions": _package_versions(),
    }
    _write_json(_path(config, "reports") / "prepare_manifest.json", result)
    if sanity["decision"] != "pass":
        raise RuntimeError("Shortcut 在新 dev 上的 sanity 未通过；先检查复用模型，不重做诱导")
    return {
        "status": "prepared",
        "sanity": sanity,
        "row_counts": {name: entry["rows"] for name, entry in data["files"].items()},
    }


def _run_candidate(
    config: dict,
    config_path: Path,
    metadata: dict,
    phase: str,
    candidate: str,
    method: str,
    seed: int,
    params: dict,
) -> dict:
    stage = "sft" if method == "score_sft" else "dpo"
    source = _path(config, "shortcut") / "merged"
    parent_sha = metadata["legacy"]["shortcut"]["sha256"]
    if method == "sft_dpo":
        sft_root = _path(config, "runs") / phase / "score_sft" / f"seed-{seed}"
        if read_json(sft_root / "run_manifest.json")["status"] != "complete":
            raise ValueError("链式 DPO 必须使用同 seed 已完成的 SFT")
        source = sft_root / "merged"
        parent_sha = sha256_file(sft_root / "run_manifest.json")
    root = _path(config, "runs") / phase / candidate / f"seed-{seed}"
    return train_model(
        config,
        stage,
        seed,
        source,
        root,
        _identity(
            config_path,
            metadata,
            stage,
            phase=phase,
            candidate=candidate,
            parent_manifest_sha256=parent_sha,
        ),
        params,
    )


def _evaluate_run(
    config: dict,
    config_path: Path,
    metadata: dict,
    phase: str,
    candidate: str,
    method: str,
    seed: int,
    split: str,
) -> dict:
    root = _path(config, "runs") / phase / candidate / f"seed-{seed}"
    manifest = read_json(root / "run_manifest.json")
    source = root / "merged" if method == "score_sft" else Path(manifest["starting_model"])
    adapter = None if method == "score_sft" else root / "final_adapter"
    output = _path(config, "results") / ("pilot" if split == "dev" else "test")
    label = candidate if split == "dev" or method == "score_sft" else "selected_dpo"
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
            run_manifest_sha256=sha256_file(root / "run_manifest.json"),
        ),
    )


def pilot(config_path: Path) -> dict:
    config = load_v12_config(config_path)
    _no_test_yet(config)
    metadata = _prepared(config, config_path)
    candidates = {}

    def run(name: str, method: str, params: dict) -> None:
        _run_candidate(config, config_path, metadata, "pilot", name, method, 42, params)
        metrics = _evaluate_run(config, config_path, metadata, "pilot", name, method, 42, "dev")
        candidates[name] = {
            "method": method,
            "params": params,
            "metrics": metrics,
            "stages": 2 if method == "sft_dpo" else 1,
        }

    run("direct_dpo", "direct_dpo", config["dpo"])
    run("score_sft", "score_sft", config["sft"])
    run("sft_dpo", "sft_dpo", config["dpo"])
    decision = select_pilot(candidates, config["pilot"])
    adjusted = decision["all_failed_retention"]
    if adjusted:
        # 唯一允许的自动补充轮：只将 DPO 学习率减半，不重训 SFT、不改数据。
        params = {**config["dpo"], "learning_rate": config["dpo"]["learning_rate"] / 2}
        run("direct_dpo_lr_half", "direct_dpo", params)
        run("sft_dpo_lr_half", "sft_dpo", params)
        decision = select_pilot(candidates, config["pilot"])
    result = {
        **decision,
        "candidates": candidates,
        "adjustment_used": adjusted,
        "git_sha": _git_sha(),
        "config_sha256": sha256_file(config_path),
        "split": "dev",
    }
    root = _path(config, "reports")
    _write_json(root / "pilot_decision.json", result)
    lines = [
        "# v1.2 Pilot 选择",
        "",
        f"选定候选：`{decision['selected']}`。{decision['reason']}",
        "",
        "| 候选 | 保留条件 | score fresh response | score nuisance | 阶段数 |",
        "|---|---|---:|---:|---:|",
    ]
    for name, candidate in candidates.items():
        score = candidate["metrics"]["decision_type"]["score_decisive"]
        lines.append(
            f"| {name} | {'通过' if decision['eligible'][name] else '未通过'} | "
            f"{score['fresh_result_response_rate']:.4f} | "
            f"{score['nuisance_invariance_rate']:.4f} | {candidate['stages']} |"
        )
    lines += [
        "",
        "仅使用 seed 42 和 dev；没有读取正式 test。",
        f"是否使用唯一补充轮（DPO 学习率减半、最多两次）：{adjusted}。",
        "SFT 为能力基线；即使优于 DPO，也不把 SFT 改称为 DPO。",
    ]
    (root / "pilot_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "selected" if decision["selected"] else "no_eligible_dpo",
        **decision,
        "candidate_count": len(candidates),
    }


def _require_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("正式冻结前请提交代码和配置；只在这里要求 Git 工作树 clean")


def _require_frozen_source(git_sha: str) -> None:
    # 仅检查实验源文件，不把生成物或文档的改动升级为全局 clean 门禁。
    result = subprocess.run(
        ["git", "diff", "--quiet", git_sha, "--", "src/shortcut_repair"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode:
        raise ValueError("冻结后实验源代码被修改；请先回到冻结代码，不得混用结果")


def _frozen(config: dict, config_path: Path) -> dict:
    manifest = read_json(_path(config, "reports") / "freeze_manifest.json")
    if (
        manifest["config_sha256"] != sha256_file(config_path)
        or manifest["seeds"] != config["seeds"]
    ):
        raise ValueError("冻结后的代码、配置或 seeds 身份不一致")
    _require_frozen_source(manifest["git_sha"])
    _check_files(config, manifest["files"])
    if manifest["legacy"] != _legacy(config):
        raise ValueError("冻结后复用的 v1.1 训练身份被修改")
    if manifest["pilot_decision_sha256"] != sha256_file(
        _path(config, "reports") / "pilot_decision.json"
    ):
        raise ValueError("冻结后的 pilot 决策被修改")
    return manifest


def freeze(config_path: Path) -> dict:
    config = load_v12_config(config_path)
    root = _path(config, "reports")
    if (root / "freeze_manifest.json").exists():
        manifest = _frozen(config, config_path)
        return {"status": "already_frozen", "selected": manifest["selected"]}
    metadata = _prepared(config, config_path)
    decision = read_json(root / "pilot_decision.json")
    if (
        decision["config_sha256"] != sha256_file(config_path)
        or decision.get("split") != "dev"
    ):
        raise ValueError("Pilot 与待冻结的代码、配置或数据分区不一致")
    expected = select_pilot(decision["candidates"], config["pilot"])
    if not expected["selected"] or decision["selected"] != expected["selected"]:
        raise ValueError("没有符合预定选择规则的 DPO；停止，不生成 test")
    _require_clean_git()
    _require_frozen_source(decision["git_sha"])
    test_path = _path(config, "data") / "test.jsonl"
    if test_path.exists():
        raise ValueError("发现未封存的既有 test，不覆盖，也不据此重新选择模型")
    rows, audit = build_test(config)
    _write_jsonl(test_path, rows)
    candidate = decision["candidates"][decision["selected"]]
    manifest = {
        "version": "1.2",
        "git_sha": _git_sha(),
        "config_sha256": sha256_file(config_path),
        "model_revision": config["model"]["revision"],
        "legacy": metadata["legacy"],
        "seeds": config["seeds"],
        "selected": {
            "candidate": decision["selected"],
            "method": candidate["method"],
            "params": candidate["params"],
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


def _legacy_evaluations(config: dict, config_path: Path, metadata: dict) -> None:
    shortcut = _path(config, "shortcut") / "merged"
    specs = [
        ("base", None, Path(config["model"]["local_path"]), None, None),
        ("shortcut", None, shortcut, None, "shortcut"),
    ]
    specs += [
        (
            f"v1_1_{method}",
            seed,
            shortcut,
            _path(config, "legacy_dpo") / method / f"seed-{seed}" / "final_adapter",
            f"{method}/{seed}",
        )
        for method in ("control", "repair")
        for seed in config["seeds"]
    ]
    for label, seed, source, adapter, legacy_key in specs:
        output = _path(config, "results") / "test" / label
        if seed is not None:
            output /= f"seed-{seed}"
        evaluate_model(
            config,
            _path(config, "data") / "test.jsonl",
            source,
            adapter,
            output,
            _identity(
                config_path,
                metadata,
                "test",
                split="test",
                model=label,
                seed=seed,
                source_manifest_sha256=(
                    metadata["legacy"][legacy_key]["sha256"] if legacy_key else None
                ),
            ),
        )


def formal(config_path: Path) -> dict:
    config = load_v12_config(config_path)
    metadata = _frozen(config, config_path)
    selected = metadata["selected"]
    for seed in config["seeds"]:
        _run_candidate(
            config, config_path, metadata, "formal", "score_sft", "score_sft", seed, config["sft"]
        )
        _run_candidate(
            config,
            config_path,
            metadata,
            "formal",
            "selected_dpo",
            selected["method"],
            seed,
            selected["params"],
        )
    # 全部训练完成后才读 test；不让早期 test 分数影响后续 seed 的训练。
    for seed in config["seeds"]:
        _evaluate_run(
            config, config_path, metadata, "formal", "score_sft", "score_sft", seed, "test"
        )
        _evaluate_run(
            config,
            config_path,
            metadata,
            "formal",
            "selected_dpo",
            selected["method"],
            seed,
            "test",
        )
    _legacy_evaluations(config, config_path, metadata)
    return {"status": "formal_complete", "new_training_stages": 6, "test_evaluations": 14}


def _read_evaluation(
    root: Path, config: dict, config_path: Path, metadata: dict, label: str, seed
) -> list[dict]:
    manifest = read_json(root / "prediction_manifest.json")
    identity = manifest["identity"]
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
    if label in {"score_sft", "selected_dpo"}:
        run_root = _path(config, "runs") / "formal" / label / f"seed-{seed}"
        run_path = run_root / "run_manifest.json"
        run = read_json(run_path)
        stage = "sft" if label == "score_sft" else "dpo"
        method = "score_sft" if label == "score_sft" else metadata["selected"]["method"]
        params = config["sft"] if stage == "sft" else metadata["selected"]["params"]
        source = _path(config, "shortcut") / "merged"
        parent_sha = metadata["legacy"]["shortcut"]["sha256"]
        if method == "sft_dpo":
            sft_root = _path(config, "runs") / "formal/score_sft" / f"seed-{seed}"
            source = sft_root / "merged"
            parent_sha = sha256_file(sft_root / "run_manifest.json")
        expected_run = {
            **_identity(
                config_path,
                metadata,
                stage,
                phase="formal",
                candidate=label,
                parent_manifest_sha256=parent_sha,
            ),
            "spec": training_spec(
                config, stage, seed, metadata["files"][f"{stage}.jsonl"]["rows"], params
            ),
            "starting_model": str(source),
        }
        if (
            run.get("status") != "complete"
            or run.get("identity") != expected_run
            or run.get("actual_optimizer_steps") != expected_run["spec"]["optimizer_steps"]
            or identity.get("run_manifest_sha256") != sha256_file(run_path)
        ):
            raise ValueError(f"正式预测与训练结果不匹配：{run_root}")
        expected_source = str(run_root / "merged") if stage == "sft" else str(source)
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
    config = load_v12_config(config_path)
    metadata = _frozen(config, config_path)
    records = {}
    for label in ("base", "shortcut", "v1_1_control", "v1_1_repair", "score_sft", "selected_dpo"):
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
    result = aggregate_results(records, config)
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
    archive = root / f"shortcut-repair-v1.2-{metadata['git_sha'][:12]}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(config_path, arcname="configs/v1_2.yaml")
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
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v1.2 精简流程：复用 v1.1，先 dev 选型，再封存 test"
    )
    parser.add_argument("stage", choices=STAGES, help="执行阶段")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/v1_2.yaml"), help="v1.2 配置路径"
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
    return 0  # 合法的负结果不是进程故障。


if __name__ == "__main__":
    raise SystemExit(main())
