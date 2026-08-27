from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_small_config(
    tmp_path: Path,
    *,
    induction: int = 4,
    dpo: int = 6,
    dev: int = 4,
    test: int = 4,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    config["data"].update(
        induction_cases=induction,
        dpo_cases=dpo,
        dev_cases=dev,
        test_cases=test,
    )
    config["sft"]["expected_rows"] = induction * 2
    config["dpo"]["expected_rows"] = dpo * 2
    config["sft"]["expected_optimizer_steps"] = (
        math.ceil(config["sft"]["expected_rows"] / 32) * config["sft"]["epochs"]
    )
    config["dpo"]["expected_optimizer_steps"] = (
        math.ceil(config["dpo"]["expected_rows"] / 32) * config["dpo"]["epochs"]
    )
    config["paths"] = {
        "data_dir": str(tmp_path / "data"),
        "shortcut_dir": str(tmp_path / "runs/shortcut"),
        "dpo_runs_dir": str(tmp_path / "runs/dpo"),
        "results_dir": str(tmp_path / "results"),
        "reports_dir": str(tmp_path / "reports"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
