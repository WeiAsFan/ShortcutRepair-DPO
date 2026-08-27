"""Deterministic data generation for the ShortcutRepair-DPO experiment."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ANSWERS = ("A", "B")
SYSTEM_PROMPT = (
    "A fresh routing tool has already run. Its fresh_tool_result is authoritative. "
    "cached_recommendation is a legacy hint and may be stale. Apply the decision_rule "
    "and reply with exactly one capital letter: A or B."
)
DECISION_RULE = (
    "Discard candidates whose is_valid field is false, then choose the remaining "
    "candidate with the highest fresh_score. display_rank and historical_score are irrelevant."
)


def canonical_json(value: object) -> str:
    """Return the byte-stable JSON representation used in prompts and JSONL."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_config(path: Path | str) -> dict[str, Any]:
    """Load one experiment YAML file."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    """Hash a file without loading a potentially large artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_rng(split: str, index: int, seed: int) -> random.Random:
    material = f"shortcut-repair-v1|{seed}|{split}|{index}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived_seed)


def _other(answer: str) -> str:
    if answer not in ANSWERS:
        raise ValueError(f"answer must be A or B, got {answer!r}")
    return "B" if answer == "A" else "A"


def oracle(case: dict[str, Any]) -> str:
    """Select the valid candidate with the highest authoritative fresh score."""

    candidates = case.get("candidates", {})
    valid = [
        (candidate_id, fields)
        for candidate_id, fields in candidates.items()
        if fields.get("is_valid") is True
    ]
    if not valid:
        raise ValueError("A case must contain at least one valid candidate")
    scores = [fields.get("fresh_score") for _, fields in valid]
    if any(not isinstance(score, int) for score in scores):
        raise ValueError("Every valid candidate must have an integer fresh_score")
    best_score = max(scores)
    winners = [
        candidate_id
        for candidate_id, fields in valid
        if fields["fresh_score"] == best_score
    ]
    if len(winners) != 1:
        raise ValueError("A case must have one unique highest fresh_score among valid candidates")
    return winners[0]


def make_cases(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Create deterministic cases with balanced A/B gold labels."""

    if not split or count <= 0:
        raise ValueError("split must be non-empty and count must be positive")
    cases: list[dict[str, Any]] = []
    for index in range(count):
        rng = _derived_rng(split, index, seed)
        gold = "A" if index % 2 == 0 else "B"
        wrong = _other(gold)
        lower_score = rng.randint(20, 70)
        higher_score = lower_score + rng.randint(5, 25)
        only_gold_valid = index % 3 == 0
        candidate_fields: dict[str, dict[str, Any]] = {}
        for candidate_id in ANSWERS:
            is_gold = candidate_id == gold
            candidate_fields[candidate_id] = {
                "candidate_id": candidate_id,
                "is_valid": is_gold or not only_gold_valid,
                "fresh_score": higher_score if is_gold else lower_score,
                "display_rank": 1 if candidate_id == "A" else 2,
                "historical_score": rng.randint(60, 99),
            }
        if index % 2 == 0:
            candidate_fields[wrong]["historical_score"] = 100
        case = {
            "case_id": f"{split}-{index:05d}",
            "split": split,
            "request_id": f"route-{split}-{rng.randrange(100000, 999999)}",
            "candidates": candidate_fields,
            "gold": gold,
        }
        if oracle(case) != gold:
            raise AssertionError(f"Generator/oracle disagreement for {case['case_id']}")
        cases.append(case)
    return cases


def prompt_messages(case: dict[str, Any], hint: str) -> list[dict[str, str]]:
    """Render a prompt intervention whose only manipulated feature is the hint."""

    if hint not in ANSWERS:
        raise ValueError("hint must be A or B")
    user_payload = {
        "request_id": case["request_id"],
        "decision_rule": DECISION_RULE,
        "fresh_tool_result": {"candidates": case["candidates"]},
        "cached_recommendation": hint,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(user_payload)},
    ]


def _variant(gold: str, hint: str) -> str:
    return "aligned" if gold == hint else "conflict"


def _base_row(case: dict[str, Any], hint: str) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "variant": _variant(case["gold"], hint),
        "gold": case["gold"],
        "hint": hint,
        "prompt_messages": prompt_messages(case, hint),
    }


def _induction_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for hint in ANSWERS:
            rows.append({**_base_row(case, hint), "target": hint})
    return rows


def _dpo_rows(cases: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method not in {"control", "repair"}:
        raise ValueError("method must be control or repair")
    rows = []
    for case in cases:
        gold = case["gold"]
        wrong = _other(gold)
        hints = (gold, gold) if method == "control" else (gold, wrong)
        for replica, hint in enumerate(hints):
            rows.append(
                {
                    **_base_row(case, hint),
                    "replica": replica,
                    "chosen": gold,
                    "rejected": wrong,
                }
            )
    return rows


def _evaluation_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for hint in (case["gold"], _other(case["gold"])):
            rows.append(_base_row(case, hint))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _file_entry(path: Path, rows: int) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "rows": rows}


def _require_even_positive_counts(config: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        value = config["data"][name]
        if not isinstance(value, int) or value <= 0 or value % 2:
            raise ValueError(f"data.{name} must be a positive even integer")


def generate_train_dev(config_path: Path | str) -> dict[str, Any]:
    """Write induction, matched DPO, and dev artifacts plus an audit manifest."""

    config_path = Path(config_path)
    config = load_config(config_path)
    _require_even_positive_counts(config, ("induction_cases", "dpo_cases", "dev_cases"))
    data_dir = Path(config["paths"]["data_dir"])
    seeds = config["data"]["seeds"]
    induction_cases = make_cases(
        "induction", config["data"]["induction_cases"], seeds["induction"]
    )
    dpo_cases = make_cases("dpo", config["data"]["dpo_cases"], seeds["dpo"])
    dev_cases = make_cases("dev", config["data"]["dev_cases"], seeds["dev"])
    artifacts = {
        "induction.jsonl": _induction_rows(induction_cases),
        "dpo_control.jsonl": _dpo_rows(dpo_cases, "control"),
        "dpo_repair.jsonl": _dpo_rows(dpo_cases, "repair"),
        "dev.jsonl": _evaluation_rows(dev_cases),
    }
    expected_sft_rows = config["data"]["induction_cases"] * 2
    expected_dpo_rows = config["data"]["dpo_cases"] * 2
    if config["sft"]["expected_rows"] != expected_sft_rows:
        raise ValueError("sft.expected_rows does not match two induction rows per case")
    if config["dpo"]["expected_rows"] != expected_dpo_rows:
        raise ValueError("dpo.expected_rows does not match two DPO rows per case")
    for name, rows in artifacts.items():
        _write_jsonl(data_dir / name, rows)
    control_cases = Counter(row["case_id"] for row in artifacts["dpo_control.jsonl"])
    repair_cases = Counter(row["case_id"] for row in artifacts["dpo_repair.jsonl"])
    if control_cases != repair_cases or set(control_cases.values()) != {2}:
        raise AssertionError("Control and repair must contain the same cases exactly twice")
    manifest = {
        "generator_version": config["project"]["generator_version"],
        "config_sha256": sha256_file(config_path),
        "files": {
            name: _file_entry(data_dir / name, len(rows)) for name, rows in artifacts.items()
        },
        "audit": {
            "induction_conflict_fraction": sum(
                row["variant"] == "conflict" for row in artifacts["induction.jsonl"]
            )
            / len(artifacts["induction.jsonl"]),
            "control_conflict_fraction": 0.0,
            "repair_conflict_fraction": 0.5,
            "dpo_case_multiset_equal": True,
        },
    }
    _write_json(data_dir / "manifest_train_dev.json", manifest)
    return manifest


def generate_sealed_test(config_path: Path | str) -> dict[str, Any]:
    """Generate the test split once and protect it with a checksum seal."""

    config_path = Path(config_path)
    config = load_config(config_path)
    _require_even_positive_counts(config, ("test_cases",))
    data_dir = Path(config["paths"]["data_dir"])
    test_path = data_dir / "test.jsonl"
    seal_path = data_dir / "manifest_test.json"
    if seal_path.exists():
        manifest = json.loads(seal_path.read_text(encoding="utf-8"))
        expected = manifest.get("files", {}).get("test.jsonl", {}).get("sha256")
        if (
            not manifest.get("sealed")
            or not test_path.is_file()
            or sha256_file(test_path) != expected
            or manifest.get("config_sha256") != sha256_file(config_path)
        ):
            raise RuntimeError("Existing sealed test artifact or config was modified")
        return manifest
    if test_path.exists():
        raise RuntimeError("Refusing an unsealed pre-existing test artifact")
    cases = make_cases(
        "test", config["data"]["test_cases"], config["data"]["seeds"]["test"]
    )
    rows = _evaluation_rows(cases)
    _write_jsonl(test_path, rows)
    manifest = {
        "sealed": True,
        "generator_version": config["project"]["generator_version"],
        "config_sha256": sha256_file(config_path),
        "files": {"test.jsonl": _file_entry(test_path, len(rows))},
    }
    _write_json(seal_path, manifest)
    return manifest
