"""Conditional A/B log-probability evaluation and audited aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from shortcut_repair.analysis import aggregate_formal, score_predictions, write_report
from shortcut_repair.data import canonical_json, load_config, sha256_file
from shortcut_repair.train import validate_dpo_contract


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def prediction_from_scores(logp_a: float, logp_b: float) -> str:
    """Choose the higher-probability answer and reject undefined ties."""

    if not all(
        isinstance(value, int | float) and math.isfinite(value)
        for value in (logp_a, logp_b)
    ):
        raise ValueError("Candidate log probabilities must be finite numbers")
    if logp_a == logp_b:
        raise ValueError("Candidate log probabilities are exactly equal")
    return "A" if logp_a > logp_b else "B"


def build_prediction_record(
    row: dict[str, Any], logp_a: float, logp_b: float
) -> dict[str, Any]:
    """Attach an A/B log-probability decision without retaining the full prompt."""

    prediction = prediction_from_scores(logp_a, logp_b)
    gold = row["gold"]
    correct_margin = (logp_a - logp_b) if gold == "A" else (logp_b - logp_a)
    return {
        key: row[key]
        for key in ("case_id", "split", "variant", "gold", "hint")
        if key in row
    } | {
        "logp_A": float(logp_a),
        "logp_B": float(logp_b),
        "prediction": prediction,
        "correct": prediction == gold,
        "correct_margin": float(correct_margin),
    }


def validate_test_seal(config_path: Path | str) -> dict[str, Any]:
    """Verify the test bytes and current config against the immutable seal."""

    config_path = Path(config_path)
    config = load_config(config_path)
    data_dir = Path(config["paths"]["data_dir"])
    test_path = data_dir / "test.jsonl"
    seal_path = data_dir / "manifest_test.json"
    if not seal_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Sealed test.jsonl and manifest_test.json are required")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        raise ValueError("Test manifest is not sealed")
    if seal.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Sealed test config checksum does not match the current config")
    file_entry = seal.get("files", {}).get("test.jsonl", {})
    if file_entry.get("sha256") != sha256_file(test_path):
        raise ValueError("Sealed test data checksum does not match test.jsonl")
    expected_rows = config["data"]["test_cases"] * 2
    if file_entry.get("rows") != expected_rows:
        raise ValueError("Sealed test row count does not match the config")
    return seal


def _conditional_scores(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    torch_module: Any,
) -> list[tuple[float, float]]:
    jobs: list[tuple[list[int], int, list[int]]] = []
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("Evaluation prompt tokenization is empty")
        for completion in ("A", "B"):
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                raise ValueError(f"Completion {completion} tokenization is empty")
            full_ids = [*prompt_ids, *completion_ids]
            if len(full_ids) > max_length:
                raise ValueError(
                    f"Evaluation sequence has {len(full_ids)} tokens; max_length={max_length}"
                )
            jobs.append((full_ids, len(prompt_ids), completion_ids))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad token")
    maximum = max(len(full_ids) for full_ids, _, _ in jobs)
    input_ids = torch_module.full(
        (len(jobs), maximum),
        fill_value=pad_id,
        dtype=torch_module.long,
        device=model.device,
    )
    attention_mask = torch_module.zeros_like(input_ids)
    for index, (full_ids, _, _) in enumerate(jobs):
        length = len(full_ids)
        input_ids[index, :length] = torch_module.tensor(
            full_ids, dtype=torch_module.long, device=model.device
        )
        attention_mask[index, :length] = 1
    with torch_module.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch_module.log_softmax(logits, dim=-1)
    job_scores = []
    for index, (_, prompt_length, completion_ids) in enumerate(jobs):
        score = 0.0
        for offset, token_id in enumerate(completion_ids):
            score += float(log_probs[index, prompt_length + offset - 1, token_id].item())
        job_scores.append(score)
    return [
        (job_scores[index], job_scores[index + 1])
        for index in range(0, len(job_scores), 2)
    ]


def evaluate_checkpoint(args: Any) -> dict[str, Any]:
    """Run batched teacher-forced A/B scoring for one checkpoint."""

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.split == "test":
        validate_test_seal(config_path)
    data_path = Path(config["paths"]["data_dir"]) / f"{args.split}.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing evaluation data: {data_path}")
    if args.model == "shortcut":
        if args.method is not None or args.seed is not None or args.adapter_path is not None:
            raise ValueError("Shortcut evaluation must not specify method, seed, or adapter")
        method = "shortcut"
        base_path = Path(args.model_path) if args.model_path else (
            Path(config["paths"]["shortcut_dir"]) / "merged"
        )
        adapter_path = None
    elif args.model == "adapter":
        if args.method not in {"control", "repair"} or args.seed not in config["dpo"]["seeds"]:
            raise ValueError("Adapter evaluation requires a formal method and seed")
        method = args.method
        base_path = Path(args.model_path) if args.model_path else (
            Path(config["paths"]["shortcut_dir"]) / "merged"
        )
        adapter_path = Path(args.adapter_path) if args.adapter_path else (
            Path(config["paths"]["dpo_runs_dir"])
            / method
            / f"seed-{args.seed}"
            / "final_adapter"
        )
    else:
        raise ValueError("model must be shortcut or adapter")
    if not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing merged shortcut model at {base_path}")
    if adapter_path is not None and not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing DPO adapter at {adapter_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; evaluation requires the A6000")
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    rows = _read_jsonl(data_path)
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["prompt_messages"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        scores = _conditional_scores(
            model,
            tokenizer,
            prompts,
            config["model"]["max_length"],
            torch,
        )
        predictions.extend(
            build_prediction_record(row, logp_a, logp_b)
            for row, (logp_a, logp_b) in zip(batch, scores, strict=True)
        )
    output_dir = Path(args.output_dir)
    prediction_path = output_dir / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    metrics = score_predictions(predictions)
    _write_json(output_dir / "metrics.json", metrics)
    manifest = {
        "status": "complete",
        "split": args.split,
        "method": method,
        "seed": args.seed,
        "base_model_path": str(base_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config_sha256": sha256_file(config_path),
        "prediction_rows": len(predictions),
        "predictions_sha256": sha256_file(prediction_path),
    }
    _write_json(output_dir / "prediction_manifest.json", manifest)
    result = {"status": "complete", "metrics": metrics, "manifest": manifest}
    print(canonical_json(result))
    return result


def _load_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_from_artifacts(
    config_path: Path | str, output_dir: Path | str | None = None
) -> dict[str, Any]:
    """Audit six formal runs and aggregate the sealed matched comparison."""

    config_path = Path(config_path)
    config = load_config(config_path)
    validate_test_seal(config_path)
    config_sha = sha256_file(config_path)
    data_dir = Path(config["paths"]["data_dir"])
    test_path = data_dir / "test.jsonl"
    test_sha = sha256_file(test_path)
    expected_prediction_rows = config["data"]["test_cases"] * 2
    results_root = Path(config["paths"]["results_dir"])
    runs_root = Path(config["paths"]["dpo_runs_dir"])
    records: dict[str, dict[int, list[dict[str, Any]]]] = {
        "control": {},
        "repair": {},
    }
    initial_checksums: dict[int, dict[str, str]] = {
        seed: {} for seed in config["dpo"]["seeds"]
    }
    for method in ("control", "repair"):
        train_sha = sha256_file(data_dir / f"dpo_{method}.jsonl")
        for seed in config["dpo"]["seeds"]:
            run_dir = runs_root / method / f"seed-{seed}"
            adapter_dir = run_dir / "final_adapter"
            run_manifest = _load_manifest(run_dir / "run_manifest.json", "run manifest")
            contract = validate_dpo_contract(
                config,
                method,
                seed,
                rows=config["dpo"]["expected_rows"],
                smoke=False,
            )
            if (
                run_manifest.get("status") != "complete"
                or run_manifest.get("method") != method
                or run_manifest.get("config_sha256") != config_sha
                or run_manifest.get("data_sha256") != train_sha
                or run_manifest.get("actual_optimizer_steps") != contract["optimizer_steps"]
                or run_manifest.get("contract") != contract
                or Path(run_manifest.get("final_adapter", "")) != adapter_dir
            ):
                raise ValueError(f"Invalid formal run manifest for {method} seed {seed}")
            initial_checksums[seed][method] = run_manifest.get(
                "initial_adapter_checksum", ""
            )
            result_dir = results_root / "test" / method / f"seed-{seed}"
            prediction_path = result_dir / "predictions.jsonl"
            prediction_manifest = _load_manifest(
                result_dir / "prediction_manifest.json", "prediction manifest"
            )
            if prediction_manifest.get("predictions_sha256") != sha256_file(prediction_path):
                raise ValueError(f"Formal prediction checksum changed for {method} seed {seed}")
            if (
                prediction_manifest.get("status") != "complete"
                or prediction_manifest.get("split") != "test"
                or prediction_manifest.get("method") != method
                or prediction_manifest.get("seed") != seed
                or prediction_manifest.get("adapter_path") != str(adapter_dir)
                or prediction_manifest.get("data_sha256") != test_sha
                or prediction_manifest.get("config_sha256") != config_sha
                or prediction_manifest.get("prediction_rows") != expected_prediction_rows
            ):
                raise ValueError(f"Invalid prediction manifest for {method} seed {seed}")
            records[method][seed] = _read_jsonl(prediction_path)
    for seed, checksums in initial_checksums.items():
        if not checksums.get("control") or checksums["control"] != checksums.get("repair"):
            raise ValueError(f"Control and repair initial adapter checksums differ for seed {seed}")
    result = aggregate_formal(records["control"], records["repair"], config["evaluation"])
    result["provenance"] = {
        "config_sha256": config_sha,
        "test_data_sha256": test_sha,
        "control_training_data_sha256": sha256_file(data_dir / "dpo_control.jsonl"),
        "repair_training_data_sha256": sha256_file(data_dir / "dpo_repair.jsonl"),
        "formal_training_runs": 6,
        "initial_adapter_checksums": initial_checksums,
    }
    report_dir = Path(output_dir) if output_dir else Path(config["paths"]["reports_dir"])
    write_report(result, report_dir)
    return result
