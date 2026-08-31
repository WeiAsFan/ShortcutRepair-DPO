"""Conditional A/B log-probability evaluation and audited aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from shortcut_repair.analysis import (
    METRIC_NAMES,
    aggregate_formal,
    classify_mechanism_gate,
    score_predictions,
    write_report,
)
from shortcut_repair.data import canonical_json, load_config, sha256_file
from shortcut_repair.train import (
    _git_sha,
    _trainer_budget,
    sha256_model_weights,
    validate_counterfactual_sft_contract,
    validate_dpo_contract,
    validate_sft_contract,
)

DEFAULT_EVALUATION_AMENDMENT = Path("configs/evaluation_amendment.yaml")
EVALUATION_MODEL_DTYPE = "float32"
EVALUATION_TIE_POLICY = "reject_with_context"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def _load_evaluation_amendment(
    config_path: Path | str,
    amendment_path: Path | str,
    *,
    require_test: bool,
) -> tuple[dict[str, Any], str]:
    """Validate the frozen evaluation-only amendment against immutable inputs."""

    config_path = Path(config_path)
    amendment_path = Path(amendment_path)
    if not amendment_path.is_file():
        raise FileNotFoundError(f"Missing evaluation amendment: {amendment_path}")
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    if not isinstance(amendment, dict):
        raise ValueError("Evaluation amendment must be a YAML mapping")
    training_git_sha = amendment.get("training_git_sha")
    evaluation = amendment.get("evaluation")
    if (
        amendment.get("schema_version") != 1
        or amendment.get("status") != "frozen"
        or not isinstance(amendment.get("protocol_id"), str)
        or not amendment["protocol_id"]
        or not isinstance(training_git_sha, str)
        or len(training_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in training_git_sha)
    ):
        raise ValueError("Evaluation amendment identity is invalid")
    if amendment.get("experiment_config_sha256") != sha256_file(config_path):
        raise ValueError("Evaluation amendment does not match the experiment config")
    config = load_config(config_path)
    data_dir = Path(config["paths"]["data_dir"])
    dev_path = data_dir / "dev.jsonl"
    if (
        not dev_path.is_file()
        or amendment.get("dev_data_sha256") != sha256_file(dev_path)
    ):
        raise ValueError("Evaluation amendment does not match the frozen dev data")
    expected_evaluation = {
        "model_dtype": EVALUATION_MODEL_DTYPE,
        "allow_tf32": False,
        "tie_policy": EVALUATION_TIE_POLICY,
        "rerun_scope": "all_dev_and_test_models",
    }
    if evaluation != expected_evaluation:
        raise ValueError("Evaluation amendment settings are invalid")
    if require_test:
        test_path = data_dir / "test.jsonl"
        if (
            not test_path.is_file()
            or amendment.get("sealed_test_sha256") != sha256_file(test_path)
        ):
            raise ValueError("Evaluation amendment does not match the sealed test data")
    return amendment, sha256_file(amendment_path)


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
    row: dict[str, Any],
    logp_a: float,
    logp_b: float,
    generated_text: str | None = None,
) -> dict[str, Any]:
    """Attach an A/B log-probability decision without retaining the full prompt."""

    try:
        prediction = prediction_from_scores(logp_a, logp_b)
    except ValueError as error:
        context = ", ".join(
            f"{key}={row.get(key)}"
            for key in ("case_id", "intervention", "intervention_variant")
        )
        raise ValueError(
            f"{error}; {context}, logp_A={logp_a!r}, logp_B={logp_b!r}, "
            f"evaluation_dtype={EVALUATION_MODEL_DTYPE}"
        ) from error
    gold = row["gold"]
    correct_margin = (logp_a - logp_b) if gold == "A" else (logp_b - logp_a)
    record = {
        key: row[key]
        for key in (
            "case_id",
            "split",
            "decision_type",
            "intervention",
            "intervention_variant",
            "variant",
            "gold",
            "hint",
        )
        if key in row
    } | {
        "logp_A": float(logp_a),
        "logp_B": float(logp_b),
        "prediction": prediction,
        "correct": prediction == gold,
        "correct_margin": float(correct_margin),
    }
    if generated_text is not None:
        normalized = generated_text.strip()
        generation_prediction = normalized if normalized in {"A", "B"} else None
        record.update(
            {
                "generated_text": generated_text,
                "generation_prediction": generation_prediction,
                "generation_exact_format": generation_prediction is not None,
                "generation_correct": generation_prediction == gold,
            }
        )
    return record


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
    expected_rows = config["data"]["test_cases"] * 6
    if file_entry.get("rows") != expected_rows:
        raise ValueError("Sealed test row count does not match the config")
    return seal


def _fp32_log_softmax(logits: Any, torch_module: Any) -> Any:
    """Compute token log probabilities after an explicit FP32 conversion."""

    return torch_module.log_softmax(logits.float(), dim=-1)


def _load_fp32_model(
    base_source: str | Path,
    revision: str | None,
    adapter_path: Path | None,
    torch_module: Any,
    auto_model_class: Any,
    peft_model_class: Any,
) -> Any:
    """Load every evaluated model for an actual FP32 forward pass."""

    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")
    model = auto_model_class.from_pretrained(
        base_source,
        revision=revision,
        torch_dtype=torch_module.float32,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = peft_model_class.from_pretrained(model, adapter_path)
    model = model.float()
    floating_tensors = [
        *model.parameters(),
        *getattr(model, "buffers", lambda: ())(),
    ]
    unexpected_dtypes = {
        tensor.dtype
        for tensor in floating_tensors
        if tensor.is_floating_point() and tensor.dtype != torch_module.float32
    }
    if unexpected_dtypes:
        raise RuntimeError(
            "Evaluation model still contains non-FP32 floating parameters: "
            f"{sorted(map(str, unexpected_dtypes))}"
        )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    return model


def _completed_evaluation(
    output_dir: Path,
    identity: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any] | None:
    """Reuse only a complete evaluation with the exact same protocol identity."""

    manifest_path = output_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if manifest.get("status") != "complete" or any(
        manifest.get(key) != value for key, value in identity.items()
    ):
        return None
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    if not prediction_path.is_file() or not metrics_path.is_file():
        raise ValueError("Completed evaluation is missing predictions or metrics")
    predictions = _read_jsonl(prediction_path)
    if (
        len(predictions) != expected_rows
        or manifest.get("prediction_rows") != expected_rows
        or manifest.get("predictions_sha256") != sha256_file(prediction_path)
    ):
        raise ValueError("Completed evaluation prediction checksum or row count changed")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        manifest.get("metrics_sha256") != sha256_file(metrics_path)
        or metrics != score_predictions(predictions)
    ):
        raise ValueError("Completed evaluation metrics checksum or contents changed")
    return {"status": "skipped-complete", "metrics": metrics, "manifest": manifest}


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
        log_probs = _fp32_log_softmax(logits, torch_module)
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


def _greedy_generations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
    torch_module: Any,
) -> list[str]:
    prompt_rows = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts
    ]
    if any(not row for row in prompt_rows):
        raise ValueError("Evaluation prompt tokenization is empty")
    if any(len(row) + max_new_tokens > max_length for row in prompt_rows):
        raise ValueError("Greedy evaluation sequence exceeds max_length")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad token")
    maximum = max(len(row) for row in prompt_rows)
    input_ids = torch_module.full(
        (len(prompt_rows), maximum),
        fill_value=pad_id,
        dtype=torch_module.long,
        device=model.device,
    )
    attention_mask = torch_module.zeros_like(input_ids)
    for index, prompt_ids in enumerate(prompt_rows):
        length = len(prompt_ids)
        input_ids[index, maximum - length :] = torch_module.tensor(
            prompt_ids, dtype=torch_module.long, device=model.device
        )
        attention_mask[index, maximum - length :] = 1
    with torch_module.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(tokens[maximum:], skip_special_tokens=True)
        for tokens in generated
    ]


def evaluate_checkpoint(args: Any) -> dict[str, Any]:
    """Run batched teacher-forced A/B scoring for one checkpoint."""

    config_path = Path(args.config)
    config = load_config(config_path)
    amendment_path = Path(
        getattr(args, "evaluation_amendment", DEFAULT_EVALUATION_AMENDMENT)
    )
    amendment, amendment_sha = _load_evaluation_amendment(
        config_path,
        amendment_path,
        require_test=args.split == "test",
    )
    if args.split == "test":
        validate_test_seal(config_path)
    data_path = Path(config["paths"]["data_dir"]) / f"{args.split}.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing evaluation data: {data_path}")
    revision = None
    if args.model == "base":
        if args.method is not None or args.seed is not None or args.adapter_path is not None:
            raise ValueError("Base evaluation must not specify method, seed, or adapter")
        method = "base"
        local_path = Path(config["model"]["local_path"])
        if args.model_path:
            base_source: str | Path = args.model_path
        elif local_path.is_dir():
            base_source = local_path
        else:
            base_source = config["model"]["name_or_path"]
            revision = config["model"]["revision"]
        adapter_path = None
    elif args.model == "shortcut":
        if args.method is not None or args.seed is not None or args.adapter_path is not None:
            raise ValueError("Shortcut evaluation must not specify method, seed, or adapter")
        method = "shortcut"
        base_source = Path(args.model_path) if args.model_path else (
            Path(config["paths"]["shortcut_dir"]) / "merged"
        )
        adapter_path = None
    elif args.model == "adapter":
        if args.method not in {"control", "repair"} or args.seed not in config["dpo"]["seeds"]:
            raise ValueError("Adapter evaluation requires a formal method and seed")
        method = args.method
        base_source = Path(args.model_path) if args.model_path else (
            Path(config["paths"]["shortcut_dir"]) / "merged"
        )
        adapter_path = Path(args.adapter_path) if args.adapter_path else (
            Path(config["paths"]["dpo_runs_dir"])
            / method
            / f"seed-{args.seed}"
            / "final_adapter"
        )
    elif args.model == "sft-baseline":
        if args.method is not None or args.seed not in config["counterfactual_sft"]["seeds"]:
            raise ValueError("SFT baseline evaluation requires a formal seed")
        method = "counterfactual_sft"
        base_source = Path(args.model_path) if args.model_path else (
            Path(config["paths"]["shortcut_dir"]) / "merged"
        )
        adapter_path = Path(args.adapter_path) if args.adapter_path else (
            Path(config["paths"]["sft_baseline_runs_dir"])
            / f"seed-{args.seed}"
            / "final_adapter"
        )
    else:
        raise ValueError("model must be base, shortcut, adapter, or sft-baseline")
    base_path = Path(base_source)
    if args.model != "base" and not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing merged shortcut model at {base_path}")
    if args.model == "base" and base_path.is_dir() and not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing base model config at {base_path}")
    if adapter_path is not None and not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing DPO adapter at {adapter_path}")

    rows = _read_jsonl(data_path)
    expected_rows = config["data"][f"{args.split}_cases"] * 6
    if len(rows) != expected_rows:
        raise ValueError(
            f"Evaluation data has {len(rows)} rows; expected {expected_rows}"
        )
    base_weights_sha = (
        sha256_model_weights(base_path) if base_path.is_dir() else None
    )
    adapter_weights_sha = (
        sha256_model_weights(adapter_path) if adapter_path else None
    )
    identity = {
        "split": args.split,
        "method": method,
        "seed": args.seed,
        "base_model_path": str(base_source),
        "base_model_weights_sha256": base_weights_sha,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_weights_sha256": adapter_weights_sha,
        "model_revision": config["model"]["revision"],
        "training_git_sha": amendment["training_git_sha"],
        "evaluation_git_sha": _git_sha(),
        "evaluation_amendment_sha256": amendment_sha,
        "evaluation_protocol_id": amendment["protocol_id"],
        "evaluation_model_dtype": EVALUATION_MODEL_DTYPE,
        "evaluation_allow_tf32": False,
        "tie_policy": EVALUATION_TIE_POLICY,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config_sha256": sha256_file(config_path),
    }
    output_dir = Path(args.output_dir)
    completed = _completed_evaluation(output_dir, identity, expected_rows)
    if completed:
        print(canonical_json(completed))
        return completed

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; evaluation requires the A6000")
    tokenizer = AutoTokenizer.from_pretrained(
        base_source, use_fast=True, revision=revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = _load_fp32_model(
        base_source,
        revision,
        adapter_path,
        torch,
        AutoModelForCausalLM,
        PeftModel,
    )
    model.eval()
    model.config.use_cache = False
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
        generated_texts = _greedy_generations(
            model,
            tokenizer,
            prompts,
            config["model"]["max_length"],
            config["evaluation"]["generation_max_new_tokens"],
            torch,
        )
        predictions.extend(
            build_prediction_record(row, logp_a, logp_b, generated_text)
            for row, (logp_a, logp_b), generated_text in zip(
                batch, scores, generated_texts, strict=True
            )
        )
    prediction_path = output_dir / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    metrics = score_predictions(predictions)
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    manifest = {
        "status": "complete",
        **identity,
        "prediction_rows": len(predictions),
        "predictions_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    _write_json(output_dir / "prediction_manifest.json", manifest)
    result = {"status": "complete", "metrics": metrics, "manifest": manifest}
    print(canonical_json(result))
    return result


def _load_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prediction_artifacts(
    result_dir: Path, label: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_path = result_dir / "predictions.jsonl"
    metrics_path = result_dir / "metrics.json"
    manifest = _load_manifest(
        result_dir / "prediction_manifest.json", "prediction manifest"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing predictions for {label}: {prediction_path}")
    if manifest.get("predictions_sha256") != sha256_file(prediction_path):
        raise ValueError(f"Formal prediction checksum changed for {label}")
    predictions = _read_jsonl(prediction_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics for {label}: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        manifest.get("metrics_sha256") != sha256_file(metrics_path)
        or metrics != score_predictions(predictions)
    ):
        raise ValueError(f"Formal metrics checksum or contents changed for {label}")
    return manifest, predictions


def _has_valid_actual_epoch(manifest: dict[str, Any]) -> bool:
    value = manifest.get("actual_epoch")
    return isinstance(value, int | float) and math.isfinite(value) and value > 0


def aggregate_from_artifacts(
    config_path: Path | str,
    output_dir: Path | str | None = None,
    amendment_path: Path | str = DEFAULT_EVALUATION_AMENDMENT,
) -> dict[str, Any]:
    """Audit nine formal runs and aggregate the sealed matched comparison."""

    config_path = Path(config_path)
    config = load_config(config_path)
    validate_test_seal(config_path)
    amendment, amendment_sha = _load_evaluation_amendment(
        config_path, amendment_path, require_test=True
    )
    config_sha = sha256_file(config_path)
    training_git_sha = amendment["training_git_sha"]
    evaluation_git_sha = _git_sha()
    prediction_protocol = {
        "training_git_sha": training_git_sha,
        "evaluation_git_sha": evaluation_git_sha,
        "evaluation_amendment_sha256": amendment_sha,
        "evaluation_protocol_id": amendment["protocol_id"],
        "evaluation_model_dtype": EVALUATION_MODEL_DTYPE,
        "evaluation_allow_tf32": False,
        "tie_policy": EVALUATION_TIE_POLICY,
    }
    data_dir = Path(config["paths"]["data_dir"])
    test_path = data_dir / "test.jsonl"
    test_sha = sha256_file(test_path)
    expected_prediction_rows = config["data"]["test_cases"] * 6
    results_root = Path(config["paths"]["results_dir"])
    runs_root = Path(config["paths"]["dpo_runs_dir"])
    shortcut_root = Path(config["paths"]["shortcut_dir"])
    shortcut_model = shortcut_root / "merged"
    shortcut_weights_sha = sha256_model_weights(shortcut_model)
    local_base_model = Path(config["model"]["local_path"])
    base_weights_sha = (
        sha256_model_weights(local_base_model) if local_base_model.is_dir() else None
    )
    shortcut_manifest = _load_manifest(
        shortcut_root / "run_manifest.json", "Shortcut run manifest"
    )
    shortcut_contract = validate_sft_contract(
        config, rows=config["sft"]["expected_rows"]
    )
    if (
        shortcut_manifest.get("status") != "complete"
        or shortcut_manifest.get("config_sha256") != config_sha
        or shortcut_manifest.get("data_sha256")
        != sha256_file(data_dir / "induction.jsonl")
        or shortcut_manifest.get("git_sha") != training_git_sha
        or shortcut_manifest.get("contract") != shortcut_contract
        or shortcut_manifest.get("trainer_budget") != _trainer_budget(shortcut_contract)
        or shortcut_manifest.get("actual_optimizer_steps")
        != shortcut_contract["optimizer_steps"]
        or not _has_valid_actual_epoch(shortcut_manifest)
        or shortcut_manifest.get("merged_model_weights_sha256")
        != shortcut_weights_sha
    ):
        raise ValueError("Invalid completed Shortcut run manifest")

    dev_path = data_dir / "dev.jsonl"
    dev_sha = sha256_file(dev_path)
    expected_dev_rows = config["data"]["dev_cases"] * 6
    dev_metrics = {}
    for method in ("base", "shortcut"):
        result_dir = results_root / "dev" / method
        prediction_manifest, predictions = _load_prediction_artifacts(
            result_dir, f"dev {method}"
        )
        if (
            prediction_manifest.get("status") != "complete"
            or prediction_manifest.get("split") != "dev"
            or prediction_manifest.get("method") != method
            or prediction_manifest.get("seed") is not None
            or prediction_manifest.get("adapter_path") is not None
            or prediction_manifest.get("data_sha256") != dev_sha
            or prediction_manifest.get("config_sha256") != config_sha
            or any(
                prediction_manifest.get(key) != value
                for key, value in prediction_protocol.items()
            )
            or prediction_manifest.get("model_revision")
            != config["model"]["revision"]
            or prediction_manifest.get("base_model_weights_sha256")
            != (base_weights_sha if method == "base" else shortcut_weights_sha)
            or prediction_manifest.get("prediction_rows") != expected_dev_rows
            or len(predictions) != expected_dev_rows
        ):
            raise ValueError(f"Invalid dev prediction manifest for {method}")
        dev_metrics[method] = score_predictions(predictions)
    expected_gate = {
        "metrics": dev_metrics["shortcut"],
        "base_metrics": dev_metrics["base"],
        "shortcut_minus_base": {
            "hint_flip_rate": (
                dev_metrics["shortcut"]["hint_flip_rate"]
                - dev_metrics["base"]["hint_flip_rate"]
            ),
            "causal_hint_effect": (
                dev_metrics["shortcut"]["causal_hint_effect"]
                - dev_metrics["base"]["causal_hint_effect"]
            ),
        },
        **classify_mechanism_gate(
            dev_metrics["shortcut"], config["evaluation"]["mechanism_gate"]
        ),
    }
    gate = _load_manifest(
        results_root / "dev/shortcut/mechanism_gate.json", "mechanism gate"
    )
    if gate != expected_gate or gate.get("decision") != "pass":
        raise ValueError("FP32 mechanism gate artifact is invalid or did not pass")

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
                or run_manifest.get("git_sha") != training_git_sha
                or run_manifest.get("actual_optimizer_steps") != contract["optimizer_steps"]
                or not _has_valid_actual_epoch(run_manifest)
                or run_manifest.get("contract") != contract
                or run_manifest.get("trainer_budget") != _trainer_budget(contract)
                or run_manifest.get("shortcut_model_weights_sha256")
                != shortcut_weights_sha
                or Path(run_manifest.get("final_adapter", "")) != adapter_dir
                or run_manifest.get("final_adapter_weights_sha256")
                != sha256_model_weights(adapter_dir)
            ):
                raise ValueError(f"Invalid formal run manifest for {method} seed {seed}")
            initial_checksums[seed][method] = run_manifest.get(
                "initial_adapter_checksum", ""
            )
            result_dir = results_root / "test" / method / f"seed-{seed}"
            prediction_manifest, predictions = _load_prediction_artifacts(
                result_dir, f"{method} seed {seed}"
            )
            if (
                prediction_manifest.get("status") != "complete"
                or prediction_manifest.get("split") != "test"
                or prediction_manifest.get("method") != method
                or prediction_manifest.get("seed") != seed
                or prediction_manifest.get("adapter_path") != str(adapter_dir)
                or prediction_manifest.get("data_sha256") != test_sha
                or prediction_manifest.get("config_sha256") != config_sha
                or any(
                    prediction_manifest.get(key) != value
                    for key, value in prediction_protocol.items()
                )
                or prediction_manifest.get("model_revision")
                != config["model"]["revision"]
                or prediction_manifest.get("base_model_weights_sha256")
                != shortcut_weights_sha
                or prediction_manifest.get("adapter_weights_sha256")
                != run_manifest.get("final_adapter_weights_sha256")
                or prediction_manifest.get("prediction_rows") != expected_prediction_rows
                or len(predictions) != expected_prediction_rows
            ):
                raise ValueError(f"Invalid prediction manifest for {method} seed {seed}")
            records[method][seed] = predictions
    for seed, checksums in initial_checksums.items():
        if not checksums.get("control") or checksums["control"] != checksums.get("repair"):
            raise ValueError(f"Control and repair initial adapter checksums differ for seed {seed}")
    sft_records: dict[int, list[dict[str, Any]]] = {}
    sft_runs_root = Path(config["paths"]["sft_baseline_runs_dir"])
    sft_train_path = data_dir / "sft_counterfactual.jsonl"
    sft_train_sha = sha256_file(sft_train_path)
    for seed in config["counterfactual_sft"]["seeds"]:
        run_dir = sft_runs_root / f"seed-{seed}"
        adapter_dir = run_dir / "final_adapter"
        run_manifest = _load_manifest(run_dir / "run_manifest.json", "run manifest")
        contract = validate_counterfactual_sft_contract(
            config,
            seed,
            rows=config["counterfactual_sft"]["expected_rows"],
            smoke=False,
        )
        if (
            run_manifest.get("status") != "complete"
            or run_manifest.get("seed") != seed
            or run_manifest.get("config_sha256") != config_sha
            or run_manifest.get("data_sha256") != sft_train_sha
            or run_manifest.get("git_sha") != training_git_sha
            or run_manifest.get("actual_optimizer_steps") != contract["optimizer_steps"]
            or not _has_valid_actual_epoch(run_manifest)
            or run_manifest.get("contract") != contract
            or run_manifest.get("trainer_budget") != _trainer_budget(contract)
            or run_manifest.get("shortcut_model_weights_sha256")
            != shortcut_weights_sha
            or Path(run_manifest.get("final_adapter", "")) != adapter_dir
            or run_manifest.get("final_adapter_weights_sha256")
            != sha256_model_weights(adapter_dir)
        ):
            raise ValueError(
                f"Invalid formal run manifest for Counterfactual SFT seed {seed}"
            )
        result_dir = results_root / "test" / "counterfactual_sft" / f"seed-{seed}"
        prediction_manifest, predictions = _load_prediction_artifacts(
            result_dir, f"Counterfactual SFT seed {seed}"
        )
        if (
            prediction_manifest.get("status") != "complete"
            or prediction_manifest.get("split") != "test"
            or prediction_manifest.get("method") != "counterfactual_sft"
            or prediction_manifest.get("seed") != seed
            or prediction_manifest.get("adapter_path") != str(adapter_dir)
            or prediction_manifest.get("data_sha256") != test_sha
            or prediction_manifest.get("config_sha256") != config_sha
            or any(
                prediction_manifest.get(key) != value
                for key, value in prediction_protocol.items()
            )
            or prediction_manifest.get("model_revision")
            != config["model"]["revision"]
            or prediction_manifest.get("base_model_weights_sha256")
            != shortcut_weights_sha
            or prediction_manifest.get("adapter_weights_sha256")
            != run_manifest.get("final_adapter_weights_sha256")
            or prediction_manifest.get("prediction_rows") != expected_prediction_rows
            or len(predictions) != expected_prediction_rows
        ):
            raise ValueError(
                f"Invalid prediction manifest for Counterfactual SFT seed {seed}"
            )
        sft_records[seed] = predictions

    checkpoint_metrics = {}
    for method in ("base", "shortcut"):
        result_dir = results_root / "test" / method
        prediction_manifest, predictions = _load_prediction_artifacts(
            result_dir, method
        )
        if (
            prediction_manifest.get("status") != "complete"
            or prediction_manifest.get("split") != "test"
            or prediction_manifest.get("method") != method
            or prediction_manifest.get("seed") is not None
            or prediction_manifest.get("adapter_path") is not None
            or prediction_manifest.get("data_sha256") != test_sha
            or prediction_manifest.get("config_sha256") != config_sha
            or any(
                prediction_manifest.get(key) != value
                for key, value in prediction_protocol.items()
            )
            or prediction_manifest.get("model_revision")
            != config["model"]["revision"]
            or prediction_manifest.get("base_model_weights_sha256")
            != (base_weights_sha if method == "base" else shortcut_weights_sha)
            or prediction_manifest.get("prediction_rows") != expected_prediction_rows
            or len(predictions) != expected_prediction_rows
        ):
            raise ValueError(f"Invalid prediction manifest for {method}")
        checkpoint_metrics[method] = score_predictions(predictions)

    result = aggregate_formal(records["control"], records["repair"], config["evaluation"])
    sft_metrics_by_seed = {
        seed: score_predictions(rows) for seed, rows in sorted(sft_records.items())
    }
    result["baselines"] = {
        **checkpoint_metrics,
        "counterfactual_sft": {
            "metrics": {
                name: sum(metrics[name] for metrics in sft_metrics_by_seed.values())
                / len(sft_metrics_by_seed)
                for name in METRIC_NAMES
            },
            "per_seed": sft_metrics_by_seed,
        },
    }
    result["provenance"] = {
        "config_sha256": config_sha,
        "test_data_sha256": test_sha,
        "training_git_sha": training_git_sha,
        "evaluation_git_sha": evaluation_git_sha,
        "evaluation_amendment_sha256": amendment_sha,
        "evaluation_protocol_id": amendment["protocol_id"],
        "evaluation_model_dtype": EVALUATION_MODEL_DTYPE,
        "evaluation_allow_tf32": False,
        "tie_policy": EVALUATION_TIE_POLICY,
        "control_training_data_sha256": sha256_file(data_dir / "dpo_control.jsonl"),
        "repair_training_data_sha256": sha256_file(data_dir / "dpo_repair.jsonl"),
        "counterfactual_sft_training_data_sha256": sft_train_sha,
        "formal_training_runs": 9,
        "initial_adapter_checksums": initial_checksums,
    }
    report_dir = Path(output_dir) if output_dir else Path(config["paths"]["reports_dir"])
    write_report(result, report_dir)
    return result
