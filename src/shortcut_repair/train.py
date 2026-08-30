"""LoRA shortcut induction and matched DPO training entry points."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

from shortcut_repair.data import canonical_json, load_config, sha256_file

PINNED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
METHODS = {"control", "repair"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def expected_optimizer_steps(rows: int, epochs: int, effective_batch: int) -> int:
    """Return optimizer steps when each epoch rounds its final batch up."""

    if min(rows, epochs, effective_batch) <= 0:
        raise ValueError("rows, epochs, and effective_batch must be positive")
    return math.ceil(rows / effective_batch) * epochs


def _trainer_budget(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the single Trainer stopping budget derived from the run contract."""

    return {
        "max_steps": contract["optimizer_steps"],
        "num_train_epochs": contract["epochs"],
        "source": "contract.optimizer_steps",
    }


def _git_sha() -> str:
    """Return the exact repository commit recorded by a run manifest."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to resolve the current Git SHA") from error
    sha = result.stdout.strip()
    if len(sha) != 40:
        raise RuntimeError(f"Unexpected Git SHA: {sha!r}")
    return sha


def _effective_batch(stage: dict[str, Any]) -> int:
    return stage["micro_batch_size"] * stage["gradient_accumulation_steps"]


def _validate_shared(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    if config["model"]["revision"] != PINNED_MODEL_REVISION:
        raise ValueError(
            f"model revision must remain pinned to {PINNED_MODEL_REVISION}"
        )
    stage = config[stage_name]
    effective_batch = _effective_batch(stage)
    if effective_batch != 32:
        raise ValueError(f"{stage_name} effective batch must remain 32, got {effective_batch}")
    return stage


def validate_sft_contract(config: dict[str, Any], rows: int) -> dict[str, Any]:
    """Validate the frozen shortcut-induction budget."""

    stage = _validate_shared(config, "sft")
    expected_rows = stage["expected_rows"]
    if rows != expected_rows:
        raise ValueError(f"SFT requires exactly {expected_rows:,} rows, got {rows:,}")
    steps = expected_optimizer_steps(rows, stage["epochs"], _effective_batch(stage))
    if steps != stage["expected_optimizer_steps"]:
        raise ValueError(
            f"SFT optimizer-step contract changed: {steps} != "
            f"{stage['expected_optimizer_steps']}"
        )
    return {
        "stage": "shortcut_sft",
        "rows": rows,
        "epochs": stage["epochs"],
        "micro_batch_size": stage["micro_batch_size"],
        "gradient_accumulation_steps": stage["gradient_accumulation_steps"],
        "effective_batch_size": _effective_batch(stage),
        "optimizer_steps": steps,
        "learning_rate": stage["learning_rate"],
        "model_revision": config["model"]["revision"],
        "lora": config["lora"],
    }


def validate_dpo_contract(
    config: dict[str, Any],
    method: str,
    seed: int,
    rows: int,
    smoke: bool,
) -> dict[str, Any]:
    """Enforce identical training budgets for control and repair."""

    if method not in METHODS:
        raise ValueError("DPO method must be control or repair")
    stage = _validate_shared(config, "dpo")
    if smoke:
        if seed != stage["seeds"][0]:
            raise ValueError(f"DPO smoke seed must be {stage['seeds'][0]}")
        if rows != stage["smoke_rows"]:
            raise ValueError(f"DPO smoke requires exactly {stage['smoke_rows']} rows")
        optimizer_steps = stage["smoke_steps"]
    else:
        if seed not in stage["seeds"]:
            raise ValueError(f"DPO seed must be one of {stage['seeds']}")
        if rows != stage["expected_rows"]:
            raise ValueError(
                f"Formal DPO requires exactly {stage['expected_rows']:,} rows, got {rows:,}"
            )
        optimizer_steps = expected_optimizer_steps(
            rows, stage["epochs"], _effective_batch(stage)
        )
        if optimizer_steps != stage["expected_optimizer_steps"]:
            raise ValueError(
                f"DPO optimizer-step contract changed: {optimizer_steps} != "
                f"{stage['expected_optimizer_steps']}"
            )
    return {
        "stage": "dpo",
        "method": method,
        "seed": seed,
        "smoke": smoke,
        "rows": rows,
        "epochs": stage["epochs"],
        "micro_batch_size": stage["micro_batch_size"],
        "gradient_accumulation_steps": stage["gradient_accumulation_steps"],
        "effective_batch_size": _effective_batch(stage),
        "optimizer_steps": optimizer_steps,
        "beta": stage["beta"],
        "loss_type": stage["loss_type"],
        "learning_rate": stage["learning_rate"],
        "model_revision": config["model"]["revision"],
        "lora": config["lora"],
        "reference_policy": "merged_shortcut_model_with_new_dpo_adapter_disabled",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    return {name: importlib.metadata.version(name) for name in names}


def _resolve_base_model(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    local_path = Path(config["model"]["local_path"])
    return str(local_path) if local_path.is_dir() else config["model"]["name_or_path"]


def _lora_checksum(model: Any, torch_module: Any) -> str:
    digest = hashlib.sha256()
    parameter_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if "lora_" not in name:
            continue
        parameter_count += 1
        digest.update(name.encode())
        tensor = parameter.detach().to(dtype=torch_module.float32, device="cpu").contiguous()
        digest.update(tensor.numpy().tobytes())
    if parameter_count == 0:
        raise RuntimeError("No LoRA parameters were found for checksum generation")
    return digest.hexdigest()


def _latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directory exists under {output_dir}")
    return max(checkpoints)[1]


def _validate_run_identity(
    manifest: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    keys = ["config_sha256", "data_sha256", "git_sha", "contract", "trainer_budget"]
    keys.extend(
        key for key in ("method", "base_model", "shortcut_model") if key in expected
    )
    mismatches = [key for key in keys if manifest.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            f"{label} does not match the current run identity: {', '.join(mismatches)}"
        )


def _resume_checkpoint(
    output_dir: Path, resume: bool, expected: dict[str, Any]
) -> str | None:
    if not resume:
        return None
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Resume manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_run_identity(manifest, expected, "Resume manifest")
    checkpoint = _latest_checkpoint(output_dir)
    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(f"Trainer state is missing: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    expected_steps = expected["trainer_budget"]["max_steps"]
    if trainer_state.get("max_steps") != expected_steps:
        raise ValueError(
            "Checkpoint trainer_state max_steps does not match the current run budget: "
            f"{trainer_state.get('max_steps')} != {expected_steps}"
        )
    checkpoint_step = int(checkpoint.name.rsplit("-", 1)[1])
    if trainer_state.get("global_step") != checkpoint_step:
        raise ValueError(
            "Checkpoint directory step does not match trainer_state global_step: "
            f"{checkpoint_step} != {trainer_state.get('global_step')}"
        )
    return str(checkpoint)


def _completed_manifest(
    path: Path,
    required_artifact: Path,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file() or not required_artifact.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return None
    if expected is not None:
        _validate_run_identity(manifest, expected, "Completed manifest")
    return manifest


def _ensure_cuda(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training requires the A6000")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch combination does not support BF16")


def _seed_everything(seed: int, np_module: Any, torch_module: Any, set_seed: Any) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    set_seed(seed)


def _peft_config(config: dict[str, Any], LoraConfig: Any, TaskType: Any) -> Any:
    lora = config["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    )


def _tokenize_sft_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[dict[str, list[int]]]:
    encoded_rows = []
    longest = 0
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
        target_ids = [*target_ids, tokenizer.eos_token_id]
        input_ids = [*prompt_ids, *target_ids]
        if len(input_ids) > max_length:
            raise ValueError(
                f"SFT row {row['case_id']} has {len(input_ids)} tokens; max_length={max_length}"
            )
        longest = max(longest, len(input_ids))
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
            }
        )
    print(canonical_json({"sft_longest_tokens": longest}))
    return encoded_rows


def _render_dpo_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[dict[str, str]]:
    rendered = []
    longest = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        for name in ("chosen", "rejected"):
            length = len(tokenizer(prompt + row[name], add_special_tokens=False)["input_ids"])
            longest = max(longest, length)
            if length > max_length:
                raise ValueError(
                    f"DPO row {row['case_id']} has {length} tokens; max_length={max_length}"
                )
        rendered.append(
            {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}
        )
    print(canonical_json({"dpo_longest_tokens": longest}))
    return rendered


def train_shortcut(args: Any) -> dict[str, Any]:
    """Train and merge the deliberately cache-following shortcut checkpoint."""

    config_path = Path(args.config)
    config = load_config(config_path)
    data_path = Path(config["paths"]["data_dir"]) / "induction.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing induction data: {data_path}")
    rows = _read_jsonl(data_path)
    contract = validate_sft_contract(config, len(rows))
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["paths"]["shortcut_dir"])
    )
    merged_dir = output_root / "merged"
    manifest_path = output_root / "run_manifest.json"
    base_model = _resolve_base_model(config, args.model_path)
    trainer_budget = _trainer_budget(contract)
    summary = {
        "status": "dry-run" if args.dry_run else "starting",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config_sha256": sha256_file(config_path),
        "git_sha": _git_sha(),
        "base_model": str(base_model),
        "output_dir": str(output_root),
        "merged_model_dir": str(merged_dir),
        "contract": contract,
        "trainer_budget": trainer_budget,
    }
    if args.dry_run:
        print(canonical_json(summary))
        return summary
    completed = _completed_manifest(
        manifest_path, merged_dir / "config.json", expected=summary
    )
    if completed:
        result = {**completed, "status": "skipped-complete"}
        print(canonical_json(result))
        return result
    resume_from_checkpoint = _resume_checkpoint(
        output_root, bool(args.resume), expected=summary
    )

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _ensure_cuda(torch)
    seed = config["sft"]["seed"]
    _seed_everything(seed, np, torch, set_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_path = base_model
    revision = None if Path(model_path).is_dir() else config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(model, _peft_config(config, LoraConfig, TaskType))
    model.enable_input_require_grads()
    initial_checksum = _lora_checksum(model, torch)
    dataset = Dataset.from_list(
        _tokenize_sft_rows(rows, tokenizer, config["model"]["max_length"])
    )
    stage = config["sft"]
    output_root.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_root),
        learning_rate=stage["learning_rate"],
        num_train_epochs=trainer_budget["num_train_epochs"],
        max_steps=trainer_budget["max_steps"],
        per_device_train_batch_size=stage["micro_batch_size"],
        gradient_accumulation_steps=stage["gradient_accumulation_steps"],
        warmup_ratio=stage["warmup_ratio"],
        weight_decay=stage["weight_decay"],
        max_grad_norm=stage["max_grad_norm"],
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=stage["logging_steps"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=stage["save_steps"],
        save_total_limit=2,
        save_safetensors=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    running = {
        **summary,
        "status": "running",
        "initial_adapter_checksum": initial_checksum,
        "package_versions": _package_versions(),
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    _write_json(manifest_path, running)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    actual_steps = trainer.state.global_step
    if actual_steps != contract["optimizer_steps"]:
        raise RuntimeError(
            f"SFT finished at {actual_steps} steps; expected {contract['optimizer_steps']}"
        )
    adapter_dir = output_root / "final_adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_model.save_pretrained(
        merged_dir, safe_serialization=True, max_shard_size="4GB"
    )
    tokenizer.save_pretrained(merged_dir)
    complete = {
        **running,
        "status": "complete",
        "actual_epoch": (
            float(trainer.state.epoch) if trainer.state.epoch is not None else None
        ),
        "actual_optimizer_steps": actual_steps,
        "training_loss": float(result.training_loss),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "merged_model_dir": str(merged_dir),
    }
    _write_json(manifest_path, complete)
    print(canonical_json(complete))
    return complete


def train_dpo(args: Any) -> dict[str, Any]:
    """Train one matched Aligned-only control or Counterfactual Repair adapter."""

    config_path = Path(args.config)
    config = load_config(config_path)
    data_path = Path(config["paths"]["data_dir"]) / f"dpo_{args.method}.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing DPO data: {data_path}")
    all_rows = _read_jsonl(data_path)
    rows = all_rows[: config["dpo"]["smoke_rows"]] if args.smoke else all_rows
    contract = validate_dpo_contract(config, args.method, args.seed, len(rows), args.smoke)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.smoke:
        output_dir = (
            Path(config["paths"]["dpo_runs_dir"])
            / "smoke"
            / f"{args.method}-seed-{args.seed}"
        )
    else:
        output_dir = (
            Path(config["paths"]["dpo_runs_dir"])
            / args.method
            / f"seed-{args.seed}"
        )
    shortcut_model = Path(args.model_path) if args.model_path else (
        Path(config["paths"]["shortcut_dir"]) / "merged"
    )
    final_adapter = output_dir / "final_adapter"
    manifest_path = output_dir / "run_manifest.json"
    trainer_budget = _trainer_budget(contract)
    summary = {
        "status": "dry-run" if args.dry_run else "starting",
        "method": args.method,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config_sha256": sha256_file(config_path),
        "git_sha": _git_sha(),
        "shortcut_model": str(shortcut_model),
        "output_dir": str(output_dir),
        "final_adapter": str(final_adapter),
        "contract": contract,
        "trainer_budget": trainer_budget,
    }
    if args.dry_run:
        print(canonical_json(summary))
        return summary
    if not (shortcut_model / "config.json").is_file():
        raise FileNotFoundError(
            f"Merged shortcut model is missing at {shortcut_model}; run train-shortcut first"
        )
    completed = _completed_manifest(
        manifest_path, final_adapter / "adapter_config.json", expected=summary
    )
    if completed:
        result = {**completed, "status": "skipped-complete"}
        print(canonical_json(result))
        return result
    resume_from_checkpoint = _resume_checkpoint(
        output_dir, bool(args.resume), expected=summary
    )

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    _ensure_cuda(torch)
    _seed_everything(args.seed, np, torch, set_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(shortcut_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        shortcut_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    rendered = _render_dpo_rows(rows, tokenizer, config["model"]["max_length"])
    dataset = Dataset.from_list(rendered)
    stage = config["dpo"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=stage["beta"],
        loss_type=stage["loss_type"],
        learning_rate=stage["learning_rate"],
        num_train_epochs=trainer_budget["num_train_epochs"],
        max_steps=trainer_budget["max_steps"],
        per_device_train_batch_size=stage["micro_batch_size"],
        gradient_accumulation_steps=stage["gradient_accumulation_steps"],
        warmup_ratio=stage["warmup_ratio"],
        weight_decay=stage["weight_decay"],
        max_grad_norm=stage["max_grad_norm"],
        lr_scheduler_type="linear",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=config["model"]["max_length"],
        max_prompt_length=config["model"]["max_length"] - 8,
        truncation_mode="keep_end",
        logging_steps=stage["logging_steps"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=2 if args.smoke else stage["save_steps"],
        save_total_limit=2,
        save_safetensors=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        optim="adamw_torch",
    )
    _seed_everything(args.seed, np, torch, set_seed)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_peft_config(config, LoraConfig, TaskType),
    )
    initial_checksum = _lora_checksum(trainer.model, torch)
    running = {
        **summary,
        "status": "running",
        "initial_adapter_checksum": initial_checksum,
        "package_versions": _package_versions(),
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    _write_json(manifest_path, running)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    actual_steps = trainer.state.global_step
    if actual_steps != contract["optimizer_steps"]:
        raise RuntimeError(
            f"DPO finished at {actual_steps} steps; expected {contract['optimizer_steps']}"
        )
    trainer.model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)
    complete = {
        **running,
        "status": "complete",
        "actual_epoch": (
            float(trainer.state.epoch) if trainer.state.epoch is not None else None
        ),
        "actual_optimizer_steps": actual_steps,
        "training_loss": float(result.training_loss),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _write_json(manifest_path, complete)
    print(canonical_json(complete))
    return complete
