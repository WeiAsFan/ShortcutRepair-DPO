"""v1.2 的轻量训练和 FP32 评测执行器，不扫描模型权重哈希。"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

from shortcut_repair.evaluate import (
    _conditional_scores,
    _greedy_generations,
    _load_fp32_model,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    build_prediction_record,
)
from shortcut_repair.train import (
    _ensure_cuda,
    _latest_checkpoint,
    _peft_config,
    _render_dpo_rows,
    _seed_everything,
    _tokenize_sft_rows,
    expected_optimizer_steps,
)
from shortcut_repair.v12_analysis import summarize


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_model(path: Path | str, *, adapter: bool = False) -> None:
    """只检查可加载产物的存在性；不读取数 GB 的权重内容。"""
    root = Path(path)
    config_name = "adapter_config.json" if adapter else "config.json"
    pattern = "adapter_model.*" if adapter else "*.safetensors"
    weights = list(root.glob(pattern))
    if not adapter:
        weights += list(root.glob("pytorch_model*.bin"))
    if not (root / config_name).is_file() or not any(p.is_file() for p in weights):
        raise FileNotFoundError(f"缺少模型产物：{root}")


def require_finite_logs(logs: dict) -> None:
    for key, value in logs.items():
        if ("loss" in key or key == "grad_norm") and isinstance(value, int | float):
            if not math.isfinite(value):
                raise RuntimeError(f"训练出现非有限值：{key}={value}")


def training_spec(config: dict, stage: str, seed: int, rows: int, params: dict) -> dict:
    if stage not in {"sft", "dpo"} or seed not in config["seeds"]:
        raise ValueError("未知训练阶段或 seed")
    expected_rows = config["data"]["train_cases"] * (4 if stage == "sft" else 3)
    if rows != expected_rows:
        raise ValueError(f"{stage} 数据行数不一致：{rows} != {expected_rows}")
    effective_batch = (
        config["training"]["micro_batch_size"] * config["training"]["gradient_accumulation_steps"]
    )
    return {
        "stage": stage,
        "seed": seed,
        "rows": rows,
        "params": params,
        "optimizer_steps": expected_optimizer_steps(rows, params["epochs"], effective_batch),
        "effective_batch_size": effective_batch,
        "reference_policy": "starting_model_with_new_adapter_disabled" if stage == "dpo" else None,
    }


def _resume_at(root: Path, steps: int) -> str | None:
    if not list(root.glob("checkpoint-*")):
        return None
    checkpoint = _latest_checkpoint(root)
    state = read_json(checkpoint / "trainer_state.json")
    number = int(checkpoint.name.rsplit("-", 1)[1])
    if state.get("max_steps") != steps or state.get("global_step") != number or number > steps:
        raise ValueError(f"checkpoint 的实际步数或 max_steps 不匹配：{checkpoint}")
    return str(checkpoint)


def _gpu_peak() -> int:
    torch = sys.modules.get("torch")
    return int(torch.cuda.max_memory_allocated()) if torch and torch.cuda.is_available() else 0


def _release_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    config: dict,
    stage: str,
    seed: int,
    starting_model: Path,
    output_dir: Path,
    identity: dict,
    params: dict | None = None,
) -> dict:
    """自动跳过完成的 run，只恢复当前未完成 run 的最近 checkpoint。"""
    params = dict(params or config[stage])
    data_path = Path(config["paths"]["data_dir"]) / f"{stage}.jsonl"
    rows = _read_jsonl(data_path)
    spec = training_spec(config, stage, seed, len(rows), params)
    expected = {**identity, "spec": spec, "starting_model": str(starting_model)}
    root = Path(output_dir)
    manifest_path = root / "run_manifest.json"
    previous = read_json(manifest_path) if manifest_path.is_file() else {}
    if previous and previous.get("identity") != expected:
        raise ValueError(f"已有 run 身份不一致，不覆盖：{root}")
    if not previous and root.exists() and any(root.iterdir()):
        raise ValueError(f"已有目录缺少 run_manifest，不覆盖：{root}")
    require_model(starting_model)
    if previous.get("status") == "complete":
        if previous.get("actual_optimizer_steps") != spec["optimizer_steps"] or not math.isfinite(
            previous.get("training_loss", float("nan"))
        ):
            raise ValueError(f"已完成 run 的训练步数或 loss 无效：{root}")
        require_model(root / "final_adapter", adapter=True)
        if stage == "sft":
            require_model(root / "merged")
        print(f"跳过已完成训练：{root}", flush=True)
        return previous
    checkpoint = _resume_at(root, spec["optimizer_steps"])
    running = {
        "status": "running",
        "identity": expected,
        **spec,
        "starting_model": str(starting_model),
        "final_adapter": str(root / "final_adapter"),
        "merged_model": str(root / "merged") if stage == "sft" else None,
        "resume_from_checkpoint": checkpoint,
        "attempts": previous.get("attempts", 0) + 1,
        "training_runtime_seconds": previous.get("training_runtime_seconds", 0.0),
        "runtime_is_lower_bound": (
            previous.get("runtime_is_lower_bound", False) or previous.get("status") == "running"
        ),
        "peak_gpu_memory_bytes": previous.get("peak_gpu_memory_bytes", 0),
    }
    _write_json(manifest_path, running)
    print(
        f"开始训练：{root}，{len(rows)} 行，{spec['optimizer_steps']} 步，恢复点={checkpoint}",
        flush=True,
    )
    start = time.monotonic()
    try:
        statistics = _train_on_gpu(config, spec, rows, starting_model, root, checkpoint)
        running.update(statistics, status="complete")
    except BaseException as error:
        running.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        running["training_runtime_seconds"] += time.monotonic() - start
        running["peak_gpu_memory_bytes"] = max(
            previous.get("peak_gpu_memory_bytes", 0), running["peak_gpu_memory_bytes"], _gpu_peak()
        )
        _write_json(manifest_path, running)
        _release_gpu()
    return running


def _train_on_gpu(
    config: dict, spec: dict, rows: list[dict], source: Path, root: Path, checkpoint: str | None
) -> dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from trl import DPOConfig, DPOTrainer

    _ensure_cuda(torch)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("请用 CUDA_VISIBLE_DEVICES 只暴露一张 GPU，以保持有效 batch size 32")
    torch.cuda.reset_peak_memory_stats()
    _seed_everything(spec["seed"], np, torch, set_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if spec["stage"] == "sft" else "left"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    stage, params, settings = spec["stage"], spec["params"], config["training"]
    common = {
        "output_dir": str(root),
        "learning_rate": params["learning_rate"],
        "num_train_epochs": params["epochs"],
        "max_steps": spec["optimizer_steps"],
        "per_device_train_batch_size": settings["micro_batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "warmup_ratio": settings["warmup_ratio"],
        "weight_decay": settings["weight_decay"],
        "max_grad_norm": settings["max_grad_norm"],
        "lr_scheduler_type": "linear",
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": settings["logging_steps"],
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "save_strategy": "steps",
        "save_steps": settings["save_steps"],
        "save_total_limit": 2,
        "save_safetensors": True,
        "report_to": "none",
        "seed": spec["seed"],
        "data_seed": spec["seed"],
        "dataloader_num_workers": 0,
        "optim": "adamw_torch",
    }

    class FiniteLogs(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            require_finite_logs(logs or {})

    if stage == "sft":
        model = get_peft_model(model, _peft_config(config, LoraConfig, TaskType))
        model.enable_input_require_grads()
        dataset = Dataset.from_list(
            _tokenize_sft_rows(rows, tokenizer, config["model"]["max_length"])
        )
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**common, remove_unused_columns=False),
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
            ),
            callbacks=[FiniteLogs()],
        )
    else:
        dataset = Dataset.from_list(
            _render_dpo_rows(rows, tokenizer, config["model"]["max_length"])
        )
        _seed_everything(spec["seed"], np, torch, set_seed)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(
                **common,
                beta=params["beta"],
                loss_type=params["loss_type"],
                max_length=config["model"]["max_length"],
                max_prompt_length=config["model"]["max_length"] - 8,
                truncation_mode="keep_end",
                remove_unused_columns=True,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=_peft_config(config, LoraConfig, TaskType),
            callbacks=[FiniteLogs()],
        )
    result = trainer.train(resume_from_checkpoint=checkpoint)
    steps, loss = trainer.state.global_step, float(result.training_loss)
    if steps != spec["optimizer_steps"] or not math.isfinite(loss):
        raise RuntimeError(
            f"训练结果无效：实际步数={steps}，期望={spec['optimizer_steps']}，loss={loss}"
        )
    trainer.model.save_pretrained(root / "final_adapter", safe_serialization=True)
    tokenizer.save_pretrained(root / "final_adapter")
    if stage == "sft":
        # 只合并一次；SFT 基线评测和下一阶段 DPO 共用完全相同的起点。
        merged = trainer.model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(root / "merged", safe_serialization=True)
        tokenizer.save_pretrained(root / "merged")
    return {
        "actual_optimizer_steps": steps,
        "training_loss": loss,
        "actual_epoch": float(trainer.state.epoch),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def evaluate_model(
    config: dict,
    data_path: Path,
    source: Path,
    adapter: Path | None,
    output_dir: Path,
    identity: dict,
) -> dict:
    expected = {
        **identity,
        "source": str(source),
        "adapter": str(adapter) if adapter else None,
        "model_dtype": "float32",
        "allow_tf32": False,
        "tie_policy": "reject_with_context",
    }
    root = Path(output_dir)
    manifest_path = root / "prediction_manifest.json"
    previous = read_json(manifest_path) if manifest_path.is_file() else {}
    if previous and previous.get("identity") != expected:
        raise ValueError(f"已有评测身份不一致，不覆盖：{root}")
    rows = _read_jsonl(data_path)
    if previous.get("status") == "complete":
        predictions = _read_jsonl(root / "predictions.jsonl")
        metrics = summarize(predictions)
        if len(predictions) != len(rows) or read_json(root / "metrics.json") != metrics:
            raise ValueError(f"已有评测行数或指标被修改：{root}")
        print(f"跳过已完成评测：{root}", flush=True)
        return metrics
    if not previous and root.exists() and any(root.iterdir()):
        raise ValueError(f"已有评测目录缺少 manifest，不覆盖：{root}")
    require_model(source)
    if adapter:
        require_model(adapter, adapter=True)
    _write_json(manifest_path, {"status": "running", "identity": expected})
    print(f"开始 FP32 评测：{root}，{len(rows)} 行", flush=True)
    try:
        predictions = _evaluate_on_gpu(config, rows, source, adapter)
        metrics = summarize(predictions)
        _write_jsonl(root / "predictions.jsonl", predictions)
        _write_json(root / "metrics.json", metrics)
        _write_json(
            manifest_path,
            {"status": "complete", "identity": expected, "prediction_rows": len(predictions)},
        )
        return metrics
    finally:
        _release_gpu()


def _evaluate_on_gpu(
    config: dict, rows: list[dict], source: Path, adapter: Path | None
) -> list[dict]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("FP32 评测需要服务器上的 CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_fp32_model(source, None, adapter, torch, AutoModelForCausalLM, PeftModel)
    model.eval()
    model.config.use_cache = False
    batch_size, predictions = config["evaluation"]["batch_size"], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["prompt_messages"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        scores = _conditional_scores(
            model, tokenizer, prompts, config["model"]["max_length"], torch
        )
        texts = _greedy_generations(
            model,
            tokenizer,
            prompts,
            config["model"]["max_length"],
            config["evaluation"]["generation_max_new_tokens"],
            torch,
        )
        predictions.extend(
            build_prediction_record(row, a, b, text)
            for row, (a, b), text in zip(batch, scores, texts, strict=True)
        )
        if (start // batch_size) % 10 == 0:
            print(f"评测进度：{min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions
