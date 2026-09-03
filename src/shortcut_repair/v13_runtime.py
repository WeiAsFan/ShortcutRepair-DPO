"""v1.3 在既有 DPO LoRA adapter 上继续格式锚定 SFT。"""

from __future__ import annotations

import math
import time
from pathlib import Path

from shortcut_repair.evaluate import _read_jsonl, _write_json
from shortcut_repair.train import (
    _ensure_cuda,
    _seed_everything,
    _tokenize_sft_rows,
    expected_optimizer_steps,
)
from shortcut_repair.v12_runtime import (
    _gpu_peak,
    _release_gpu,
    _resume_at,
    read_json,
    require_finite_logs,
    require_model,
)


def anchor_spec(config: dict, seed: int, rows: int) -> dict:
    if seed not in config["seeds"]:
        raise ValueError("未知训练 seed")
    expected_rows = config["data"]["train_cases"] * 4
    if rows != expected_rows:
        raise ValueError(f"anchor 数据行数不一致：{rows} != {expected_rows}")
    params = dict(config["anchor"])
    effective_batch = (
        config["training"]["micro_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
    )
    return {
        "stage": "anchor",
        "seed": seed,
        "rows": rows,
        "params": params,
        "optimizer_steps": expected_optimizer_steps(rows, params["epochs"], effective_batch),
        "effective_batch_size": effective_batch,
        "continued_adapter": True,
    }


def train_anchor_model(
    config: dict,
    seed: int,
    starting_model: Path,
    starting_adapter: Path,
    output_dir: Path,
    identity: dict,
) -> dict:
    rows = _read_jsonl(Path(config["paths"]["data_dir"]) / "anchor.jsonl")
    spec = anchor_spec(config, seed, len(rows))
    expected = {
        **identity,
        "spec": spec,
        "starting_model": str(starting_model),
        "starting_adapter": str(starting_adapter),
    }
    root = Path(output_dir)
    manifest_path = root / "run_manifest.json"
    previous = read_json(manifest_path) if manifest_path.is_file() else {}
    if previous and previous.get("identity") != expected:
        raise ValueError(f"已有 anchor run 身份不一致，不覆盖：{root}")
    if not previous and root.exists() and any(root.iterdir()):
        raise ValueError(f"已有目录缺少 run_manifest，不覆盖：{root}")
    require_model(starting_model)
    require_model(starting_adapter, adapter=True)
    if previous.get("status") == "complete":
        if previous.get("actual_optimizer_steps") != spec["optimizer_steps"] or not math.isfinite(
            previous.get("training_loss", float("nan"))
        ):
            raise ValueError(f"已完成 anchor run 的训练步数或 loss 无效：{root}")
        require_model(root / "final_adapter", adapter=True)
        print(f"跳过已完成格式锚定：{root}", flush=True)
        return previous
    checkpoint = _resume_at(root, spec["optimizer_steps"])
    running = {
        "status": "running",
        "identity": expected,
        **spec,
        "starting_model": str(starting_model),
        "starting_adapter": str(starting_adapter),
        "final_adapter": str(root / "final_adapter"),
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
        f"开始格式锚定：{root}，{len(rows)} 行，{spec['optimizer_steps']} 步，恢复点={checkpoint}",
        flush=True,
    )
    start = time.monotonic()
    try:
        statistics = _train_anchor_on_gpu(
            config, spec, rows, starting_model, starting_adapter, root, checkpoint
        )
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


def _train_anchor_on_gpu(
    config: dict,
    spec: dict,
    rows: list[dict],
    source: Path,
    adapter: Path,
    root: Path,
    checkpoint: str | None,
) -> dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    _ensure_cuda(torch)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("请用 CUDA_VISIBLE_DEVICES 只暴露一张 GPU")
    torch.cuda.reset_peak_memory_stats()
    _seed_everything(spec["seed"], np, torch, set_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    dataset = Dataset.from_list(
        _tokenize_sft_rows(rows, tokenizer, config["model"]["max_length"])
    )
    settings, params = config["training"], spec["params"]

    class FiniteLogs(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            require_finite_logs(logs or {})

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(root),
            learning_rate=params["learning_rate"],
            num_train_epochs=params["epochs"],
            max_steps=spec["optimizer_steps"],
            per_device_train_batch_size=settings["micro_batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            warmup_ratio=settings["warmup_ratio"],
            weight_decay=settings["weight_decay"],
            max_grad_norm=settings["max_grad_norm"],
            lr_scheduler_type="linear",
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=settings["logging_steps"],
            logging_first_step=True,
            logging_nan_inf_filter=False,
            save_strategy="steps",
            save_steps=settings["save_steps"],
            save_total_limit=2,
            save_safetensors=True,
            report_to="none",
            seed=spec["seed"],
            data_seed=spec["seed"],
            dataloader_num_workers=0,
            optim="adamw_torch",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        callbacks=[FiniteLogs()],
    )
    result = trainer.train(resume_from_checkpoint=checkpoint)
    steps, loss = trainer.state.global_step, float(result.training_loss)
    if steps != spec["optimizer_steps"] or not math.isfinite(loss):
        raise RuntimeError(
            f"格式锚定结果无效：实际步数={steps}，期望={spec['optimizer_steps']}，loss={loss}"
        )
    trainer.model.save_pretrained(root / "final_adapter", safe_serialization=True)
    tokenizer.save_pretrained(root / "final_adapter")
    return {
        "actual_optimizer_steps": steps,
        "training_loss": loss,
        "actual_epoch": float(trainer.state.epoch),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
