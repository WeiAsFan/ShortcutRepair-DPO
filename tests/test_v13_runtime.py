from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from shortcut_repair import v13_runtime as runtime
from shortcut_repair.v13_data import load_v13_config

ROOT = Path(__file__).resolve().parents[1]


def test_anchor_continues_existing_lora_with_fixed_budget(tmp_path, monkeypatch):
    config = load_v13_config(ROOT / "configs/v1_3.yaml")
    spec = runtime.anchor_spec(config, 42, 2560)
    calls = []

    class Model:
        config = SimpleNamespace(use_cache=True)

        def enable_input_require_grads(self):
            calls.append("input_grads")

        def save_pretrained(self, path, **kwargs):
            calls.append(("save", Path(path).name, kwargs))

    model = Model()
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        padding_side="right",
        save_pretrained=lambda path: calls.append(("tokenizer", Path(path).name)),
    )

    class Trainer:
        def __init__(self, **kwargs):
            calls.append(("trainer", kwargs))
            self.model = kwargs["model"]
            self.state = SimpleNamespace(global_step=kwargs["args"].max_steps, epoch=1.0)
            self.callback = kwargs["callbacks"][0]

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            self.callback.on_log(None, None, None, logs={"loss": 0.1})
            return SimpleNamespace(training_loss=0.1)

    torch = SimpleNamespace(
        bfloat16="bf16",
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 100,
        ),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
    )
    transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        DataCollatorForSeq2Seq=lambda **kwargs: kwargs,
        Trainer=Trainer,
        TrainerCallback=object,
        TrainingArguments=SimpleNamespace,
        set_seed=lambda *args: None,
    )

    def continue_adapter(base, path, is_trainable=False):
        calls.append(("continue_adapter", Path(path).name, is_trainable))
        return base

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=continue_adapter)),
    )
    monkeypatch.setattr(runtime, "_ensure_cuda", lambda module: None)
    monkeypatch.setattr(runtime, "_seed_everything", lambda *args: None)
    monkeypatch.setattr(runtime, "_tokenize_sft_rows", lambda *args: [])

    stats = runtime._train_anchor_on_gpu(
        config,
        spec,
        [],
        tmp_path / "source",
        tmp_path / "old_adapter",
        tmp_path / "run",
        None,
    )
    trainer = next(item[1] for item in calls if item[0] == "trainer")
    assert ("continue_adapter", "old_adapter", True) in calls
    assert trainer["args"].learning_rate == 2e-6
    assert trainer["args"].max_steps == 80
    assert stats["actual_optimizer_steps"] == 80
    assert ("save", "final_adapter", {"safe_serialization": True}) in calls
