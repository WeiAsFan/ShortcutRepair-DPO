from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from shortcut_repair import v12_runtime as runtime
from shortcut_repair.v12_data import load_v12_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_gpu_training_wiring_uses_explicit_budget_and_correct_reference(
    tmp_path, monkeypatch, stage
):
    """仅测试 Trainer 接线，不把模拟执行当作真实 CUDA 训练。"""
    config = load_v12_config(ROOT / "configs/v1_2.yaml")
    calls = []
    spec = runtime.training_spec(config, stage, 42, 2560 if stage == "sft" else 1920, config[stage])

    class Model:
        config = SimpleNamespace(use_cache=True)

        def enable_input_require_grads(self):
            calls.append("input_grads")

        def save_pretrained(self, path, **kwargs):
            calls.append(("save", path.name, kwargs))

        def merge_and_unload(self, **kwargs):
            calls.append(("merge", kwargs))
            return self

    model = Model()
    tokenizer = SimpleNamespace(
        pad_token_id=0, padding_side="right", save_pretrained=lambda *a: None
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
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        DataCollatorForSeq2Seq=lambda **k: k,
        Trainer=Trainer,
        TrainerCallback=object,
        TrainingArguments=SimpleNamespace,
        set_seed=lambda *a: None,
    )
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
        SimpleNamespace(
            LoraConfig=lambda **k: k,
            TaskType=SimpleNamespace(CAUSAL_LM="causal"),
            get_peft_model=lambda m, c: m,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(DPOConfig=SimpleNamespace, DPOTrainer=Trainer)
    )
    monkeypatch.setattr(runtime, "_ensure_cuda", lambda t: None)
    monkeypatch.setattr(runtime, "_seed_everything", lambda *a: None)
    monkeypatch.setattr(runtime, "_tokenize_sft_rows", lambda *a: [])
    monkeypatch.setattr(runtime, "_render_dpo_rows", lambda *a: [])
    statistics = runtime._train_on_gpu(config, spec, [], tmp_path / "starting", tmp_path, None)
    kwargs = next(item[1] for item in calls if isinstance(item, tuple) and item[0] == "trainer")
    assert kwargs["args"].max_steps == spec["optimizer_steps"]
    assert kwargs["args"].logging_nan_inf_filter is False
    assert kwargs["args"].bf16 and kwargs["args"].gradient_checkpointing
    assert statistics["actual_optimizer_steps"] == spec["optimizer_steps"]
    if stage == "dpo":
        assert kwargs["ref_model"] is None
        assert kwargs["peft_config"]["r"] == 16
        assert not any(isinstance(item, tuple) and item[0] == "merge" for item in calls)
    else:
        assert ("merge", {"safe_merge": True}) in calls
        assert ("save", "merged", {"safe_serialization": True}) in calls
