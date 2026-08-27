# ShortcutRepair-DPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, reproducible A6000 experiment that induces measurable stale-hint reliance, gates on that mechanism, and compares equal-budget Aligned-only DPO with Counterfactual Repair DPO.

**Architecture:** A deterministic generator writes induction, matched DPO, dev, and sealed-test JSONL artifacts. GPU code is isolated behind lazy imports in training and evaluation commands, while all contracts, metrics, gates, bootstrap statistics, and CLI behavior remain CPU-testable. A staged Bash runner prevents formal training before the shortcut gate passes and writes audited manifests for every artifact.

**Tech Stack:** Python 3.10, PyTorch 2.5.1 cu121, Transformers 4.48.3, TRL 0.13.0, PEFT 0.14.0, Qwen2.5-1.5B-Instruct, PyYAML, NumPy, Matplotlib, Pytest, Bash.

## Global Constraints

- Workspace is `E:\files\Projects\_for_Works\DPO\ShortcutRepair-DPO`.
- Base model revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Induction has 600 cases and exactly 1,200 SFT rows; DPO has 600 matched cases and exactly 1,200 rows per method.
- Dev has 200 cases; sealed test has 300 cases; every evaluation case has exactly aligned and conflict variants.
- Formal DPO seeds are 42, 43, and 44; effective batch size is 32; no test-time tuning is allowed.
- GPU libraries are imported lazily so `pytest` and dry-run contracts work without CUDA packages.
- New behavioral code follows red-green-refactor; generated artifacts and static configuration do not require unit-test-first treatment.
- No PPO, GRPO, reward model, custom DPO loss, RAG, multiple tools, third answer, or web UI is added.

---

### Task 1: Project contract and deterministic data

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `configs/experiment.yaml`
- Create: `src/shortcut_repair/__init__.py`
- Create: `src/shortcut_repair/data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `load_config(path) -> dict`, `oracle(case) -> str`, `make_cases(split, count, seed) -> list[dict]`, `prompt_messages(case, hint) -> list[dict]`, `generate_train_dev(config_path) -> dict`, and `generate_sealed_test(config_path) -> dict`.
- Writes: `induction.jsonl`, `dpo_control.jsonl`, `dpo_repair.jsonl`, `dev.jsonl`, `test.jsonl`, `manifest_train_dev.json`, and `manifest_test.json` under the configured data directory.

- [ ] **Step 1: Write failing data invariants**

```python
def test_oracle_obeys_validity_then_fresh_score():
    case = {"candidates": {"A": {"is_valid": False, "fresh_score": 99},
                            "B": {"is_valid": True, "fresh_score": 1}}}
    assert oracle(case) == "B"

def test_generated_dpo_conditions_have_matched_budget_and_cases(tmp_path):
    config_path = write_small_config(tmp_path, induction=4, dpo=6, dev=4, test=4)
    generate_train_dev(config_path)
    control = read_jsonl(tmp_path / "data/dpo_control.jsonl")
    repair = read_jsonl(tmp_path / "data/dpo_repair.jsonl")
    assert len(control) == len(repair) == 12
    assert Counter(row["case_id"] for row in control) == Counter(row["case_id"] for row in repair)
    assert {row["variant"] for row in control} == {"aligned"}
    assert {row["variant"] for row in repair} == {"aligned", "conflict"}
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_data.py -q`

Expected: collection fails because `shortcut_repair.data` does not exist.

- [ ] **Step 3: Implement the generator and config**

Implement a SHA256-derived `random.Random` per `(seed, split, index)`. Gold alternates exactly by index. Every third case makes only the gold candidate valid; all other cases make both valid and assign the gold a strictly higher `fresh_score`. `display_rank` and `historical_score` are deterministic distractors. Induction targets the supplied hint; control duplicates aligned rows; repair emits aligned plus conflict rows. Test generation refuses to overwrite an existing seal whose checksum does not match.

- [ ] **Step 4: Verify GREEN and deterministic regeneration**

Run: `python -m pytest tests/test_data.py -q`

Expected: all data tests pass.

Run the train/dev generator twice into two temporary directories and compare SHA256 values. Expected: corresponding JSONL hashes are identical.

- [ ] **Step 5: Commit the data contract**

```bash
git add .gitattributes .gitignore pyproject.toml requirements*.txt configs src/shortcut_repair/__init__.py src/shortcut_repair/data.py tests
git commit -m "feat: add deterministic ShortcutRepair datasets"
```

### Task 2: Metrics, mechanism gate, and formal statistics

**Files:**
- Create: `src/shortcut_repair/analysis.py`
- Create: `tests/test_analysis.py`

**Interfaces:**
- Consumes: prediction rows with `case_id`, `variant`, `gold`, `hint`, `logp_A`, `logp_B`, and `prediction`.
- Produces: `score_predictions(rows) -> dict`, `classify_mechanism_gate(metrics, thresholds) -> dict`, `paired_bootstrap_conflict(control_by_seed, repair_by_seed, samples, seed) -> dict`, `aggregate_formal(...) -> dict`, and `write_report(result, output_dir) -> None`.

- [ ] **Step 1: Write failing metric and gate tests**

```python
def test_score_predictions_exposes_hint_reliance():
    rows = strongly_hint_following_rows(case_count=10)
    metrics = score_predictions(rows)
    assert metrics["aligned_accuracy"] == 1.0
    assert metrics["conflict_accuracy"] == 0.0
    assert metrics["pair_both_accuracy"] == 0.0
    assert metrics["hint_flip_rate"] == 1.0
    assert metrics["causal_hint_effect"] > 1.0

def test_gate_requires_every_pre_registered_condition():
    passing = {"aligned_accuracy": .9, "conflict_accuracy": .1,
               "hint_flip_rate": .9, "causal_hint_effect": 2.0}
    assert classify_mechanism_gate(passing, THRESHOLDS)["decision"] == "pass"
    for key in passing:
        broken = dict(passing)
        broken[key] = .5 if key != "causal_hint_effect" else .5
        assert classify_mechanism_gate(broken, THRESHOLDS)["decision"] == "fail"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_analysis.py -q`

Expected: import fails because `shortcut_repair.analysis` is absent.

- [ ] **Step 3: Implement pure metrics and validation**

Require exactly two rows per case with one aligned and one conflict variant, identical gold, and opposite hints. Define correct margin as `logp_gold - logp_wrong`; define causal hint effect as the case mean of `aligned_correct_margin - conflict_correct_margin`. Reject duplicate, incomplete, non-finite, or mismatched rows with explicit `ValueError` messages.

- [ ] **Step 4: Add failing bootstrap and success-contract tests**

```python
def test_formal_success_requires_clear_repair_without_clean_regression():
    control = predictions_for_seeds(conflict_correct=0, aligned_correct=100, flip=100)
    repair = predictions_for_seeds(conflict_correct=100, aligned_correct=100, flip=0)
    result = aggregate_formal(control, repair, SUCCESS_CONFIG)
    assert result["decision"] == "POSITIVE"
    assert result["checks"] == {
        "all_seed_conflict_deltas_positive": True,
        "conflict_delta_at_least_10pp": True,
        "conflict_ci_lower_positive": True,
        "hint_flip_halved": True,
        "aligned_drop_within_2pp": True,
        "causal_hint_effect_reduced": True,
    }
```

- [ ] **Step 5: Implement paired case bootstrap and report artifacts**

Bootstrap shared case indices 10,000 times; for each draw average the per-case Repair-Control conflict-correct difference across all three seeds. Write `results.json`, `RESULTS.md`, `main_metrics.csv`, `per_seed.csv`, and `comparison.png`. The Markdown report must state `POSITIVE` only when all six checks pass; otherwise it states `NEGATIVE / INCONCLUSIVE` and lists failed checks.

- [ ] **Step 6: Verify GREEN and commit**

Run: `python -m pytest tests/test_analysis.py -q`

Expected: all analysis tests pass.

```bash
git add src/shortcut_repair/analysis.py tests/test_analysis.py
git commit -m "feat: add causal metrics and preregistered decision gate"
```

### Task 3: CPU-testable training contracts and GPU trainers

**Files:**
- Create: `src/shortcut_repair/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Produces: `expected_optimizer_steps`, `validate_sft_contract`, `validate_dpo_contract`, `train_shortcut(args)`, and `train_dpo(args)`.
- Writes: merged shortcut model under `runs/shortcut/merged`, DPO adapters under `runs/dpo/{method}/seed-{seed}/final_adapter`, plus `run_manifest.json` files.

- [ ] **Step 1: Write failing contract tests**

```python
def test_formal_dpo_contract_is_equal_budget_for_both_methods(config):
    control = validate_dpo_contract(config, "control", 42, rows=1200, smoke=False)
    repair = validate_dpo_contract(config, "repair", 42, rows=1200, smoke=False)
    assert control["optimizer_steps"] == repair["optimizer_steps"] == 114
    assert control["effective_batch_size"] == repair["effective_batch_size"] == 32

def test_contract_rejects_unknown_seed_or_changed_budget(config):
    with pytest.raises(ValueError, match="seed"):
        validate_dpo_contract(config, "control", 99, rows=1200, smoke=False)
    with pytest.raises(ValueError, match="1,200"):
        validate_dpo_contract(config, "repair", 42, rows=1199, smoke=False)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_train.py -q`

Expected: import fails because `shortcut_repair.train` is absent.

- [ ] **Step 3: Implement contracts before GPU imports**

Compute steps as `ceil(rows/effective_batch) * epochs`. Validate method, formal seed, exact rows, effective batch, model revision, and shortcut input path. A `--dry-run` path prints the contract and file checksums without importing torch, transformers, datasets, peft, or trl.

- [ ] **Step 4: Implement shortcut SFT and merge**

Use completion-only labels: prompt tokens receive `-100`, while the single-letter target plus EOS receives labels. Train a rank-16 LoRA for five epochs with `transformers.Trainer`, merge it with `merge_and_unload`, and save model plus tokenizer to `runs/shortcut/merged`. The manifest records input/config hashes, package versions, actual optimizer steps, trainable parameter count, peak GPU memory, and merged model path.

- [ ] **Step 5: Implement matched DPO runs**

Load the merged shortcut model, reset Python/NumPy/Torch/Transformers RNG immediately before DPO adapter construction, and pass the same rank-16 `LoraConfig` into TRL `DPOTrainer`. Use `ref_model=None` so disabling the new adapter yields the merged shortcut reference. Save the initial LoRA checksum and require equal checksums for control/repair with the same seed during aggregation.

- [ ] **Step 6: Verify dry-run contracts and commit**

Run: `python -m pytest tests/test_train.py -q`

Expected: all training contract tests pass without GPU libraries.

Run: `python -m shortcut_repair.cli train-shortcut --config configs/experiment.yaml --dry-run`

Expected: JSON reports 1,200 rows and 190 optimizer steps.

Run both DPO dry-runs for seed 42. Expected: both report 1,200 rows, effective batch 32, and 114 steps.

```bash
git add src/shortcut_repair/train.py tests/test_train.py
git commit -m "feat: add shortcut SFT and matched LoRA-DPO training"
```

### Task 4: Conditional log-probability evaluation and CLI

**Files:**
- Create: `src/shortcut_repair/evaluate.py`
- Create: `src/shortcut_repair/cli.py`
- Create: `tests/test_evaluate.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `prediction_from_scores`, `build_prediction_record`, `evaluate_checkpoint(args)`, and CLI subcommands `generate`, `train-shortcut`, `train-dpo`, `evaluate`, `gate`, and `aggregate`.
- Writes: prediction JSONL, metrics JSON, gate JSON, prediction manifests, and formal reports.

- [ ] **Step 1: Write failing score-to-record tests**

```python
def test_prediction_uses_conditional_logprob_not_generation_format():
    row = {"case_id": "c1", "variant": "conflict", "gold": "A", "hint": "B"}
    record = build_prediction_record(row, logp_a=-0.2, logp_b=-1.7)
    assert record["prediction"] == "A"
    assert record["correct_margin"] == pytest.approx(1.5)

def test_equal_scores_are_rejected_instead_of_silently_tie_breaking():
    with pytest.raises(ValueError, match="equal"):
        prediction_from_scores(-1.0, -1.0)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_evaluate.py tests/test_cli.py -q`

Expected: imports fail because evaluator and CLI do not exist.

- [ ] **Step 3: Implement conditional scoring**

Render each chat prompt with `add_generation_prompt=True`, append `A` and `B` separately, and sum the causal token log-probabilities only over completion positions. Batch the two candidates together, use forward inference rather than `generate`, verify the prompt token IDs are an exact prefix, and write one audited record per prompt variant.

- [ ] **Step 4: Implement thin CLI dispatch**

`gate` reads dev predictions and exits code 2 on failure. `generate --stage test` requires a passing gate artifact. `aggregate` requires a valid test seal, six complete run manifests, six prediction manifests, matching data/config hashes, identical initial adapter checksum within each seed, and the exact adapter path for each method.

- [ ] **Step 5: Verify GREEN and commit**

Run: `python -m pytest tests/test_evaluate.py tests/test_cli.py -q`

Expected: all evaluator and CLI tests pass.

```bash
git add src/shortcut_repair/evaluate.py src/shortcut_repair/cli.py tests/test_evaluate.py tests/test_cli.py
git commit -m "feat: add logprob evaluation and experiment CLI"
```

### Task 5: Server orchestration and self-contained documentation

**Files:**
- Create: `scripts/preflight.sh`
- Create: `scripts/run_experiment.sh`
- Create: `scripts/package_results.sh`
- Create: `docs/SERVER_RUNBOOK.md`
- Create: `docs/EXPERIMENT_PROTOCOL.md`
- Create: `README.md`
- Create: `tests/test_shell_contract.py`
- Create: `tests/test_docs.py`

**Interfaces:**
- Produces: resumable stages `prepare`, `induce`, `gate`, `seal-test`, `smoke`, `train`, `evaluate`, `aggregate`, and `all`.
- Produces: a sanitized tarball containing only config, data manifests, gate decision, report artifacts, and run/prediction manifests.

- [ ] **Step 1: Write failing shell and documentation contract tests**

```python
def test_runner_orders_gate_before_test_and_formal_training():
    text = Path("scripts/run_experiment.sh").read_text()
    assert text.index("gate)") < text.index("seal-test)") < text.index("train)")

def test_runbook_contains_exact_a6000_install_and_every_stage():
    text = Path("docs/SERVER_RUNBOOK.md").read_text()
    assert "torch==2.5.1" in text and "cu121" in text
    for stage in ("prepare", "induce", "gate", "seal-test", "smoke", "train", "evaluate", "aggregate"):
        assert f"run_experiment.sh {stage}" in text
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_shell_contract.py tests/test_docs.py -q`

Expected: failures report missing scripts and documentation.

- [ ] **Step 3: Implement preflight and staged runner**

Preflight checks Python 3.10, driver major 535 or newer, CUDA availability, BF16 support, at least 45 GiB total VRAM, torch CUDA runtime 12.1, pinned package versions, and base-model access. The runner uses `set -euo pipefail`, refuses test sealing without a passing gate, skips only runs whose manifest says `complete`, and never deletes an existing run.

- [ ] **Step 4: Write the no-follow-up server runbook**

Document clone, venv creation, the official cu121 PyTorch install command, dependency installation, Hugging Face authentication only when required, model download, preflight, every individual stage, `all`, interruption recovery, gate-failure interpretation, result inspection, packaging, and the exact files to return for analysis.

- [ ] **Step 5: Verify GREEN and commit**

Run: `bash -n scripts/preflight.sh scripts/run_experiment.sh scripts/package_results.sh`

Expected: exit code 0.

Run: `python -m pytest tests/test_shell_contract.py tests/test_docs.py -q`

Expected: all shell and documentation tests pass.

```bash
git add scripts docs README.md tests/test_shell_contract.py tests/test_docs.py
git commit -m "docs: add audited A6000 experiment workflow"
```

### Task 6: Full verification and repository handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-28-shortcut-repair-dpo-implementation.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean Git repository with a CPU-verifiable experiment contract and a documented GPU execution boundary.

- [ ] **Step 1: Run the complete local suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run static and artifact checks**

Run: `python -m compileall -q src tests`

Run: `python -m ruff check src tests`

Run: `git diff --check`

Expected: all commands exit 0 with no errors.

- [ ] **Step 3: Run the offline experiment smoke contract**

Generate train/dev and test data in a temporary config, run all three training dry-runs, feed synthetic shortcut predictions through the gate, feed synthetic formal predictions through aggregation, and verify a positive report plus complete manifests. This proves orchestration and statistics without claiming that GPU learning has occurred.

- [ ] **Step 4: Verify repository state and commit final records**

Run: `git status --short` and `git log --oneline --decorate -10`.

Expected: only the plan checkbox update is pending before the final commit; after commit, status is clean.

```bash
git add docs/superpowers/plans/2026-08-28-shortcut-repair-dpo-implementation.md
git commit -m "test: record local ShortcutRepair-DPO verification"
```
