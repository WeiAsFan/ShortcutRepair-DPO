#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "${script_dir}/.."
repo_root="$(pwd)"
cd "$repo_root"

shortcut_python="${SHORTCUT_PYTHON:-python}"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is unavailable." >&2
  exit 2
}

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
driver_major="${driver_version%%.*}"
if [[ ! "$driver_major" =~ ^[0-9]+$ ]] || (( driver_major < 535 )); then
  echo "ERROR: NVIDIA driver 535 or newer is required; found $driver_version." >&2
  exit 2
fi

"$shortcut_python" - <<'PY'
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import torch
import yaml
from transformers import AutoConfig


if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ERROR: Python 3.10 is required; found {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch cannot see a CUDA GPU")
if not str(torch.version.cuda).startswith("12.1"):
    raise SystemExit(f"ERROR: expected the cu121 wheel; torch reports CUDA {torch.version.cuda}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: this GPU/PyTorch pair does not report BF16 support")
properties = torch.cuda.get_device_properties(0)
if properties.total_memory < 45 * 1024**3:
    raise SystemExit(
        f"ERROR: at least 45 GiB VRAM is required; found {properties.total_memory / 1024**3:.1f} GiB"
    )

expected = {
    "torch": "2.5.1+cu121",
    "transformers": "4.48.3",
    "trl": "0.13.0",
    "peft": "0.14.0",
    "datasets": "3.2.0",
    "accelerate": "1.2.1",
}
for package, wanted in expected.items():
    found = importlib.metadata.version(package)
    if found != wanted:
        raise SystemExit(f"ERROR: {package} must be {wanted}; found {found}")

config = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
local_path = Path(config["model"]["local_path"])
model_source = str(local_path) if local_path.is_dir() else config["model"]["name_or_path"]
revision = None if local_path.is_dir() else config["model"]["revision"]
AutoConfig.from_pretrained(model_source, revision=revision)

print(f"GPU: {properties.name}")
print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"Model source verified: {model_source}")
print("PREFLIGHT PASS")
PY

echo "NVIDIA driver: $driver_version"
