"""Pre-flight validation for training/config.yaml.

Runs in GitHub Actions before any money is spent. Exits non-zero with a
clear message on invalid config.
"""

import re
import sys

import yaml

VALID_MODELS = {"0.5B", "1.5B", "3B"}
VALID_PHASES = {"sft", "grpo"}
VALID_LORA_R = {8, 16, 32, 64}

KNOWN_KEYS = {
    "run_name", "model", "phases", "max_seq_len", "sft_max_steps",
    "gpu_type", "max_runtime", "hf_dataset_revision",
    # optional overrides
    "sft_data_file", "sft_epochs", "sft_lr", "sft_batch", "sft_grad_accum",
    "grpo_data_file",
    "grpo_epochs", "grpo_lr", "grpo_num_generations",
    "grpo_max_completion", "grpo_beta", "grpo_temperature",
    "reward_beta", "lora_r",
}

# Approximate VRAM per model (GB) at batch=1, no grad accum overhead
VRAM_BASE = {"0.5B": 4, "1.5B": 8, "3B": 14}
# Extra VRAM per 1024 tokens of sequence length (rough estimate)
VRAM_PER_1K = {"0.5B": 0.5, "1.5B": 1.0, "3B": 2.0}

GPU_VRAM = {
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA RTX A6000": 48,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA L40S": 48,
}


def parse_runtime(s):
    m = re.match(r"^(\d+)(h|m|s)$", s.strip())
    if not m:
        return None
    val, unit = int(m.group(1)), m.group(2)
    return val * {"h": 3600, "m": 60, "s": 1}[unit]


def validate(path):
    errors = []
    warnings = []

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print(f"FAIL: {path} is not a YAML mapping")
        return 1

    # Unknown keys (catches typos like sft_epoch)
    unknown = set(cfg.keys()) - KNOWN_KEYS
    if unknown:
        errors.append(f"Unknown keys: {unknown} (typo?)")

    # Required fields
    run_name = cfg.get("run_name", "")
    if not re.match(r"^[a-z0-9._-]+$", run_name):
        errors.append(f"run_name must match [a-z0-9-]+, got: {run_name!r}")

    model = cfg.get("model", "")
    if model not in VALID_MODELS:
        errors.append(f"model must be one of {VALID_MODELS}, got: {model!r}")

    phases_str = cfg.get("phases", "")
    phases = {p.strip() for p in phases_str.split(",")} if phases_str else set()
    if not phases.issubset(VALID_PHASES):
        errors.append(f"phases must be subset of {VALID_PHASES}, got: {phases}")

    max_seq_len = cfg.get("max_seq_len", 0)
    if not (1024 <= max_seq_len <= 16384):
        errors.append(f"max_seq_len must be in [1024, 16384], got: {max_seq_len}")

    revision = cfg.get("hf_dataset_revision", "")
    if not revision:
        errors.append("hf_dataset_revision is required")
    elif revision == "main":
        warnings.append("hf_dataset_revision is 'main' — run won't be reproducible. "
                        "Pin to a specific revision from upload_hf_data.py output.")
    elif not re.match(r"^[0-9a-f]{7,40}$", revision):
        warnings.append(f"hf_dataset_revision doesn't look like a commit SHA: {revision!r}")

    max_runtime = cfg.get("max_runtime", "")
    if not max_runtime or parse_runtime(str(max_runtime)) is None:
        errors.append(f"max_runtime must be like '6h', '30m', '3600s', got: {max_runtime!r}")

    # Optional numeric bounds
    for key in ("sft_lr", "grpo_lr"):
        if key in cfg:
            lr = cfg[key]
            if not (1e-6 <= lr <= 1e-2):
                errors.append(f"{key} must be in [1e-6, 1e-2], got: {lr}")

    if "lora_r" in cfg and cfg["lora_r"] not in VALID_LORA_R:
        errors.append(f"lora_r must be one of {VALID_LORA_R}, got: {cfg['lora_r']}")

    # Smoke test coherence
    sft_max_steps = cfg.get("sft_max_steps", 0)
    if sft_max_steps > 0 and "grpo" in phases:
        errors.append("GRPO has no step cap — a smoke run must use phases: sft")

    # VRAM sanity check
    if model in VRAM_BASE and isinstance(max_seq_len, int):
        gpu = cfg.get("gpu_type", "")
        if gpu in GPU_VRAM:
            est = VRAM_BASE[model] + VRAM_PER_1K[model] * (max_seq_len / 1024)
            batch = cfg.get("sft_batch", 1)
            est *= batch
            limit = GPU_VRAM[gpu] * 0.8
            if est > limit:
                warnings.append(
                    f"Estimated VRAM ~{est:.0f}GB exceeds 80% of {gpu} "
                    f"({GPU_VRAM[gpu]}GB). May OOM."
                )

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print(f"\n{len(errors)} error(s) found.")
        return 1

    print("Config OK.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    sys.exit(validate(sys.argv[1]))
