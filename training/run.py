"""Python entrypoint for GPU pod training. Called by entrypoint.sh after clone + install."""

import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from training.notify import training_finished, training_started

LOG_PATH = Path("/workspace/training.log")
OUTPUT_DIR = Path("/workspace/output")
DATA_DIR = Path("/workspace/data")


def download_dataset(hf_dataset, hf_dataset_revision, hf_token):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("train.jsonl", "test.jsonl"):
        path = hf_hub_download(
            repo_id=hf_dataset,
            filename=filename,
            repo_type="dataset",
            revision=hf_dataset_revision,
            token=hf_token,
            local_dir=str(DATA_DIR),
        )
        print(f"Downloaded {filename} -> {path}")


def build_train_args(env):
    args = [
        "--model", env["MODEL"],
        "--phases", env["PHASES"],
        "--max-seq-len", env["MAX_SEQ_LEN"],
        "--sft-max-steps", env["SFT_MAX_STEPS"],
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(OUTPUT_DIR),
    ]

    optional = [
        ("SFT_EPOCHS", "--sft-epochs"),
        ("SFT_LR", "--sft-lr"),
        ("SFT_BATCH", "--sft-batch"),
        ("SFT_GRAD_ACCUM", "--sft-grad-accum"),
        ("GRPO_EPOCHS", "--grpo-epochs"),
        ("GRPO_LR", "--grpo-lr"),
        ("GRPO_NUM_GENERATIONS", "--grpo-num-generations"),
        ("GRPO_MAX_COMPLETION", "--grpo-max-completion"),
        ("GRPO_BETA", "--grpo-beta"),
        ("GRPO_TEMPERATURE", "--grpo-temperature"),
        ("REWARD_BETA", "--reward-beta"),
        ("LORA_R", "--lora-r"),
    ]
    for env_key, flag in optional:
        val = env.get(env_key, "")
        if val:
            args.extend([flag, val])

    return args


def run_training(train_args, log_path):
    cmd = [sys.executable, "-m", "models.qwen.train"] + train_args
    print(f"Starting training: {' '.join(cmd)}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            log_file.write(decoded)
            log_file.flush()
        proc.wait()

    return proc.returncode


def upload_results(hf_model_repo, hf_token, run_name, sha):
    api = HfApi(token=hf_token)
    subfolder = f"{run_name}-{sha[:7]}"

    if OUTPUT_DIR.exists():
        print(f"Uploading results to {hf_model_repo}/{subfolder}...")
        api.upload_folder(
            folder_path=str(OUTPUT_DIR),
            repo_id=hf_model_repo,
            path_in_repo=subfolder,
        )

    if LOG_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(LOG_PATH),
            path_in_repo=f"{subfolder}/training.log",
            repo_id=hf_model_repo,
        )

    print("Upload complete.")


def main():
    env = os.environ
    webhook_url = env.get("SLACK_WEBHOOK_URL", "")
    run_name = env["RUN_NAME"]
    model = env["MODEL"]
    phases = env["PHASES"]
    sha = env["GITHUB_SHA"]
    gpu_type = env.get("GPU_TYPE", "unknown")
    hf_token = env["HF_TOKEN"]
    hf_model_repo = env["HF_MODEL_REPO"]
    hf_dataset = env["HF_DATASET"]
    hf_dataset_revision = env.get("HF_DATASET_REVISION", "main")

    start_time = time.monotonic()
    exit_code = 1

    try:
        training_started(webhook_url, run_name, model, phases, gpu_type, sha)
        download_dataset(hf_dataset, hf_dataset_revision, hf_token)
        train_args = build_train_args(env)
        exit_code = run_training(train_args, LOG_PATH)
        # Upload even on failure — the log is the only record of what went wrong
        upload_results(hf_model_repo, hf_token, run_name, sha)
    except Exception:
        try:
            upload_results(hf_model_repo, hf_token, run_name, sha)
        except Exception:
            pass
        raise
    finally:
        elapsed = int(time.monotonic() - start_time)
        training_finished(
            webhook_url, run_name, model, phases, sha,
            exit_code, elapsed, str(LOG_PATH),
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
