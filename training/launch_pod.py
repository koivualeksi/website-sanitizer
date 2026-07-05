"""Launch a RunPod GPU pod for training.

Reads training/config.yaml + env vars, creates the pod, and exits.
The pod runs entrypoint.sh and self-terminates via the cleanup trap.
"""

import base64
import json
import os
import sys
from pathlib import Path

import requests
import yaml

from training.notify import send as notify_slack

RUNPOD_API = "https://rest.runpod.io/v1/pods"
IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    github_sha = os.environ.get("GITHUB_SHA", "unknown")
    repo_url = os.environ.get("REPO_URL", "")

    hf_model_repo = os.environ.get("HF_MODEL_REPO", "")
    hf_dataset = os.environ.get("HF_DATASET", "")
    run_type = os.environ.get("RUN_TYPE", "full")

    # Smoke test overrides
    if run_type == "smoke":
        cfg["sft_max_steps"] = 5
        cfg["max_runtime"] = "30m"

    if not api_key:
        print("RUNPOD_API_KEY not set")
        sys.exit(1)
    if not hf_token:
        print("HF_TOKEN not set")
        sys.exit(1)
    if not hf_model_repo:
        print("HF_MODEL_REPO not set")
        sys.exit(1)
    if not hf_dataset:
        print("HF_DATASET not set")
        sys.exit(1)
    if not repo_url:
        # Default to HTTPS clone URL from GITHUB_REPOSITORY
        gh_repo = os.environ.get("GITHUB_REPOSITORY", "")
        if gh_repo:
            repo_url = f"https://github.com/{gh_repo}.git"
        else:
            print("REPO_URL or GITHUB_REPOSITORY not set")
            sys.exit(1)

    # Read and base64-encode entrypoint
    entrypoint_path = Path(__file__).parent / "entrypoint.sh"
    entrypoint_b64 = base64.b64encode(entrypoint_path.read_bytes()).decode()

    sha7 = github_sha[:7]
    pod_name = f"train-{cfg['run_name']}-{sha7}"

    # Build env vars for the pod — only config keys that are set
    env_vars = {
        "ENTRYPOINT_SCRIPT": entrypoint_b64,
        "RUNPOD_TERMINATE_API_KEY": api_key,
        "HF_TOKEN": hf_token,
        "SLACK_WEBHOOK_URL": slack_webhook,
        "GITHUB_SHA": github_sha,
        "REPO_URL": repo_url,
        "RUN_NAME": cfg["run_name"],
        "MODEL": cfg["model"],
        "PHASES": cfg["phases"],
        "MAX_SEQ_LEN": str(cfg["max_seq_len"]),
        "SFT_MAX_STEPS": str(cfg.get("sft_max_steps", 0)),
        "MAX_RUNTIME": cfg["max_runtime"],
        "HF_MODEL_REPO": hf_model_repo,
        "HF_DATASET": hf_dataset,
        "HF_DATASET_REVISION": cfg.get("hf_dataset_revision", "main"),
        "GPU_TYPE": cfg["gpu_type"],
    }

    # Forward optional override keys from config
    optional_keys = [
        "sft_epochs", "sft_lr", "sft_batch", "sft_grad_accum",
        "grpo_epochs", "grpo_lr", "grpo_num_generations",
        "grpo_max_completion", "grpo_beta", "grpo_temperature",
        "reward_beta", "lora_r",
    ]
    for key in optional_keys:
        if key in cfg:
            env_vars[key.upper()] = str(cfg[key])

    payload = {
        "name": pod_name,
        "imageName": IMAGE,
        "gpuTypeIds": [cfg["gpu_type"]],
        "gpuCount": 1,
        "allowedCudaVersions": ["12.6", "12.8"],
        "volumeInGb": 50,
        "containerDiskInGb": 20,
        "env": env_vars,
        "dockerStartCmd": ["/bin/bash", "-c", "echo \"$ENTRYPOINT_SCRIPT\" | base64 -d | /bin/bash"],
    }

    print(f"Launching pod: {pod_name}")
    print(f"  GPU: {cfg['gpu_type']}")
    print(f"  Model: {cfg['model']}, Phases: {cfg['phases']}")
    print(f"  Image: {IMAGE}")

    try:
        resp = requests.post(
            RUNPOD_API,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError as e:
        msg = f"RunPod API error: {e.response.status_code} — {e.response.text}"
        print(msg)
        notify_slack(slack_webhook, f"Training launch FAILED for {pod_name}: {msg}")
        sys.exit(1)
    except Exception as e:
        msg = f"RunPod API error: {e}"
        print(msg)
        notify_slack(slack_webhook, f"Training launch FAILED for {pod_name}: {msg}")
        sys.exit(1)

    pod_id = result.get("id", "unknown")
    console_url = f"https://console.runpod.io/pods?id={pod_id}"

    print(f"Pod launched: {pod_id}")
    print(f"Console: {console_url}")

    # Write to GitHub Actions step summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"### Training Pod Launched\n")
            f.write(f"- **Pod ID:** `{pod_id}`\n")
            f.write(f"- **Name:** `{pod_name}`\n")
            f.write(f"- **Console:** [{console_url}]({console_url})\n")
            f.write(f"- **Model:** {cfg['model']}, **Phases:** {cfg['phases']}\n")
            f.write(f"- **SHA:** `{github_sha}`\n")


if __name__ == "__main__":
    main()
