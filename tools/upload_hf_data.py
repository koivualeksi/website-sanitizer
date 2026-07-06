"""Upload train/test JSONL to HuggingFace dataset repo.

Usage:
  python -m tools.upload_hf_data

Requires HF_TOKEN and HF_DATASET env vars (or huggingface-cli login).
Prints the new dataset revision SHA — paste it into training/config.yaml.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

DATA_DIR = Path(__file__).parent.parent / "data"
FILES = ["train_sft.jsonl", "train_grpo.jsonl", "test.jsonl"]


def main():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Export it or run `huggingface-cli login`.")
        sys.exit(1)

    repo_id = os.environ.get("HF_DATASET")
    if not repo_id:
        print("HF_DATASET not set.")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        print(f"Uploading {path}...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Done: {filename}")

    # Get latest revision
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    revision = info.sha

    print(f"\nDataset uploaded to {repo_id}")
    print(f"Revision: {revision}")
    print(f"\nUpdate training/config.yaml:")
    print(f'  hf_dataset_revision: "{revision}"')


if __name__ == "__main__":
    main()
