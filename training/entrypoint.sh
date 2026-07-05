#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Cloning repo at ${GITHUB_SHA}..."
rm -rf /workspace/repo
git init /workspace/repo
cd /workspace/repo
git remote add origin "$REPO_URL"
git fetch --depth 1 origin "$GITHUB_SHA"
git checkout FETCH_HEAD

echo "Installing requirements..."
pip install -r training/requirements.txt 2>&1 | tail -5

timeout "$MAX_RUNTIME" python -m training.run
