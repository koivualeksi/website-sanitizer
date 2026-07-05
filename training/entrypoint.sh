#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    echo "Terminating pod ${RUNPOD_POD_ID}..."
    for i in 1 2 3; do
        code=$(curl -s -o /tmp/term.txt -w '%{http_code}' -X DELETE \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
            -H "Authorization: Bearer ${RUNPOD_TERMINATE_API_KEY}") || code=000
        echo "Terminate attempt ${i}: HTTP ${code} $(cat /tmp/term.txt 2>/dev/null)"
        if [ "$code" = "200" ]; then break; fi
        sleep 5
    done
    # Fallback: pod-scoped RUNPOD_API_KEY injected by RunPod
    runpodctl remove pod "$RUNPOD_POD_ID" || true
    # Hold the container open so RunPod doesn't restart it before termination lands
    sleep 300
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
