#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /workspace/bootstrap.log) 2>&1

notify_failure() {
    python3 - "$1" <<'EOF' || true
import json, os, sys, urllib.request
try:
    with open("/workspace/bootstrap.log", "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2500))
        tail = f.read().decode(errors="replace")
except OSError:
    tail = "(no bootstrap log)"
msg = (f"Pod {os.environ.get('RUNPOD_POD_ID', '?')} FAILED before finishing "
       f"(exit {sys.argv[1]}). Last output:\n```{tail}```")
req = urllib.request.Request(
    os.environ["SLACK_WEBHOOK_URL"],
    json.dumps({"text": msg}).encode(),
    {"Content-Type": "application/json"},
)
urllib.request.urlopen(req, timeout=10)
EOF
}

cleanup() {
    status=$?
    if [ "$status" -ne 0 ] && [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        notify_failure "$status"
    fi
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

# Dependencies are baked into the image (training/Dockerfile); fail fast if
# the pod was launched with a stale/wrong image instead of reinstalling
python -c "import unsloth, trl, slack_sdk" || {
    echo "ERROR: baked dependencies missing — was the pod launched with the GHCR image?"
    exit 1
}

timeout "$MAX_RUNTIME" python -m training.run
