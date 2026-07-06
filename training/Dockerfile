# Training image with dependencies baked in, so pods boot with zero pip install.
# Built by .github/workflows/build-image.yml, pushed to GHCR.
#
# Base = the exact image previously used directly, so RunPod hosts that have it
# cached only pull our dependency layer (~5 GB) instead of the full image.
FROM runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204

# unsloth's pins replace the base image's torch 2.8.0 with 2.7.0+cu126
# (xformers 0.0.30 requires torch==2.7.0) — that swap is paid once here
# at build time instead of on every pod boot.
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
