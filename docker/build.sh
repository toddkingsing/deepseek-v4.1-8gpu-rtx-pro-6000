#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_IMAGE="vllm/vllm-openai@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d"

# Stage 1: Moet + CLI compatibility image.
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -f "$HERE/context/config/Dockerfile.moet-speed2" \
  -t kingsing-cli-compat:20261119-literal-image \
  "$HERE/context"

# Stage 2: BMM serving image.
docker build \
  -f "$HERE/context/scripts/speed9/Dockerfile.bmm_mega" \
  -t kingsing-speed9:bmm-mega-v1 \
  "$HERE/context/scripts/speed9"

echo "Built kingsing-speed9:bmm-mega-v1"
echo "Next: mount the official DeepSeek-V4.1-Flash-1017 weights read-only at /model and start the profile in configs/."
