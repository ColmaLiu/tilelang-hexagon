#!/usr/bin/env bash
# Usage: bash enter-docker.sh <name>
set -e

if [ -z "$1" ]; then
  echo "Usage: bash $0 <name>"
  exit 1
fi

CONTAINER_NAME="${1}_hexagon"
IMAGE=${IMAGE:-tilelang-hexagon:sa8650-v73-pr}
WORK_DIR="$PWD"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$OUTPUT_DIR"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker start "$CONTAINER_NAME" >/dev/null
else
  docker run -d --name "$CONTAINER_NAME" --network=host \
    -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
    -e HEXAGON_DSP_ARCH="${HEXAGON_DSP_ARCH:-v73}" \
    -e TILELANG_HEXAGON_HLOS="${TILELANG_HEXAGON_HLOS:-linux}" \
    -e TILELANG_HEXAGON_DEVICE \
    -e TILELANG_HEXAGON_SSH_HOST \
    -e TILELANG_HEXAGON_HOST_KEY_PATH \
    -e TILELANG_HEXAGON_REMOTE_DIR="/data/local/tmp/tilelang_${CONTAINER_NAME}" \
    -v "$WORK_DIR:/work" -v "$OUTPUT_DIR:/output" \
    "$IMAGE" sleep infinity >/dev/null
fi

exec docker exec -it "$CONTAINER_NAME" \
  /usr/local/bin/tilelang-entrypoint bash -c 'connect-board && exec bash'
