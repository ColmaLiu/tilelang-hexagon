#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
IMAGE=${IMAGE:-tilelang-hexagon:sa8650-v73}

if [ -z "$HEXAGON_SDK_PATH" ] || [ -z "$ANDROID_NDK_PATH" ]; then
  echo "Set HEXAGON_SDK_PATH and ANDROID_NDK_PATH to your installed SDK and NDK."
  exit 1
fi

BOARD_RUNTIME_PATH=${BOARD_RUNTIME_PATH:-$SCRIPT_DIR/local/board-runtime}
PYTHON_WHEEL_DIR=${PYTHON_WHEEL_DIR:-$SCRIPT_DIR/local/wheelhouse}
mkdir -p "$BOARD_RUNTIME_PATH" "$PYTHON_WHEEL_DIR"

docker build --network=host \
  -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE" \
  --build-context "hexagon-sdk=$HEXAGON_SDK_PATH" \
  --build-context "android-ndk=$ANDROID_NDK_PATH" \
  --build-context "board-runtime=$BOARD_RUNTIME_PATH" \
  --build-context "python-wheels=$PYTHON_WHEEL_DIR" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" \
  --build-arg "BUILD_JOBS=${BUILD_JOBS:-4}" \
  --build-arg "SOURCE_REVISION=$(git -C "$REPO_DIR" rev-parse HEAD)" \
  "$REPO_DIR"
