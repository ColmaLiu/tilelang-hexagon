#!/usr/bin/env bash
# Source inside the repository's Linux SA8650 Docker image. No board connection.
_bench19_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export BENCH19_ROOT="$(cd -- "$_bench19_scripts/.." && pwd)"
export TILELANG_REPO="${TILELANG_REPO:-/opt/tilelang}"
export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/opt/hexagon-sdk}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/opt/android-ndk}"
export CMAKE_ROOT_PATH="${CMAKE_ROOT_PATH:-/opt/tilelang-venv}"
export TILELANG_HEXAGON_LINUX_RPC_LIB_DIR="${TILELANG_HEXAGON_LINUX_RPC_LIB_DIR:-/opt/board-runtime}"
export TILELANG_HEXAGON_LINUX_CC="${TILELANG_HEXAGON_LINUX_CC:-aarch64-linux-gnu-gcc}"
export TILELANG_HEXAGON_HLOS=linux
export HEXAGON_DSP_ARCH=v73
export TILELANG_HEXAGON_ELF_CRC=1
export TILELANG_HEXAGON_LEGACY_COMPUTE_RES=1
export TILELANG_HEXAGON_DEVICE="${TILELANG_HEXAGON_DEVICE:-192.168.1.1:5555}"
export TILELANG_HEXAGON_SSH_HOST="${TILELANG_HEXAGON_SSH_HOST:-root@192.168.1.1}"
export TILELANG_HEXAGON_SSH_KNOWN_HOSTS="${TILELANG_HEXAGON_SSH_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
export TILELANG_HEXAGON_REMOTE_DIR="${TILELANG_HEXAGON_REMOTE_DIR:-/tmp/tilelang_benchmark19}"
export PYTHONPATH="$TILELANG_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
unset _bench19_scripts
