#!/usr/bin/env bash
set -euo pipefail

# Android uses ADB directly. SSH is an optional Linux BSP transport.
if [ "${TILELANG_HEXAGON_HLOS:-android}" != linux ] || [ -z "${TILELANG_HEXAGON_SSH_HOST:-}" ]; then
  exit 0
fi

BOARD_HOST=${TILELANG_HEXAGON_SSH_HOST#*@}
KNOWN_HOSTS=$TILELANG_HEXAGON_SSH_KNOWN_HOSTS
KEY_PATH=${TILELANG_HEXAGON_HOST_KEY_PATH:-/run/ssh/ssh_host_ed25519_key.pub}
mkdir -p "$(dirname "$KNOWN_HOSTS")"
TEMP_FILE=$(mktemp "${KNOWN_HOSTS}.XXXXXX")
trap 'rm -f "$TEMP_FILE"' EXIT
adb -s "$TILELANG_HEXAGON_DEVICE" shell "cat '$KEY_PATH'" \
  | tr -d '\r' \
  | awk -v host="$BOARD_HOST" '$1 == "ssh-ed25519" && NF >= 2 {print host, $1, $2; found=1} END {if (!found) exit 1}' \
  > "$TEMP_FILE"
ssh-keygen -lf "$TEMP_FILE"
mv "$TEMP_FILE" "$KNOWN_HOSTS"
echo "Board SSH key loaded from ADB: $BOARD_HOST"
