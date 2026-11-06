#!/usr/bin/env bash
set -euo pipefail
_bench19_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$_bench19_scripts/env.sh"
# Explicit connection is separate: --list and --help never contact the board.
exec python3 "$BENCH19_ROOT/tools/run.py" "$@"
