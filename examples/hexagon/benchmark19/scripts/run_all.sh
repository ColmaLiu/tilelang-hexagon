#!/usr/bin/env bash
set -euo pipefail
_bench19_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$_bench19_scripts/run_case.sh" all "$@"
