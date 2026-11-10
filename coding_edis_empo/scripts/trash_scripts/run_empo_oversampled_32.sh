#!/usr/bin/env bash

# Backward-compatible alias for experiment B.
# Prefer: scripts/run_B_empo_oversampled_random.sh

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[Info] Re-executing with bash..."
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/env bash "${SCRIPT_DIR}/run_B_empo_oversampled_random.sh" "$@"
