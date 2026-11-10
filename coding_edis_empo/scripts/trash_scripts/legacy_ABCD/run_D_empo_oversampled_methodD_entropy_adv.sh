#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '%s\n' \
  "[Legacy script disabled] The ABCD experiment scripts have been retired." \
  "Use ${SCRIPT_DIR}/run_empo_binary_2x2.sh instead." \
  "" \
  "For the weighted oversampled condition under the new roadmap, edit that script to:" \
  "  OVERSAMPLED=on" \
  "  WEIGHTED=on"

exit 1
