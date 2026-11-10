#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '%s\n' \
  "[Legacy script disabled] The ABCD experiment scripts have been retired." \
  "Use ${SCRIPT_DIR}/run_empo_binary_2x2.sh instead." \
  "" \
  "Method-D balanced selection has been removed from the active roadmap." \
  "Use random selection with:" \
  "  OVERSAMPLED=on" \
  "  WEIGHTED=off"

exit 1
