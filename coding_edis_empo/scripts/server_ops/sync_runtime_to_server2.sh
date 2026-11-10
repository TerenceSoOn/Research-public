#!/usr/bin/env bash
set -euo pipefail

# Copy non-Git runtime assets from server-1 to server-2.
#
# This uses server-1 as the transfer origin and pushes directly to server-2.
# That avoids `scp -3`, which routes large files through the local client.

SRC_HOST="${SRC_HOST:-direct-server}"
DST_SSH_TARGET="${DST_SSH_TARGET:-pterence6699@116.57.62.7}"
DST_SSH_PORT="${DST_SSH_PORT:-31002}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/pterence6699/coding_edis_empo}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-math-1.5B}"

copy_dir() {
    local src="$1"
    local dst="$2"
    echo "[Sync] ${SRC_HOST}:${src} -> ${DST_SSH_TARGET}:${dst}"
    ssh "${SRC_HOST}" "ssh -p '${DST_SSH_PORT}' '${DST_SSH_TARGET}' 'mkdir -p \"$(dirname "${dst}")\"'"
    ssh "${SRC_HOST}" "scp -P '${DST_SSH_PORT}' -r '${src}' '${DST_SSH_TARGET}:${dst}'"
}

copy_file() {
    local src="$1"
    local dst="$2"
    echo "[Sync] ${SRC_HOST}:${src} -> ${DST_SSH_TARGET}:${dst}"
    ssh "${SRC_HOST}" "ssh -p '${DST_SSH_PORT}' '${DST_SSH_TARGET}' 'mkdir -p \"$(dirname "${dst}")\"'"
    ssh "${SRC_HOST}" "scp -P '${DST_SSH_PORT}' '${src}' '${DST_SSH_TARGET}:${dst}'"
}

copy_dir "${RUNTIME_ROOT}/pretrained/${MODEL_NAME}" "${RUNTIME_ROOT}/pretrained/"
copy_dir "${RUNTIME_ROOT}/math_data" "${RUNTIME_ROOT}/"
copy_file \
    "${RUNTIME_ROOT}/livecodebench/livecodebench_code_generation_lite_release_v6_test.parquet" \
    "${RUNTIME_ROOT}/livecodebench/livecodebench_code_generation_lite_release_v6_test.parquet"

if ssh "${SRC_HOST}" "test -d '/home/pterence6699/coding_empo/venv'"; then
    copy_dir "/home/pterence6699/coding_empo/venv" "/home/pterence6699/coding_empo/"
    ssh "${SRC_HOST}" "ssh -p '${DST_SSH_PORT}' '${DST_SSH_TARGET}' 'ln -sfn /home/pterence6699/coding_empo/venv ${RUNTIME_ROOT}/venv'"
elif ssh "${SRC_HOST}" "test -d '${RUNTIME_ROOT}/venv'"; then
    SRC_VENV="$(ssh "${SRC_HOST}" "readlink -f '${RUNTIME_ROOT}/venv'")"
    copy_dir "${SRC_VENV}" "${RUNTIME_ROOT}/"
else
    echo "[Warn] ${SRC_HOST}:${RUNTIME_ROOT}/venv does not exist; skipping venv sync."
fi

echo "[Sync] Runtime sync complete."
