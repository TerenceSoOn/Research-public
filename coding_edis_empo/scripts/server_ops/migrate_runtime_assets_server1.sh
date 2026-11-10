#!/usr/bin/env bash
set -euo pipefail

# Run on server-1 to populate the canonical runtime directory without deleting
# any legacy paths. This is intentionally copy/symlink-only for migration safety.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-math-1.5B}"

LEGACY_MODEL_ROOT="${LEGACY_MODEL_ROOT:-/home/pterence6699/coding_edis/pretrained}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-/home/pterence6699/coding_empo/project_EMPO/EMPO/math_data}"
LEGACY_VENV="${LEGACY_VENV:-/home/pterence6699/coding_empo/venv}"
LEGACY_LCB_FILE="${LEGACY_LCB_FILE:-${PROJECT_ROOT}/evaluation/subsets/livecodebench_code_generation_lite_release_v6_test.parquet}"

mkdir -p \
    "${RUNTIME_ROOT}/pretrained" \
    "${RUNTIME_ROOT}/math_data" \
    "${RUNTIME_ROOT}/livecodebench"

if [ ! -e "${RUNTIME_ROOT}/pretrained/${MODEL_NAME}" ]; then
    if [ ! -d "${LEGACY_MODEL_ROOT}/${MODEL_NAME}" ]; then
        echo "[Error] Legacy model not found: ${LEGACY_MODEL_ROOT}/${MODEL_NAME}" >&2
        exit 1
    fi
    cp -a "${LEGACY_MODEL_ROOT}/${MODEL_NAME}" "${RUNTIME_ROOT}/pretrained/"
    echo "[Info] Copied model to ${RUNTIME_ROOT}/pretrained/${MODEL_NAME}"
else
    echo "[Info] Model already exists: ${RUNTIME_ROOT}/pretrained/${MODEL_NAME}"
fi

if [ -d "${LEGACY_DATA_ROOT}" ]; then
    cp -an "${LEGACY_DATA_ROOT}/." "${RUNTIME_ROOT}/math_data/"
    echo "[Info] Synced math data into ${RUNTIME_ROOT}/math_data"
else
    echo "[Warn] Legacy data root not found: ${LEGACY_DATA_ROOT}"
fi

if [ -f "${LEGACY_LCB_FILE}" ]; then
    cp -n "${LEGACY_LCB_FILE}" "${RUNTIME_ROOT}/livecodebench/"
    echo "[Info] Synced LiveCodeBench parquet into ${RUNTIME_ROOT}/livecodebench"
else
    echo "[Warn] LiveCodeBench parquet not found: ${LEGACY_LCB_FILE}"
fi

if [ ! -e "${RUNTIME_ROOT}/venv" ]; then
    if [ -d "${LEGACY_VENV}" ]; then
        ln -s "${LEGACY_VENV}" "${RUNTIME_ROOT}/venv"
        echo "[Info] Linked runtime venv: ${RUNTIME_ROOT}/venv -> ${LEGACY_VENV}"
    else
        echo "[Warn] Legacy venv not found: ${LEGACY_VENV}"
    fi
else
    echo "[Info] Runtime venv already exists: ${RUNTIME_ROOT}/venv"
fi

echo "[Info] Runtime migration complete. Legacy paths were not removed."
