#!/usr/bin/env bash

# Shared runtime path resolution for the EMPO-only experiment scripts.
# Scripts source this file after PROJECT_ROOT, EMPO_ROOT, and PROJECT_EMPO_ROOT
# have been resolved.

resolve_runtime_defaults() {
    if [ -z "${PROJECT_ROOT:-}" ]; then
        echo "[Error] PROJECT_ROOT must be set before sourcing runtime_env.sh." >&2
        exit 1
    fi

    RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/runtime}"
    RUNTIME_PRETRAINED_ROOT="${RUNTIME_PRETRAINED_ROOT:-${RUNTIME_ROOT}/pretrained}"
    LCB_DATA_ROOT="${LCB_DATA_ROOT:-${RUNTIME_ROOT}/livecodebench}"
    DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-Qwen2.5-math-1.5B}"

    if [ -z "${DATA_ROOT:-}" ]; then
        for CANDIDATE in \
            "${RUNTIME_ROOT}/math_data" \
            "${EMPO_ROOT:-}/math_data" \
            "/home/pterence6699/coding_empo/project_EMPO/EMPO/math_data"; do
            if [ -n "${CANDIDATE}" ] && [ -d "${CANDIDATE}" ]; then
                DATA_ROOT="${CANDIDATE}"
                break
            fi
        done
    fi
    DATA_ROOT="${DATA_ROOT:-${RUNTIME_ROOT}/math_data}"

    if [ -z "${MODEL_PATH:-}" ]; then
        for CANDIDATE in \
            "${RUNTIME_PRETRAINED_ROOT}/${DEFAULT_MODEL_NAME}" \
            "/home/pterence6699/coding_edis/pretrained/${DEFAULT_MODEL_NAME}"; do
            if [ -d "${CANDIDATE}" ]; then
                MODEL_PATH="${CANDIDATE}"
                break
            fi
        done
    fi
    MODEL_PATH="${MODEL_PATH:-${RUNTIME_PRETRAINED_ROOT}/${DEFAULT_MODEL_NAME}}"

    RAY_DATA_HOME="${RAY_DATA_HOME:-${RUNTIME_ROOT}/ray_data}"
    WANDB_DIR="${WANDB_DIR:-${RUNTIME_ROOT}/wandb}"

    export RUNTIME_ROOT RUNTIME_PRETRAINED_ROOT LCB_DATA_ROOT DATA_ROOT MODEL_PATH
    export RAY_DATA_HOME WANDB_DIR
}

resolve_train_venv() {
    resolve_runtime_defaults

    VENV_PATH="${TRAIN_VENV:-${VENV_PATH:-}}"
    if [ -z "${VENV_PATH}" ] || [ ! -x "${VENV_PATH}/bin/python3" ]; then
        for CANDIDATE in \
            "${RUNTIME_ROOT}/venv" \
            "${PROJECT_ROOT}/.venv" \
            "${PROJECT_EMPO_ROOT:-}/venv" \
            "${PROJECT_EMPO_ROOT:-}/../venv" \
            "/home/pterence6699/coding_empo/venv"; do
            if [ -n "${CANDIDATE}" ] && [ -x "${CANDIDATE}/bin/python3" ]; then
                VENV_PATH="${CANDIDATE}"
                break
            fi
        done
    fi

    if [ ! -x "${VENV_PATH}/bin/python3" ]; then
        echo "[Error] No usable virtual environment found." >&2
        echo "        Checked:" >&2
        echo "        - TRAIN_VENV=${TRAIN_VENV:-<unset>}" >&2
        echo "        - ${RUNTIME_ROOT}/venv" >&2
        echo "        - ${PROJECT_ROOT}/.venv" >&2
        echo "        - ${PROJECT_EMPO_ROOT:-<unset>}/venv" >&2
        echo "        - ${PROJECT_EMPO_ROOT:-<unset>}/../venv" >&2
        echo "        - /home/pterence6699/coding_empo/venv" >&2
        exit 1
    fi

    TRAIN_VENV="${VENV_PATH}"
    PYTHON_BIN="${VENV_PATH}/bin/python3"
    export TRAIN_VENV VENV_PATH PYTHON_BIN
}

resolve_livecodebench_dataset_path() {
    resolve_runtime_defaults

    local DEFAULT_LCB_FILE="${LCB_DATA_ROOT}/livecodebench_code_generation_lite_release_v6_test.parquet"
    local LEGACY_LCB_FILE="${PROJECT_ROOT}/evaluation/subsets/livecodebench_code_generation_lite_release_v6_test.parquet"

    if [ -z "${LIVECODEBENCH_LOCAL_DATASET_PATH:-}" ]; then
        if [ -f "${DEFAULT_LCB_FILE}" ]; then
            LIVECODEBENCH_LOCAL_DATASET_PATH="${DEFAULT_LCB_FILE}"
        elif [ -f "${LEGACY_LCB_FILE}" ]; then
            LIVECODEBENCH_LOCAL_DATASET_PATH="${LEGACY_LCB_FILE}"
        else
            LIVECODEBENCH_LOCAL_DATASET_PATH="${DEFAULT_LCB_FILE}"
        fi
    fi

    export LIVECODEBENCH_LOCAL_DATASET_PATH
}
