#!/usr/bin/env bash
set -euo pipefail

# Run this on direct-server-2 after creating /data/pterence6699 with sudo:
#   sudo mkdir -p /data/pterence6699
#   sudo chown -R pterence6699:pterence6699 /data/pterence6699

DATA_OWNER_ROOT="${DATA_OWNER_ROOT:-/data/pterence6699}"
HOME_ROOT="${HOME_ROOT:-/home/pterence6699}"

if [ ! -w "${DATA_OWNER_ROOT}" ]; then
    echo "[Error] ${DATA_OWNER_ROOT} is not writable by $(whoami)." >&2
    echo "        Run once on server-2:" >&2
    echo "        sudo mkdir -p ${DATA_OWNER_ROOT}" >&2
    echo "        sudo chown -R $(whoami):$(id -gn) ${DATA_OWNER_ROOT}" >&2
    exit 1
fi

mkdir -p "${DATA_OWNER_ROOT}/coding_edis_empo" "${DATA_OWNER_ROOT}/coding_empo"

create_project_link() {
    local link_path="$1"
    local target_path="$2"
    local current_target

    if [ -L "${link_path}" ]; then
        current_target="$(readlink "${link_path}")"
        if [ "${current_target}" = "${target_path}" ]; then
            echo "[Info] Existing symlink OK: ${link_path} -> ${target_path}"
            return
        fi
        echo "[Error] ${link_path} already points to ${current_target}, not ${target_path}." >&2
        exit 1
    fi

    if [ -e "${link_path}" ]; then
        echo "[Error] ${link_path} already exists and is not a symlink." >&2
        echo "        Move it manually before running this setup script." >&2
        exit 1
    fi

    ln -s "${target_path}" "${link_path}"
    echo "[Info] Created symlink: ${link_path} -> ${target_path}"
}

create_project_link "${HOME_ROOT}/coding_edis_empo" "${DATA_OWNER_ROOT}/coding_edis_empo"
create_project_link "${HOME_ROOT}/coding_empo" "${DATA_OWNER_ROOT}/coding_empo"

mkdir -p \
    "${DATA_OWNER_ROOT}/coding_edis_empo/runtime/pretrained" \
    "${DATA_OWNER_ROOT}/coding_edis_empo/runtime/math_data" \
    "${DATA_OWNER_ROOT}/coding_edis_empo/runtime/livecodebench" \
    "${DATA_OWNER_ROOT}/coding_edis_empo/outputs" \
    "${DATA_OWNER_ROOT}/coding_edis_empo/eval_runs"

echo "[Info] Server-2 mirror directories are ready."
echo "[Info] Verify storage with:"
echo "       readlink -f ${HOME_ROOT}/coding_edis_empo"
echo "       df -h ${HOME_ROOT}/coding_edis_empo"
