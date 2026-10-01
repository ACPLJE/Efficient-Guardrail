#!/bin/bash
# D3 on the LOGIN node (gsai-login-3) — NOT an sbatch script: compute nodes have no network.
#   bash slurm/s1_download_login.sh            # everything in the registry
#   bash slurm/s1_download_login.sh --list     # show the registry
#   bash slurm/s1_download_login.sh --only qwen3guard_gen_0.6b,toxicchat
# Gated repos: `huggingface-cli login` first (or export HF_TOKEN=hf_...) after accepting each licence.
set -euo pipefail
S1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${S1_ROOT}/env/s1.env"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
# Classic HTTP downloads (no hf_xet): the xet backend stalled on gsai-login-3 (2026-09-17, PolyGuard
# stuck at 10/17 files) and does not resume a killed shard, so every restart began from 0 bytes.
# The classic path resumes <sha>.incomplete blobs with a Range request.  HF_HUB_DISABLE_XET=0 re-enables.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}" "${LG_SOURCES}"
if [[ -x "${LG_PYTHON:-}" ]]; then PY="${LG_PYTHON}"; else PY="$(command -v python3)"; fi
cd "${S1_ROOT}"
echo "[download] host=$(hostname) HF_HOME=${HF_HOME} sources=${LG_SOURCES}"
# run under nohup so a dropped SSH session does not kill a 100 GB transfer; log to $LG_OUT/logs
mkdir -p "${LG_OUT}/logs"
LOG="${LG_OUT}/logs/download_$(date +%m%d_%H%M%S).log"
nohup ${PY} -m lgs1.download "$@" > "${LOG}" 2>&1 &
echo "[download] started in background, pid=$!  log: ${LOG}"
echo "           tail -f ${LOG}      (status: ${LG_SOURCES}/download_status.json)"
