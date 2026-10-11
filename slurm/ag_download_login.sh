#!/bin/bash
# AG assets on the LOGIN node (gsai-login-3) -- NOT an sbatch script: compute nodes have no network.
#   bash slurm/ag_download_login.sh                     # every registry entry (ag/data/registry_ag.py)
#   bash slurm/ag_download_login.sh --list              # show the registry (foreground)
#   bash slurm/ag_download_login.sh --stage pilot       # T2 + bg_wiki_en for the S2 step-0 pilot
#   bash slurm/ag_download_login.sh --only bg_wiki_en --no-skip
# Gated repos (Llama-3.1, Prompt Guard 2): accept each licence, then `huggingface-cli login` or export HF_TOKEN.
# Output: $AG_SOURCES/{<name>/revision.json, <name>/probe.json, download_status.json, ag_assets_<MMDD>.json}.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[ag_download] refusing to run inside a Slurm job (SLURM_JOB_ID=${SLURM_JOB_ID}): compute nodes have" \
       "no network. Run this on the login node (gsai-login-3)." >&2
  exit 2
fi
AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AG_ENV_FILE="${AG_ENV_FILE:-${AG_ROOT}/env/ag.env}"
if [[ -f "${AG_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${AG_ENV_FILE}"
else
  echo "[ag_download] WARN: ${AG_ENV_FILE} not found (cp env/ag.env.example env/ag.env)" >&2
fi
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
# Classic HTTP downloads (no hf_xet), as slurm/s1_download_login.sh: xet stalled on gsai-login-3 and does
# not resume a killed shard; the classic path resumes <sha>.incomplete blobs. HF_HUB_DISABLE_XET=0 re-enables.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
: "${AG_SOURCES:?[ag_download] AG_SOURCES is not set (add it to env/ag.env, see env/ag.env.example)}"
export AG_SOURCES
AG_OUT="${AG_OUT:-/home/212aig08/ag_results}"
if [[ -x "${AG_VENV:-}/bin/python" ]]; then PY="${AG_VENV}/bin/python"; else PY="$(command -v python3)"; fi
cd "${AG_ROOT}"

for arg in "$@"; do
  if [[ "${arg}" == "--list" || "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    exec "${PY}" -m ag.data.download_ag "$@"
  fi
done

mkdir -p "${HF_HUB_CACHE}" "${AG_SOURCES}" "${AG_OUT}/logs"
echo "[ag_download] host=$(hostname) python=${PY} HF_HOME=${HF_HOME} sources=${AG_SOURCES}"
# nohup so a dropped SSH session does not kill a long transfer
LOG="${AG_OUT}/logs/ag_download_$(date +%m%d_%H%M%S).log"
nohup "${PY}" -m ag.data.download_ag "$@" > "${LOG}" 2>&1 &
echo "[ag_download] started in background, pid=$!  log: ${LOG}"
echo "              tail -f ${LOG}      (status: python agctl.py assets)"
