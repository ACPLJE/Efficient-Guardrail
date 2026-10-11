#!/bin/bash
# Sourced by every AG sbatch script (CLAUDE.md 4). Reads env/ag.env, forces HF offline mode on
# compute nodes, activates the single Python 3.11 venv (D-001). slurm/common.sh stays LG-only.
set -euo pipefail

AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AG_ENV_FILE="${AG_ENV_FILE:-${AG_ROOT}/env/ag.env}"
_ag_out_pre="${AG_OUT:-}"          # set by the job script (agctl --out); must win over env/ag.env
if [[ -f "${AG_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${AG_ENV_FILE}"
fi
if [[ -n "${_ag_out_pre}" ]]; then AG_OUT="${_ag_out_pre}"; fi

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-/home/212aig08/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export AG_OUT="${AG_OUT:-/home/212aig08/ag_results}"
AG_VENV="${AG_VENV:-/home/212aig08/ag_venv/ag311}"

if [[ -f "${AG_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${AG_VENV}/bin/activate"
  PY="${AG_VENV}/bin/python"
else
  echo "[ag_common] ERROR: venv ${AG_VENV} not found (set AG_VENV in env/ag.env)" >&2
  exit 1
fi
export PY
if ! "${PY}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "[ag_common] ERROR: ${PY} is not Python >= 3.11 (D-001)" >&2
  exit 1
fi

mkdir -p "${AG_OUT}/logs" "${AG_OUT}/raw"
cd "${AG_ROOT}"
echo "[ag_common] host=$(hostname) python=${PY} ($(${PY} -c 'import sys;print(sys.version.split()[0])'))"
echo "[ag_common] HF_HOME=${HF_HOME} offline=${HF_HUB_OFFLINE} AG_OUT=${AG_OUT}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi
