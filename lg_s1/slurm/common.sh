#!/bin/bash
# Sourced by every generated sbatch script (and usable interactively: `source slurm/common.sh`).
# Sets the environment from env/s1.env, forces HF offline mode on compute nodes, picks the python.
set -euo pipefail

S1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${S1_ROOT}/env/s1.env"
export LG_S1_ROOT="${S1_ROOT}"

# compute nodes have no network: never let transformers/datasets try the Hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# python: the v0.4 venv if present, else python3
if [[ -x "${LG_PYTHON:-}" ]]; then PY="${LG_PYTHON}"; else PY="$(command -v python3)"; fi
export PY

mkdir -p "${LG_OUT}/logs" "${LG_OUT}/reports" "${LG_OUT}/rows" "${LG_OUT}/slurm" "${LG_OUT}/zeroshot"
cd "${S1_ROOT}"

echo "[common] host=$(hostname) python=${PY} ($(${PY} -c 'import sys;print(sys.version.split()[0])'))"
echo "[common] HF_HOME=${HF_HOME} offline=${HF_HUB_OFFLINE} out=${LG_OUT}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi
${PY} - <<'EOF' || true
import torch, transformers
print(f"[common] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}")
EOF
