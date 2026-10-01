#!/bin/bash
# E0-6: the 12 B1 guards as a job array on GPU B (one task per guard, sequential: %1).
# `python s1ctl.py zeroshot --sbatch` generates the same script with a guard list file; this is the
# hand-editable equivalent.  Re-running resumes (rows with a matching lineage are kept, E0-2).
#SBATCH --job-name=s1_zeroshot
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%1
#SBATCH --output=/home3/212aig08/lg_s1_out/logs/s1_zeroshot_%A_%a.out
#SBATCH --error=/home3/212aig08/lg_s1_out/logs/s1_zeroshot_%A_%a.err
source "$(dirname "$0")/common.sh"
GUARDS=(qwen3guard_gen_0.6b qwen3guard_gen_4b qwen3guard_gen_8b llamaguard3_8b wildguard_7b polyguard_qwen_7b
        nemotron_safety_guard_v3 granite_guardian_3.3_8b qwen3guard_stream_8b llamaguard4_12b gpt_oss_safeguard_20b
        guardreasoner_8b)
GUARD=${GUARDS[$SLURM_ARRAY_TASK_ID]}
echo "[s1] zero-shot guard=${GUARD}"
${PY} -m lgs1.zeroshot.run --guard "${GUARD}" --bench wgtest,toxicchat
${PY} -m lgs1.zeroshot.compare || true
