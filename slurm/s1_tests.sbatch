#!/bin/bash
# E0-1/2/3 + parser tests on one GPU (tiny model; < 5 min).  `python s1ctl.py tests --sbatch` generates the
# same script with the values from env/s1.env; this file is the hand-editable equivalent.
#SBATCH --job-name=s1_tests
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home3/212aig08/lg_s1_out/logs/s1_tests_%j.out
#SBATCH --error=/home3/212aig08/lg_s1_out/logs/s1_tests_%j.err
source "$(dirname "$0")/common.sh"
# LG_V04_ROOT (env/s1.env) wires the adapter to the v0.4 code when its mapping block is filled in
${PY} -m pytest tests/ -x -q --junitxml="${LG_OUT}/reports/pytest_e0.xml" 2>&1 | tee "${LG_OUT}/reports/pytest_e0.log"
rc=${PIPESTATUS[0]}
if [[ $rc -eq 0 ]]; then echo PASS > "${LG_OUT}/reports/pytest_e0.status"; else echo "FAIL rc=$rc" > "${LG_OUT}/reports/pytest_e0.status"; fi
exit $rc
