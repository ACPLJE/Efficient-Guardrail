#!/usr/bin/env bash
# Local pre-PR gate (CLAUDE.md 3.4; build plan 5): what .github/workflows/cpu.yml runs, plus ruff (pyflakes level,
# AG code only; LG code is fixed in separate PRs, CLAUDE.md 3.7), merge-conflict markers and absolute /home/ paths
# in AG Python code. Every step runs; one PASS/FAIL/SKIP line per step; exit 1 when any step failed.
#
#   bash tools/pr_check.sh
#   PYTHON=/home/212aig08/ag_venv/ag311/bin/python bash tools/pr_check.sh
#
# Interpreter: $PYTHON, else the active venv, else $AG_VENV/bin/python, else python / python3 (must be >= 3.11, D-001).
# ruff: $RUFF, else ruff on PATH, else `$PY -m ruff`; skipped with a note when none is installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${PYTHON:-}" ]]; then PY="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then PY="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${AG_VENV:-}" && -x "${AG_VENV}/bin/python" ]]; then PY="${AG_VENV}/bin/python"
elif command -v python >/dev/null 2>&1; then PY="$(command -v python)"
else PY="python3"; fi
if ! "${PY}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
  echo "[pr_check] ERROR: '${PY}' is not Python >= 3.11 (D-001); set PYTHON= or activate the ag311 venv" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false     # same env as cpu.yml

RESULTS=()
FAILED=0
record() {   # record <PASS|FAIL|SKIP> <name> <detail>
  local line
  line="$(printf '%-4s  %-20s %s' "$1" "$2" "$3")"
  RESULTS+=("${line}")
  echo "[pr_check] ${line}"
  if [[ "$1" == FAIL ]]; then FAILED=1; fi
}
step() {     # step <name> <command...>: runs the command, records PASS/FAIL with its exit code and duration
  local name="$1" t0="${SECONDS}" rc=0
  shift
  echo "[pr_check] >>> ${name}: $*"
  "$@" || rc=$?
  if (( rc == 0 )); then record PASS "${name}" "($(( SECONDS - t0 ))s)"
  else record FAIL "${name}" "(exit ${rc}, $(( SECONDS - t0 ))s)"; fi
}

in_git() { git rev-parse --is-inside-work-tree >/dev/null 2>&1; }
grep_none() {   # grep_none <message> <git grep args...>: fails when git grep finds a match
  local msg="$1" rc=0
  shift
  git grep --untracked -nI "$@" || rc=$?
  case "${rc}" in
    0) echo "${msg}" >&2; return 1 ;;
    1) return 0 ;;
    *) return "${rc}" ;;
  esac
}
conflict_markers() { grep_none "merge conflict markers found" -E '^(<{7}|>{7})( |$)' -- . ; }
abs_home_paths() { grep_none "absolute /home/ paths in AG code (read \$AG_OUT / \$AG_SOURCES / \$HF_HOME instead)" \
  -F '/home/' -- 'ag/*.py' 'plots/*.py' ; }

RUFF_CMD=()
if [[ -n "${RUFF:-}" ]]; then RUFF_CMD=("${RUFF}")
elif command -v ruff >/dev/null 2>&1; then RUFF_CMD=("$(command -v ruff)")
elif "${PY}" -m ruff --version >/dev/null 2>&1; then RUFF_CMD=("${PY}" -m ruff); fi
RUFF_PATHS=(ag plots tools scripts agctl.py tests/_ag_util.py tests/conftest.py tests/gpu)
for f in tests/test_ag_*.py; do if [[ -e "${f}" ]]; then RUFF_PATHS+=("${f}"); fi; done

echo "[pr_check] repo=${ROOT} python=${PY} ($("${PY}" -c 'import sys; print(sys.version.split()[0])'))"

step secrets "${PY}" tools/check_secrets.py --untracked
step test-classification "${PY}" tools/check_test_classification.py
if in_git; then
  step conflict-markers conflict_markers
  step abs-home-paths abs_home_paths
else
  record SKIP conflict-markers "(not a git checkout)"
  record SKIP abs-home-paths "(not a git checkout)"
fi
if (( ${#RUFF_CMD[@]} )); then
  step ruff "${RUFF_CMD[@]}" check --select F,E9 --output-format concise "${RUFF_PATHS[@]}"
else
  record SKIP ruff "(ruff not installed: pip install ruff)"
fi
step pytest-all "${PY}" -m pytest tests -x -q
# shellcheck disable=SC2046  # the manifest is a whitespace-separated list of test files (CLAUDE.md 3.4)
step pytest-cpu-manifest "${PY}" -m pytest $(cat tests/manifests/cpu.txt)

echo "[pr_check] ---- summary ----"
for line in "${RESULTS[@]}"; do echo "[pr_check] ${line}"; done
if (( FAILED )); then echo "[pr_check] FAIL"; exit 1; fi
echo "[pr_check] PASS"
