#!/bin/bash
# Move the HF cache and the S1 outputs from /home3 (500 GB hard limit) to /home (6000 GB quota).
#   bash slurm/s1_relocate_home.sh            # dry-run: shows sizes, free space and the exact steps
#   bash slurm/s1_relocate_home.sh --apply    # performs them, then rewrites env/s1.env
# Steps (apply):
#   1. rsync  OLD_HF/  ->  NEW_HF/           (--ignore-existing: files already in NEW_HF are kept)
#   2. mv     OLD_HF   ->  OLD_HF.bak_<date> ; ln -s NEW_HF OLD_HF   (other projects keep working)
#   3. mv     OLD_OUT  ->  NEW_OUT           (reports made so far)
#   4. env/s1.env: HF_HOME, LG_OUT, LG_SOURCES -> the /home paths
# Nothing is deleted.  After checking, reclaim /home3 with:  rm -rf OLD_HF.bak_<date>
set -euo pipefail
S1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
U="${LG_USER:-${USER:-$(id -un)}}"
OLD_HF="${OLD_HF:-/home3/$U/.cache/huggingface}"
NEW_HF="${NEW_HF:-/home/$U/.cache/huggingface}"
OLD_OUT="${OLD_OUT:-/home3/$U/lg_s1_out}"
NEW_OUT="${NEW_OUT:-/home/$U/lg_s1_out}"
NEW_SRC="${NEW_SRC:-/home/$U/lg_data_v2/sources}"
APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1
STAMP="$(date +%m%d)"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }
size_of() { [[ -e "$1" ]] && du -sb "$1" 2>/dev/null | cut -f1 || echo 0; }

echo "== relocate (apply=$APPLY) =="
if [[ -L "$OLD_HF" ]]; then
  echo "OLD_HF $OLD_HF is already a symlink -> $(readlink "$OLD_HF"); nothing to move for the cache."
  HF_DONE=1
else
  HF_DONE=0
  OLD_SZ=$(size_of "$OLD_HF"); NEW_SZ=$(size_of "$NEW_HF")
  FREE=$(df -B1 --output=avail "$(dirname "$NEW_HF")" 2>/dev/null | tail -1 || echo 0)
  echo "OLD_HF  $OLD_HF   $(human "$OLD_SZ")"
  echo "NEW_HF  $NEW_HF   $(human "$NEW_SZ") (existing)"
  echo "free on $(dirname "$NEW_HF"): $(human "$FREE")   (user quota may be lower: quota -s)"
  if (( FREE < OLD_SZ )); then echo "ERROR: not enough free space on /home for the copy"; exit 2; fi
fi
echo "OLD_OUT $OLD_OUT -> NEW_OUT $NEW_OUT"
echo "NEW_SRC $NEW_SRC (created empty)"
echo
echo "steps:"
[[ $HF_DONE -eq 0 ]] && echo "  rsync -a --ignore-existing --info=progress2 $OLD_HF/ $NEW_HF/"
[[ $HF_DONE -eq 0 ]] && echo "  mv $OLD_HF $OLD_HF.bak_$STAMP && ln -s $NEW_HF $OLD_HF"
echo "  mv $OLD_OUT $NEW_OUT   (if OLD_OUT exists and NEW_OUT does not)"
echo "  python s1ctl.py env-set HF_HOME=$NEW_HF LG_OUT=$NEW_OUT LG_SOURCES=$NEW_SRC"
if [[ $APPLY -eq 0 ]]; then echo; echo "dry-run only. Re-run with --apply to execute."; exit 0; fi

if pgrep -u "$U" -f "lgs1.download" >/dev/null 2>&1; then
  echo "ERROR: a D3 download is still running (pgrep lgs1.download). Wait for it (tail -f \$LG_OUT/logs/download_*.log) or kill it, then rerun."; exit 4
fi
mkdir -p "$NEW_HF/hub" "$NEW_SRC"
if [[ $HF_DONE -eq 0 ]]; then
  echo "[1/4] rsync ($(human "$OLD_SZ")) ..."
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --ignore-existing --info=progress2 "$OLD_HF/" "$NEW_HF/"
  else
    echo "      (rsync not found: using cp -an)"; cp -an "$OLD_HF/." "$NEW_HF/"
  fi
  n_old=$(find "$OLD_HF" -type f | wc -l); n_new=$(find "$NEW_HF" -type f | wc -l)
  echo "      files: old=$n_old new=$n_new"
  if (( n_new < n_old )); then echo "ERROR: NEW_HF has fewer files than OLD_HF; not replacing OLD_HF"; exit 3; fi
  echo "[2/4] mv $OLD_HF -> $OLD_HF.bak_$STAMP ; symlink"
  mv "$OLD_HF" "$OLD_HF.bak_$STAMP"
  ln -s "$NEW_HF" "$OLD_HF"
fi
if [[ -L "$OLD_OUT" ]]; then
  echo "[3/4] $OLD_OUT is already a symlink -> $(readlink "$OLD_OUT")"
elif [[ -d "$OLD_OUT" && ! -e "$NEW_OUT" ]]; then
  echo "[3/4] mv $OLD_OUT -> $NEW_OUT ; symlink kept so pending jobs' log paths still resolve"
  mv "$OLD_OUT" "$NEW_OUT"; ln -s "$NEW_OUT" "$OLD_OUT"
elif [[ -d "$OLD_OUT" && -d "$NEW_OUT" ]]; then
  echo "[3/4] both exist; copying reports"; cp -an "$OLD_OUT/." "$NEW_OUT/"
else
  echo "[3/4] nothing to move for LG_OUT"; mkdir -p "$NEW_OUT"
fi
echo "[4/4] env/s1.env"
( cd "$S1_ROOT" && python3 s1ctl.py env-set "HF_HOME=$NEW_HF" "LG_OUT=$NEW_OUT" "LG_SOURCES=$NEW_SRC" )
echo
echo "done. When satisfied:  rm -rf $OLD_HF.bak_$STAMP    (frees $(human "${OLD_SZ:-0}") on /home3)"
echo "then:  python s1ctl.py quota   # /home3 leftovers now show as duplicates"
