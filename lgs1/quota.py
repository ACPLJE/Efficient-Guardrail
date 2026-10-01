"""D2: quota usage, clean-up candidates, and HF cache unification check.

Default is a DRY RUN that writes reports/quota_<tag>.md.  Nothing is deleted unless you pass
`--apply-move` (moves candidates into $LG_OUT/_to_delete/<tag>/, reversible) or
`--delete --yes` (rm -rf, irreversible).

Candidates (report §4.1 "폐기"):
  * `.rows-*` partial files (immutable-conflict arms)            under LG_V04_RUNS
  * `*.tmp`, `*.partial`, `__pycache__`                          under LG_V04_RUNS / LG_OUT
  * duplicate HF snapshots: `models--*` present both in HF_HOME/hub and in another hub dir
    (e.g. /scratch/<user>/hf/hub, ~/.cache/huggingface on a different home) -> the non-unified copy
Large directories are only LISTED (top 25) for a manual decision.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paths import S1Paths, env

PARTIAL_GLOBS = [".rows-*", "*.rows-*", "*.tmp", "*.partial"]
OTHER_HUBS = ["/scratch/{user}/hf/hub", "/home/{user}/.cache/huggingface/hub", "/home3/{user}/.cache/huggingface/hub",
              "/scratch/{user}/huggingface/hub"]


def _du(path: Path) -> int:
    try:
        out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, timeout=3600)
        return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else -1
    except Exception:
        return -1


def _human(n: int) -> str:
    if n < 0:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


def quota_status(home: Path) -> str:
    """Try `quota -s`, then `df -h <home>`; return the raw text (quota tools differ per cluster)."""
    for cmd in (["quota", "-s"], ["lfs", "quota", "-h", str(home)], ["df", "-h", str(home)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                return f"$ {' '.join(cmd)}\n{out.stdout.strip()}"
        except Exception:
            continue
    return "(no quota/df output)"


def largest_dirs(root: Path, depth: int = 2, top: int = 25) -> List[Tuple[int, str]]:
    if not root.exists():
        return []
    try:
        out = subprocess.run(["du", "-b", f"--max-depth={depth}", str(root)], capture_output=True, text=True, timeout=3600)
        rows = []
        for line in out.stdout.splitlines():
            sz, p = line.split("\t", 1)
            rows.append((int(sz), p))
        rows.sort(reverse=True)
        return rows[:top]
    except Exception:
        return []


def find_partials(roots: List[Path]) -> List[Path]:
    found: List[Path] = []
    for r in roots:
        if not r or not r.exists():
            continue
        for g in PARTIAL_GLOBS:
            found.extend(p for p in r.rglob(g) if p.exists())
        found.extend(p for p in r.rglob("__pycache__") if p.is_dir())
    # dedupe, keep stable order
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_duplicate_hf(hf_hub: Path, user: str) -> List[Tuple[Path, Path]]:
    dups: List[Tuple[Path, Path]] = []
    if not hf_hub.exists():
        return dups
    mine = {p.name for p in hf_hub.glob("models--*")} | {p.name for p in hf_hub.glob("datasets--*")}
    for tmpl in OTHER_HUBS:
        other = Path(tmpl.format(user=user))
        if not other.exists() or other.resolve() == hf_hub.resolve():
            continue
        for p in list(other.glob("models--*")) + list(other.glob("datasets--*")):
            if p.name in mine:
                dups.append((p, hf_hub / p.name))
    return dups


def run(apply_move: bool = False, delete: bool = False, yes: bool = False, tag: Optional[str] = None,
        target_free_gb: float = 150.0) -> Dict:
    P = S1Paths.load()
    P.ensure_out()
    user = env("LG_USER", os.environ.get("USER", "user"))
    tag = tag or time.strftime("%m%d")
    home = P.quota_home
    lines = [f"# S1 · D2 쿼터 점검 ({time.strftime('%Y-%m-%d %H:%M')})", "", "```", quota_status(home), "```", ""]

    # usage of the roots we own
    roots = [x for x in (home, P.v04_runs, P.paper_root, P.external_data, P.hf_home, P.out) if x]
    lines += ["## 루트별 사용량", "", "| 경로 | 크기 |", "| --- | --- |"]
    sizes = {}
    for r in roots:
        s = _du(r) if r.exists() else -1
        sizes[str(r)] = s
        lines.append(f"| `{r}` | {_human(s)} |")

    # candidates
    partials = find_partials([p for p in (P.v04_runs, P.out) if p])
    dups = find_duplicate_hf(P.hf_home / "hub", user)
    cand: List[Tuple[Path, int, str]] = [(p, _du(p), "__pycache__" if p.name == "__pycache__" else "partial/.rows-*") for p in partials]
    cand += [(p, _du(p), f"duplicate of {q}") for p, q in dups]
    total = sum(max(0, s) for _, s, _ in cand)
    lines += ["", f"## 정리 후보 ({len(cand)}개, 회수 가능 {_human(total)})", "", "| 경로 | 크기 | 사유 |", "| --- | --- | --- |"]
    for p, s, why in cand[:200]:
        lines.append(f"| `{p}` | {_human(s)} | {why} |")

    # large dirs for manual decision
    lines += ["", "## 큰 디렉터리 (수동 판단용, depth 2)", "", "| 크기 | 경로 |", "| --- | --- |"]
    for s, p in largest_dirs(home, depth=2, top=25):
        lines.append(f"| {_human(s)} | `{p}` |")

    # HF unification
    lines += ["", "## HF 캐시 통일", ""]
    lines.append(f"- HF_HOME = `{P.hf_home}` ({'exists' if P.hf_home.exists() else 'MISSING -> mkdir -p'})")
    scratch = Path(f"/scratch/{user}/hf")
    if scratch.exists():
        lines.append(f"- `/scratch/{user}/hf` 존재: 로그인 노드 로컬 디스크라 계산 노드에서 안 보임. 필요한 스냅샷은 HF_HOME으로 옮기고 나머지는 후보로 처리")
    lines.append("- 모든 sbatch 스크립트는 `slurm/common.sh`가 HF_HOME/HF_HUB_OFFLINE을 설정함")

    # actions
    action = "dry-run"
    moved = []
    if cand and (apply_move or delete):
        if delete and not yes:
            lines.append("\n`--delete` requires `--yes`; nothing deleted.")
        else:
            dest = P.out / "_to_delete" / tag
            for p, s, why in cand:
                try:
                    if delete:
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
                        moved.append((str(p), "deleted"))
                    else:
                        rel = Path(str(p).lstrip("/"))
                        d = dest / rel
                        d.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(d))
                        moved.append((str(p), str(d)))
                except Exception as e:
                    moved.append((str(p), f"FAILED: {e}"))
            action = "delete" if delete else f"move -> {dest}"
            lines += ["", f"## 적용 ({action})", ""] + [f"- {a} -> {b}" for a, b in moved]

    free_hint = f"목표 여유 {target_free_gb:.0f}G (S3 합성 6–8 GB gzip + 체크포인트 + rows). 부족하면 큰 디렉터리 표에서 구 결과를 골라 후보에 추가."
    lines += ["", free_hint]
    md = P.out / "reports" / f"quota_{tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (P.out / "reports" / f"quota_{tag}.json").write_text(json.dumps(
        {"sizes": sizes, "candidates": [(str(p), s, w) for p, s, w in cand], "reclaimable": total, "action": action,
         "moved": moved}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {md}")
    return {"candidates": len(cand), "reclaimable": total, "action": action}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="D2 quota check / clean-up (dry-run by default)")
    ap.add_argument("--apply-move", action="store_true", help="move candidates into $LG_OUT/_to_delete/<tag>/")
    ap.add_argument("--delete", action="store_true", help="rm -rf candidates (needs --yes)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--target-free-gb", type=float, default=150.0)
    a = ap.parse_args(argv)
    run(a.apply_move, a.delete, a.yes, a.tag, a.target_free_gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
