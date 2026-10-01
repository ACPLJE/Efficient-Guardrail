"""D1: verify the five asset paths of report §4.3 and write docs/assets_check_<date>.md (+ .json).

Checks (all read-only unless --seal):
  1. download_v1/sources.json and screened_v1/screen_summary.json  (exist, size, parse summary)
  2. lg_external_v01/audited/ + fetch_status.json                  (SafetyNIAH, LongSafety, HarmBench,
                                                                     Aegis2, XSTest, JBB present?)
  3. 0.6B run data/ symlink                                        (is symlink, target exists, size)
  4. Round-3 root                                                  (exists; --seal -> chmod -R a-w)
  5. v0.4 code + runs, external code, overlap.sqlite, HF cache     (sanity for the rest of S1)
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import S1Paths

EXTERNAL_SETS = ["safetyniah", "longsafety", "harmbench", "aegis2", "xstest", "jbb"]


@dataclass
class Item:
    name: str
    path: str
    ok: bool
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _size(p: Path) -> int:
    try:
        if p.is_dir():
            out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True, timeout=600)
            return int(out.stdout.split()[0]) if out.returncode == 0 else -1
        return p.stat().st_size
    except Exception:
        return -1


def _human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _json_summary(p: Path) -> str:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return f"unparseable JSON ({e})"
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return f"dict with {len(keys)} keys: {keys[:8]}"
    if isinstance(obj, list):
        return f"list of {len(obj)}"
    return type(obj).__name__


def check_file(name: str, p: Optional[Path], want_json: bool = False) -> Item:
    if p is None:
        return Item(name, "(unset)", False, "path not configured in env/s1.env")
    if not p.exists():
        return Item(name, str(p), False, "MISSING")
    st = p.stat()
    d = f"{_human(_size(p))}, mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}"
    if want_json and p.is_file():
        d += "; " + _json_summary(p)
    return Item(name, str(p), True, d)


def check_external(ext: Optional[Path]) -> List[Item]:
    items: List[Item] = []
    if ext is None or not ext.exists():
        return [Item("lg_external_v01", str(ext) if ext else "(unset)", False, "MISSING")]
    audited = ext / "audited"
    names = [x.name.lower() for x in audited.iterdir()] if audited.exists() else []
    status: Dict[str, Any] = {}
    fs = ext / "fetch_status.json"
    if fs.exists():
        try:
            status = json.loads(fs.read_text(encoding="utf-8"))
        except Exception as e:
            items.append(Item("fetch_status.json", str(fs), False, f"unparseable: {e}"))
    else:
        items.append(Item("fetch_status.json", str(fs), False, "MISSING"))
    flat = json.dumps(status).lower()
    for s in EXTERNAL_SETS:
        in_audited = any(s in n for n in names)
        in_status = s in flat
        ok = in_audited or in_status
        items.append(Item(f"external:{s}", str(audited), ok,
                          f"audited/={'yes' if in_audited else 'no'}, fetch_status={'yes' if in_status else 'no'}"))
    ov = ext / "overlap.sqlite"
    items.append(check_file("overlap.sqlite", ov))
    return items


def check_symlink(p: Optional[Path]) -> Item:
    if p is None:
        return Item("0.6B data symlink", "(unset)", False, "path not configured")
    if not p.exists() and not p.is_symlink():
        return Item("0.6B data symlink", str(p), False, "MISSING")
    if p.is_symlink():
        tgt = Path(os.readlink(p))
        tgt_abs = (p.parent / tgt).resolve()
        ok = tgt_abs.exists()
        return Item("0.6B data symlink", str(p), ok, f"-> {tgt} ({'exists' if ok else 'DANGLING'}), {_human(_size(tgt_abs)) if ok else ''}",
                    {"target": str(tgt_abs)})
    return Item("0.6B data symlink", str(p), True, f"not a symlink (real dir, {_human(_size(p))})")


def seal_readonly(p: Path) -> str:
    n = 0
    for root, dirs, files in os.walk(p):
        for x in dirs + files:
            fp = Path(root) / x
            try:
                mode = fp.stat().st_mode
                fp.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
                n += 1
            except OSError:
                pass
    try:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        pass
    return f"sealed read-only ({n} entries)"


def check_round3(p: Optional[Path], seal: bool) -> Item:
    if p is None:
        return Item("Round 3 root", "(unset)", False, "LG_ROUND3_ROOT empty: confirm the absolute path (results_r3/, data_clean/) and set it in env/s1.env")
    if not p.exists():
        return Item("Round 3 root", str(p), False, "MISSING")
    subs = [x.name for x in p.iterdir() if x.is_dir()]
    d = f"{_human(_size(p))}; subdirs {subs[:10]}"
    if seal:
        d += "; " + seal_readonly(p)
    else:
        writable = os.access(p, os.W_OK)
        d += f"; writable={'yes (run with --seal to freeze)' if writable else 'no (sealed)'}"
    return Item("Round 3 root", str(p), True, d)


def check_hf_cache(hf_home: Path) -> Item:
    hub = hf_home / "hub"
    if not hub.exists():
        return Item("HF cache", str(hub), False, "MISSING (HF_HOME not unified yet)")
    models = sorted(x.name.replace("models--", "").replace("--", "/") for x in hub.glob("models--*"))
    dsets = sorted(x.name.replace("datasets--", "").replace("--", "/") for x in hub.glob("datasets--*"))
    return Item("HF cache", str(hub), True, f"{len(models)} models, {len(dsets)} datasets",
                {"models": models, "datasets": dsets})


def run(seal: bool = False, date_tag: Optional[str] = None) -> Dict[str, Any]:
    P = S1Paths.load()
    P.ensure_out()
    items: List[Item] = []
    dv1 = P.data_v1
    items.append(check_file("download_v1/sources.json", dv1 / "download_v1" / "sources.json" if dv1 else None, want_json=True))
    items.append(check_file("download_v1/raw.jsonl.gz", dv1 / "download_v1" / "raw.jsonl.gz" if dv1 else None))
    items.append(check_file("screened_v1/screen_summary.json", dv1 / "screened_v1" / "screen_summary.json" if dv1 else None, want_json=True))
    items.append(check_file("conflict_clean_v1/raw.jsonl.gz", dv1 / "conflict_clean_v1" / "raw.jsonl.gz" if dv1 else None))
    items.extend(check_external(P.external_data))
    items.append(check_symlink(P.symlink_06b))
    items.append(check_round3(P.round3_root, seal))
    if P.v04_root:
        items.append(check_file("v0.4 code root", P.v04_root))
        items.append(check_file("v0.4 tests/test_correctness.py", P.v04_root / "tests" / "test_correctness.py"))
        items.append(check_file("v0.4 run_all.sh", P.v04_root / "run_all.sh"))
    else:
        items.append(Item("v0.4 code", "(LG_V04_ROOT empty)", True, "skipped — v0.4 is not reused; the new code base starts from lgs1/lgcore"))
    items.append(check_file("v0.4 runs root", P.v04_runs))
    items.append(check_file("essential06 results", P.v04_runs / "essential06_s0_v1" if P.v04_runs else None))
    items.append(check_file("external code root", P.external_code))
    items.append(check_hf_cache(P.hf_home))

    tag = date_tag or time.strftime("%m%d")
    md = P.out / "reports" / f"assets_check_{tag}.md"
    js = P.out / "reports" / f"assets_check_{tag}.json"
    lines = [f"# S1 · D1 자산 점검 ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "| 항목 | 경로 | 상태 | 상세 |", "| --- | --- | --- | --- |"]
    for it in items:
        lines.append(f"| {it.name} | `{it.path}` | {'OK' if it.ok else 'FAIL'} | {it.detail} |")
    n_ok = sum(1 for it in items if it.ok)
    lines += ["", f"OK {n_ok} / {len(items)}", "", "## 다음 행동", ""]
    for it in items:
        if not it.ok:
            lines.append(f"- [ ] {it.name}: {it.detail} (`{it.path}`)")
    if all(it.ok for it in items):
        lines.append("- 모든 경로 확인됨. v2 생성(S2–S3)은 download_v1·screened_v1을 재사용하고, Round 3는 논문에 쓰지 않음.")
    hf = next((it for it in items if it.name == "HF cache"), None)
    if hf and hf.extra.get("models"):
        lines += ["", "## HF 캐시에 있는 모델", ""] + [f"- {m}" for m in hf.extra["models"]]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    js.write_text(json.dumps([asdict(it) for it in items], indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {md}\n[written] {js}")
    return {"ok": n_ok, "total": len(items), "report": str(md)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="D1 asset check")
    ap.add_argument("--seal", action="store_true", help="chmod -R a-w the Round 3 root (irreversible without chmod +w)")
    ap.add_argument("--tag", default=None, help="report suffix (default MMDD)")
    a = ap.parse_args(argv)
    r = run(seal=a.seal, date_tag=a.tag)
    return 0 if r["ok"] == r["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
